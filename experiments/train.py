#!/usr/bin/env python3
"""
Unified entry point for symmetry-regularized dynamics experiments.

Wraps existing in_progress/ infrastructure with Hydra config dispatch,
supporting sweep, consensus, and multi-animal experiment types.

Usage:
    python experiments/train.py +experiment=regularizer_comparison
    python experiments/train.py +experiment=consensus training.device=cuda
    python experiments/train.py +experiment=sphere_so3 training.n_epochs=2
"""

import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer

import hydra
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Path setup: make in_progress/ and xfads importable without modifying them
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_PROGRESS = PROJECT_ROOT / "in_progress"
sys.path.insert(0, str(IN_PROGRESS))
sys.path.insert(0, str(PROJECT_ROOT))

import xfads.utils as utils
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL

from regularized_ssm import RegularizedSSM, RegularizedLightningSSM
from symmetry_examples import rotation_symmetry, translation_symmetry, scaling_symmetry
from test_dynamics import PerturbedRingAttractorRNN
from sphere_dynamics import PerturbedSphereAttractorRNN, create_so3_generators, create_combined_so3_vector_field

log = logging.getLogger(__name__)


# ============================================================================
# Config bridge: hierarchical YAML -> flat OmegaConf expected by existing code
# ============================================================================

def build_flat_cfg(cfg: DictConfig) -> DictConfig:
    """
    Bridge between our hierarchical config and the flat cfg that existing
    in_progress/ functions expect (e.g. cfg.n_latents, cfg.batch_sz).

    Why: The upstream XFADS Lightning trainer reads cfg.n_samples, cfg.lr, etc.
    as flat attributes. Rather than modifying upstream code, we produce a flat
    DictConfig that satisfies all access patterns.
    """
    flat = OmegaConf.create({
        # Model
        "n_latents": cfg.model.n_latents,
        "n_latents_read": cfg.model.n_latents_read,
        "rank_local": cfg.model.rank_local,
        "rank_backward": cfg.model.rank_backward,
        "n_hidden_dynamics": cfg.model.n_hidden_dynamics,
        "n_samples": cfg.model.n_samples,
        "n_hidden_local": cfg.model.n_hidden_local,
        "n_hidden_backward": cfg.model.n_hidden_backward,
        # Inference
        "use_cd": cfg.inference.use_cd,
        "p_mask_a": cfg.inference.p_mask_a,
        "p_mask_b": cfg.inference.p_mask_b,
        "p_mask_apb": cfg.inference.p_mask_apb,
        "p_mask_y_in": cfg.inference.p_mask_y_in,
        "p_local_dropout": cfg.inference.p_local_dropout,
        "p_backward_dropout": cfg.inference.p_backward_dropout,
        # Training
        "lr": cfg.training.lr,
        "n_epochs": cfg.training.n_epochs,
        "batch_sz": cfg.training.batch_sz,
        "lr_gamma_decay": cfg.training.lr_gamma_decay,
        "patience": cfg.training.patience,
        "grad_clip": cfg.training.grad_clip,
        # Device (resolved below)
        "device": cfg.training.device,
        "data_device": cfg.training.device,
    })
    return flat


def resolve_device(cfg: DictConfig) -> str:
    """Resolve 'auto' device to concrete device string."""
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


# ============================================================================
# Symmetry vector field factory
# ============================================================================

def build_symmetry_field(sym_cfg: dict):
    """
    Build a target vector field from a symmetry config dict.

    Args:
        sym_cfg: dict with 'type' and 'params' keys.

    Returns:
        Callable vector field g(x), or None for rotation_invariance type.

    The 'rotation_invariance' type is handled separately in training since it
    uses the RotationInvarianceRegularizer rather than a target vector field.
    """
    sym_type = sym_cfg["type"]
    params = sym_cfg.get("params", {})

    if sym_type == "rotation":
        return rotation_symmetry(speed=params.get("speed", 1.0))
    elif sym_type == "translation":
        direction = torch.tensor(params["direction"], dtype=torch.float32)
        return translation_symmetry(direction)
    elif sym_type == "scaling":
        return scaling_symmetry(rate=params.get("rate", 1.0))
    elif sym_type == "so3":
        weights = params.get("weights", [1.0, 1.0, 1.0])
        return create_combined_so3_vector_field(torch.tensor(weights))
    elif sym_type == "rotation_invariance":
        # Handled via lambda_rotation, not a target vector field
        return None
    else:
        raise ValueError(f"Unknown symmetry type: {sym_type}")


# ============================================================================
# Data generation
# ============================================================================

def generate_ring_data(flat_cfg, cfg):
    """Generate ring attractor data with perturbations."""
    device = flat_cfg.device
    n_trials = cfg.data.n_trials
    n_neurons = cfg.data.n_neurons
    n_time_bins = cfg.data.n_time_bins

    # Isolate RNG state from perturbation sampling
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()

    mean_fn = PerturbedRingAttractorRNN(
        bin_sz=cfg.dynamics.bin_sz,
        lengthscale=cfg.dynamics.lengthscale,
        perturbation_magnitude=cfg.dynamics.perturbation_magnitude,
    ).to(device)

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(np_state)

    C = utils.FanInLinear(flat_cfg.n_latents, n_neurons, device=device).requires_grad_(False)
    Q_diag = cfg.data.Q_diag * torch.ones(flat_cfg.n_latents, device=device)
    Q_0_diag = cfg.data.Q_0_diag * torch.ones(flat_cfg.n_latents, device=device)
    R_diag = cfg.data.R_diag * torch.ones(n_neurons, device=device)
    m_0 = torch.zeros(flat_cfg.n_latents, device=device)

    z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
    y = C(z) + torch.sqrt(R_diag) * torch.randn(
        (n_trials, n_time_bins, n_neurons), device=device
    )
    y = y.detach()

    return y, z, mean_fn, C, Q_diag, Q_0_diag, R_diag, m_0


def generate_sphere_data(flat_cfg, cfg):
    """Generate 3D sphere attractor data."""
    device = flat_cfg.device
    n_trials = cfg.data.n_trials
    n_neurons = cfg.data.n_neurons
    n_time_bins = cfg.data.n_time_bins

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()

    mean_fn = PerturbedSphereAttractorRNN(
        bin_sz=cfg.dynamics.bin_sz,
        perturbation_magnitude=cfg.dynamics.perturbation_magnitude,
        lengthscale=cfg.dynamics.lengthscale,
    ).to(device)

    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(np_state)

    C = utils.FanInLinear(flat_cfg.n_latents, n_neurons, device=device).requires_grad_(False)
    Q_diag = cfg.data.Q_diag * torch.ones(flat_cfg.n_latents, device=device)
    Q_0_diag = cfg.data.Q_0_diag * torch.ones(flat_cfg.n_latents, device=device)
    R_diag = cfg.data.R_diag * torch.ones(n_neurons, device=device)
    m_0 = torch.zeros(flat_cfg.n_latents, device=device)

    z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
    y = C(z) + torch.sqrt(R_diag) * torch.randn(
        (n_trials, n_time_bins, n_neurons), device=device
    )
    y = y.detach()

    return y, z, mean_fn, C, Q_diag, Q_0_diag, R_diag, m_0


def generate_data(flat_cfg, cfg):
    """Dispatch to correct data generator based on dynamics type."""
    if cfg.dynamics.type == "ring":
        return generate_ring_data(flat_cfg, cfg)
    elif cfg.dynamics.type == "sphere":
        return generate_sphere_data(flat_cfg, cfg)
    else:
        raise ValueError(f"Unknown dynamics type: {cfg.dynamics.type}")


# ============================================================================
# Dataloaders
# ============================================================================

def create_dataloaders(y, flat_cfg, train_split=0.667):
    """Create train/validation dataloaders (mirrors regularized_experiments.py)."""
    device = flat_cfg.device

    def collate_fn(batch):
        elem = batch[0]
        if isinstance(elem, (tuple, list)):
            return tuple(torch.stack([b[i] for b in batch]).to(device) for i in range(len(elem)))
        return torch.stack(batch).to(device)

    n_trials = y.shape[0]
    split_idx = int(train_split * n_trials)
    y_train, y_valid = y[:split_idx], y[split_idx:]

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_train),
        batch_size=flat_cfg.batch_sz,
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_valid),
        batch_size=flat_cfg.batch_sz,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return train_loader, valid_loader, y_train, y_valid


# ============================================================================
# Model creation
# ============================================================================

def create_model_components(flat_cfg, n_neurons, C, Q_diag, Q_0_diag, R_diag, m_0):
    """Create all model components (mirrors regularized_experiments.py)."""
    device = flat_cfg.device

    H = utils.ReadoutLatentMask(flat_cfg.n_latents, flat_cfg.n_latents_read)
    readout_fn = nn.Sequential(H, C)
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=device, fix_R=True)

    dynamics_fn = utils.build_gru_dynamics_function(
        flat_cfg.n_latents, flat_cfg.n_hidden_dynamics, device=device
    )
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, flat_cfg.n_latents, Q_diag, device=device)
    initial_condition_pdf = DenseGaussianInitialCondition(
        flat_cfg.n_latents, m_0, Q_0_diag, device=device
    )

    backward_encoder = BackwardEncoderLRMvn(
        flat_cfg.n_latents, flat_cfg.n_hidden_backward, flat_cfg.n_latents,
        rank_local=flat_cfg.rank_local, rank_backward=flat_cfg.rank_backward, device=device,
    )
    local_encoder = LocalEncoderLRMvn(
        flat_cfg.n_latents, n_neurons, flat_cfg.n_hidden_local, flat_cfg.n_latents,
        rank=flat_cfg.rank_local, device=device, dropout=flat_cfg.p_local_dropout,
    )
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=device)

    return {
        "likelihood_pdf": likelihood_pdf,
        "dynamics_mod": dynamics_mod,
        "initial_condition_pdf": initial_condition_pdf,
        "backward_encoder": backward_encoder,
        "local_encoder": local_encoder,
        "nl_filter": nl_filter,
    }


# ============================================================================
# Single training run
# ============================================================================

def train_single(
    flat_cfg,
    cfg,
    y,
    C,
    Q_diag,
    Q_0_diag,
    R_diag,
    m_0,
    target_vector_field,
    lambda_lie,
    lambda_curvature,
    lambda_rotation,
    run_dir: Path,
):
    """
    Train a single model with specified regularization parameters.

    Returns a dict with final metrics and timing.
    """
    device = flat_cfg.device
    n_neurons = cfg.data.n_neurons

    # Build model
    components = create_model_components(flat_cfg, n_neurons, C, Q_diag, Q_0_diag, R_diag, m_0)

    lie_normalize = cfg.regularization.lie_normalize
    if lie_normalize == "false" or lie_normalize is False:
        lie_normalize = False

    regularized_ssm = RegularizedSSM(
        dynamics_mod=components["dynamics_mod"],
        likelihood_pdf=components["likelihood_pdf"],
        initial_c_pdf=components["initial_condition_pdf"],
        backward_encoder=components["backward_encoder"],
        local_encoder=components["local_encoder"],
        nl_filter=components["nl_filter"],
        target_vector_field=target_vector_field,
        lambda_lie=lambda_lie,
        lambda_curvature=lambda_curvature,
        lambda_rotation=lambda_rotation,
        lie_normalize=lie_normalize,
        curvature_order=cfg.regularization.curvature_order,
        n_rotations=cfg.regularization.n_rotations,
        device=device,
    )

    # Dataloaders
    train_loader, valid_loader, y_train, y_valid = create_dataloaders(
        y, flat_cfg, train_split=cfg.data.train_split
    )

    # Lightning wrapper + trainer
    seq_vae = RegularizedLightningSSM(regularized_ssm, flat_cfg)

    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    ckpt_dir = run_dir / "ckpts"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    csv_logger = CSVLogger(str(log_dir), name="", version="")
    ckpt_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="valid_loss",
        mode="min",
        dirpath=str(ckpt_dir),
        filename="{epoch:02d}_{valid_loss:.4f}",
    )
    early_stop = EarlyStopping(
        monitor="valid_elbo",
        min_delta=0.001,
        patience=flat_cfg.patience,
        verbose=True,
        mode="min",
    )
    timer = Timer()

    trainer = pl.Trainer(
        max_epochs=flat_cfg.n_epochs,
        gradient_clip_val=flat_cfg.grad_clip,
        default_root_dir=str(run_dir),
        callbacks=[ckpt_callback, timer, early_stop],
        accelerator=device,
        logger=csv_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    t0 = time.time()
    trainer.fit(model=seq_vae, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    wall_time = time.time() - t0

    # Gather final validation metrics
    best_path = ckpt_callback.best_model_path
    final_valid_loss = ckpt_callback.best_model_score
    if final_valid_loss is not None:
        final_valid_loss = final_valid_loss.item()

    # Evaluate final loss components
    # Ensure model and data are on the same device after Lightning training
    regularized_ssm.to(device)
    regularized_ssm.eval()
    with torch.no_grad():
        y_valid_device = y_valid.to(device)
        total_loss, _, stats = regularized_ssm(y_valid_device, flat_cfg.n_samples)

    result = {
        "lambda_lie": lambda_lie,
        "lambda_curvature": lambda_curvature,
        "lambda_rotation": lambda_rotation,
        "final_valid_loss": final_valid_loss,
        "final_total_loss": stats["total_loss"].item(),
        "final_elbo_loss": stats["elbo_loss"].item(),
        "final_lie_loss": stats["lie_loss"].item(),
        "final_curvature_loss": stats["curvature_loss"].item(),
        "final_rotation_loss": stats["rotation_loss"].item(),
        "best_ckpt": best_path,
        "wall_time_s": wall_time,
        "epochs_trained": trainer.current_epoch,
    }

    log.info(
        f"  Done: ELBO={result['final_elbo_loss']:.4f}, "
        f"Lie={result['final_lie_loss']:.4f}, "
        f"time={wall_time:.1f}s"
    )
    return result


# ============================================================================
# Experiment dispatchers
# ============================================================================

def run_sweep(cfg: DictConfig, flat_cfg: DictConfig):
    """
    Sweep experiment: iterate over lambda values and symmetries.

    For each (symmetry, lambda_lie) pair, trains a model and records results.
    """
    device = flat_cfg.device
    sweep_cfg = cfg.sweep
    lambda_values = list(sweep_cfg.lambda_lie)
    symmetries = OmegaConf.to_container(sweep_cfg.symmetries, resolve=True)
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Sweep: {len(symmetries)} symmetries x {len(lambda_values)} lambda values")

    # Seed and generate data (shared across all runs in the sweep)
    pl.seed_everything(cfg.training.seed, workers=True)
    y, z, mean_fn, C, Q_diag, Q_0_diag, R_diag, m_0 = generate_data(flat_cfg, cfg)

    all_results = []

    for sym_cfg in symmetries:
        sym_name = sym_cfg["name"]
        target_field = build_symmetry_field(sym_cfg)

        for lam in lambda_values:
            run_name = f"{sym_name}_lambda_{lam:.2e}"
            run_dir = output_dir / run_name
            log.info(f"--- {run_name} ---")

            # Reset seed for reproducibility across runs
            pl.seed_everything(cfg.training.seed, workers=True)

            # Determine which lambda to set based on symmetry type
            lambda_lie = lam if sym_cfg["type"] != "rotation_invariance" else 0.0
            lambda_rotation = lam if sym_cfg["type"] == "rotation_invariance" else 0.0

            result = train_single(
                flat_cfg, cfg, y, C, Q_diag, Q_0_diag, R_diag, m_0,
                target_vector_field=target_field,
                lambda_lie=lambda_lie,
                lambda_curvature=cfg.regularization.lambda_curvature,
                lambda_rotation=lambda_rotation,
                run_dir=run_dir,
            )
            result["symmetry"] = sym_name
            all_results.append(result)

    # Save summary CSV
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "sweep_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Sweep results saved to {csv_path}")
    return df


def run_consensus(cfg: DictConfig, flat_cfg: DictConfig):
    """
    Consensus experiment: pool data from multiple perturbation seeds
    (same C matrix) to show regularization recovers the perfect attractor.

    Key idea: each seed produces a different GP perturbation realization.
    Pooling the data averages out perturbations, and regularization pushes
    the learned dynamics toward the clean ring attractor.
    """
    device = flat_cfg.device
    consensus_cfg = cfg.consensus
    seeds = list(consensus_cfg.perturbation_seeds)
    sweep_cfg = cfg.sweep
    lambda_values = list(sweep_cfg.lambda_lie)
    symmetries = OmegaConf.to_container(sweep_cfg.symmetries, resolve=True)
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Consensus: pooling {len(seeds)} perturbation seeds")

    # Generate shared C matrix with the main seed
    pl.seed_everything(cfg.training.seed, workers=True)
    n_neurons = cfg.data.n_neurons
    C = utils.FanInLinear(flat_cfg.n_latents, n_neurons, device=device).requires_grad_(False)

    Q_diag = cfg.data.Q_diag * torch.ones(flat_cfg.n_latents, device=device)
    Q_0_diag = cfg.data.Q_0_diag * torch.ones(flat_cfg.n_latents, device=device)
    R_diag = cfg.data.R_diag * torch.ones(n_neurons, device=device)
    m_0 = torch.zeros(flat_cfg.n_latents, device=device)

    # Generate data per seed and pool
    all_y = []
    for seed in seeds:
        torch.manual_seed(seed)
        mean_fn = PerturbedRingAttractorRNN(
            bin_sz=cfg.dynamics.bin_sz,
            lengthscale=cfg.dynamics.lengthscale,
            perturbation_magnitude=cfg.dynamics.perturbation_magnitude,
        ).to(device)

        z_i = utils.sample_gauss_z(
            mean_fn, Q_diag, m_0, Q_0_diag,
            cfg.data.n_trials, cfg.data.n_time_bins,
        )
        y_i = C(z_i) + torch.sqrt(R_diag) * torch.randn(
            (cfg.data.n_trials, cfg.data.n_time_bins, n_neurons), device=device,
        )
        all_y.append(y_i.detach())
        log.info(f"  Generated {cfg.data.n_trials} trials with seed {seed}")

    y_pooled = torch.cat(all_y, dim=0)
    log.info(f"  Pooled data: {y_pooled.shape[0]} total trials")

    # Sweep over lambda values
    all_results = []
    for sym_cfg in symmetries:
        sym_name = sym_cfg["name"]
        target_field = build_symmetry_field(sym_cfg)

        for lam in lambda_values:
            run_name = f"consensus_{sym_name}_lambda_{lam:.2e}"
            run_dir = output_dir / run_name
            log.info(f"--- {run_name} ---")

            pl.seed_everything(cfg.training.seed, workers=True)

            result = train_single(
                flat_cfg, cfg, y_pooled, C, Q_diag, Q_0_diag, R_diag, m_0,
                target_vector_field=target_field,
                lambda_lie=lam,
                lambda_curvature=cfg.regularization.lambda_curvature,
                lambda_rotation=0.0,
                run_dir=run_dir,
            )
            result["symmetry"] = sym_name
            result["n_seeds_pooled"] = len(seeds)
            all_results.append(result)

    df = pd.DataFrame(all_results)
    csv_path = output_dir / "consensus_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Consensus results saved to {csv_path}")
    return df


def run_multi_animal(cfg: DictConfig, flat_cfg: DictConfig):
    """
    Multi-animal experiment: different C (observation) matrices per animal,
    same underlying dynamics.

    Two strategies:
    - shared_dynamics: Pool all observations, train one model with shared dynamics.
    - independent: Train a separate model per animal.
    """
    device = flat_cfg.device
    ma_cfg = cfg.multi_animal
    animal_seeds = list(ma_cfg.animal_seeds)
    n_animals = ma_cfg.n_animals
    strategy = ma_cfg.strategy
    n_trials_per_animal = ma_cfg.n_trials_per_animal
    sweep_cfg = cfg.sweep
    lambda_values = list(sweep_cfg.lambda_lie)
    symmetries = OmegaConf.to_container(sweep_cfg.symmetries, resolve=True)
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Multi-animal: {n_animals} animals, strategy={strategy}")

    n_neurons = cfg.data.n_neurons

    # Shared dynamics (same perturbation realization)
    pl.seed_everything(cfg.training.seed, workers=True)
    mean_fn = PerturbedRingAttractorRNN(
        bin_sz=cfg.dynamics.bin_sz,
        lengthscale=cfg.dynamics.lengthscale,
        perturbation_magnitude=cfg.dynamics.perturbation_magnitude,
    ).to(device)

    Q_diag = cfg.data.Q_diag * torch.ones(flat_cfg.n_latents, device=device)
    Q_0_diag = cfg.data.Q_0_diag * torch.ones(flat_cfg.n_latents, device=device)
    R_diag = cfg.data.R_diag * torch.ones(n_neurons, device=device)
    m_0 = torch.zeros(flat_cfg.n_latents, device=device)

    # Generate shared latents
    z = utils.sample_gauss_z(
        mean_fn, Q_diag, m_0, Q_0_diag,
        n_trials_per_animal, cfg.data.n_time_bins,
    )

    # Generate observations per animal with different C matrices
    per_animal_y = []
    per_animal_C = []
    for i, a_seed in enumerate(animal_seeds[:n_animals]):
        torch.manual_seed(a_seed)
        C_i = utils.FanInLinear(flat_cfg.n_latents, n_neurons, device=device).requires_grad_(False)
        y_i = C_i(z) + torch.sqrt(R_diag) * torch.randn(
            (n_trials_per_animal, cfg.data.n_time_bins, n_neurons), device=device,
        )
        per_animal_y.append(y_i.detach())
        per_animal_C.append(C_i)
        log.info(f"  Animal {i}: {n_trials_per_animal} trials (C seed={a_seed})")

    all_results = []

    for sym_cfg in symmetries:
        sym_name = sym_cfg["name"]
        target_field = build_symmetry_field(sym_cfg)

        for lam in lambda_values:
            if strategy == "shared_dynamics":
                # Pool all animals, use C from animal 0 as the readout
                # (the model learns a single dynamics shared across animals)
                y_pooled = torch.cat(per_animal_y, dim=0)
                C_shared = per_animal_C[0]

                run_name = f"multi_shared_{sym_name}_lambda_{lam:.2e}"
                run_dir = output_dir / run_name
                log.info(f"--- {run_name} (pooled {n_animals} animals) ---")

                pl.seed_everything(cfg.training.seed, workers=True)
                result = train_single(
                    flat_cfg, cfg, y_pooled, C_shared, Q_diag, Q_0_diag, R_diag, m_0,
                    target_vector_field=target_field,
                    lambda_lie=lam,
                    lambda_curvature=cfg.regularization.lambda_curvature,
                    lambda_rotation=0.0,
                    run_dir=run_dir,
                )
                result["symmetry"] = sym_name
                result["strategy"] = "shared_dynamics"
                result["n_animals"] = n_animals
                all_results.append(result)

            elif strategy == "independent":
                # Train separate models per animal
                for i in range(n_animals):
                    run_name = f"multi_ind_animal{i}_{sym_name}_lambda_{lam:.2e}"
                    run_dir = output_dir / run_name
                    log.info(f"--- {run_name} ---")

                    pl.seed_everything(cfg.training.seed, workers=True)
                    result = train_single(
                        flat_cfg, cfg, per_animal_y[i], per_animal_C[i],
                        Q_diag, Q_0_diag, R_diag, m_0,
                        target_vector_field=target_field,
                        lambda_lie=lam,
                        lambda_curvature=cfg.regularization.lambda_curvature,
                        lambda_rotation=0.0,
                        run_dir=run_dir,
                    )
                    result["symmetry"] = sym_name
                    result["strategy"] = "independent"
                    result["animal_id"] = i
                    all_results.append(result)
            else:
                raise ValueError(f"Unknown multi-animal strategy: {strategy}")

    df = pd.DataFrame(all_results)
    csv_path = output_dir / "multi_animal_results.csv"
    df.to_csv(csv_path, index=False)
    log.info(f"Multi-animal results saved to {csv_path}")
    return df


# ============================================================================
# Hydra entry point
# ============================================================================

@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="base",
)
def main(cfg: DictConfig):
    """
    Unified experiment entry point.

    Hydra loads configs/base.yaml merged with any +experiment=... override.
    We resolve device, build the flat config bridge, save the resolved config,
    and dispatch to the appropriate experiment runner.
    """
    log.info(f"Experiment type: {cfg.experiment_type}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Resolve device
    device = resolve_device(cfg)
    cfg.training.device = device
    log.info(f"Device: {device}")

    # Seed
    pl.seed_everything(cfg.training.seed, workers=True)
    torch.set_default_dtype(torch.float32)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Disable cuDNN for GRU - the fused CUDA GRU cell doesn't support
        # forward-mode autodiff (torch.func.jvp) used by the Lie regularizer
        torch.backends.cudnn.enabled = False

    # Build flat config for existing in_progress/ functions
    flat_cfg = build_flat_cfg(cfg)

    # Save resolved config (Hydra changes cwd, so output is relative to run dir)
    output_dir = Path(cfg.output.base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    log.info(f"Resolved config saved to {config_path}")

    # Dispatch
    if cfg.experiment_type == "sweep":
        run_sweep(cfg, flat_cfg)
    elif cfg.experiment_type == "consensus":
        run_consensus(cfg, flat_cfg)
    elif cfg.experiment_type == "multi_animal":
        run_multi_animal(cfg, flat_cfg)
    else:
        raise ValueError(f"Unknown experiment_type: {cfg.experiment_type}")


if __name__ == "__main__":
    main()
