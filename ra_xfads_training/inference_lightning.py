import os
import random

import numpy as np
from copy import deepcopy
from typing import Dict, Any
import tempfile
import shutil


import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xfads.utils as utils
from hydra import compose, initialize
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from pytorch_lightning.loggers import CSVLogger
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL, LowRankNonlinearStateSpaceModel
from xfads.ssm_modules.dynamics import DenseGaussianDynamics, DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.ssm_modules.likelihoods import GaussianLikelihood

# Dynamically load the ring-attractor module
# script_dir = Path(__file__).parent
# spec = importlib.util.spec_from_file_location("test_dynamics", (script_dir / Path(
#     "../in_brogress/in_progress/test_dynamics.py")).resolve())
# test_dynamics = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(test_dynamics)
# PerturbedRingAttractorODE = test_dynamics.PerturbedRingAttractorODE
# PerturbedRingAttractorRNN = test_dynamics.PerturbedRingAttractorRNN
from in_progress.test_dynamics import PerturbedRingAttractorRNN, sample_gauss_z

class AddObjectsToBestCheckpoint(Callback):
    def __init__(self, extra_objects: Dict[str, Any]):
        super().__init__()
        self.extra_objects = extra_objects  # dict: name -> object

    def on_fit_end(self, trainer, pl_module):
        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        if best_ckpt_path and os.path.isfile(best_ckpt_path):
            # Load best checkpoint
            checkpoint = torch.load(best_ckpt_path, map_location="cpu")

            # Add extra objects (Tensors, nn.Modules, anything serializable)
            for key, obj in self.extra_objects.items():
                checkpoint[key] = obj

            # Save back
            torch.save(checkpoint, best_ckpt_path)
            print(f"✅ Added {list(self.extra_objects.keys())} to {best_ckpt_path}")


def assert_no_cpu_params(model: nn.Module,device=None):
    if device == "cpu":
        return
    cpu_tensors = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == "cpu":
            cpu_tensors.append((name, tensor.shape, tensor.device, tensor.dtype))

    if cpu_tensors:
        message_lines = ["Some model parameters or buffers are on CPU:"]
        for name, shape, device, dtype in cpu_tensors:
            message_lines.append(f" - {name}: shape={tuple(shape)}, device={device}, dtype={dtype}")
        raise AssertionError("\n".join(message_lines))

def initialize_rnn(perturbation_magnitude: float, device: str, seed: int, *, bin_sz:float=1e-1):
    """
    Instantiate PerturbedRingAttractorODE with a fixed seed, restoring global RNG state.
    """
    # Backup RNG states
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_state = random.getstate()
    np_state = np.random.get_state()

    # Seed for ODE
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Initialize ODE
    rnn = PerturbedRingAttractorRNN(bin_sz, grid_size=20,
                                    perturbation_magnitude=perturbation_magnitude, lengthscale=0.3).to(device)

    # Restore RNG states
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    random.setstate(py_state)
    np.random.set_state(np_state)

    return rnn


def plot_two_d_vector_field(dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=100, device="cpu"):
    import numpy as np
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        X, Y = np.meshgrid(x, y)

        XY = torch.zeros((X.shape[0]**2, 2), device=device)
        XY[:, 0] = torch.from_numpy(X).flatten()
        XY[:, 1] = torch.from_numpy(Y).flatten()
        XY_out = dynamics_fn(XY.to(device))
        s = XY_out - XY
        u = s[:, 0].reshape(X.shape[0], X.shape[1])
        v = s[:, 1].reshape(Y.shape[0], Y.shape[1])
        (X,Y, u, v) = [t.to("cpu") if isinstance(t, torch.Tensor) else t for t in (X,Y,u,v) ]

        axs.streamplot(X, Y, u, v, color="black", linewidth=0.5, arrowsize=0.5)

def data_gen(cfg, n_neurons, n_time_bins, n_trials,
             perturbation_magnitude: float, ode_seed: int):
    """
    Generate synthetic data and sample latent trajectories for the ring attractor.
    """
    if cfg.device != "cpu":
        loader_kwargs = Dict(num_workers=32, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    else:
        loader_kwargs = {}
    device = "cpu"
    cfg.device=device
    bin_sz = 1e-1

    # Initialize the perturbed ring ODE with fixed seed
    mean_fn = initialize_rnn(perturbation_magnitude, device, ode_seed, bin_sz=bin_sz)
    # mean_fn = lambda z: z + bin_sz * ode(z)

    # Build observation model
    C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(
        False
    )
    Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents, device=cfg.device)
    R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)

    # Sample latent and observations
    z = sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
    y = C(z) + torch.sqrt(R_diag) * torch.randn(
        (n_trials, n_time_bins, n_neurons), device=cfg.device
    )
    y = y.detach()

    # Split into train/validation
    y_train, z_train = y[: 2 * n_trials // 3], z[: 2 * n_trials // 3]
    y_valid, z_valid = y[2 * n_trials // 3 :], z[2 * n_trials // 3 :]

    # DataLoader helpers
    def collate_fn(batch):
        elem = batch[0]
        if isinstance(elem, (tuple, list)):
            return tuple(torch.stack([b[i] for b in batch]) for i in range(len(elem)))
        return torch.stack(batch)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_train),
        batch_size=cfg.batch_sz, shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_valid),
        batch_size=cfg.batch_sz, shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs
    )
    return train_loader, valid_loader, {
        "C": C,
        "Q_diag":Q_diag,
        "Q_0_diag":Q_0_diag,
        "R_diag":R_diag,
        "m_0":m_0,
        "mean_fn_gt":mean_fn,
        "z_train": z_train,
        "z_valid": z_valid
    }
def run_experiment(cfg, n_trials, n_neurons, n_time_bins,
                   n_ex_samples, n_ex_trials, n_ex_time_bins,
                   perturbation_magnitude: float, ode_seed: int):
    """
    Execute training and evaluation for a given perturbation magnitude.
    """
    # Create result directories
    base_dir_tmp = tempfile.mkdtemp()
    base_dir = f"results/perturb_{perturbation_magnitude}"
    ckpt_dir = os.path.join(base_dir_tmp, "ckpts")
    log_dir = os.path.join(base_dir_tmp, "logs")
    fig_dir = os.path.join(base_dir_tmp, "figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Data generation and sample plot
    train_loader, valid_loader, params = data_gen(
        deepcopy(cfg), n_neurons, n_time_bins,
        n_trials, perturbation_magnitude, ode_seed
    )
    device = cfg.device
    Q_diag = params["Q_diag"].to(device=device)
    C = params["C"].to(device=device)
    Q_0_diag = params["Q_0_diag"].to(device=device)
    R_diag = params["R_diag"].to(device=device)
    m_0 = params["m_0"].to(device=device)

    readout_fn = nn.Sequential(
        utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read, device=device),
        C
    ).to(device)
    likelihood_pdf = GaussianLikelihood(
        readout_fn, n_neurons, R_diag, device=cfg.device, fix_R=True
    )
    dynamics_fn = utils.build_gru_dynamics_function(
        cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device
    )
    dynamics_mod = DenseGaussianDynamics(
        dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device
    )

    # Initial condition
    initial_condition_pdf = DenseGaussianInitialCondition(
        cfg.n_latents, m_0, Q_0_diag, device=cfg.device
    )

    # Encoders
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents,
        cfg.n_hidden_backward,
        cfg.n_latents,
        rank_local=cfg.rank_local,
        rank_backward=cfg.rank_backward,
        device=cfg.device,
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents,
        n_neurons,
        cfg.n_hidden_local,
        cfg.n_latents,
        rank=cfg.rank_local,
        device=cfg.device,
        dropout=cfg.p_local_dropout,
    )

    # Nonlinear filtering
    nl_filter = NonlinearFilterSmallL(
        dynamics_mod, initial_condition_pdf, device=cfg.device
    )

    # Lightning training setup
    ssm = LowRankNonlinearStateSpaceModel(
        dynamics_mod,
        likelihood_pdf,
        initial_condition_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        device=cfg.device,
    )

    assert_no_cpu_params(ssm, device=cfg.device)

    seq_vae = LightningNonlinearSSM(ssm, cfg)

    logger = CSVLogger(save_dir=log_dir,
                       name=f"perturb_{perturbation_magnitude}")
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"perturb_{perturbation_magnitude}" + "_{epoch:0}_{valid_loss}",
        save_top_k=3, monitor="valid_loss", mode="min"
    )
    extra_objects = {
        "mean_fn": params["mean_fn_gt"],
        "train_dataset": train_loader.dataset,
        "valid_dataset": valid_loader.dataset,
        "z_train": params["z_train"],
        "z_valid": params["z_valid"],
    }
    extra_callback = AddObjectsToBestCheckpoint(extra_objects)

    early_stop_callback = EarlyStopping(monitor="valid_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir=base_dir_tmp,
        callbacks=[ckpt_callback,early_stop_callback,extra_callback],
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        logger=logger
    )

    # Train
    trainer.fit(
        model=seq_vae,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader
    )
    seq_vae = LightningNonlinearSSM.load_from_checkpoint(
        ckpt_callback.best_model_path, ssm=ssm, cfg=cfg
    )
    # ckpt = torch.load(ckpt_callback.best_model_path, map_location="cpu", weights_only=False)
    # mean_fn_gt = ckpt["mean_fn"]
    # train_loader = ckpt["train_dataset"]
    # valid_loader = ckpt["valid_dataset"]
    
    # Evaluation: autonomous rollouts
    z0 = torch.zeros((n_ex_samples, n_ex_trials, cfg.n_latents))
    z0[:, ::2] = 0.2 * torch.randn_like(z0[:, ::2])
    z0[:, 1::2] = 2.0 * torch.randn_like(z0[:, 1::2])
    z_prd = seq_vae.ssm.predict_forward(z0, n_ex_time_bins).detach()

    # Plot learned dynamics + trajectories
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_box_aspect(1.0)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Learned Dynamics and Autonomous Trajectories")
    plot_two_d_vector_field(seq_vae.ssm.dynamics_mod.mean_fn,
                            ax, min_xy=-2, max_xy=2)
    for i in range(min(n_ex_trials, z_prd.shape[1])):
        ax.plot(z_prd[0, i, :, 0].cpu(),
                z_prd[0, i, :, 1].cpu(), lw=0.5, alpha=0.6)
    fig.savefig(
        os.path.join(fig_dir, f"learned_traj_perturb_{perturbation_magnitude}.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    os.makedirs(base_dir, exist_ok=True)
    # shutil.copyfile(os.path.join(fig_dir, f"latent_traj_perturb_{perturbation_magnitude}.png"),os.path.join(base_dir,f"latent_traj_perturb_{perturbation_magnitude}.png"))
    shutil.copyfile(os.path.join(fig_dir, f"learned_traj_perturb_{perturbation_magnitude}.png"),os.path.join(base_dir,f"learned_traj_perturb_{perturbation_magnitude}.png"))
    shutil.copyfile(
        ckpt_callback.best_model_path,
        os.path.join(base_dir, "best_model.ckpt"),
    )
    shutil.rmtree(base_dir_tmp)


    print(f"Temp directory {base_dir_tmp}")

def main():
    # Load config
    initialize(version_base=None, config_path="", job_name="ra_experiment")
    cfg = compose(config_name="config")
    print(f"Using device: {cfg.device}")

    # Global seeding
    pl.seed_everything(cfg.seed, workers=True)
    torch.set_default_dtype(torch.float32)

    # Experiment parameters
    n_trials, n_neurons, n_time_bins = 2000, 100, 75
    n_ex_samples, n_ex_trials, n_ex_time_bins = 1, 50, 50

    # Smoke test option: two epochs, two perturbation values
    smoke = getattr(cfg, "smoke", False)

    smoke = True
    if smoke:
        print("Smoke test mode: training for 2 epochs on perturbations [0.0, 0.1]")
        cfg.n_epochs = 100
        cfg.device = "cpu"
        perturbations = [0.1]
    else:
        perturbations = [0.0, 0.1, 0.2, 0.5]

    for pm in perturbations:
        run_experiment(
            cfg,
            n_trials, n_neurons, n_time_bins,
            n_ex_samples, n_ex_trials, n_ex_time_bins,
            perturbation_magnitude=pm,
            ode_seed=cfg.seed
        )


if __name__ == "__main__":
    main()
