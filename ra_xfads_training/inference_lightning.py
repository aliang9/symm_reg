import os

import numpy as np
from typing import Dict
import tempfile
import shutil


import pytorch_lightning as pl
import torch
import torch.nn as nn
import xfads.utils as utils
from hydra import compose, initialize
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import (
    NonlinearFilterSmallL,
    LowRankNonlinearStateSpaceModel,
)
from xfads.ssm_modules.dynamics import (
    DenseGaussianDynamics,
    DenseGaussianInitialCondition,
)
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from in_progress.utils.lightning import StoreObjectsInBestCheckpoint
from in_progress.utils.ring_attractor_data import data_gen


def collate_fn(batch):
    elem = batch[0]
    if isinstance(elem, (tuple, list)):
        return tuple(torch.stack([b[i] for b in batch]) for i in range(len(elem)))
    return torch.stack(batch)


def assert_no_cpu_params(model: nn.Module, device=None):
    if device == "cpu":
        return
    cpu_tensors = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == "cpu":
            cpu_tensors.append((name, tensor.shape, tensor.device, tensor.dtype))

    if cpu_tensors:
        message_lines = ["Some ode parameters or buffers are on CPU:"]
        for name, shape, device, dtype in cpu_tensors:
            message_lines.append(
                f" - {name}: shape={tuple(shape)}, device={device}, dtype={dtype}"
            )
        raise AssertionError("\n".join(message_lines))


def plot_two_d_vector_field(
    dynamics_fn, axs, min_xy=-3, max_xy=3, n_pts=100, device="cpu"
):
    with torch.no_grad():
        x = np.linspace(min_xy, max_xy, n_pts)
        y = np.linspace(min_xy, max_xy, n_pts)
        X, Y = np.meshgrid(x, y)

        XY = torch.zeros((X.shape[0] ** 2, 2), device=device)
        XY[:, 0] = torch.from_numpy(X).flatten()
        XY[:, 1] = torch.from_numpy(Y).flatten()
        XY_out = dynamics_fn(XY.to(device))
        s = XY_out - XY
        u = s[:, 0].reshape(X.shape[0], X.shape[1])
        v = s[:, 1].reshape(Y.shape[0], Y.shape[1])
        (X, Y, u, v) = [
            t.to("cpu") if isinstance(t, torch.Tensor) else t for t in (X, Y, u, v)
        ]

        axs.streamplot(X, Y, u, v, color="black", linewidth=0.5, arrowsize=0.5)


def run_experiment(
    cfg,
    n_trials,
    n_neurons,
    n_time_bins,
    n_ex_samples,
    n_ex_trials,
    n_ex_time_bins,
    perturbation_magnitude: float,
    ode_seed: int,
):
    """
    Execute training and evaluation for old given perturbation magnitude.
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
    data = data_gen(
        n_neurons, n_time_bins, n_trials, perturbation_magnitude, deepcopy(cfg)
    )

    # DataLoader helpers
    if cfg.device != "cpu":
        loader_kwargs = Dict(
            num_workers=32, pin_memory=True, persistent_workers=True, prefetch_factor=2
        )
    else:
        loader_kwargs = {}

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["y_train"]),
        batch_size=cfg.batch_sz,
        shuffle=True,
        collate_fn=collate_fn,
        **loader_kwargs,
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data["y_valid"]),
        batch_size=cfg.batch_sz,
        shuffle=False,
        collate_fn=collate_fn,
        **loader_kwargs,
    )

    device = cfg.device
    Q_diag = data["Q_diag"].to(device=device)
    C = data["C"].to(device=device)
    Q_0_diag = data["Q_0_diag"].to(device=device)
    R_diag = data["R_diag"].to(device=device)
    m_0 = data["m_0"].to(device=device)

    readout_fn = nn.Sequential(
        utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read, device=device), C
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

    # assert_no_cpu_params(ssm, device=cfg.device)

    seq_vae = LightningNonlinearSSM(ssm, cfg)

    logger = CSVLogger(save_dir=log_dir, name=f"perturb_{perturbation_magnitude}")
    ckpt_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"perturb_{perturbation_magnitude}" + "_{epoch:0}_{valid_loss}",
        save_top_k=3,
        monitor="valid_loss",
        mode="min",
    )
    store_data = {
        "mean_fn": data["mean_fn"],
        "z_train": data["z_train"],
        "z_valid": data["z_valid"],
    }
    store_object_callback = StoreObjectsInBestCheckpoint(store_data)

    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=2, verbose=False, mode="min"
    )
    callback_list = [ckpt_callback, early_stop_callback, store_object_callback]
    trainer = pl.Trainer(
        max_epochs=cfg.n_epochs,
        gradient_clip_val=1.0,
        default_root_dir=base_dir_tmp,
        callbacks=callback_list,
        accelerator="gpu" if cfg.device == "cuda" else "cpu",
        logger=logger,
    )

    # Train
    trainer.fit(
        model=seq_vae, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )
    seq_vae = LightningNonlinearSSM.load_from_checkpoint(
        ckpt_callback.best_model_path, ssm=ssm, cfg=cfg
    )

    # Plot learned dynamics + trajectories
    shutil.copyfile(
        os.path.join(fig_dir, f"learned_traj_perturb_{perturbation_magnitude}.png"),
        os.path.join(base_dir, f"learned_traj_perturb_{perturbation_magnitude}.png"),
    )
    shutil.copyfile(
        ckpt_callback.best_model_path, os.path.join(base_dir, "best_model.ckpt")
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
            n_trials,
            n_neurons,
            n_time_bins,
            n_ex_samples,
            n_ex_trials,
            n_ex_time_bins,
            perturbation_magnitude=pm,
            ode_seed=cfg.seed,
        )


if __name__ == "__main__":
    main()
