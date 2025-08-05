from inspect import isclass
from typing import TypedDict, Dict, Any, Optional, Callable  # noqa
from typing_extensions import Required, NotRequired  # noqa
from xfads import utils as utils

from in_progress.test_dynamics import WarpedRingAttractorRNN, PerturbedRingAttractorRNN  # noqa
from in_progress.utils.random import rng_state_noop
from copy import deepcopy
import torch
from omegaconf import DictConfig
from math import sqrt, ceil  # noqa


@rng_state_noop
def initialize_rnn(
    bin_sz: float, cfg: DictConfig, rnn_kwargs: Optional[Dict[str, Any]] = None
):
    """
    Instantiate PerturbedRingAttractorODE with old fixed seed, restoring global RNG state.
    """
    if not rnn_kwargs:
        rnn_kwargs = {}
    rnn_init = rnn_kwargs.pop("dynamics", WarpedRingAttractorRNN)
    # bin_sz = rnn_kwargs.pop("bin_sz", 1e-1)
    if isclass(rnn_init):
        rnn = rnn_init(bin_sz=bin_sz, **rnn_kwargs).to(cfg.device)
    else:
        rnn = rnn_init

    return rnn


def data_gen(
    n_neurons: int,
    n_time_bins: int,
    n_trials: int,
    bin_sz: float,
    rnn_kwargs: Dict[str, Any],
    cfg: DictConfig,
):
    """
    Generate synthetic data and sample latent trajectories for the ring attractor.
    """
    cfg = deepcopy(cfg)
    cfg.device = "cpu"

    mean_fn = initialize_rnn(bin_sz, cfg, rnn_kwargs)

    # Build observation ode
    C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(
        False
    )
    # Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)
    # Q_0_diag = 1.0 * torch.ones(cfg.n_latents, device=cfg.device)
    Q_diag = 5e-2 * torch.ones(cfg.n_latents, device=cfg.device)
    Q_0_diag = 2.0 * torch.ones(cfg.n_latents, device=cfg.device)
    R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)

    # Sample latent and observations
    with torch.no_grad():
        z = sample_gauss_z(
            mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins, bin_sz=bin_sz
        )
    y = C(z) + torch.sqrt(R_diag) * torch.randn(
        (n_trials, n_time_bins, n_neurons), device=cfg.device
    )
    y = y.detach()

    # Split into train/validation
    y_train, z_train = y[: 2 * n_trials // 3], z[: 2 * n_trials // 3]
    y_valid, z_valid = y[2 * n_trials // 3 :], z[2 * n_trials // 3 :]

    return {
        "C": C,
        "Q_diag": Q_diag,
        "Q_0_diag": Q_0_diag,
        "R_diag": R_diag,
        "m_0": m_0,
        "mean_fn": mean_fn,
        "z_train": z_train,
        "z_valid": z_valid,
        "y_train": y_train,
        "y_valid": y_valid,
    }


def sample_gauss_z(
    mean_fn: Callable[[torch.Tensor], torch.Tensor],
    Q_diag: torch.Tensor,
    m_0: torch.Tensor,
    Q_0_diag: torch.Tensor,
    n_trials: int,
    n_time_bins: int,
    *,
    bin_sz: Optional[float] = None,
) -> torch.Tensor:
    if bin_sz is None:
        bin_sz = 1.0

    dt = bin_sz
    device, dtype = Q_diag.device, Q_diag.dtype
    n_latents = Q_diag.shape[-1]
    z = torch.empty((n_trials, n_time_bins, n_latents), device=device)

    # Pre-sample all noise
    eps = torch.randn((n_trials, n_time_bins, n_latents), device=device)

    # First time step
    z[:, 0] = m_0 + eps[:, 0] * torch.sqrt(Q_0_diag)
    z_prev = torch.zeros(n_trials, n_latents, device=device, dtype=dtype)

    # Recursive sampling
    for t in range(1, n_time_bins):
        z_prev.copy_(z[:, t - 1])
        z[:, t] = mean_fn(z_prev) + eps[:, t] * sqrt(dt) * torch.sqrt(Q_diag)

    return z
