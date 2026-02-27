"""
Time-aware data generation for time-varying perturbation experiments.

This module extends the standard sample_gauss_z function from xfads.utils
to pass time information through the dynamics function, enabling time-varying
perturbations during data generation.

The key difference from static perturbations:
- Standard: mean_fn(z[:, t-1]) - dynamics don't know what time it is
- Time-varying: mean_fn(z[:, t-1], t) - dynamics can vary with time

Usage:
    from time_varying_utils import TimeVaryingPerturbedRingAttractorRNN
    from utils.time_varying_data import sample_gauss_z_time_varying

    mean_fn = TimeVaryingPerturbedRingAttractorRNN(...)
    z = sample_gauss_z_time_varying(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins, bin_sz)
"""

from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor


def sample_gauss_z_time_varying(
    mean_fn: Callable[[Tensor, float], Tensor],
    Q_diag: Tensor,
    m_0: Tensor,
    Q_0_diag: Tensor,
    n_trials: int,
    n_time_bins: int,
    bin_sz: float = 0.1,
) -> Tensor:
    """
    Sample latent trajectories with time-varying dynamics.

    This is the time-aware version of xfads.utils.sample_gauss_z. The key
    difference is that we pass the current time t to the dynamics function:
        z[:, t] = mean_fn(z[:, t-1], t * bin_sz) + noise

    This enables the dynamics to change over time, which is needed for
    time-varying perturbation experiments.

    Args:
        mean_fn: Dynamics function f(x, t) -> x' that takes both state and time.
            Should be a TimeVaryingPerturbedRingAttractorRNN or similar.
        Q_diag: Process noise variance (n_latents,)
        m_0: Initial mean (n_latents,)
        Q_0_diag: Initial variance (n_latents,)
        n_trials: Number of trajectories to sample
        n_time_bins: Number of time steps per trajectory
        bin_sz: Time step size for computing continuous time: t = bin_idx * bin_sz

    Returns:
        z: Latent trajectories (n_trials, n_time_bins, n_latents)
    """
    device = Q_diag.device
    n_latents = Q_diag.shape[-1]
    z = torch.zeros((n_trials, n_time_bins, n_latents), device=device)

    for t_idx in range(n_time_bins):
        # Compute continuous time for this bin
        t = t_idx * bin_sz

        if t_idx == 0:
            # Initial condition: sample from N(m_0, Q_0_diag)
            z[:, 0] = m_0 + torch.sqrt(Q_0_diag) * torch.randn_like(z[:, 0])
        else:
            # Dynamics step: z_t = f(z_{t-1}, t) + noise
            # Pass time to dynamics for time-varying perturbations
            z[:, t_idx] = mean_fn(z[:, t_idx - 1], t) + \
                          torch.sqrt(Q_diag) * torch.randn_like(z[:, t_idx - 1])

    return z


def generate_time_varying_ring_data(
    mean_fn: nn.Module,
    C: nn.Module,
    Q_diag: Tensor,
    Q_0_diag: Tensor,
    R_diag: Tensor,
    m_0: Tensor,
    n_trials: int,
    n_time_bins: int,
    bin_sz: float = 0.1,
) -> Tuple[Tensor, Tensor]:
    """
    Generate ring attractor data with time-varying perturbations.

    This wraps sample_gauss_z_time_varying with observation generation,
    mirroring the standard data generation flow in XFADS.

    Args:
        mean_fn: Time-varying dynamics module (TimeVaryingPerturbedRingAttractorRNN)
        C: Observation matrix module (typically FanInLinear)
        Q_diag: Process noise variance (n_latents,)
        Q_0_diag: Initial variance (n_latents,)
        R_diag: Observation noise variance (n_neurons,)
        m_0: Initial mean (n_latents,)
        n_trials: Number of trajectories
        n_time_bins: Time steps per trajectory
        bin_sz: Time step size

    Returns:
        y: Observations (n_trials, n_time_bins, n_neurons)
        z: Latent states (n_trials, n_time_bins, n_latents)
    """
    device = Q_diag.device

    # Determine n_neurons from C
    if hasattr(C, 'out_features'):
        n_neurons = C.out_features
    elif hasattr(C, 'weight'):
        n_neurons = C.weight.shape[0]
    else:
        # Fallback: infer from R_diag
        n_neurons = R_diag.shape[0]

    # Sample latent trajectories with time-varying dynamics
    z = sample_gauss_z_time_varying(
        mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins, bin_sz
    )

    # Generate observations: y = C(z) + noise
    y = C(z) + torch.sqrt(R_diag) * torch.randn(
        (n_trials, n_time_bins, n_neurons), device=device
    )

    return y.detach(), z
