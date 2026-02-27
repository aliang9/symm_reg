"""
Time-varying GP perturbations using Random Fourier Features (RFF).

This module provides PyTorch implementations of time-varying perturbations
for ring attractor dynamics, converted from the NumPy prototype in
experiments/time_varying.ipynb.

Key classes:
- RFFTimeVaryingPerturbation: PyTorch RFF approximation of spatiotemporal GP
- TimeVaryingRbfRingAttractorODE: ODE with time-dependent perturbations
- TimeVaryingPerturbedRingAttractorRNN: Euler wrapper for XFADS compatibility
"""

import math
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor


class RFFTimeVaryingPerturbation(nn.Module):
    """
    Random Fourier Features approximation of a spatiotemporal GP.

    Perturbation formula:
        g(x, t) = scale * sqrt(2/D) * theta @ cos(Wx @ x + Wt * t + b)

    This approximates a zero-mean GP with kernel:
        k((x,t), (x',t')) = exp(-||x-x'||^2 / (2*length_scale_x^2))
                         * exp(-|t-t'|^2 / (2*length_scale_t^2))

    Separate length scales for spatial and temporal dimensions provide
    independent control over perturbation correlation structure:
    - length_scale_x: Smaller values = faster spatial variation (more wiggly)
    - length_scale_t: Smaller values = faster temporal drift (quicker changes)

    Args:
        n_features: Number of random Fourier features (D). Higher = better
            approximation but more compute. Typical: 50-200.
        scale: Output scale controlling perturbation magnitude.
        length_scale_x: Spatial correlation length. Should be on order of
            the attractor radius for meaningful perturbations.
        length_scale_t: Temporal correlation length in time units.
            Larger = slower perturbation drift.
        x_dim: Spatial dimension (2 for ring, 3 for sphere).
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_features: int = 100,
        scale: float = 0.5,
        length_scale_x: float = 1.0,
        length_scale_t: float = 10.0,
        x_dim: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.n_features = n_features
        self.scale = scale
        self.length_scale_x = length_scale_x
        self.length_scale_t = length_scale_t
        self.x_dim = x_dim

        # Normalization factor for RFF approximation
        self.norm = math.sqrt(2.0 / n_features)

        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        else:
            generator = None

        # Spatial frequency weights: Wx ~ N(0, 1/length_scale_x^2)
        # Shape: (n_features, x_dim)
        Wx = torch.randn(n_features, x_dim, generator=generator) / length_scale_x
        self.register_buffer("Wx", Wx)

        # Temporal frequency weights: Wt ~ N(0, 1/length_scale_t^2)
        # Shape: (n_features,) - squeeze to scalar per feature
        Wt = torch.randn(n_features, generator=generator) / length_scale_t
        self.register_buffer("Wt", Wt)

        # Phase shifts: b ~ Uniform(0, 2*pi)
        # Shape: (n_features,)
        b = torch.rand(n_features, generator=generator) * 2 * math.pi
        self.register_buffer("b", b)

        # Output projection weights for each spatial dimension
        # Shape: (x_dim, n_features) - one set of weights per output component
        theta = torch.randn(x_dim, n_features, generator=generator)
        self.register_buffer("theta", theta)

    def forward(
        self,
        x: Tensor,
        t: Union[Tensor, float]
    ) -> Tensor:
        """
        Evaluate time-varying perturbation.

        Args:
            x: Spatial coordinates (..., x_dim)
            t: Time scalar or tensor broadcastable to x.shape[:-1]

        Returns:
            Perturbation vectors (..., x_dim)
        """
        # Convert t to tensor if needed
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device=x.device, dtype=x.dtype)

        # Store original shape for reshaping output
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.x_dim)  # (B, x_dim)
        batch_size = x_flat.shape[0]

        # Spatial projection: (B, n_features)
        spatial_proj = x_flat @ self.Wx.T

        # Temporal projection: broadcast Wt * t to (B, n_features)
        # Wt has shape (n_features,), t is scalar or (B,)
        if t.dim() == 0:
            # Scalar t: broadcast to all samples
            temporal_proj = t * self.Wt  # (n_features,) broadcasts to (B, n_features)
        else:
            # Tensor t: shape should be (B,) or broadcastable
            t_flat = t.reshape(-1)
            if t_flat.shape[0] == 1:
                temporal_proj = t_flat[0] * self.Wt  # (n_features,)
            else:
                temporal_proj = t_flat.unsqueeze(-1) * self.Wt.unsqueeze(0)  # (B, n_features)

        # Combined RFF features: cos(Wx @ x + Wt * t + b)
        # Shape: (B, n_features)
        features = torch.cos(spatial_proj + temporal_proj + self.b)

        # Output: scale * norm * features @ theta.T
        # Shape: (B, x_dim)
        output_flat = self.scale * self.norm * (features @ self.theta.T)

        # Reshape to match original batch dimensions
        return output_flat.reshape(*orig_shape)

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, scale={self.scale}, "
            f"length_scale_x={self.length_scale_x}, length_scale_t={self.length_scale_t}"
        )


class TimeVaryingRbfRingAttractorODE(nn.Module):
    """
    Ring attractor ODE with time-varying RFF perturbations.

    Vector field:
        x' = x * (1 - ||x||) + g(x, t)

    where g(x, t) is a time-varying perturbation from RFFTimeVaryingPerturbation.

    The base dynamics x * (1 - ||x||) create a limit cycle at radius 1:
    - Inside ring (||x|| < 1): radial expansion
    - Outside ring (||x|| > 1): radial contraction
    - On ring (||x|| = 1): neutral stability (zero radial force)

    The perturbation g(x, t) breaks the rotational symmetry in a
    continuously time-varying manner.

    Args:
        perturbation_magnitude: Scale of perturbation (same as 'scale' in RFF).
            Set to 0 to disable perturbations.
        n_features: Number of RFF features for perturbation.
        length_scale_x: Spatial correlation length of perturbation.
        length_scale_t: Temporal correlation length of perturbation.
        seed: Optional RNG seed for perturbation reproducibility.
    """

    def __init__(
        self,
        perturbation_magnitude: float = 0.2,
        n_features: int = 100,
        length_scale_x: float = 1.0,
        length_scale_t: float = 10.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.perturbation_magnitude = perturbation_magnitude

        if perturbation_magnitude > 0:
            self.perturbation = RFFTimeVaryingPerturbation(
                n_features=n_features,
                scale=perturbation_magnitude,
                length_scale_x=length_scale_x,
                length_scale_t=length_scale_t,
                x_dim=2,
                seed=seed,
            )
        else:
            self.perturbation = None

    def forward(
        self,
        x: Tensor,
        t: Union[Tensor, float] = 0.0
    ) -> Tensor:
        """
        Evaluate the time-varying ring attractor vector field.

        Args:
            x: State (..., 2)
            t: Current time (scalar or tensor)

        Returns:
            Vector field dx/dt (..., 2)
        """
        # Base ring attractor dynamics: x * (1 - ||x||)
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        base = x * (1 - r)

        if self.perturbation is not None:
            perturb = self.perturbation(x, t)
            return base + perturb
        else:
            return base


class TimeVaryingPerturbedRingAttractorRNN(nn.Module):
    """
    RNN wrapper for time-varying ring attractor dynamics.

    Implements Euler integration:
        x_{t+1} = x_t + bin_sz * f(x_t, t)

    Key difference from static PerturbedRingAttractorRNN: forward() accepts
    both x and t, enabling time-varying perturbations during data generation.

    Compatible with modified XFADS data generation (sample_gauss_z_time_varying)
    that passes time through the dynamics.

    Args:
        bin_sz: Integration time step (Euler step size).
        perturbation_magnitude: Perturbation scale.
        n_features: Number of RFF features.
        length_scale_x: Spatial perturbation correlation length.
        length_scale_t: Temporal perturbation correlation length.
        seed: Optional RNG seed.
    """

    def __init__(
        self,
        bin_sz: float = 0.1,
        perturbation_magnitude: float = 0.2,
        n_features: int = 100,
        length_scale_x: float = 1.0,
        length_scale_t: float = 10.0,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.bin_sz = bin_sz
        self.ode = TimeVaryingRbfRingAttractorODE(
            perturbation_magnitude=perturbation_magnitude,
            n_features=n_features,
            length_scale_x=length_scale_x,
            length_scale_t=length_scale_t,
            seed=seed,
        )

    def forward(
        self,
        x: Tensor,
        t: Optional[Union[Tensor, float]] = None
    ) -> Tensor:
        """
        Euler step with time-varying dynamics.

        Args:
            x: Current state (..., 2)
            t: Current time (if None, defaults to 0)

        Returns:
            Next state (..., 2)
        """
        if t is None:
            t = 0.0
        return x + self.ode(x, t) * self.bin_sz

    def vectorfield(
        self,
        x: Tensor,
        t: Optional[Union[Tensor, float]] = None
    ) -> Tensor:
        """
        Return the vector field at given state and time (for visualization).

        Args:
            x: State (..., 2)
            t: Time (defaults to 0)

        Returns:
            Vector field (..., 2)
        """
        if t is None:
            t = 0.0
        return self.ode(x, t)

    @property
    def perturbation_magnitude(self) -> float:
        """Return the perturbation magnitude."""
        return self.ode.perturbation_magnitude
