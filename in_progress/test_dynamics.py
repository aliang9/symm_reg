from typing import Callable, Optional, Tuple, Union, TypeAlias

import gpytorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from functools import wraps

from torch import Tensor
from torchdyn.numerics import odeint as _td_odeint

from in_progress import VectorField

nn_activation: TypeAlias = torch.nn.modules.activation
_odeint = lambda f_, x_, t_, **kwargs:_td_odeint(f_, x_, t_, solver ="tsit5", **kwargs)[-1]

def deprecated(reason: str = "This will be removed in a future version."):
    def decorator(cls):
        orig_init = cls.__init__

        @wraps(orig_init)
        def new_init(self, *args, **kwargs):
            warnings.warn(
                f"{cls.__name__} is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return orig_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls
    return decorator

class _GPPerturbation(nn.Module):
    """
    Simple zero-mean GP prior with RBF kernel using GPyTorch.
    """
    def __init__(self, train_x: torch.Tensor, lengthscale: float = 1.0, noise: float = 1e-6):
        super().__init__()
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=lengthscale, batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2]),
        )
        self.train_x = train_x # Jitter for numerical stability
        self.noise = noise

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Add jitter to diagonal
        covar = covar_x + self.noise * torch.eye(x.size(0), device=x.device)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar))

@deprecated(reason="Does not support autodiff, use RbfPerturbedRingAttractorODE instead.")
class PerturbedRingAttractorODE(nn.Module):
    """
    PyTorch module for a perturbed radial-attractor vector field.

    Vector field:
        x' = x*(1 - ||x||_2) + perturb(x)

    The perturbation is a zero-mean GP sample on a grid, then bilinearly interpolated.
    """

    def __init__(
            self,
            perturbation_magnitude: float = 0.2,
            grid_size: Union[int, Tuple[int, int]] = 20,
            domain_extent: Union[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]] = (-2.0, 2.0),
            lengthscale: float = 1.0,
            noise: float = 1e-6
    ):
        super().__init__()
        self.perturbation_magnitude = perturbation_magnitude

        # Build grid
        if isinstance(grid_size, int):
            Nx = Ny = grid_size
        else:
            Nx, Ny = grid_size

        if isinstance(domain_extent[0], (int, float)):
            d_min, d_max = domain_extent
            x_min, x_max = -abs(d_min), abs(d_max)
            y_min, y_max = -abs(d_min), abs(d_max)
        else:
            (x_min, x_max), (y_min, y_max) = domain_extent

        x_grid = np.linspace(x_min, x_max, Nx)
        y_grid = np.linspace(y_min, y_max, Ny)
        X, Y = np.meshgrid(x_grid, y_grid)
        pts = np.column_stack([X.ravel(), Y.ravel()])  # (Nx*Ny, 2)
        pts_t = torch.tensor(pts.astype(np.float32))

        # Sample GP prior for U and V components
        gp_model = _GPPerturbation(train_x=pts_t, lengthscale=lengthscale, noise=noise)
        gp_model.eval()
        with torch.no_grad():
            UV_grid = gp_model(pts_t).sample()

        U_grid, V_grid = UV_grid.renorm(p=2, dim=0, maxnorm=perturbation_magnitude).vsplit(2)

        # Register buffers
        self.register_buffer("x_grid", torch.tensor(x_grid.astype(np.float32)))
        self.register_buffer("y_grid", torch.tensor(y_grid.astype(np.float32)))
        self.register_buffer("perturb_u", U_grid.unsqueeze(0).unsqueeze(0))
        self.register_buffer("perturb_v", V_grid.unsqueeze(0).unsqueeze(0))
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def _bilinear_interpolate(self, field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [-1,1]
        xi = 2 * (x[..., 0] - self.x_min) / (self.x_max - self.x_min) - 1
        yi = 2 * (x[..., 1] - self.y_min) / (self.y_max - self.y_min) - 1
        pts = torch.stack((xi, yi), dim=-1)
        pts = pts.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(field, pts, align_corners=True, mode="bilinear")
        return sampled.squeeze()

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = torch.tensor(0)) -> torch.Tensor:
        # Radial-attractor base field
        r = torch.norm(x, dim=-1, keepdim=True)
        base = x * (1 - r)
        # Perturbation
        u = self._bilinear_interpolate(self.perturb_u, x)
        v = self._bilinear_interpolate(self.perturb_v, x)
        perturb = torch.stack((u, v), dim=-1)
        return base + perturb

class RbfPerturbedRingAttractorODE(nn.Module):
    def __init__(
        self,
        perturbation_magnitude: Union[float,int] = 0.2,
        grid_size: int = 20,
        domain_extent: tuple = (-2.0, 2.0),
        lengthscale: float = 1.0,
        noise: float = 1e-6
    ):
        super().__init__()
        if perturbation_magnitude != 0.:
            self.perturbation_magnitude = perturbation_magnitude
        else:
            self.perturbation_magnitude = 0

        x_vals = torch.linspace(domain_extent[0], domain_extent[1], grid_size)
        y_vals = x_vals
        X, Y = torch.meshgrid(x_vals, y_vals, indexing="xy")
        centers = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=1)  # (M,2)
        M = centers.size(0)

        gp = _GPPerturbation(train_x=centers, lengthscale=lengthscale, noise=noise)
        gp.eval()
        with torch.no_grad():
            uv = gp(centers).sample()  # (M,2)
        uv = uv.renorm(p=2, dim=0, maxnorm=perturbation_magnitude)

        dist2 = torch.cdist(centers, centers,p=2,compute_mode="donot_use_mm_for_euclid_dist").pow(2)  # (M,M)
        K = torch.exp(-dist2 / (2 * lengthscale**2))
        K += noise * torch.eye(M)  # regularization

        u_vals = uv[:, 0]
        v_vals = uv[:, 1]
        coeff_u = torch.linalg.solve(K, u_vals)  # (M,)
        coeff_v = torch.linalg.solve(K, v_vals)  # (M,)

        self.register_buffer("centers", centers)    # (M,2)
        self.register_buffer("coeff_u", coeff_u)    # (M,)
        self.register_buffer("coeff_v", coeff_v)    # (M,)
        self.lengthscale = lengthscale

    @staticmethod
    def cdist2norm(x: torch.Tensor,
                   y: torch.Tensor,
                   eps: float = 1e-10) -> torch.Tensor:
        """
        Compute pairwise distances between rows of x and rows of y.

        Args:
            x: Tensor of shape (..., P, D)
            y: Tensor of shape (..., R, D)
            p: the norm degree (only p == 2 is optimized)
            eps: small value to clamp negative due to numerical error

        Returns:
            Tensor of shape (..., P, R) where out[..., i, j] = ||x[..., i] - y[..., j]||_p
        """
        # shapes
        # *batch, P, D = x.shape
        # _, R, _ = y.shape

        # ‖u - v‖_2 = sqrt(‖u‖² + ‖v‖² - 2 u·v)
        x_sq = x.pow(2).sum(dim=-1, keepdim=True)          # (..., P, 1)
        y_sq = y.pow(2).sum(dim=-1).unsqueeze(-2)          # (..., 1, R)
        xy   = torch.matmul(x, y.transpose(-2, -1))        # (..., P, R)
        # make sure no tiny negatives under the sqrt
        dist_sq = (x_sq + y_sq - 2.0 * xy).clamp_min(eps)  # (..., P, R)
        return dist_sq

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """
        x: (..., 2) input points
        returns: same shape as x, the vector field at x
        """
        if self.perturbation_magnitude > 0:
            x_flat = x.reshape(-1, 2)  # (B,2)
            dist2 = self.cdist2norm(x_flat, self.centers)
            phi = torch.exp(-dist2 / (2 * self.lengthscale**2))  # (B,M)

            u_flat = phi @ self.coeff_u  # (B,)
            v_flat = phi @ self.coeff_v  # (B,)
            uv_flat = torch.stack([u_flat, v_flat], dim=1)  # (B,2)

            uv = uv_flat.reshape(*x.shape[:-1], 2)  # (...,2)

            r = x.norm(dim=-1, keepdim=True)
            base = x * (1 - r)

            return base + uv
        else:
            r = x.norm(dim=-1, keepdim=True)
            base = x * (1 - r)

            return base

class PerturbedRingAttractorRNN(RbfPerturbedRingAttractorODE):
    def __init__(self,bin_sz:float=1e-1,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.bin_sz = bin_sz

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        return x + super().forward(x) * self.bin_sz

def sample_gauss_z(
        mean_fn: Callable[[torch.Tensor], torch.Tensor],
        Q_diag: torch.Tensor,
        m_0: torch.Tensor,
        Q_0_diag: torch.Tensor,
        n_trials: int,
        n_time_bins: int,
) -> torch.Tensor:
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
        z[:, t] = mean_fn(z_prev) + eps[:, t] * torch.sqrt(Q_diag)

    return z


class Phi(nn.Module):
    def __init__(self, dim: int, *args, hidden: int = 96, activation: nn_activation = nn.GELU, repetitions: int = 0,
                 **kwargs):
        super().__init__(*args,**kwargs)
        layers = [nn.Linear(dim, hidden), activation()]
        for _ in range(repetitions):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(activation())
        layers.append(nn.Linear(hidden, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class DiffeoWrapper(nn.Module):
    def __init__(self, diffeo_model: Callable[[Tensor], Tensor],t: float,solver: Callable, **solver_kwargs):
        super().__init__()
        self.diffeo = diffeo_model
        self.T = t
        self.solver = solver
        self.solver_kwargs = solver_kwargs

    def forward(self, t, x):  # noqa
        return self.diffeo(x)

    def phi(self, x: Tensor) -> Tensor:
        t = torch.tensor([0.0, self.T], dtype=x.dtype, device=x.device)
        traj = self.solver(self, x, t, **self.solver_kwargs)
        return traj[-1]

    def phi_inverse(self, y: Tensor) -> Tensor:
        t = torch.tensor([self.T, 0.0], dtype=y.dtype, device=y.device)
        traj = self.solver(self, y, t, **self.solver_kwargs)
        return traj[-1]


class ConjugateSystem(nn.Module):
    def __init__(
        self,
        f: VectorField,
        diffeo_model: nn.Module,
        t: float = 1.0,
        *,
        solver=_odeint,
        **solver_kwargs,
    ):
        super().__init__()
        self.F = f
        self.model = DiffeoWrapper(diffeo_model,t, solver=solver, **solver_kwargs)

    def phi(self, x: Tensor) -> Tensor:
        return self.model.phi(x)

    def phi_inverse(self, y: Tensor) -> Tensor:
        return self.model.phi_inverse(y)

    def conjugate_vector_field(self, y: Tensor) -> Tensor:
        # 1) pull back y to x = ϕ⁻¹(y)
        x = self.phi_inverse(y)
        jvp = self.tangent_map(x)
        return jvp

    def tangent_map(self, x):
        dx = self.F(x)
        jvp = torch.func.jvp(self.phi, (x,), (dx,))[-1]
        return jvp

    def forward(self, x: Tensor) -> Tensor:
        return self.tangent_map(x)

    def conjugate_vector_field_batch(self, Y: Tensor) -> Tensor:
        return torch.stack([self.conjugate_vector_field(y_i) for y_i in Y], dim=0)


def dynamics_factory(A = None) -> Callable[[Tensor], Tensor]:
    if A is None:
        A = torch.eye(2)
    def f(x: Tensor) -> Tensor:
        r = torch.sqrt(torch.sum((x @ A) * x, dim=-1, keepdim=True))
        return x * (1 - r)

    return f
