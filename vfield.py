import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gpytorch
from typing import Tuple, Union, Optional,Callable


class GPPrior(nn.Module):
    """
    Simple zero-mean GP prior with RBF kernel using GPyTorch.
    """
    def __init__(self, train_x: torch.Tensor, lengthscale: float = 1.0, noise: float = 1e-6):
        super().__init__()
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([2]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(lengthscale=lengthscale,batch_shape=torch.Size([2])),
            batch_shape=torch.Size([2]),
        )
        self.train_x = train_x
        # Jitter for numerical stability
        self.noise = noise

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Add jitter to diagonal
        covar = covar_x + self.noise * torch.eye(x.size(0), device=x.device)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar))

class PerturbedVectorField(nn.Module):
    """
    PyTorch module for a perturbed radial-attractor vector field.

    Vector field:
        x' = x*(1 - ||x||_2) + perturb(x)

    The perturbation is a zero-mean GP sample on a grid, then bilinearly interpolated.
    """
    def __init__(
        self,
        perturbation_magnitude: float = 1.0,
        grid_size: Union[int, Tuple[int, int]] = 100,
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
        pts_t = torch.from_numpy(pts.astype(np.float32))

        # Sample GP prior for U and V components
        gp_model = GPPrior(train_x=pts_t, lengthscale=lengthscale, noise=noise)
        gp_model.eval()
        with torch.no_grad():
            UV_grid = gp_model(pts_t).sample()
            # dist_u = gp_model(pts_t)
            # dist_v = gp_model(pts_t)
            # U_samples = dist_u.sample().cpu()
            # V_samples = dist_v.sample().cpu()

        U_grid,V_grid = UV_grid.renorm(p=2,dim=1,maxnorm=perturbation_magnitude).vsplit(2)

        # Register buffers
        self.register_buffer('x_grid', torch.from_numpy(x_grid.astype(np.float32)))
        self.register_buffer('y_grid', torch.from_numpy(y_grid.astype(np.float32)))
        self.register_buffer('perturb_u', U_grid.unsqueeze(0).unsqueeze(0))
        self.register_buffer('perturb_v', V_grid.unsqueeze(0).unsqueeze(0))
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max

    def _bilinear_interpolate(self, field: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Normalize to [-1,1]
        xi = 2 * (x[...,0] - self.x_min) / (self.x_max - self.x_min) - 1
        yi = 2 * (x[...,1] - self.y_min) / (self.y_max - self.y_min) - 1
        pts = torch.stack((xi, yi), dim=-1)
        pts = pts.unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(field, pts, align_corners=True, mode='bilinear')
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
    z_prev = torch.zeros(n_trials,n_time_bins,n_latents,device=device,dtype=dtype)

    # Recursive sampling
    for t in range(1, n_time_bins):
        z_prev.copy_(z[:, t - 1])
        z[:, t] = mean_fn(z_prev) + eps[:, t] * torch.sqrt(Q_diag)

    return z

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Visualize with a stream plot
    vb = PerturbedVectorField(perturbation_magnitude=0.0,
                              grid_size=(40),
                              domain_extent=(-3.0,3.0),
                              lengthscale=1.0)

    # Create plotting grid
    x = vb.x_grid.cpu().numpy()
    y = vb.y_grid.cpu().numpy()
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    pts_t = torch.from_numpy(pts.astype(np.float32))

    # Evaluate vector field
    with torch.no_grad():
        vals = vb(pts_t)
    U = vals[:,0].cpu().numpy().reshape(Y.shape)
    V = vals[:,1].cpu().numpy().reshape(Y.shape)

    # Plot
    plt.figure(figsize=(6,6))
    plt.streamplot(X, Y, U, V, density=1.2)
    plt.title('Perturbed Radial-Attractor Vector Field')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
