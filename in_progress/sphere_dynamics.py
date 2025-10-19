"""
3D Sphere Attractor Dynamics with SO(3) Regularization Support

This module implements a 3D sphere attractor system analogous to the 2D ring attractor,
but operating on the 2-sphere (unit sphere surface) in 3D space.

Key features:
- Sphere attractor dynamics: x' = x * (1 - ||x||)  
- RBF-based perturbations in 3D
- Support for SO(3) rotational symmetries
- Compatible with existing XFADS infrastructure
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple
import gpytorch


class _GP3DPerturbation(nn.Module):
    """
    3D zero-mean GP prior with RBF kernel using GPyTorch.
    Generates perturbation fields in 3D space.
    """

    def __init__(
        self, train_x: torch.Tensor, lengthscale: float = 1.0, noise: float = 1e-6
    ):
        super().__init__()
        self.mean_module = gpytorch.means.ZeroMean(batch_shape=torch.Size([3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale=lengthscale, batch_shape=torch.Size([3])
            ),
            batch_shape=torch.Size([3]),
        )
        self.train_x = train_x
        self.noise = noise

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Add jitter to diagonal
        covar = covar_x + self.noise * torch.eye(x.size(0), device=x.device)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar)
        )


class RbfPerturbedSphereAttractorODE(nn.Module):
    """
    3D Sphere attractor with RBF-based perturbations.
    
    Vector field: x' = x * (1 - ||x||) + perturbation(x)
    
    The base dynamics attract trajectories to the unit sphere, while
    perturbations add structured noise for more realistic dynamics.
    """
    
    def __init__(
        self,
        perturbation_magnitude: Union[float, int] = 0.2,
        grid_size: int = 15,  # Fewer points in 3D to avoid memory issues
        domain_extent: tuple = (-2.0, 2.0),
        lengthscale: float = 0.3,
        noise: float = 1e-6,
    ):
        super().__init__()
        
        self.perturbation_magnitude = perturbation_magnitude
        
        if perturbation_magnitude > 0:
            # Create 3D grid of centers
            vals = torch.linspace(domain_extent[0], domain_extent[1], grid_size)
            X, Y, Z = torch.meshgrid(vals, vals, vals, indexing="ij")
            centers = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)  # (M,3)
            M = centers.size(0)

            # Generate GP perturbation field
            gp = _GP3DPerturbation(train_x=centers, lengthscale=lengthscale, noise=noise)
            gp.eval()
            with torch.no_grad():
                uvw = gp(centers).sample()  # (M,3) - u,v,w components
            
            # Normalize perturbations
            uvw = uvw.renorm(p=2, dim=0, maxnorm=perturbation_magnitude)
            
            # Compute RBF kernel matrix for interpolation
            dist2 = torch.cdist(
                centers, centers, p=2, compute_mode="donot_use_mm_for_euclid_dist"
            ).pow(2)  # (M,M)
            K = torch.exp(-dist2 / (2 * lengthscale**2))
            K += noise * torch.eye(M)  # regularization

            # Solve for RBF coefficients for each component
            u_vals, v_vals, w_vals = uvw[:, 0], uvw[:, 1], uvw[:, 2]
            coeff_u = torch.linalg.solve(K, u_vals)  # (M,)
            coeff_v = torch.linalg.solve(K, v_vals)  # (M,)
            coeff_w = torch.linalg.solve(K, w_vals)  # (M,)

            self.register_buffer("centers", centers)  # (M,3)
            self.register_buffer("coeff_u", coeff_u)  # (M,)
            self.register_buffer("coeff_v", coeff_v)  # (M,)
            self.register_buffer("coeff_w", coeff_w)  # (M,)
            self.lengthscale = lengthscale

    @staticmethod
    def cdist2norm(
        x: torch.Tensor, y: torch.Tensor, eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute pairwise squared distances between rows of x and rows of y.
        
        Args:
            x: Tensor of shape (..., P, D)
            y: Tensor of shape (..., R, D)
            eps: small value to clamp negatives
            
        Returns:
            Tensor of shape (..., P, R) with squared distances
        """
        x_sq = x.pow(2).sum(dim=-1, keepdim=True)  # (..., P, 1)
        y_sq = y.pow(2).sum(dim=-1).unsqueeze(-2)  # (..., 1, R)
        xy = torch.matmul(x, y.transpose(-2, -1))  # (..., P, R)
        dist_sq = (x_sq + y_sq - 2.0 * xy).clamp_min(eps)  # (..., P, R)
        return dist_sq

    def forward(self, x: torch.Tensor, t=None) -> torch.Tensor:
        """
        Compute 3D sphere attractor vector field.
        
        Args:
            x: (..., 3) input points
            t: time (unused, for compatibility)
            
        Returns:
            Vector field values of same shape as x
        """
        # Base sphere attractor dynamics
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        base = x * (1 - r)
        
        if self.perturbation_magnitude > 0:
            # Add RBF perturbations
            x_flat = x.reshape(-1, 3)  # (B,3)
            dist2 = self.cdist2norm(x_flat, self.centers)
            phi = torch.exp(-dist2 / (2 * self.lengthscale**2))  # (B,M)

            u_flat = phi @ self.coeff_u  # (B,)
            v_flat = phi @ self.coeff_v  # (B,)
            w_flat = phi @ self.coeff_w  # (B,)
            uvw_flat = torch.stack([u_flat, v_flat, w_flat], dim=1)  # (B,3)

            uvw = uvw_flat.reshape(*x.shape[:-1], 3)  # (...,3)
            return base + uvw
        else:
            return base


class PerturbedSphereAttractorRNN(nn.Module):
    """
    RNN wrapper for the sphere attractor ODE, using Euler integration.
    This makes it compatible with the existing XFADS infrastructure.
    """
    
    def __init__(self, bin_sz: float = 1e-1, **ode_kwargs):
        super().__init__()
        self.ode = RbfPerturbedSphereAttractorODE(**ode_kwargs)
        self.bin_sz = bin_sz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Euler step: x_{t+1} = x_t + bin_sz * f(x_t)"""
        return x + self.ode(x) * self.bin_sz

    def vectorfield(self, x: torch.Tensor) -> torch.Tensor:
        """Return the vector field (for visualization/analysis)"""
        return self.ode(x)


# SO(3) generators for regularization
def create_so3_generators():
    """
    Create the three generators of SO(3) - infinitesimal rotation matrices.
    
    These correspond to rotations about the x, y, and z axes:
    - J1: rotation about x-axis  
    - J2: rotation about y-axis
    - J3: rotation about z-axis
    
    Returns:
        List of 3 generator functions, each taking x: (N,3) -> (N,3)
    """
    
    def rotation_x(x: torch.Tensor) -> torch.Tensor:
        """Infinitesimal rotation about x-axis: [0, -z, y]"""
        if x.dim() == 1:
            return torch.tensor([0.0, -x[2], x[1]], device=x.device, dtype=x.dtype)
        else:
            zeros = torch.zeros_like(x[..., 0])
            return torch.stack([zeros, -x[..., 2], x[..., 1]], dim=-1)
    
    def rotation_y(x: torch.Tensor) -> torch.Tensor:
        """Infinitesimal rotation about y-axis: [z, 0, -x]"""
        if x.dim() == 1:
            return torch.tensor([x[2], 0.0, -x[0]], device=x.device, dtype=x.dtype)
        else:
            zeros = torch.zeros_like(x[..., 0])
            return torch.stack([x[..., 2], zeros, -x[..., 0]], dim=-1)
    
    def rotation_z(x: torch.Tensor) -> torch.Tensor:
        """Infinitesimal rotation about z-axis: [-y, x, 0]"""
        if x.dim() == 1:
            return torch.tensor([-x[1], x[0], 0.0], device=x.device, dtype=x.dtype)
        else:
            zeros = torch.zeros_like(x[..., 0])
            return torch.stack([-x[..., 1], x[..., 0], zeros], dim=-1)
    
    return [rotation_x, rotation_y, rotation_z]


def create_combined_so3_vector_field(weights: Optional[torch.Tensor] = None):
    """
    Create a combined SO(3) vector field as a linear combination of generators.
    
    Args:
        weights: (3,) tensor of combination weights. If None, uses [1,1,1].
        
    Returns:
        Combined vector field function
    """
    if weights is None:
        weights = torch.tensor([1.0, 1.0, 1.0])
    
    generators = create_so3_generators()
    
    def combined_field(x: torch.Tensor) -> torch.Tensor:
        result = weights[0] * generators[0](x)
        result += weights[1] * generators[1](x) 
        result += weights[2] * generators[2](x)
        return result
    
    return combined_field


# Helper functions for data generation compatible with existing infrastructure
def sample_sphere_initial_conditions(n_trials: int, radius_range: Tuple[float, float] = (0.5, 1.5)) -> torch.Tensor:
    """
    Sample initial conditions around the sphere.
    
    Args:
        n_trials: Number of initial conditions to sample
        radius_range: (min_radius, max_radius) for sampling
        
    Returns:
        (n_trials, 3) tensor of initial positions
    """
    # Sample random directions on unit sphere
    directions = torch.randn(n_trials, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    
    # Sample radii
    radii = torch.rand(n_trials, 1) * (radius_range[1] - radius_range[0]) + radius_range[0]
    
    return directions * radii


if __name__ == "__main__":
    # Test the 3D sphere attractor
    print("Testing 3D Sphere Attractor...")
    
    # Create dynamics
    sphere_dynamics = PerturbedSphereAttractorRNN(
        bin_sz=0.1,
        perturbation_magnitude=0.1,
        grid_size=10,  # Smaller for testing
        lengthscale=0.3
    )
    
    # Test with sample points
    test_points = sample_sphere_initial_conditions(5)
    print(f"Test points shape: {test_points.shape}")
    
    # Test forward pass
    output = sphere_dynamics(test_points)
    print(f"Output shape: {output.shape}")
    
    # Test SO(3) generators
    so3_generators = create_so3_generators()
    combined_field = create_combined_so3_vector_field()
    
    for i, gen in enumerate(so3_generators):
        gen_output = gen(test_points)
        print(f"SO(3) generator {i} output shape: {gen_output.shape}")
    
    combined_output = combined_field(test_points)
    print(f"Combined SO(3) field shape: {combined_output.shape}")
    
    print("✅ 3D Sphere Attractor implementation ready!")