import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from typing import Callable, TypeAlias, Union
from functools import partial
import torchdiffeq

VectorField = Union[nn.Module, Callable[[Tensor], Tensor]]
nn_activation: TypeAlias = torch.nn.modules.activation
odeint = partial(torchdiffeq.odeint, options={"dtype": torch.float32})


class LieDerivative(nn.Module):
    """
    Computes the Lie bracket [X, Y] = X·∇Y − Y·∇X of two vector fields X, Y : ℝⁿ → ℝⁿ,
    in a fully differentiable way.
    X, Y may be nn.Modules or plain callables.
    """
    def __init__(self, f: VectorField, g: VectorField) -> None:
        super().__init__()

        if isinstance(f, nn.Module):
            self.X = f
        else:
            mod = nn.Module()
            mod.forward = f  # type: ignore[attr-defined]
            self.X = mod

        if isinstance(g, nn.Module):
            self.Y = g
        else:
            mod = nn.Module()
            mod.forward = g  # type: ignore[attr-defined]
            self.Y = mod

    def forward(self, x: Tensor) -> Tensor:
        f, g, x = self._eval_vfields(x)

        dg = torch.autograd.grad(
            outputs=g,
            inputs=x,
            grad_outputs=f,
            create_graph=True,
            retain_graph=True
        )[0]

        # directional derivative dX = ∇X(x) · Y_x
        df = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=g,
            create_graph=True
        )[0]

        return dg - df

    def regularizer(self, x, *, normalize: bool = True) -> Tensor:
        f, g, x = self._eval_vfields(x)

        dg = torch.autograd.grad(
            outputs=g,
            inputs=x,
            grad_outputs=f,
            create_graph=True,
            retain_graph=True
        )[0]

        # directional derivative dX = ∇X(x) · Y_x
        df = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=g,
            create_graph=True
        )[0]
        eps = torch.finfo(df.dtype).eps
        ell2 = torch.norm(dg - df, p=2, dim=-1)
        ell2 = torch.nn.functional.mse_loss(dg, df, reduction="sum")
        if normalize:
            dg.square().sum(dim=-1)
            dg_norm = dg.square().sum(dim=-1)
            return torch.sum(ell2 / (dg_norm + eps))
        else:
            return torch.sum(ell2)

    def _eval_vfields(self, x):
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        f = self.X(x)
        g = self.Y(x)
        return f, g, x


def dynamics_factory(A) -> Callable[[Tensor], Tensor]:
    def f(x: Tensor) -> Tensor:
        r = torch.sqrt(torch.sum((x @ A) * x, dim=-1, keepdim=True))
        return x * (1 - r)
    return f


def visualize_lie_derivative_regularizer():
    """
    Visualize the Lie derivative regularizer as a heatmap overlaid on the vector field.
    """
    # Set up the dynamics
    A = torch.eye(2).requires_grad_(True)
    f = dynamics_factory(A)

    # Define rotational symmetry generator
    def v(x: Tensor) -> Tensor:
        return x @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])

    # Create Lie derivative object
    lie = LieDerivative(f, v)

    # Create a grid of points to evaluate the regularizer
    n_grid = 50
    x_range = torch.linspace(-2.5, 2.5, n_grid)
    y_range = torch.linspace(-2.5, 2.5, n_grid)
    X_grid, Y_grid = torch.meshgrid(x_range, y_range, indexing='xy')
    grid_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=-1).requires_grad_(True)

    # Compute regularizer values at each grid point
    regularizer_values = []
    for i in range(grid_points.shape[0]):
        point = grid_points[i:i+1]  # Keep batch dimension
        reg_val = lie.regularizer(point, normalize=False)
        regularizer_values.append(reg_val.item())

    regularizer_values = torch.tensor(regularizer_values).reshape(n_grid, n_grid)

    # Create the visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Vector field streamplot
    x_np = X_grid.detach().numpy()
    y_np = Y_grid.detach().numpy()

    # Evaluate vector field
    with torch.no_grad():
        f_vals = torch.stack([f(point) for point in grid_points])
        U = f_vals[:, 0].reshape(n_grid, n_grid).numpy()
        V = f_vals[:, 1].reshape(n_grid, n_grid).numpy()

    ax1.streamplot(x_np, y_np, U, V, density=1.5, color='blue')
    ax1.set_title('Ring Attractor Vector Field', fontsize=14)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Regularizer heatmap with streamplot overlay
    reg_vals_np = regularizer_values.detach().numpy()
    # Use log scale to better visualize the variations
    reg_vals_log = np.log10(reg_vals_np + 1e-10)

    im = ax2.imshow(reg_vals_log, extent=[-2.5, 2.5, -2.5, 2.5], 
                    origin='lower', cmap='hot', alpha=0.8)
    ax2.streamplot(x_np, y_np, U, V, density=1.0, color='white', linewidth=0.8)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Log₁₀(Lie Derivative Regularizer)', rotation=270, labelpad=20)

    ax2.set_title('Lie Derivative Regularizer Heatmap\n(with Vector Field Overlay)', fontsize=14)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Regularizer statistics:")
    print(f"  Min: {reg_vals_np.min():.3e}")
    print(f"  Max: {reg_vals_np.max():.3e}")
    print(f"  Mean: {reg_vals_np.mean():.3e}")
    print(f"  Std: {reg_vals_np.std():.3e}")

    return regularizer_values, X_grid, Y_grid


if __name__ == "__main__":
    visualize_lie_derivative_regularizer()