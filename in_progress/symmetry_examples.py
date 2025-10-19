"""
Examples of different symmetry vector fields that can be used with the Lie derivative regularizer.

Each function defines a vector field G(x) that generates a specific type of symmetry transformation.
When used as target_vector_field in RegularizedSSM, the learned dynamics F will be encouraged 
to commute with G, i.e., [F, G] = F·∇G - G·∇F ≈ 0.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def rotation_symmetry(speed: float = 1.0) -> Callable:
    """
    Rotational symmetry: G(x) = speed * [-y, x]
    Generates rotations around the origin.
    
    Physical interpretation: System should be rotationally invariant
    Example: Ring attractor, circular dynamics
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return speed * torch.tensor([-x[1], x[0]], device=x.device, dtype=x.dtype)
        else:
            return speed * torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    return G


def translation_symmetry(direction: torch.Tensor) -> Callable:
    """
    Translational symmetry: G(x) = direction (constant vector field)
    Generates translations in the specified direction.
    
    Physical interpretation: System should be translation invariant
    Example: Traveling waves, drift-invariant dynamics
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return direction.to(x.device)
        else:
            return direction.to(x.device).expand_as(x)
    return G


def scaling_symmetry(center: torch.Tensor = None, rate: float = 1.0) -> Callable:
    """
    Scaling symmetry: G(x) = rate * (x - center)
    Generates scaling transformations around a center point.
    
    Physical interpretation: System should be scale invariant
    Example: Self-similar dynamics, fractal patterns
    """
    if center is None:
        center = torch.zeros(2)
        
    def G(x: torch.Tensor) -> torch.Tensor:
        center_expanded = center.to(x.device)
        if x.dim() == 1:
            return rate * (x - center_expanded)
        else:
            return rate * (x - center_expanded.expand_as(x))
    return G


def spiral_symmetry(rotation_speed: float = 1.0, expansion_rate: float = 0.1) -> Callable:
    """
    Spiral symmetry: combines rotation and scaling
    G(x) = rotation_speed * [-y, x] + expansion_rate * x
    
    Physical interpretation: Log-spiral invariant dynamics  
    Example: Galaxy arms, nautilus shells, hurricane patterns
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            rot_part = rotation_speed * torch.tensor([-x[1], x[0]], device=x.device, dtype=x.dtype)
            scale_part = expansion_rate * x
            return rot_part + scale_part
        else:
            rot_part = rotation_speed * torch.stack([-x[..., 1], x[..., 0]], dim=-1)
            scale_part = expansion_rate * x
            return rot_part + scale_part
    return G


def shear_symmetry(shear_direction: str = 'horizontal', rate: float = 1.0) -> Callable:
    """
    Shear symmetry: G(x) = [rate * y, 0] or [0, rate * x]
    Generates shear transformations.
    
    Physical interpretation: Shear-invariant flow patterns
    Example: Laminar flow, geological formations
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if shear_direction == 'horizontal':
            if x.dim() == 1:
                return torch.tensor([rate * x[1], 0.0], device=x.device, dtype=x.dtype)
            else:
                return torch.stack([rate * x[..., 1], torch.zeros_like(x[..., 0])], dim=-1)
        else:  # vertical
            if x.dim() == 1:
                return torch.tensor([0.0, rate * x[0]], device=x.device, dtype=x.dtype)
            else:
                return torch.stack([torch.zeros_like(x[..., 1]), rate * x[..., 0]], dim=-1)
    return G


def radial_symmetry(power: float = 1.0, inward: bool = False) -> Callable:
    """
    Radial symmetry: G(x) = ±|x|^(power-1) * x
    Generates radial flows (inward or outward).
    
    Physical interpretation: Radially symmetric dynamics
    Example: Central force fields, explosion/implosion patterns
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        r = torch.norm(x, dim=-1, keepdim=True)
        direction = -1.0 if inward else 1.0
        eps = 1e-8  # Avoid division by zero
        
        if power == 1.0:
            return direction * x
        else:
            factor = direction * (r + eps) ** (power - 1.0)
            return factor * x
    return G


def hyperbolic_symmetry(saddle_strength: float = 1.0) -> Callable:
    """
    Hyperbolic symmetry: G(x) = [saddle_strength * x, -saddle_strength * y]
    Generates hyperbolic/saddle-point dynamics.
    
    Physical interpretation: Saddle-point invariant systems
    Example: Critical points, phase transitions
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return torch.tensor([saddle_strength * x[0], -saddle_strength * x[1]], 
                              device=x.device, dtype=x.dtype)
        else:
            return torch.stack([saddle_strength * x[..., 0], -saddle_strength * x[..., 1]], dim=-1)
    return G


def oscillatory_symmetry(frequency: torch.Tensor, phase: torch.Tensor = None) -> Callable:
    """
    Oscillatory symmetry with spatially varying frequency.
    G(x) = frequency * [-sin(x·phase), cos(x·phase)]
    
    Physical interpretation: Spatially modulated oscillations
    Example: Wave patterns, neural oscillations with spatial variation
    """
    if phase is None:
        phase = torch.tensor([1.0, 0.0])
        
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            phase_val = torch.dot(x, phase.to(x.device))
            freq_val = frequency.to(x.device)
            return freq_val * torch.tensor([-torch.sin(phase_val), torch.cos(phase_val)], 
                                         device=x.device, dtype=x.dtype)
        else:
            phase_val = torch.sum(x * phase.to(x.device).expand_as(x), dim=-1)
            freq_val = frequency.to(x.device)
            return freq_val.unsqueeze(-1) * torch.stack([-torch.sin(phase_val), torch.cos(phase_val)], dim=-1)
    return G


def custom_nonlinear_symmetry() -> Callable:
    """
    Example of a complex nonlinear symmetry.
    G(x) = [x²y - y³, -x³ + xy²] (generates complex invariant curves)
    
    Physical interpretation: Nonlinear transformation group
    Example: Complex dynamical systems with curved invariant manifolds
    """
    def G(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x_val, y_val = x[0], x[1]
            return torch.tensor([x_val**2 * y_val - y_val**3, -x_val**3 + x_val * y_val**2], 
                              device=x.device, dtype=x.dtype)
        else:
            x_val, y_val = x[..., 0], x[..., 1]
            return torch.stack([x_val**2 * y_val - y_val**3, -x_val**3 + x_val * y_val**2], dim=-1)
    return G


# Visualization utilities
def visualize_symmetry_fields(symmetries: dict, grid_size: int = 20, domain: tuple = (-2, 2)):
    """Visualize multiple symmetry vector fields side by side."""
    n_symmetries = len(symmetries)
    fig, axes = plt.subplots(1, n_symmetries, figsize=(5*n_symmetries, 5))
    if n_symmetries == 1:
        axes = [axes]
    
    # Create grid
    x = np.linspace(domain[0], domain[1], grid_size)
    y = np.linspace(domain[0], domain[1], grid_size)
    X, Y = np.meshgrid(x, y)
    points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
    
    for ax, (name, symmetry_fn) in zip(axes, symmetries.items()):
        # Evaluate vector field
        vectors = symmetry_fn(points).numpy()
        U = vectors[:, 0].reshape(grid_size, grid_size)
        V = vectors[:, 1].reshape(grid_size, grid_size)
        
        # Plot
        ax.streamplot(X, Y, U, V, density=1.5, color='blue', alpha=0.7)
        ax.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3], 
                 color='red', alpha=0.8, scale=50)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(domain)
        ax.set_ylim(domain)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def test_symmetry_examples():
    """Test and visualize various symmetry examples."""
    
    # Define different symmetries
    symmetries = {
        'Rotation': rotation_symmetry(speed=1.0),
        'Translation': translation_symmetry(torch.tensor([1.0, 0.5])),
        'Scaling': scaling_symmetry(rate=0.5),
        'Spiral': spiral_symmetry(rotation_speed=1.0, expansion_rate=0.2),
        'Radial Inward': radial_symmetry(power=1.0, inward=True),
        'Hyperbolic': hyperbolic_symmetry(saddle_strength=0.8)
    }
    
    print("🌟 Testing symmetry vector fields...")
    
    # Test each symmetry
    test_point = torch.tensor([1.0, 0.5])
    for name, symmetry_fn in symmetries.items():
        result = symmetry_fn(test_point)
        print(f"  {name}: G([1.0, 0.5]) = [{result[0]:.3f}, {result[1]:.3f}]")
    
    # Visualize
    fig = visualize_symmetry_fields(symmetries)
    fig.savefig('symmetry_examples.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Symmetry examples saved as 'symmetry_examples.png'")
    
    return symmetries


if __name__ == "__main__":
    # Run the test
    test_symmetry_examples()
    
    print("\n🎯 Usage in RegularizedSSM:")
    print("target_field = rotation_symmetry(speed=1.0)")
    print("regularized_ssm = RegularizedSSM(..., target_vector_field=target_field)")