"""
Extended Regularizers for SO(3) Symmetry

This module extends the existing regularizer framework to support multiple
symmetry generators, specifically for SO(3) Lie group regularization.

Key features:
- MultiGeneratorLieDerivativeRegularizer: Regularizes against multiple symmetry generators
- SO(3) support with 3 rotation generators  
- Weighted combinations of different symmetries
- Compatible with existing regularizer interface
"""

import torch
from typing import List, Callable, Union, Optional, Dict, Any
from regularizers import LieDerivativeRegularizer, AbstractRegularizer
from __init__ import VectorField


class MultiGeneratorLieDerivativeRegularizer(AbstractRegularizer):
    """
    Lie derivative regularizer for multiple symmetry generators.
    
    This regularizer enforces that the learned vector field f(x) commutes
    with multiple symmetry generators g1(x), g2(x), ..., gk(x).
    
    The total regularization loss is:
    R = Σᵢ λᵢ * ||[f, gᵢ]||² 
    
    where [f, gᵢ] is the Lie bracket and λᵢ are the weights.
    """
    
    def __init__(
        self,
        f: VectorField,
        generators: List[VectorField],
        weights: Optional[torch.Tensor] = None,
        normalize: Union[bool, str] = False,
    ):
        """
        Args:
            f: The learned vector field to regularize
            generators: List of symmetry generators
            weights: (k,) tensor of weights for each generator. If None, uses equal weights.
            normalize: Normalization strategy for Lie derivatives
        """
        self.generators = generators
        self.n_generators = len(generators)
        
        if weights is None:
            self.weights = torch.ones(self.n_generators)
        else:
            assert len(weights) == self.n_generators
            self.weights = weights
            
        # Create individual regularizers for each generator
        self.individual_regularizers = []
        for generator in generators:
            reg = LieDerivativeRegularizer(f, generator, normalize=normalize)
            self.individual_regularizers.append(reg)
    
    @property 
    def f(self) -> VectorField:
        """Expose the wrapped vector field"""
        return self.individual_regularizers[0].f
    
    def eval_regularizer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate regularizer at points x.
        
        Args:
            x: (N, d) tensor of points
            
        Returns:
            (N,) tensor of regularization values
        """
        total_reg = torch.zeros(x.shape[0], device=x.device)
        
        for i, reg in enumerate(self.individual_regularizers):
            reg_values = reg.eval_regularizer(x)  # (N,)
            total_reg += self.weights[i] * reg_values
            
        return total_reg
    
    def regularizer(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sum regularization over all points.
        
        Args:
            x: (N, d) tensor of points
            
        Returns:
            Scalar regularization loss
        """
        return self.eval_regularizer(x).sum()
    
    def get_individual_losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get regularization loss for each individual generator.
        
        Args:
            x: (N, d) tensor of points
            
        Returns:
            Dictionary with individual losses
        """
        losses = {}
        for i, reg in enumerate(self.individual_regularizers):
            loss = reg.regularizer(x)
            losses[f"generator_{i}"] = loss
        
        losses["total"] = sum(losses.values())
        return losses


class SO3LieDerivativeRegularizer(MultiGeneratorLieDerivativeRegularizer):
    """
    Specialized regularizer for SO(3) symmetry using rotation generators.
    
    Enforces rotational symmetry by regularizing against the three
    infinitesimal rotation generators of SO(3).
    """
    
    def __init__(
        self,
        f: VectorField,
        rotation_generators: Optional[List[VectorField]] = None,
        generator_weights: Optional[torch.Tensor] = None,
        normalize: Union[bool, str] = False,
    ):
        """
        Args:
            f: The learned vector field to regularize
            rotation_generators: List of 3 SO(3) generators. If None, creates standard ones.
            generator_weights: (3,) weights for [rot_x, rot_y, rot_z]. If None, uses [1,1,1].
            normalize: Normalization strategy for Lie derivatives
        """
        if rotation_generators is None:
            from sphere_dynamics import create_so3_generators
            rotation_generators = create_so3_generators()
            
        if generator_weights is None:
            generator_weights = torch.tensor([1.0, 1.0, 1.0])
            
        super().__init__(f, rotation_generators, generator_weights, normalize)
        
        # Store generator names for interpretability
        self.generator_names = ["rotation_x", "rotation_y", "rotation_z"]
    
    def get_individual_losses(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get losses with meaningful names for SO(3) generators."""
        losses = {}
        for i, reg in enumerate(self.individual_regularizers):
            loss = reg.regularizer(x)
            losses[self.generator_names[i]] = loss
        
        losses["total_so3"] = sum(losses.values())
        return losses


# Factory functions for easy creation
def create_so3_regularizer(
    f: VectorField,
    generator_weights: Optional[torch.Tensor] = None,
    normalize: Union[bool, str] = "yang",
) -> SO3LieDerivativeRegularizer:
    """
    Factory function to create an SO(3) regularizer.
    
    Args:
        f: Vector field to regularize
        generator_weights: Weights for [x_rot, y_rot, z_rot] generators
        normalize: Normalization strategy
        
    Returns:
        Configured SO(3) regularizer
    """
    return SO3LieDerivativeRegularizer(
        f=f,
        generator_weights=generator_weights,
        normalize=normalize
    )


def create_single_axis_regularizer(
    f: VectorField,
    axis: str = "z",
    normalize: Union[bool, str] = "yang",
) -> LieDerivativeRegularizer:
    """
    Create a regularizer for rotation about a single axis.
    
    Args:
        f: Vector field to regularize
        axis: "x", "y", or "z" rotation axis
        normalize: Normalization strategy
        
    Returns:
        Single-generator Lie derivative regularizer
    """
    from sphere_dynamics import create_so3_generators
    
    generators = create_so3_generators()
    axis_map = {"x": 0, "y": 1, "z": 2}
    
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Must be 'x', 'y', or 'z'.")
    
    generator = generators[axis_map[axis]]
    
    return LieDerivativeRegularizer(f, generator, normalize=normalize)


if __name__ == "__main__":
    # Test the SO(3) regularizer
    print("Testing SO(3) Regularizers...")
    
    # Create a simple test vector field (sphere attractor)
    def test_field(x: torch.Tensor) -> torch.Tensor:
        r = torch.linalg.norm(x, dim=-1, keepdim=True)
        return x * (1 - r)
    
    # Test points
    test_points = torch.randn(10, 3)
    
    # Test SO(3) regularizer
    so3_reg = create_so3_regularizer(test_field)
    
    # Evaluate regularizer
    reg_values = so3_reg.eval_regularizer(test_points)
    total_loss = so3_reg.regularizer(test_points)
    individual_losses = so3_reg.get_individual_losses(test_points)
    
    print(f"Regularizer values shape: {reg_values.shape}")
    print(f"Total regularization loss: {total_loss.item():.6f}")
    print(f"Individual losses: {individual_losses}")
    
    # Test single-axis regularizer  
    z_reg = create_single_axis_regularizer(test_field, axis="z")
    z_loss = z_reg.regularizer(test_points)
    
    print(f"Z-axis regularization loss: {z_loss.item():.6f}")
    
    print("✅ SO(3) regularizers ready!")