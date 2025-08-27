"""
Utilities for analyzing loss components and regularization strength in RegularizedSSM.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import seaborn as sns

from regularized_ssm import RegularizedSSM
from regularizers import LieDerivativeRegularizer


class LossComponentAnalyzer:
    """
    Analyzer for tracking and visualizing loss components during training.
    """
    
    def __init__(self, regularized_ssm: RegularizedSSM, device: str = 'cpu'):
        self.ssm = regularized_ssm
        self.device = device
        self.loss_history = []
        
    def evaluate_loss_components(self, 
                                y: torch.Tensor, 
                                n_samples: int = 10,
                                batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate all loss components on a dataset.
        
        Args:
            y: Observations [n_trials, n_time, n_neurons]
            n_samples: Number of samples for Monte Carlo
            batch_size: Process in batches if specified
            
        Returns:
            Dictionary with loss components
        """
        self.ssm.eval()
        
        if batch_size is None:
            with torch.no_grad():
                return self.ssm.get_loss_components(y, n_samples)
        
        # Process in batches
        n_trials = y.shape[0]
        n_batches = (n_trials + batch_size - 1) // batch_size
        
        total_losses = {
            'elbo_loss': 0.0,
            'lie_loss': 0.0, 
            'curvature_loss': 0.0,
            'total_reg_loss': 0.0,
            'total_loss': 0.0
        }
        
        with torch.no_grad():
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_trials)
                batch_y = y[start_idx:end_idx]
                
                batch_losses = self.ssm.get_loss_components(batch_y, n_samples)
                batch_weight = (end_idx - start_idx) / n_trials
                
                for key in total_losses:
                    total_losses[key] += batch_weight * batch_losses[key]
        
        return total_losses
    
    def track_training_step(self, 
                           y_train: torch.Tensor, 
                           y_valid: torch.Tensor,
                           n_samples: int = 10,
                           epoch: int = 0):
        """Track loss components for one training step."""
        train_losses = self.evaluate_loss_components(y_train, n_samples, batch_size=100)
        valid_losses = self.evaluate_loss_components(y_valid, n_samples, batch_size=100)
        
        step_data = {'epoch': epoch}
        for key in train_losses:
            step_data[f'train_{key}'] = train_losses[key]
            step_data[f'valid_{key}'] = valid_losses[key]
            
        self.loss_history.append(step_data)
        return step_data
    
    def plot_loss_components(self, 
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot all loss components over training.
        """
        if not self.loss_history:
            raise ValueError("No loss history recorded. Call track_training_step first.")
            
        df = pd.DataFrame(self.loss_history)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.ravel()
        
        components = ['total_loss', 'elbo_loss', 'lie_loss', 'curvature_loss', 'total_reg_loss']
        colors = ['black', 'blue', 'red', 'green', 'purple']
        
        for i, (component, color) in enumerate(zip(components, colors)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Plot train and validation
            train_col = f'train_{component}'
            valid_col = f'valid_{component}'
            
            if train_col in df.columns:
                ax.plot(df['epoch'], df[train_col], f'{color}-', 
                       label='Train', linewidth=2, alpha=0.8)
            if valid_col in df.columns:
                ax.plot(df['epoch'], df[valid_col], f'{color}--', 
                       label='Valid', linewidth=2, alpha=0.8)
            
            ax.set_title(component.replace('_', ' ').title(), fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        # Remove empty subplot
        if len(axes) > len(components):
            fig.delaxes(axes[-1])
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def analyze_lie_derivative_spatial(self, 
                                     grid_size: int = 50,
                                     domain: Tuple[float, float] = (-2, 2)) -> Dict[str, np.ndarray]:
        """
        Analyze Lie derivative loss spatially across the latent space.
        
        Returns:
            Dictionary with spatial analysis results
        """
        # Create grid
        x = np.linspace(domain[0], domain[1], grid_size)
        y = np.linspace(domain[0], domain[1], grid_size) 
        X, Y = np.meshgrid(x, y)
        points = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=torch.float32)
        
        with torch.no_grad():
            # Evaluate Lie derivative components
            lie_values = self.ssm.lie_regularizer.eval_regularizer(points)
            lie_bracket = self.ssm.lie_regularizer(points)
            
            # Evaluate learned and target vector fields
            F_values = self.ssm.dynamics_mod.mean_fn(points)
            G_values = self.ssm.target_vector_field(points)
        
        results = {
            'X': X,
            'Y': Y, 
            'lie_loss': lie_values.numpy().reshape(grid_size, grid_size),
            'lie_bracket_x': lie_bracket[:, 0].numpy().reshape(grid_size, grid_size),
            'lie_bracket_y': lie_bracket[:, 1].numpy().reshape(grid_size, grid_size),
            'F_x': F_values[:, 0].numpy().reshape(grid_size, grid_size),
            'F_y': F_values[:, 1].numpy().reshape(grid_size, grid_size),
            'G_x': G_values[:, 0].numpy().reshape(grid_size, grid_size),
            'G_y': G_values[:, 1].numpy().reshape(grid_size, grid_size)
        }
        
        return results
    
    def plot_spatial_analysis(self, 
                            spatial_data: Dict[str, np.ndarray],
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spatial analysis of Lie derivative loss.
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        X, Y = spatial_data['X'], spatial_data['Y']
        
        # 1. Learned vector field F
        ax = axes[0, 0]
        ax.streamplot(X, Y, spatial_data['F_x'], spatial_data['F_y'], 
                     color='blue', density=1.5)
        ax.set_title('Learned Vector Field F', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 2. Target vector field G
        ax = axes[0, 1]
        ax.streamplot(X, Y, spatial_data['G_x'], spatial_data['G_y'], 
                     color='red', density=1.5)
        ax.set_title('Target Vector Field G', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 3. Lie bracket [F, G]
        ax = axes[0, 2]
        ax.streamplot(X, Y, spatial_data['lie_bracket_x'], spatial_data['lie_bracket_y'], 
                     color='purple', density=1.5)
        ax.set_title('Lie Bracket [F, G]', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 4. Lie derivative loss magnitude
        ax = axes[1, 0]
        im = ax.contourf(X, Y, spatial_data['lie_loss'], levels=20, cmap='viridis')
        ax.set_title('Lie Derivative Loss ||[F, G]||²', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # 5. Log-scale loss
        ax = axes[1, 1]
        im = ax.contourf(X, Y, np.log10(spatial_data['lie_loss'] + 1e-8), 
                        levels=20, cmap='plasma')
        ax.set_title('Log₁₀(Lie Loss)', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # 6. Difference magnitude |F| - |G|
        F_mag = np.sqrt(spatial_data['F_x']**2 + spatial_data['F_y']**2)
        G_mag = np.sqrt(spatial_data['G_x']**2 + spatial_data['G_y']**2)
        ax = axes[1, 2]
        im = ax.contourf(X, Y, F_mag - G_mag, levels=20, cmap='RdBu_r')
        ax.set_title('Magnitude Difference |F| - |G|', fontweight='bold')
        ax.set_xlabel('x₁')
        ax.set_ylabel('x₂')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


class RegularizationStrengthAnalyzer:
    """
    Analyze the effect of different regularization strengths.
    """
    
    def __init__(self):
        self.results = []
        
    def sweep_lambda_values(self, 
                          lambda_values: List[float],
                          ssm_factory: Callable,
                          y_data: torch.Tensor,
                          n_samples: int = 10) -> List[Dict]:
        """
        Sweep over different lambda values and evaluate loss components.
        
        Args:
            lambda_values: List of regularization strengths to test
            ssm_factory: Function that creates SSM given lambda_lie parameter  
            y_data: Test data
            n_samples: Monte Carlo samples
            
        Returns:
            List of results for each lambda value
        """
        results = []
        
        for lambda_lie in lambda_values:
            print(f"Testing λ_lie = {lambda_lie}")
            
            # Create SSM with this lambda
            ssm = ssm_factory(lambda_lie=lambda_lie)
            analyzer = LossComponentAnalyzer(ssm)
            
            # Evaluate loss components
            losses = analyzer.evaluate_loss_components(y_data, n_samples)
            
            result = {
                'lambda_lie': lambda_lie,
                **losses
            }
            results.append(result)
            
        self.results = results
        return results
    
    def plot_lambda_sweep(self, 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot how loss components vary with regularization strength.
        """
        if not self.results:
            raise ValueError("No results to plot. Run sweep_lambda_values first.")
            
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. All loss components
        ax = axes[0]
        ax.loglog(df['lambda_lie'], df['total_loss'], 'k-o', label='Total', linewidth=2)
        ax.loglog(df['lambda_lie'], df['elbo_loss'], 'b-s', label='ELBO', linewidth=2)
        ax.loglog(df['lambda_lie'], df['lie_loss'], 'r-^', label='Lie', linewidth=2)
        ax.loglog(df['lambda_lie'], df['total_reg_loss'], 'g-d', label='Total Reg', linewidth=2)
        ax.set_xlabel('λ_lie')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components vs λ_lie', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Relative contribution
        ax = axes[1]
        elbo_fraction = df['elbo_loss'] / df['total_loss']
        reg_fraction = df['total_reg_loss'] / df['total_loss']
        ax.semilogx(df['lambda_lie'], elbo_fraction, 'b-o', label='ELBO fraction', linewidth=2)
        ax.semilogx(df['lambda_lie'], reg_fraction, 'r-s', label='Reg fraction', linewidth=2)
        ax.set_xlabel('λ_lie')
        ax.set_ylabel('Fraction of Total Loss')
        ax.set_title('Loss Composition', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 3. Trade-off curve
        ax = axes[2]
        ax.loglog(df['elbo_loss'], df['lie_loss'], 'ko-', linewidth=2, markersize=6)
        
        # Add lambda labels
        for i, row in df.iterrows():
            if i % 2 == 0:  # Label every other point
                ax.annotate(f'λ={row["lambda_lie"]:.1e}', 
                          (row['elbo_loss'], row['lie_loss']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('ELBO Loss')
        ax.set_ylabel('Lie Loss')
        ax.set_title('ELBO vs Regularization Trade-off', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def find_optimal_lambda(self, 
                           target_lie_loss: float = 1e-3,
                           method: str = 'target') -> Dict:
        """
        Find optimal lambda value based on different criteria.
        
        Args:
            target_lie_loss: Target Lie loss value
            method: 'target', 'elbow', or 'balanced'
            
        Returns:
            Dictionary with optimal lambda and analysis
        """
        if not self.results:
            raise ValueError("No results available. Run sweep_lambda_values first.")
            
        df = pd.DataFrame(self.results)
        
        if method == 'target':
            # Find lambda that achieves target Lie loss
            distances = np.abs(df['lie_loss'] - target_lie_loss)
            best_idx = distances.argmin()
            
        elif method == 'elbow':
            # Find elbow in ELBO vs Lie trade-off curve
            elbo_normalized = (df['elbo_loss'] - df['elbo_loss'].min()) / (df['elbo_loss'].max() - df['elbo_loss'].min())
            lie_normalized = (df['lie_loss'] - df['lie_loss'].min()) / (df['lie_loss'].max() - df['lie_loss'].min())
            distances = np.sqrt(elbo_normalized**2 + lie_normalized**2)
            best_idx = distances.argmin()
            
        elif method == 'balanced':
            # Find where ELBO and regularization contribute equally
            elbo_fraction = df['elbo_loss'] / df['total_loss']
            distances = np.abs(elbo_fraction - 0.5)
            best_idx = distances.argmin()
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        optimal_result = df.iloc[best_idx].to_dict()
        
        return {
            'method': method,
            'optimal_lambda': optimal_result['lambda_lie'],
            'optimal_result': optimal_result,
            'analysis': {
                'elbo_loss': optimal_result['elbo_loss'],
                'lie_loss': optimal_result['lie_loss'],
                'total_loss': optimal_result['total_loss'],
                'elbo_fraction': optimal_result['elbo_loss'] / optimal_result['total_loss'],
                'reg_fraction': optimal_result['total_reg_loss'] / optimal_result['total_loss']
            }
        }


# Example usage functions
def create_perfect_rotation_test():
    """
    Create a test to find lambda that creates perfect rotation symmetry.
    """
    
    def rotation_field(x):
        """Perfect rotation field G(x) = [-y, x]"""
        if x.dim() == 1:
            return torch.tensor([-x[1], x[0]], device=x.device, dtype=x.dtype)
        else:
            return torch.stack([-x[..., 1], x[..., 0]], dim=-1)
    
    # Test points on circles of different radii
    n_points = 100
    radii = [0.5, 1.0, 1.5, 2.0]
    test_points = []
    
    for r in radii:
        angles = torch.linspace(0, 2*np.pi, n_points//len(radii))
        points = r * torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        test_points.append(points)
    
    test_points = torch.cat(test_points, dim=0)
    
    def evaluate_rotation_symmetry(learned_field, target_field, points):
        """Evaluate how well learned field satisfies rotation symmetry."""
        with torch.no_grad():
            # Compute Lie bracket [F, G]
            regularizer = LieDerivativeRegularizer(learned_field, target_field)
            lie_bracket = regularizer(points)
            lie_loss = regularizer.eval_regularizer(points)
            
            # Perfect rotation symmetry means [F, G] = 0 everywhere
            max_violation = lie_loss.max().item()
            mean_violation = lie_loss.mean().item()
            
            return {
                'max_violation': max_violation,
                'mean_violation': mean_violation,
                'lie_bracket': lie_bracket,
                'lie_loss': lie_loss
            }
    
    return test_points, rotation_field, evaluate_rotation_symmetry


if __name__ == "__main__":
    print("🔍 Loss Analysis Utilities Ready!")
    print("\nExample usage:")
    print("analyzer = LossComponentAnalyzer(regularized_ssm)")
    print("analyzer.track_training_step(y_train, y_valid, epoch=0)")
    print("analyzer.plot_loss_components('loss_components.png')")
    print("\nstrength_analyzer = RegularizationStrengthAnalyzer()")
    print("strength_analyzer.sweep_lambda_values([0.1, 1.0, 10.0], ssm_factory, y_data)")
    print("strength_analyzer.plot_lambda_sweep('lambda_sweep.png')")