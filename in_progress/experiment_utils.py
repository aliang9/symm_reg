"""
Utility functions and classes for regularized SSM experiments.

This module contains helper functions and classes that are used by
regularized_experiments.py to keep the main file cleaner and more focused.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import importlib.util

import xfads.plot_utils as plot_utils


class ComprehensiveLossLogger:
    """Enhanced logger that tracks loss components and creates comprehensive vector field visualizations."""
    
    def __init__(self, experiment_name=None, save_vector_fields=True, perfect_ring_dynamics=None, 
                 perturbed_ring_dynamics=None, target_field=None):
        self.experiment_name = experiment_name or "unknown_experiment"
        self.save_vector_fields = save_vector_fields
        self.vector_field_save_dir = None
        
        # Store reference dynamics for comparison
        self.perfect_ring_dynamics = perfect_ring_dynamics
        self.perturbed_ring_dynamics = perturbed_ring_dynamics  
        self.target_field = target_field
        
        self.reset()
        
        # Set up vector field saving directory
        if self.save_vector_fields:
            self.vector_field_save_dir = Path(f'in_progress/{self.experiment_name}/vector_fields')
            self.vector_field_save_dir.mkdir(parents=True, exist_ok=True)
            print(f"📁 Vector field plots will be saved to: {self.vector_field_save_dir}")
    
    def reset(self):
        self.train_logs = defaultdict(list)
        self.valid_logs = defaultdict(list)
        self.epochs = []
    
    def compute_vector_field_differences(self, learned_dynamics, grid_size=50):
        """
        Compute differences between learned dynamics and reference fields on a grid.
        
        Returns:
            dict: Contains grid points and difference magnitudes for each comparison
        """
        try:
            # Create evaluation grid
            x = np.linspace(-2, 2, grid_size)
            y = np.linspace(-2, 2, grid_size)
            X, Y = np.meshgrid(x, y)
            grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
            
            # Evaluate learned dynamics
            with torch.no_grad():
                learned_output = learned_dynamics(grid_points)
                learned_field = learned_output - grid_points  # Convert to vector field
            
            differences = {}
        except Exception as e:
            print(f"   ⚠️ Could not compute vector field differences: {e}")
            return {}
        
        # Difference from perfect ring attractor
        if self.perfect_ring_dynamics is not None:
            with torch.no_grad():
                perfect_output = self.perfect_ring_dynamics(grid_points)
                perfect_field = perfect_output - grid_points
                diff_perfect = learned_field - perfect_field
                diff_perfect_magnitude = torch.norm(diff_perfect, dim=1)
                differences['perfect_ring'] = {
                    'grid_points': grid_points.numpy(),
                    'grid_shape': (grid_size, grid_size),
                    'difference_magnitude': diff_perfect_magnitude.numpy(),
                    'difference_vector': diff_perfect.numpy(),
                    'max_diff': diff_perfect_magnitude.max().item(),
                    'mean_diff': diff_perfect_magnitude.mean().item()
                }
        
        # Difference from perturbed ring attractor (data generation)
        if self.perturbed_ring_dynamics is not None:
            with torch.no_grad():
                perturbed_output = self.perturbed_ring_dynamics(grid_points)
                perturbed_field = perturbed_output - grid_points
                diff_perturbed = learned_field - perturbed_field
                diff_perturbed_magnitude = torch.norm(diff_perturbed, dim=1)
                differences['perturbed_ring'] = {
                    'grid_points': grid_points.numpy(),
                    'grid_shape': (grid_size, grid_size),
                    'difference_magnitude': diff_perturbed_magnitude.numpy(),
                    'difference_vector': diff_perturbed.numpy(),
                    'max_diff': diff_perturbed_magnitude.max().item(),
                    'mean_diff': diff_perturbed_magnitude.mean().item()
                }
        
        # Difference from target rotational field
        if self.target_field is not None:
            with torch.no_grad():
                target_output = self.target_field(grid_points)
                diff_target = learned_field - target_output
                diff_target_magnitude = torch.norm(diff_target, dim=1)
                differences['target_rotation'] = {
                    'grid_points': grid_points.numpy(),
                    'grid_shape': (grid_size, grid_size),
                    'difference_magnitude': diff_target_magnitude.numpy(),
                    'difference_vector': diff_target.numpy(),
                    'max_diff': diff_target_magnitude.max().item(),
                    'mean_diff': diff_target_magnitude.mean().item()
                }
        
        return differences
    
    def plot_and_save_vector_field(self, ssm, epoch, label=None):
        """Plot and save vector field with optional comprehensive analysis."""
        if not self.save_vector_fields or self.vector_field_save_dir is None:
            return
        
        try:
            # Basic vector field plot
            fig, ax = plt.subplots(figsize=(8, 8))
            if hasattr(ssm, 'dynamics_mod') and hasattr(ssm.dynamics_mod, 'mean_fn'):
                plot_utils.plot_two_d_vector_field(ssm.dynamics_mod.mean_fn, ax, min_xy=-2, max_xy=2)
            else:
                ax.text(0.5, 0.5, 'Vector field not available', ha='center', va='center', transform=ax.transAxes)
            
            if label:
                title = f'Vector Field - {label.title()} (Epoch {epoch})'
            else:
                title = f'Vector Field - Epoch {epoch}'
                
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Latent Dim 1')
            ax.set_ylabel('Latent Dim 2')
            ax.set_xlim(-2, 2)
            ax.set_ylim(-2, 2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            if label:
                filename = f'vector_field_epoch_{epoch:03d}_{label}.png'
            else:
                filename = f'vector_field_epoch_{epoch:03d}.png'
                
            save_path = self.vector_field_save_dir / filename
            fig.savefig(str(save_path), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            return save_path
        except Exception as e:
            print(f"   ⚠️ Could not save vector field plot: {e}")
            return None
    
    def log_epoch(self, epoch, train_stats, valid_stats):
        """Log key loss components for both training and validation."""
        self.epochs.append(epoch)
        
        # Define which losses to track
        loss_keys = ['total_loss', 'elbo_loss', 'lie_loss', 'recon_loss', 'kl_loss', 'total_reg_loss', 'curvature_loss']
        
        # Log training losses
        for key in loss_keys:
            if key in train_stats:
                value = train_stats[key]
                if isinstance(value, torch.Tensor):
                    self.train_logs[key].append(value.item())
                else:
                    self.train_logs[key].append(value)
            else:
                # For baseline runs (lambda=0), some losses might be zero or missing
                if key in ['lie_loss', 'curvature_loss', 'total_reg_loss']:
                    self.train_logs[key].append(0.0)
        
        # Log validation losses
        for key in loss_keys:
            if key in valid_stats:
                value = valid_stats[key]
                if isinstance(value, torch.Tensor):
                    self.valid_logs[key].append(value.item())
                else:
                    self.valid_logs[key].append(value)
            else:
                # For baseline runs (lambda=0), some losses might be zero or missing
                if key in ['lie_loss', 'curvature_loss', 'total_reg_loss']:
                    self.valid_logs[key].append(0.0)
    
    def get_dataframe(self):
        """Convert logs to pandas DataFrame for analysis."""
        data = {'epoch': self.epochs}
        
        # Add training data with 'train_' prefix
        for key, values in self.train_logs.items():
            data[f'train_{key}'] = values
            
        # Add validation data with 'valid_' prefix
        for key, values in self.valid_logs.items():
            data[f'valid_{key}'] = values
            
        return pd.DataFrame(data)


def train_with_comprehensive_logging(ssm, y_train, y_valid, logger, cfg, lambda_lie=1.0):
    """Enhanced training function with comprehensive loss logging and vector field plotting."""
    
    optimizer = torch.optim.Adam(ssm.parameters(), lr=1e-3)
    
    # Early stopping based on validation ELBO
    best_valid_elbo = float('inf')
    patience_counter = 0
    patience = getattr(cfg, 'patience', 5)
    max_epochs = getattr(cfg, 'max_epochs', getattr(cfg, 'n_epochs', 50))
    grad_clip = getattr(cfg, 'grad_clip', 1.0)
    batch_sz = getattr(cfg, 'batch_sz', 32)
    n_samples = getattr(cfg, 'n_samples', 5)
    best_model_state = None
    
    print(f"🚀 Training with λ_lie = {lambda_lie:.2e}")
    print(f"   Max epochs: {max_epochs}, Patience: {patience}")
    print(f"   Early stopping on: validation ELBO loss")
    
    # Plot and save initial vector field (before training)
    if logger:
        logger.plot_and_save_vector_field(ssm, epoch=-1, label="initial")
    
    # Get initial losses before any training (epoch 0)
    ssm.eval()
    with torch.no_grad():
        # Initial training loss (small batch to match training)
        train_batch = y_train[:batch_sz]
        initial_train_loss, _, initial_train_stats = ssm(train_batch, n_samples)
        
        # Initial validation loss
        initial_valid_loss, _, initial_valid_stats = ssm(y_valid, n_samples)
    
    # Log initial state
    if logger:
        logger.log_epoch(0, initial_train_stats, initial_valid_stats)
    print(f"   Initial: Train ELBO = {initial_train_stats['elbo_loss']:.3f}, "
          f"Valid ELBO = {initial_valid_stats['elbo_loss']:.3f}, "
          f"Lie = {initial_valid_stats['lie_loss']:.3f}")
    
    for epoch in range(1, max_epochs + 1):
        # Training phase
        ssm.train()
        train_losses = []
        
        # Shuffle training data
        indices = torch.randperm(len(y_train))
        
        for i in range(0, len(y_train), batch_sz):
            batch_indices = indices[i:i+batch_sz]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            total_loss, _, stats = ssm(y_batch, n_samples)
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(ssm.parameters(), grad_clip)
            optimizer.step()
            
            train_losses.append(stats)
        
        # Average training statistics
        train_stats = {}
        for key in train_losses[0].keys():
            train_stats[key] = torch.stack([batch[key] for batch in train_losses]).mean()
        
        # Validation phase
        ssm.eval()
        with torch.no_grad():
            valid_total_loss, _, valid_stats = ssm(y_valid, n_samples)
        
        # Log epoch
        if logger:
            logger.log_epoch(epoch, train_stats, valid_stats)
            # Plot and save vector field after this epoch (every 5 epochs to save space)
            if epoch % 5 == 0 or epoch == max_epochs:
                logger.plot_and_save_vector_field(ssm, epoch)
        
        # Early stopping check (using validation ELBO)
        current_valid_elbo = valid_stats['elbo_loss'].item()
        
        if current_valid_elbo < best_valid_elbo:
            best_valid_elbo = current_valid_elbo
            patience_counter = 0
            best_model_state = ssm.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print progress every 10 epochs
        if epoch % 10 == 0 or patience_counter == 0:
            print(f"   Epoch {epoch:3d}: Train ELBO = {train_stats['elbo_loss']:.3f}, "
                  f"Valid ELBO = {current_valid_elbo:.3f}, "
                  f"Lie = {valid_stats['lie_loss']:.3f}, "
                  f"Patience = {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"   🛑 Early stopping at epoch {epoch} (best valid ELBO: {best_valid_elbo:.6f})")
            break
    
    # Restore best model
    if best_model_state is not None:
        ssm.load_state_dict(best_model_state)
        print(f"   ✅ Restored model with best validation ELBO: {best_valid_elbo:.6f}")
    
    # Plot and save final vector field
    if logger:
        logger.plot_and_save_vector_field(ssm, epoch, label="final")
    
    return {
        'final_epoch': epoch,
        'best_valid_elbo': best_valid_elbo,
        'converged': patience_counter < patience
    }


def create_reference_dynamics(perturbation_magnitude, device):
    """Create reference dynamics for comparison (perfect and perturbed ring attractors)."""
    try:
        file_path = "test_dynamics.py"
        spec = importlib.util.spec_from_file_location("test_dynamics", file_path)
        test_dynamics = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(test_dynamics)
        PerturbedRingAttractorDynamics = test_dynamics.PerturbedRingAttractorRNN
        
        perfect_ring_dynamics = PerturbedRingAttractorDynamics(
            bin_sz=1e-1, lengthscale=0.2, perturbation_magnitude=0.0
        ).to(device)
        perturbed_ring_dynamics = PerturbedRingAttractorDynamics(
            bin_sz=1e-1, lengthscale=0.2, perturbation_magnitude=perturbation_magnitude
        ).to(device)
        
        return perfect_ring_dynamics, perturbed_ring_dynamics
    except Exception as e:
        print(f"   ⚠️ Could not create reference dynamics: {e}")
        return None, None


def find_optimal_lambda_criteria(df_results):
    """Find optimal regularization values using different criteria."""
    results = {}
    
    if df_results.empty:
        print("   ⚠️ No results to analyze")
        return results
    
    # 1. Minimum total loss
    min_total_idx = df_results['final_total_loss'].idxmin()
    results['min_total_loss'] = {
        'lambda_lie': df_results.loc[min_total_idx, 'lambda_lie'],
        'total_loss': df_results.loc[min_total_idx, 'final_total_loss'],
        'elbo_loss': df_results.loc[min_total_idx, 'final_elbo_loss'],
        'lie_loss': df_results.loc[min_total_idx, 'final_lie_loss'] if 'final_lie_loss' in df_results.columns else 0.0
    }
    
    # 2. Minimum ELBO loss (data fitting)
    min_elbo_idx = df_results['final_elbo_loss'].idxmin()
    results['min_elbo_loss'] = {
        'lambda_lie': df_results.loc[min_elbo_idx, 'lambda_lie'],
        'total_loss': df_results.loc[min_elbo_idx, 'final_total_loss'],
        'elbo_loss': df_results.loc[min_elbo_idx, 'final_elbo_loss'],
        'lie_loss': df_results.loc[min_elbo_idx, 'final_lie_loss'] if 'final_lie_loss' in df_results.columns else 0.0
    }
    
    # 3. Best lie loss while keeping ELBO reasonable (< 2x minimum) - only if lie loss exists
    if 'final_lie_loss' in df_results.columns and (df_results['final_lie_loss'] > 0).any():
        min_elbo = df_results['final_elbo_loss'].min()
        reasonable_elbo_mask = df_results['final_elbo_loss'] <= 2 * min_elbo
        if reasonable_elbo_mask.any():
            reasonable_df = df_results[reasonable_elbo_mask]
            # Only consider non-zero lie losses
            non_zero_lie_mask = reasonable_df['final_lie_loss'] > 0
            if non_zero_lie_mask.any():
                reasonable_df_nonzero = reasonable_df[non_zero_lie_mask]
                best_lie_idx = reasonable_df_nonzero['final_lie_loss'].idxmin()
                results['best_lie_reasonable_elbo'] = {
                    'lambda_lie': df_results.loc[best_lie_idx, 'lambda_lie'],
                    'total_loss': df_results.loc[best_lie_idx, 'final_total_loss'],
                    'elbo_loss': df_results.loc[best_lie_idx, 'final_elbo_loss'],
                    'lie_loss': df_results.loc[best_lie_idx, 'final_lie_loss']
                }
    
    # 4. Balanced approach: minimize normalized sum (only if regularization exists)
    if 'final_lie_loss' in df_results.columns and (df_results['final_lie_loss'] > 0).any():
        try:
            elbo_std = df_results['final_elbo_loss'].std()
            lie_std = df_results['final_lie_loss'].std()
            
            if elbo_std > 0 and lie_std > 0:
                elbo_normalized = (df_results['final_elbo_loss'] - df_results['final_elbo_loss'].min()) / elbo_std
                lie_normalized = (df_results['final_lie_loss'] - df_results['final_lie_loss'].min()) / lie_std
                balanced_score = elbo_normalized + lie_normalized
                balanced_idx = balanced_score.idxmin()
                results['balanced'] = {
                    'lambda_lie': df_results.loc[balanced_idx, 'lambda_lie'],
                    'total_loss': df_results.loc[balanced_idx, 'final_total_loss'],
                    'elbo_loss': df_results.loc[balanced_idx, 'final_elbo_loss'],
                    'lie_loss': df_results.loc[balanced_idx, 'final_lie_loss']
                }
        except Exception as e:
            print(f"   ⚠️ Could not compute balanced criterion: {e}")
    
    return results


def run_regularization_sweep(experiment_base_name, vector_field_name='rotation_1.0', 
                           lambda_values=None, perturbation_magnitude=0.1,
                           n_epochs=50, seed=1234, comprehensive_analysis=True):
    """Run experiments across multiple regularization values."""
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-4, 0, 5)))
    
    print(f"🔬 Running regularization sweep:")
    print(f"   Base name: {experiment_base_name}")
    print(f"   Lambda values: {len(lambda_values)} points")
    print(f"   Vector field: {vector_field_name}")
    print(f"   Perturbation: {perturbation_magnitude}")
    
    all_results = []
    
    for i, lambda_lie in enumerate(lambda_values):
        experiment_name = f"{experiment_base_name}_lambda_{lambda_lie:.2e}".replace(".", "_").replace("+", "").replace("-", "neg")
        
        print(f"\n--- Experiment {i+1}/{len(lambda_values)}: λ = {lambda_lie:.2e} ---")
        
        # Import the main experiment function
        from regularized_experiments import run_experiment
        
        result = run_experiment(
            experiment_name=experiment_name,
            vector_field_name=vector_field_name,
            lambda_lie=lambda_lie,
            lambda_curvature=0.0,
            perturbation_magnitude=perturbation_magnitude,
            n_epochs=n_epochs,
            seed=seed,
            comprehensive_analysis=comprehensive_analysis
        )
        
        all_results.append(result)
    
    return all_results


def run_perturbation_sweep(experiment_base_name, vector_field_name='rotation_0.5',
                         perturbation_magnitudes=None, lambda_values=None,
                         n_epochs=25, seed=41, comprehensive_analysis=True):
    """Run experiments across multiple perturbation magnitudes and regularization values."""
    if perturbation_magnitudes is None:
        perturbation_magnitudes = [0.2, 0.3, 0.4, 0.5]
    
    if lambda_values is None:
        lambda_values = np.concatenate(([0.0], np.logspace(-4, 0, 3)))
    
    print(f"🔬 Running perturbation × regularization sweep:")
    print(f"   Base name: {experiment_base_name}")
    print(f"   Perturbation magnitudes: {len(perturbation_magnitudes)} values")
    print(f"   Lambda values: {len(lambda_values)} points")
    print(f"   Total experiments: {len(perturbation_magnitudes)} × {len(lambda_values)} = {len(perturbation_magnitudes) * len(lambda_values)}")
    
    all_results = []
    
    for perturb_idx, perturbation_magnitude in enumerate(perturbation_magnitudes):
        print(f"\n{'='*80}")
        print(f"🎯 PERTURBATION MAGNITUDE {perturbation_magnitude} ({perturb_idx+1}/{len(perturbation_magnitudes)})")
        print(f"{'='*80}")
        
        for lambda_idx, lambda_lie in enumerate(lambda_values):
            experiment_name = f"{experiment_base_name}_perturb_{perturbation_magnitude}_lambda_{lambda_lie:.2e}".replace(".", "_").replace("+", "").replace("-", "neg")
            
            print(f"\n--- Experiment {lambda_idx+1}/{len(lambda_values)}: λ = {lambda_lie:.2e}, perturbation = {perturbation_magnitude} ---")
            
            # Import the main experiment function
            from regularized_experiments import run_experiment
            
            result = run_experiment(
                experiment_name=experiment_name,
                vector_field_name=vector_field_name,
                lambda_lie=lambda_lie,
                lambda_curvature=0.0,
                perturbation_magnitude=perturbation_magnitude,
                n_epochs=n_epochs,
                seed=seed,
                comprehensive_analysis=comprehensive_analysis
            )
            
            # Add perturbation magnitude to result
            result['perturbation_magnitude'] = perturbation_magnitude
            all_results.append(result)
    
    return all_results