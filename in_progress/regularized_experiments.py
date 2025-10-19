#!/usr/bin/env python3
"""
Regularized State Space Model Experiments

This file contains all the infrastructure from demo_regularize.ipynb
converted into reusable functions for easily running tweaked experiments.

Enhanced with comprehensive analysis capabilities from the analysis notebooks.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import importlib.util
from collections import defaultdict

# Add xfads to path
sys.path.append('../..')

import xfads.utils as utils
import xfads.plot_utils as plot_utils

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Timer
from xfads.ssm_modules.dynamics import DenseGaussianDynamics
from xfads.ssm_modules.likelihoods import GaussianLikelihood
from xfads.ssm_modules.dynamics import DenseGaussianInitialCondition
from xfads.ssm_modules.encoders import LocalEncoderLRMvn, BackwardEncoderLRMvn
from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import NonlinearFilterSmallL

# Import our regularized SSM
from regularized_ssm import RegularizedSSM, create_rotation_vector_field, RegularizedLightningSSM
from loss_analysis_utils import (
    LossComponentAnalyzer, 
    RegularizationStrengthAnalyzer,
    create_perfect_rotation_test
)
from experiment_utils import (
    ComprehensiveLossLogger,
    train_with_comprehensive_logging,
    create_reference_dynamics,
    find_optimal_lambda_criteria,
    run_regularization_sweep,
    run_perturbation_sweep
)


def setup_config(n_epochs=50, seed=1234, device='auto'):
    """Setup configuration for experiments."""
    # Clear any existing hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    initialize(version_base=None, config_path='', job_name="lds")
    cfg = compose(config_name="config")
    
    pl.seed_everything(seed, workers=True)
    torch.set_default_dtype(torch.float32)
    torch.cuda.empty_cache()
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    cfg['device'] = device
    cfg['n_epochs'] = n_epochs
    
    return cfg


def generate_data(cfg, perturbation_magnitude=0.1, n_trials=3000, n_neurons=100, n_time_bins=75):
    """Generate ring attractor data with perturbations."""
    # Import dynamics
    file_path = "test_dynamics.py"
    spec = importlib.util.spec_from_file_location("test_dynamics", file_path)
    test_dynamics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_dynamics)
    PerturbedRingAttractorDynamics = test_dynamics.PerturbedRingAttractorRNN
    
    # Backup RNG states
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_state = np.random.get_state()
    
    mean_fn = PerturbedRingAttractorDynamics(
        bin_sz=1e-1, 
        lengthscale=0.2, 
        perturbation_magnitude=perturbation_magnitude
    ).to(cfg.device)
    
    # Restore RNG states
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    np.random.set_state(np_state)
    
    C = utils.FanInLinear(cfg.n_latents, n_neurons, device=cfg.device).requires_grad_(False)
    
    Q_diag = 5e-3 * torch.ones(cfg.n_latents, device=cfg.device)
    Q_0_diag = 1.0 * torch.ones(cfg.n_latents, device=cfg.device)
    R_diag = 1e-1 * torch.ones(n_neurons, device=cfg.device)
    m_0 = torch.zeros(cfg.n_latents, device=cfg.device)
    
    z = utils.sample_gauss_z(mean_fn, Q_diag, m_0, Q_0_diag, n_trials, n_time_bins)
    y = C(z) + torch.sqrt(R_diag) * torch.randn((n_trials, n_time_bins, n_neurons), device=cfg.device)
    y = y.detach()
    
    return y, z, mean_fn, C, Q_diag, Q_0_diag, R_diag, m_0


def create_dataloaders(y, cfg, train_split=2/3):
    """Create train/validation dataloaders."""
    def collate_fn(batch):
        elem = batch[0]
        if isinstance(elem, (tuple, list)):
            return tuple(torch.stack([b[i] for b in batch]).to(cfg.device) for i in range(len(elem)))
        else:
            return torch.stack(batch).to(cfg.device)
    
    n_trials = y.shape[0]
    split_idx = int(train_split * n_trials)
    y_train, y_valid = y[:split_idx], y[split_idx:]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_train),
        batch_size=cfg.batch_sz,
        shuffle=True,
        collate_fn=collate_fn
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(y_valid),
        batch_size=cfg.batch_sz,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, valid_loader, y_train, y_valid


def create_model_components(cfg, n_neurons, C, Q_diag, Q_0_diag, R_diag, m_0):
    """Create all model components."""
    # Likelihood
    H = utils.ReadoutLatentMask(cfg.n_latents, cfg.n_latents_read)
    readout_fn = nn.Sequential(H, C)
    likelihood_pdf = GaussianLikelihood(readout_fn, n_neurons, R_diag, device=cfg.device, fix_R=True)
    
    # Dynamics (learnable)
    dynamics_fn = utils.build_gru_dynamics_function(cfg.n_latents, cfg.n_hidden_dynamics, device=cfg.device)
    dynamics_mod = DenseGaussianDynamics(dynamics_fn, cfg.n_latents, Q_diag, device=cfg.device)
    
    # Initial condition
    initial_condition_pdf = DenseGaussianInitialCondition(cfg.n_latents, m_0, Q_0_diag, device=cfg.device)
    
    # Encoders
    backward_encoder = BackwardEncoderLRMvn(
        cfg.n_latents, cfg.n_hidden_backward, cfg.n_latents,
        rank_local=cfg.rank_local, rank_backward=cfg.rank_backward, device=cfg.device
    )
    local_encoder = LocalEncoderLRMvn(
        cfg.n_latents, n_neurons, cfg.n_hidden_local, cfg.n_latents,
        rank=cfg.rank_local, device=cfg.device, dropout=cfg.p_local_dropout
    )
    
    # Nonlinear filtering
    nl_filter = NonlinearFilterSmallL(dynamics_mod, initial_condition_pdf, device=cfg.device)
    
    return {
        'likelihood_pdf': likelihood_pdf,
        'dynamics_mod': dynamics_mod,
        'initial_condition_pdf': initial_condition_pdf,
        'backward_encoder': backward_encoder,
        'local_encoder': local_encoder,
        'nl_filter': nl_filter
    }


def create_vector_fields():
    """Create different target vector fields for experimentation."""
    vector_fields = {}
    
    # Rotation fields
    vector_fields['rotation_0.5'] = create_rotation_vector_field(rotation_speed=0.5)
    vector_fields['rotation_1.0'] = create_rotation_vector_field(rotation_speed=1.0)
    vector_fields['rotation_2.0'] = create_rotation_vector_field(rotation_speed=2.0)
    
    # Radial fields
    def radial_expansion(x, strength=0.1):
        return strength * x
    
    def radial_contraction(x, strength=-0.1):
        return strength * x
    
    vector_fields['expansion'] = lambda x: radial_expansion(x, 0.1)
    vector_fields['contraction'] = lambda x: radial_contraction(x, -0.1)
    
    # Harmonic oscillator
    def harmonic_oscillator(x, freq=1.0):
        # [x_dot, y_dot] = [y, -freq^2 * x]
        return torch.stack([-freq**2 * x[..., 0], x[..., 1]], dim=-1)
    
    vector_fields['harmonic'] = lambda x: harmonic_oscillator(x, 1.0)
    
    # Spiral field (rotation + expansion)
    def spiral_field(x, rotation_speed=0.5, expansion_rate=0.1):
        rotation = create_rotation_vector_field(rotation_speed)(x)
        expansion = expansion_rate * x
        return rotation + expansion
    
    vector_fields['spiral'] = lambda x: spiral_field(x, 0.5, 0.05)
    
    return vector_fields


def run_experiment(experiment_name, vector_field_name='rotation_0.5', lambda_lie=0.0, lambda_curvature=0.1,
                   perturbation_magnitude=0.1, n_epochs=50, seed=1234, 
                   n_trials=3000, n_neurons=100, n_time_bins=75, 
                   save_vector_fields=True, comprehensive_analysis=False):
    """Run a complete regularized SSM experiment."""
    
    print(f"🚀 Starting experiment: {experiment_name}")
    print(f"   Vector field: {vector_field_name}")
    print(f"   λ_lie: {lambda_lie}, λ_curvature: {lambda_curvature}")
    print(f"   Perturbation: {perturbation_magnitude}")
    print(f"   Comprehensive analysis: {comprehensive_analysis}")
    
    # Check if this is a baseline run (no regularization)
    is_baseline = (lambda_lie == 0.0 and lambda_curvature == 0.0)
    if is_baseline:
        print(f"   🔍 Baseline run detected (no regularization)")
    
    # Create experiment directory structure
    exp_dir = Path(f"{experiment_name}")
    exp_dir.mkdir(exist_ok=True)
    
    # Setup
    cfg = setup_config(n_epochs=n_epochs, seed=seed)
    
    # Generate data
    y, z, mean_fn, C, Q_diag, Q_0_diag, R_diag, m_0 = generate_data(
        cfg, perturbation_magnitude, n_trials, n_neurons, n_time_bins
    )
    
    # Create dataloaders
    train_loader, valid_loader, y_train, y_valid = create_dataloaders(y, cfg)
    
    # Create model components
    components = create_model_components(cfg, n_neurons, C, Q_diag, Q_0_diag, R_diag, m_0)
    
    # Get target vector field
    vector_fields = create_vector_fields()
    if vector_field_name not in vector_fields:
        raise ValueError(f"Unknown vector field: {vector_field_name}. Available: {list(vector_fields.keys())}")
    target_vector_field = vector_fields[vector_field_name]
    
    # Create regularized SSM
    regularized_ssm = RegularizedSSM(
        dynamics_mod=components['dynamics_mod'],
        likelihood_pdf=components['likelihood_pdf'],
        initial_c_pdf=components['initial_condition_pdf'],
        backward_encoder=components['backward_encoder'],
        local_encoder=components['local_encoder'],
        nl_filter=components['nl_filter'],
        target_vector_field=target_vector_field,
        lambda_lie=lambda_lie,
        lambda_curvature=lambda_curvature,
        lie_normalize=False,
        curvature_order=1,
        device=cfg.device
    )
    
    # Setup subdirectories
    log_dir = exp_dir / 'logs'
    ckpt_dir = exp_dir / 'ckpts'
    lightning_dir = exp_dir / 'lightning'
    plots_dir = exp_dir / 'plots'
    
    # Create subdirectories
    log_dir.mkdir(exist_ok=True)
    ckpt_dir.mkdir(exist_ok=True)
    lightning_dir.mkdir(exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Choose training approach based on comprehensive_analysis flag
    if comprehensive_analysis:
        # Use comprehensive training with detailed logging
        print("🔬 Using comprehensive analysis training...")
        
        # Create reference dynamics if available
        perfect_ring_dynamics, perturbed_ring_dynamics = create_reference_dynamics(
            perturbation_magnitude, cfg.device
        )
        
        # Create comprehensive logger
        logger = ComprehensiveLossLogger(
            experiment_name=experiment_name,
            save_vector_fields=save_vector_fields,
            perfect_ring_dynamics=perfect_ring_dynamics,
            perturbed_ring_dynamics=perturbed_ring_dynamics,
            target_field=target_vector_field
        )
        
        # Train with comprehensive logging
        training_result = train_with_comprehensive_logging(
            regularized_ssm, y_train, y_valid, logger, cfg, lambda_lie=lambda_lie
        )
        
        # Save training results
        best_model_path = str(ckpt_dir / 'best_model.pt')
        torch.save(regularized_ssm.state_dict(), best_model_path)
        
        print(f"✅ Comprehensive training completed!")
        print(f"   Final epoch: {training_result['final_epoch']}")
        print(f"   Best valid ELBO: {training_result['best_valid_elbo']:.6f}")
        print(f"   Converged: {training_result['converged']}")
        
        # Get final statistics
        regularized_ssm.eval()
        with torch.no_grad():
            final_total_loss, _, final_stats = regularized_ssm(y_valid, cfg.n_samples)
        
        result = {
            'cfg': cfg,
            'data': (y, z, mean_fn),
            'model': regularized_ssm,
            'best_path': best_model_path,
            'target_vector_field': target_vector_field,
            'experiment_name': experiment_name,
            'exp_dir': exp_dir,
            'log_dir': log_dir,
            'ckpt_dir': ckpt_dir,
            'plots_dir': plots_dir,
            'logger': logger,
            'training_result': training_result,
            'final_stats': final_stats
        }
        
    else:
        # Use standard Lightning training
        print("⚡ Using standard Lightning training...")
        
        # Create Lightning module
        seq_vae = RegularizedLightningSSM(regularized_ssm, cfg)
        
        csv_logger = CSVLogger(str(log_dir), name='', version='')
        ckpt_callback = ModelCheckpoint(
            save_top_k=3,
            monitor='valid_loss',
            mode='min',
            dirpath=str(ckpt_dir),
            filename='{epoch:02d}_{valid_loss:.4f}'
        )
        early_stop_callback = EarlyStopping(
            monitor="valid_elbo",
            min_delta=0.001,
            patience=10,
            verbose=True,
            mode="min"
        )
        timer = Timer()
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=cfg.n_epochs,
            gradient_clip_val=1.0,
            default_root_dir=str(lightning_dir),
            callbacks=[ckpt_callback, timer, early_stop_callback],
            accelerator=cfg.device,
            logger=csv_logger,
            log_every_n_steps=10
        )
        
        # Train
        print("🔥 Training...")
        trainer.fit(
            model=seq_vae,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader
        )
        
        # Save best model path
        torch.save(ckpt_callback.best_model_path, str(ckpt_dir / 'best_model_path.pt'))
        best_model_path = ckpt_callback.best_model_path
        
        print(f"✅ Training completed!")
        print(f"   Best model: {ckpt_callback.best_model_path}")
        print(f"   Training time: {timer.time_elapsed('train'):.1f}s")
        
        result = {
            'cfg': cfg,
            'data': (y, z, mean_fn),
            'model': seq_vae,
            'best_path': best_model_path,
            'target_vector_field': target_vector_field,
            'experiment_name': experiment_name,
            'exp_dir': exp_dir,
            'log_dir': log_dir,
            'ckpt_dir': ckpt_dir,
            'plots_dir': plots_dir,
            'trainer': trainer
        }
    
    return result


def analyze_experiment(result):
    """Analyze experiment results and generate plots."""
    cfg = result['cfg']
    y, z, mean_fn = result['data']
    model = result['model']
    target_vector_field = result['target_vector_field']
    experiment_name = result['experiment_name']
    log_dir = result['log_dir']
    plots_dir = result['plots_dir']
    
    print(f"🔍 Analyzing experiment: {experiment_name}")
    
    # Load best model if needed
    if hasattr(model, 'ssm'):
        # Lightning model
        best_model_path = torch.load(str(result['ckpt_dir'] / 'best_model_path.pt'))
        model = RegularizedLightningSSM.load_from_checkpoint(
            best_model_path,
            regularized_ssm=model.regularized_ssm if hasattr(model, 'regularized_ssm') else model.ssm,
            cfg=cfg
        )
        model = model.to('cpu')
        model.eval()
        ssm = model.ssm
    else:
        # Direct SSM model
        ssm = model
        ssm.eval()
    
    # Infer latent trajectories
    with torch.no_grad():
        if hasattr(model, 'ssm'):
            _, z_inferred_samples, _ = model.ssm(y, cfg.n_samples)
        else:
            _, z_inferred_samples, _ = ssm(y, cfg.n_samples)
        z_inferred = z_inferred_samples.mean(dim=0)
    
    # Compute recovery metrics
    mse = torch.mean((z_inferred - z)**2).item()
    correlation = torch.corrcoef(torch.stack([z.flatten(), z_inferred.flatten()]))[0, 1].item()
    
    print(f"📊 Recovery metrics:")
    print(f"   MSE: {mse:.6f}")
    print(f"   Correlation: {correlation:.4f}")
    
    # Plot loss curves
    try:
        plot_loss_curves(log_dir, plots_dir)
    except Exception as e:
        print(f"   ⚠️ Could not plot loss curves: {e}")
    
    # Plot dynamics comparison
    try:
        plot_dynamics_comparison(ssm, mean_fn, target_vector_field, plots_dir)
    except Exception as e:
        print(f"   ⚠️ Could not plot dynamics comparison: {e}")
    
    # Final loss analysis
    try:
        if hasattr(ssm, 'get_loss_components'):
            final_losses = ssm.get_loss_components(y[2000:], cfg.n_samples)  # Use validation set
            plot_final_losses(final_losses, plots_dir)
        else:
            print("   ⚠️ Cannot compute loss components for this model type")
            final_losses = None
    except Exception as e:
        print(f"   ⚠️ Could not compute final losses: {e}")
        final_losses = None
    
    return {
        'mse': mse,
        'correlation': correlation,
        'final_losses': final_losses
    }


def plot_loss_curves(log_dir, plots_dir):
    """Plot training loss curves."""
    import glob
    csv_files = glob.glob(f'{log_dir}/**/metrics.csv', recursive=True)
    if not csv_files:
        print("⚠️ No metrics CSV file found")
        return
        
    df = pd.read_csv(csv_files[0])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    # Plot each loss component
    loss_types = [
        ('train_loss', 'valid_loss', 'Total Loss'),
        ('train_elbo', 'valid_elbo', 'ELBO Loss'),
        ('train_lie', 'valid_lie', 'Lie Derivative Loss'),
        ('train_curvature', 'valid_curvature', 'Curvature Loss')
    ]
    
    for i, (train_col, valid_col, title) in enumerate(loss_types):
        # Check if columns exist in dataframe
        has_train = train_col in df.columns
        has_valid = valid_col in df.columns
        
        if has_train:
            train_mask = ~df[train_col].isna() & (df[train_col] > 0)  # Also exclude zeros for log scale
            if train_mask.any():
                axes[i].plot(df[train_mask]['epoch'], df[train_mask][train_col], 'b-', label='Train', linewidth=2)
        
        if has_valid:
            valid_mask = ~df[valid_col].isna() & (df[valid_col] > 0)  # Also exclude zeros for log scale
            if valid_mask.any():
                axes[i].plot(df[valid_mask]['epoch'], df[valid_mask][valid_col], 'r-', label='Valid', linewidth=2)
            
        axes[i].set_title(title)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Only use log scale if we have positive values to plot
        if (has_train and (df[train_col] > 0).any()) or (has_valid and (df[valid_col] > 0).any()):
            axes[i].set_yscale('log')
        else:
            # If no data or all zeros (baseline case), show message
            axes[i].text(0.5, 0.5, 'No regularization\n(λ = 0)', 
                        ha='center', va='center', transform=axes[i].transAxes,
                        fontsize=12, alpha=0.6)
    
    plt.tight_layout()
    plot_path = plots_dir / 'loss_curves.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Loss curves saved as '{plot_path}'")


def plot_dynamics_comparison(ssm, mean_fn, target_vector_field, plots_dir):
    """Plot comparison of true, learned, and target dynamics."""
    # Generate autonomous trajectories
    n_rollout_trials = 20
    n_rollout_time = 30
    
    z_0 = torch.zeros((1, n_rollout_trials, 2))
    z_0[:, ::2] = 0.3 * torch.randn_like(z_0[:, ::2])
    z_0[:, 1::2] = 1.5 * torch.randn_like(z_0[:, 1::2])
    
    if hasattr(ssm, 'predict_forward'):
        z_predicted = ssm.predict_forward(z_0, n_rollout_time).detach()
    else:
        # Fallback for different model types
        z_predicted = z_0.repeat(1, 1, n_rollout_time, 1)  # Placeholder
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # True dynamics
    axes[0].set_title("True Dynamics", fontsize=14, fontweight='bold')
    plot_utils.plot_two_d_vector_field(mean_fn.to('cpu'), axes[0], min_xy=-2, max_xy=2)
    axes[0].set_xlim(-2, 2)
    axes[0].set_ylim(-2, 2)
    axes[0].set_box_aspect(1.0)
    axes[0].grid(True, alpha=0.3)
    
    # Learned dynamics
    axes[1].set_title("Learned Dynamics", fontsize=14, fontweight='bold')
    if hasattr(ssm, 'dynamics_mod'):
        plot_utils.plot_two_d_vector_field(ssm.dynamics_mod.mean_fn, axes[1], min_xy=-2, max_xy=2)
        for i in range(min(n_rollout_trials, z_predicted.shape[1])):
            if z_predicted.ndim == 4:  # [batch, trial, time, dim]
                axes[1].plot(z_predicted[0, i, :, 0].cpu(), z_predicted[0, i, :, 1].cpu(), 'r-', alpha=0.6, linewidth=1)
    axes[1].set_xlim(-2, 2)
    axes[1].set_ylim(-2, 2)
    axes[1].set_box_aspect(1.0)
    axes[1].grid(True, alpha=0.3)
    
    # Target vector field
    axes[2].set_title("Target Vector Field", fontsize=14, fontweight='bold')
    dt = 1e-2
    plot_utils.plot_two_d_vector_field(
        lambda x: x + dt * target_vector_field(x), axes[2], min_xy=-2, max_xy=2
    )
    axes[2].set_xlim(-2, 2)
    axes[2].set_ylim(-2, 2)
    axes[2].set_box_aspect(1.0)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / 'dynamics_comparison.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"🌟 Dynamics comparison saved as '{plot_path}'")


def plot_final_losses(final_losses, plots_dir):
    """Plot final loss components."""
    # Build components and values based on what's available
    components = []
    values = []
    colors = []
    
    # Always include ELBO if available
    if 'elbo_loss' in final_losses:
        components.append('ELBO')
        values.append(final_losses['elbo_loss'])
        colors.append('blue')
    
    # Include regularization losses if non-zero
    if 'lie_loss' in final_losses and final_losses['lie_loss'] > 0:
        components.append('Lie Derivative')
        values.append(final_losses['lie_loss'])
        colors.append('red')
    
    if 'curvature_loss' in final_losses and final_losses['curvature_loss'] > 0:
        components.append('Curvature')
        values.append(final_losses['curvature_loss'])
        colors.append('green')
    
    # If we have no values to plot, skip
    if not values:
        print("   ⚠️ No loss values to plot")
        return
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(components, values, color=colors, alpha=0.7)
    plt.title('Final Loss Components', fontsize=14, fontweight='bold')
    plt.ylabel('Loss Value')
    
    # Only use log scale if we have varying orders of magnitude
    if max(values) / min(values) > 10:
        plt.yscale('log')
    
    plt.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                 f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = plots_dir / 'final_losses.png'
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"📊 Final losses saved as '{plot_path}'")


# Example experiments using the imported sweep functions
def experiment_rotation_sweep():
    """Run experiments with different rotation speeds."""
    return run_regularization_sweep(
        experiment_base_name='rotation_sweep',
        vector_field_name='rotation_1.0',
        lambda_values=[0.0, 1e-4, 1e-3, 1e-2, 1e-1],
        n_epochs=30,
        comprehensive_analysis=True
    )


def experiment_vector_field_comparison():
    """Compare different target vector fields."""
    results = []
    vector_fields = ['rotation_1.0', 'expansion', 'contraction', 'harmonic', 'spiral']
    
    for vf_name in vector_fields:
        result = run_experiment(
            experiment_name=f'vf_{vf_name}',
            vector_field_name=vf_name,
            lambda_lie=1.0,
            lambda_curvature=0.1,
            n_epochs=30,
            comprehensive_analysis=True
        )
        analysis = analyze_experiment(result)
        results.append((result, analysis))
    
    return results


def test_baseline_robustness():
    """Test that the code handles baseline runs (λ=0) robustly."""
    print("🧪 Testing baseline robustness...")
    
    try:
        # Test with no regularization
        result = run_experiment(
            experiment_name='test_baseline',
            vector_field_name='rotation_0.5',
            lambda_lie=0.0,
            lambda_curvature=0.0,
            n_epochs=2,
            n_trials=100,  # Small for testing
            comprehensive_analysis=False
        )
        
        # Test analysis on baseline
        analysis = analyze_experiment(result)
        
        print("✅ Baseline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Baseline test failed: {e}")
        return False


if __name__ == "__main__":
    # Example usage
    print("🌀 Regularized SSM Experiments")
    print("Available functions:")
    print("- run_experiment(): Run a single experiment")
    print("- experiment_rotation_sweep(): Compare different regularization strengths")
    print("- experiment_vector_field_comparison(): Compare vector field types")
    print("- run_regularization_sweep(): Comprehensive regularization analysis")
    print("- run_perturbation_sweep(): Perturbation magnitude analysis")
    print("- test_baseline_robustness(): Test handling of zero regularization")
    
    # # Test baseline robustness first
    # baseline_ok = test_baseline_robustness()
    
    # if baseline_ok:
    #     # Run a simple regularized example
    #     result = run_experiment(
    #         experiment_name='example_rotation',
    #         vector_field_name='rotation_0.5',
    #         lambda_lie=1e-4,
    #         lambda_curvature=0.0,
    #         n_epochs=1
    #     )
        
    #     analysis = analyze_experiment(result)
    #     print(f"✅ Example experiment completed!")

    result = run_experiment(
        experiment_name='example_rotation',
        vector_field_name='rotation_0.5',
        lambda_lie=1e-4,
        lambda_curvature=0.0,
        n_epochs=1
    )
    
    analysis = analyze_experiment(result)