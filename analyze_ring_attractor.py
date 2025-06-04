import numpy as np
import matplotlib.pyplot as plt
from ra import build_perturbed_ringattractor
import seaborn as sns
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple

class ObservationModel(Enum):
    GAUSSIAN = "gaussian"
    POISSON = "poisson"

@dataclass
class ObservationSettings:
    model: ObservationModel
    noise_std: float = 0.1  # for Gaussian
    baseline_rate: float = 1.0  # for Poisson
    scale_factor: float = 1.0  # scaling factor for Poisson input
    observation_matrix: Optional[np.ndarray] = None
    snr_target: Optional[float] = None  # target SNR for automatic scaling

def plot_vector_field(X, Y, U, V, title="Vector Field"):
    """Plot the vector field with streamlines."""
    plt.figure(figsize=(10, 10))
    # Convert density to integer
    plt.streamplot(X, Y, U, V, density=2, color='gray', linewidth=0.5)
    plt.quiver(X, Y, U, V, color='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')

def plot_trajectories(trajectories, title="Example Trajectories"):
    """Plot example trajectories."""
    plt.figure(figsize=(10, 10))
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.3)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')

def plot_invariant_manifold(inv_man, title="Invariant Manifold"):
    """Plot the invariant manifold."""
    plt.figure(figsize=(10, 10))
    plt.plot(inv_man[:, 0], inv_man[:, 1], 'r-', linewidth=2)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')

def calculate_metrics(trajectories):
    """Calculate basic metrics for the trajectories."""
    # Calculate mean radius
    radii = np.sqrt(trajectories[:, :, 0]**2 + trajectories[:, :, 1]**2)
    mean_radius = np.mean(radii)
    std_radius = np.std(radii)
    
    # Calculate angular velocity
    angles = np.arctan2(trajectories[:, :, 1], trajectories[:, :, 0])
    angular_velocities = np.diff(angles, axis=1)
    mean_angular_velocity = np.mean(angular_velocities)
    std_angular_velocity = np.std(angular_velocities)
    
    return {
        'mean_radius': mean_radius,
        'std_radius': std_radius,
        'mean_angular_velocity': mean_angular_velocity,
        'std_angular_velocity': std_angular_velocity
    }

def generate_observations(trajectories: np.ndarray, settings: ObservationSettings) -> Tuple[np.ndarray, np.float64]:
    """
    Generate noisy observations from trajectories using specified observation model.
    
    Args:
        trajectories: Array of shape (n_trajectories, n_timesteps, n_states)
        settings: ObservationSettings object containing model parameters
    
    Returns:
        Tuple of (noisy_observations, actual_snr)
    """
    n_trajectories, n_timesteps, n_states = trajectories.shape
    
    # If no observation matrix provided, use identity matrix
    if settings.observation_matrix is None:
        settings.observation_matrix = np.eye(n_states)
    
    n_observations = settings.observation_matrix.shape[0]
    noisy_observations = np.zeros((n_trajectories, n_timesteps, n_observations))
    
    # Generate observations for each trajectory and timestep
    for i in range(n_trajectories):
        for t in range(n_timesteps):
            # Linear transformation
            observation = settings.observation_matrix @ trajectories[i, t]
            
            if settings.model == ObservationModel.GAUSSIAN:
                # Add Gaussian noise
                noise = np.random.normal(0, settings.noise_std, size=n_observations)
                noisy_observations[i, t] = observation + noise
                
            elif settings.model == ObservationModel.POISSON:
                # Log-linear Poisson model with scaled input
                rate = settings.baseline_rate * np.exp(settings.scale_factor * observation)
                noisy_observations[i, t] = np.random.poisson(rate)
    
    # Calculate actual SNR
    if settings.model == ObservationModel.GAUSSIAN:
        signal_var = np.var(trajectories)
        noise_var = np.var(noisy_observations - trajectories)
        actual_snr = signal_var / noise_var if noise_var > 0 else float('inf')
    else:  # Poisson
        # Calculate rates from trajectories
        rates = settings.baseline_rate * np.exp(settings.scale_factor * trajectories)
        
        # For Poisson, we use a modified SNR calculation that considers:
        # 1. The mean rate as signal
        # 2. The Fano factor (variance/mean) as a measure of noise
        # 3. The relative change in rate as a measure of signal strength
        mean_rate = np.mean(rates)
        rate_std = np.std(rates)
        
        # Calculate relative rate variation (signal)
        relative_rate_variation = rate_std / mean_rate
        
        # Calculate Fano factor (noise)
        # For Poisson, variance = mean, so Fano factor should be close to 1
        fano_factor = np.var(noisy_observations) / np.mean(noisy_observations)
        
        # Modified SNR for Poisson: relative rate variation / deviation from ideal Fano factor
        actual_snr = relative_rate_variation / abs(fano_factor - 1) if fano_factor != 1 else float('inf')
    
    return noisy_observations, actual_snr

def plot_noisy_observations(trajectories, noisy_observations, settings: ObservationSettings, title="Noisy Observations"):
    """Plot original trajectories and their noisy observations."""
    plt.figure(figsize=(10, 10))
    
    # Plot original trajectories
    for traj in trajectories:
        plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.1, label='Original' if traj is trajectories[0] else "")
    
    # Plot noisy observations
    for obs in noisy_observations:
        if settings.model == ObservationModel.GAUSSIAN:
            plt.scatter(obs[:, 0], obs[:, 1], color='red', alpha=0.1, s=10, label='Noisy' if obs is noisy_observations[0] else "")
        else:  # Poisson
            plt.scatter(obs[:, 0], obs[:, 1], color='green', alpha=0.1, s=10, label='Poisson' if obs is noisy_observations[0] else "")
    
    plt.title(f"{title} ({settings.model.value})")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

def plot_poisson_analysis(trajectories: np.ndarray, noisy_observations: np.ndarray, settings: ObservationSettings):
    """Plot analysis of Poisson rates and observations."""
    # Calculate rates from trajectories with scaling
    rates = settings.baseline_rate * np.exp(settings.scale_factor * trajectories)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Scatter plot of true state vs rate
    plt.subplot(2, 2, 1)
    plt.scatter(trajectories[:, :, 0].flatten(), rates[:, :, 0].flatten(), 
                alpha=0.1, s=10, label='x-component')
    plt.scatter(trajectories[:, :, 1].flatten(), rates[:, :, 1].flatten(), 
                alpha=0.1, s=10, label='y-component')
    plt.xlabel('True State')
    plt.ylabel('Poisson Rate')
    plt.title(f'True State vs Poisson Rate (scale={settings.scale_factor:.2f})')
    plt.legend()
    
    # Plot 2: Histogram of rates
    plt.subplot(2, 2, 2)
    plt.hist(rates.flatten(), bins=50, alpha=0.5, label='Rates')
    plt.xlabel('Rate Value')
    plt.ylabel('Count')
    plt.title('Distribution of Poisson Rates')
    
    # Plot 3: Scatter plot of rate vs observations
    plt.subplot(2, 2, 3)
    plt.scatter(rates[:, :, 0].flatten(), noisy_observations[:, :, 0].flatten(), 
                alpha=0.1, s=10, label='x-component')
    plt.scatter(rates[:, :, 1].flatten(), noisy_observations[:, :, 1].flatten(), 
                alpha=0.1, s=10, label='y-component')
    plt.xlabel('Poisson Rate')
    plt.ylabel('Observations')
    plt.title('Rate vs Observations')
    plt.legend()
    
    # Plot 4: Histogram of observations
    plt.subplot(2, 2, 4)
    plt.hist(noisy_observations.flatten(), bins=50, alpha=0.5, label='Observations')
    plt.xlabel('Observation Value')
    plt.ylabel('Count')
    plt.title('Distribution of Observations')
    
    plt.tight_layout()
    return fig

def main():
    # Set parameters
    perturbation_norm = 0.1
    random_seed = 313
    min_val_sim = 3
    n_grid = 40
    num_points_invman = 200
    maxT = 5
    tsteps = 100
    number_of_target_trajectories = 100
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Generate the perturbed ring attractor
    X, Y, U_pert, V_pert, grid_u, grid_v, perturb_grid_u, perturb_grid_v, full_grid_u, full_grid_v, inv_man, trajectories = build_perturbed_ringattractor(
        perturbation_norm=perturbation_norm,
        random_seed=random_seed,
        min_val_sim=min_val_sim,
        n_grid=n_grid,
        num_points_invman=num_points_invman,
        maxT=maxT,
        tsteps=tsteps,
        number_of_target_trajectories=number_of_target_trajectories
    )
    
    # Plot vector field
    plot_vector_field(X, Y, U_pert, V_pert, "Perturbed Vector Field")
    plt.savefig('vector_field.png')
    plt.close()
    
    # Plot invariant manifold
    plot_invariant_manifold(inv_man)
    plt.savefig('invariant_manifold.png')
    plt.close()
    
    # Plot trajectories
    plot_trajectories(trajectories)
    plt.savefig('trajectories.png')
    plt.close()
    
    # Generate observations with both models
    gaussian_settings = ObservationSettings(
        model=ObservationModel.GAUSSIAN,
        noise_std=0.1,
        snr_target=50.0
    )
    
    poisson_settings = ObservationSettings(
        model=ObservationModel.POISSON,
        baseline_rate=5.0,
        scale_factor=0.2,  # Scale down the input to exponential
        snr_target=50.0
    )
    
    # Generate and plot Gaussian observations
    gaussian_observations, gaussian_snr = generate_observations(trajectories, gaussian_settings)
    plot_noisy_observations(trajectories, gaussian_observations, gaussian_settings)
    plt.savefig('gaussian_observations.png')
    plt.close()
    
    # Generate and plot Poisson observations
    poisson_observations, poisson_snr = generate_observations(trajectories, poisson_settings)
    plot_noisy_observations(trajectories, poisson_observations, poisson_settings)
    plt.savefig('poisson_observations.png')
    plt.close()
    
    # Plot Poisson analysis
    poisson_fig = plot_poisson_analysis(trajectories, poisson_observations, poisson_settings)
    poisson_fig.savefig('poisson_analysis.png')
    plt.close()
    
    # Calculate and print metrics
    metrics = calculate_metrics(trajectories)
    print("\nSystem Metrics:")
    print(f"Mean radius: {metrics['mean_radius']:.3f} ± {metrics['std_radius']:.3f}")
    print(f"Mean angular velocity: {metrics['mean_angular_velocity']:.3f} ± {metrics['std_angular_velocity']:.3f}")
    
    # Calculate observation noise metrics
    print("\nGaussian Observation Metrics:")
    gaussian_errors = gaussian_observations - trajectories
    gaussian_mean_error = np.mean(np.abs(gaussian_errors))
    gaussian_std_error = np.std(gaussian_errors)
    print(f"Mean absolute error: {gaussian_mean_error:.3f} ± {gaussian_std_error:.3f}")
    print(f"Signal-to-Noise Ratio (SNR): {gaussian_snr:.3f}")
    
    print("\nPoisson Observation Metrics:")
    poisson_errors = poisson_observations - trajectories
    poisson_mean_error = np.mean(np.abs(poisson_errors))
    poisson_std_error = np.std(poisson_errors)
    print(f"Mean absolute error: {poisson_mean_error:.3f} ± {poisson_std_error:.3f}")
    print(f"Signal-to-Noise Ratio (SNR): {poisson_snr:.3f}")
    
    # Print Poisson rate statistics
    rates = poisson_settings.baseline_rate * np.exp(poisson_settings.scale_factor * trajectories)
    print("\nPoisson Rate Statistics:")
    print(f"Mean rate: {np.mean(rates):.3f}")
    print(f"Std rate: {np.std(rates):.3f}")
    print(f"Min rate: {np.min(rates):.3f}")
    print(f"Max rate: {np.max(rates):.3f}")
    
    # Print additional Poisson metrics
    fano_factor = np.var(poisson_observations) / np.mean(poisson_observations)
    relative_rate_variation = np.std(rates) / np.mean(rates)
    print("\nAdditional Poisson Metrics:")
    print(f"Fano factor: {fano_factor:.3f}")
    print(f"Relative rate variation: {relative_rate_variation:.3f}")

if __name__ == "__main__":
    main() 