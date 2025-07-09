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
    target_rate_per_bin: float = 0.1  # target firing rate for Poisson
    p_coherence: float = 0.5  # probability of coherence for loading matrix
    p_sparsity: float = 0.1  # probability of sparsity for loading matrix

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

def compute_firing_rate(x: np.ndarray, C: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the firing rate of a log-linear Poisson neuron model"""
    return np.exp(x @ C + b)

def update_bias_to_match_target_rate(current_rate: np.ndarray, current_bias: np.ndarray, target_rate: float) -> np.ndarray:
    """Update bias to match target firing rate"""
    return current_bias + np.log(target_rate / current_rate)

def compute_snr(latent_traj: np.ndarray, C: np.ndarray, b: np.ndarray, target_rate: float) -> Tuple[float, np.ndarray]:
    """Compute SNR using Fisher information matrix"""
    # Compute firing rates
    firing_rates = compute_firing_rate(latent_traj, C, b)
    
    # Update bias to match target rate
    b = update_bias_to_match_target_rate(np.mean(firing_rates, axis=0), b, target_rate)
    firing_rates = compute_firing_rate(latent_traj, C, b)
    
    # Compute SNR using Fisher information
    SNR = 0
    for i, firing_rate in enumerate(firing_rates):
        SNR += np.trace(np.linalg.inv(C @ np.diag(firing_rate) @ C.T))
    SNR = SNR / firing_rates.shape[0]
    SNR = 10 * np.log10(C.shape[0] / SNR)
    
    return SNR, b

def generate_poisson_observations(trajectories: np.ndarray, settings: ObservationSettings) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate Poisson observations with target SNR"""
    n_trajectories, n_timesteps, n_states = trajectories.shape
    
    # Reshape trajectories for processing
    latent_traj = trajectories.reshape(-1, n_states)
    
    # Generate random loading matrix if not provided
    if settings.observation_matrix is None:
        # Simple random matrix for now (we can add the more sophisticated generation later)
        C = np.random.randn(n_states, n_states)
        C = C / np.linalg.norm(C, axis=0)
    else:
        C = settings.observation_matrix
    
    # Initialize bias
    # TODO Investigate why subtracting changes the mean firing rate by so much
    b = np.zeros((1, n_states)) - np.log(settings.target_rate_per_bin) - 5.0
    
    # Compute initial firing rates and SNR
    firing_rates = compute_firing_rate(latent_traj, C, b)
    
    if settings.snr_target is not None:
        current_snr, b = compute_snr(latent_traj, C, b, settings.target_rate_per_bin)
        if current_snr > 0:  # Only scale if SNR is positive
            gain = np.sqrt(settings.snr_target / current_snr)
            # Limit the maximum gain to prevent too large rates
            max_gain = 10.0
            gain = min(gain, max_gain)
            C = C * gain
            firing_rates = compute_firing_rate(latent_traj, C, b)
            current_snr, b = compute_snr(latent_traj, C, b, settings.target_rate_per_bin)
        else:
            current_snr = 0.0
    else:
        current_snr = 0.0


    # Generate Poisson observations
    observations = np.random.poisson(firing_rates)
    
    # Reshape back to original dimensions
    observations = observations.reshape(n_trajectories, n_timesteps, n_states)
    firing_rates = firing_rates.reshape(n_trajectories, n_timesteps, n_states)
    
    return observations, C, b, firing_rates, current_snr

def generate_observations(trajectories: np.ndarray, settings: ObservationSettings) -> Tuple[np.ndarray, float]:
    """Generate noisy observations from trajectories using specified observation model."""
    if settings.model == ObservationModel.GAUSSIAN:
        n_trajectories, n_timesteps, n_states = trajectories.shape
        
        if settings.observation_matrix is None:
            settings.observation_matrix = np.eye(n_states)
        
        n_observations = settings.observation_matrix.shape[0]
        noisy_observations = np.zeros((n_trajectories, n_timesteps, n_observations))
        
        for i in range(n_trajectories):
            for t in range(n_timesteps):
                observation = settings.observation_matrix @ trajectories[i, t]
                noise = np.random.normal(0, settings.noise_std, size=n_observations)
                noisy_observations[i, t] = observation + noise
        
        # For SNR calculation, project observations back to latent space or use clean observations
        clean_observations = np.zeros((n_trajectories, n_timesteps, n_observations))
        for i in range(n_trajectories):
            for t in range(n_timesteps):
                clean_observations[i, t] = settings.observation_matrix @ trajectories[i, t]
        
        signal_var = np.var(clean_observations)
        noise_var = np.var(noisy_observations - clean_observations)
        snr = float(signal_var / noise_var) if noise_var > 0 else float('inf')
        
        return noisy_observations, snr
        
    elif settings.model == ObservationModel.POISSON:
        observations, C, b, firing_rates, snr = generate_poisson_observations(trajectories, settings)
        return observations, float(snr)
    else:
        return np.zeros((1, 1)), 0.0

def plot_noisy_observations(trajectories, noisy_observations, settings: ObservationSettings, title="Noisy Observations"):
    """Plot original trajectories and their noisy observations."""
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original trajectories in latent space
    plt.subplot(1, 3, 1)
    for i, traj in enumerate(trajectories[:10]):  # Show first 10 trajectories
        plt.plot(traj[:, 0], traj[:, 1], color='blue', alpha=0.7, linewidth=1, 
                label='Trajectories' if i == 0 else "")
    plt.title('Original Latent Trajectories')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # Plot 2: Observation dimensionality and statistics
    plt.subplot(1, 3, 2)
    obs_means = np.mean(noisy_observations, axis=(0, 1))
    obs_stds = np.std(noisy_observations, axis=(0, 1))
    neuron_indices = np.arange(len(obs_means))
    
    plt.errorbar(neuron_indices, obs_means, yerr=obs_stds, alpha=0.7, capsize=2)
    plt.title(f'Neural Observations\n({noisy_observations.shape[2]} neurons)')
    plt.xlabel('Neuron Index')
    plt.ylabel('Mean ± Std Activity')
    plt.grid(True)
    
    # Plot 3: Time series for first few neurons
    plt.subplot(1, 3, 3)
    n_neurons_plot = min(5, noisy_observations.shape[2])
    for i in range(n_neurons_plot):
        plt.plot(noisy_observations[0, :, i], alpha=0.7, label=f'Neuron {i+1}')
    plt.title('Neural Activity Over Time\n(First trial)')
    plt.xlabel('Time Steps')
    plt.ylabel('Neural Activity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()

def plot_poisson_analysis(trajectories: np.ndarray, noisy_observations: np.ndarray, firing_rates: np.ndarray, settings: ObservationSettings):
    """Plot analysis of Poisson rates and observations."""
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Scatter plot of true state vs rate
    plt.subplot(2, 2, 1)
    plt.scatter(trajectories[:, :, 0].flatten(), firing_rates[:, :, 0].flatten(), 
                alpha=0.1, s=10, label='x-component')
    plt.scatter(trajectories[:, :, 1].flatten(), firing_rates[:, :, 1].flatten(), 
                alpha=0.1, s=10, label='y-component')
    plt.xlabel('True State')
    plt.ylabel('Firing Rate')
    plt.title(f'True State vs Firing Rate (SNR={settings.snr_target:.1f})')
    plt.legend()
    
    # Plot 2: Histogram of rates
    plt.subplot(2, 2, 2)
    plt.hist(firing_rates.flatten(), bins=50, alpha=0.5, label='Rates')
    plt.xlabel('Rate Value')
    plt.ylabel('Count')
    plt.title('Distribution of Firing Rates')
    
    # Plot 3: Scatter plot of rate vs observations
    plt.subplot(2, 2, 3)
    plt.scatter(firing_rates[:, :, 0].flatten(), noisy_observations[:, :, 0].flatten(), 
                alpha=0.1, s=10, label='x-component')
    plt.scatter(firing_rates[:, :, 1].flatten(), noisy_observations[:, :, 1].flatten(), 
                alpha=0.1, s=10, label='y-component')
    plt.xlabel('Firing Rate')
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

def main(return_data=False):
    # Set parameters
    perturbation_norm = 0.1
    random_seed = 313
    min_val_sim = 3
    n_grid = 40
    num_points_invman = 200
    maxT = 5
    tsteps = 100
    number_of_target_trajectories = 500
    
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
    n_neurons = 50  # Number of neurons to simulate
    n_latents = 2   # Latent dimensionality
    
    # Create random observation matrix (neurons x latents)
    np.random.seed(42)  # For reproducibility
    observation_matrix = np.random.randn(n_neurons, n_latents) * 0.5
    
    gaussian_settings = ObservationSettings(
        model=ObservationModel.GAUSSIAN,
        noise_std=0.1,
        snr_target=50.0,
        observation_matrix=observation_matrix
    )
    
    poisson_settings = ObservationSettings(
        model=ObservationModel.POISSON,
        target_rate_per_bin=0.1,
        snr_target=10.0,  # Target SNR in dB
        p_coherence=0.5,
        p_sparsity=0.1
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
    
    # Get firing rates for Poisson analysis
    _, _, _, firing_rates, _ = generate_poisson_observations(trajectories, poisson_settings)
    
    # Plot Poisson analysis
    poisson_fig = plot_poisson_analysis(trajectories, poisson_observations, firing_rates, poisson_settings)
    poisson_fig.savefig('poisson_analysis.png')
    plt.close()
    
    # Calculate and print metrics
    metrics = calculate_metrics(trajectories)
    print("\nSystem Metrics:")
    print(f"Mean radius: {metrics['mean_radius']:.3f} ± {metrics['std_radius']:.3f}")
    print(f"Mean angular velocity: {metrics['mean_angular_velocity']:.3f} ± {metrics['std_angular_velocity']:.3f}")
    
    # Calculate observation noise metrics
    print("\nGaussian Observation Metrics:")
    # Compute clean observations for error calculation
    clean_gaussian_obs = np.zeros_like(gaussian_observations)
    for i in range(gaussian_observations.shape[0]):
        for t in range(gaussian_observations.shape[1]):
            clean_gaussian_obs[i, t] = observation_matrix @ trajectories[i, t]
    
    gaussian_errors = gaussian_observations - clean_gaussian_obs
    gaussian_mean_error = np.mean(np.abs(gaussian_errors))
    gaussian_std_error = np.std(gaussian_errors)
    print(f"Mean absolute observation error: {gaussian_mean_error:.3f} ± {gaussian_std_error:.3f}")
    print(f"Observation dimensionality: {gaussian_observations.shape[2]}D")
    print(f"Signal-to-Noise Ratio (SNR): {gaussian_snr:.3f}")
    
    print("\nPoisson Observation Metrics:")
    print(f"Signal-to-Noise Ratio (SNR): {poisson_snr:.3f}")
    print(f"Mean firing rate: {np.mean(firing_rates):.3f}")
    print(f"Std firing rate: {np.std(firing_rates):.3f}")
    print(f"Min firing rate: {np.min(firing_rates):.3f}")
    print(f"Max firing rate: {np.max(firing_rates):.3f}")

    if return_data:
        return {
            'trajectories': trajectories,
            'gaussian_observations': gaussian_observations,
            'gaussian_settings': gaussian_settings,
            'poisson_observations': poisson_observations,
            'poisson_settings': poisson_settings,
            'inv_man': inv_man,
            'vector_field': {
                'X': X,
                'Y': Y,
                'U': U_pert,
                'V': V_pert
            }
        }

if __name__ == "__main__":
    main()