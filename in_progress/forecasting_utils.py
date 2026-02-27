"""
Forecasting evaluation for symmetry-regularized dynamics.

This module provides utilities to evaluate how well a learned model
recovers the underlying attractor structure by testing forecasting
from off-manifold initial conditions.

Experiment setup:
1. Initialize points off the ring attractor manifold (via radial perturbation
   or strong GP kick)
2. Turn off external input (run pure learned dynamics)
3. Roll out trajectories and measure return to manifold
4. Compare baseline (lambda=0) vs regularized (lambda>0) systems

Key metrics:
- Return-to-manifold: How quickly does the system return to the ring?
- Radial attraction quality: Is motion directed toward the ring?
- Attractor smoothness: How smooth is the recovered attractor surface?
"""

from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def initialize_off_manifold_points(
    n_points: int,
    manifold_radius: float = 1.0,
    perturbation_type: str = "radial",
    delta_r: float = 0.3,
    kick_magnitude: float = 0.5,
    seed: Optional[int] = None,
    device: str = "cpu",
) -> Tensor:
    """
    Initialize points off the ring attractor manifold.

    Two initialization strategies:
    - radial: Perturb radius by +/- delta_r (half inside, half outside ring)
    - strong_gp_kick: Start on manifold, add random displacement

    Args:
        n_points: Number of evaluation points
        manifold_radius: Target ring radius (default 1.0)
        perturbation_type: "radial" or "strong_gp_kick"
        delta_r: Radial perturbation magnitude (for radial type)
        kick_magnitude: Random displacement magnitude (for gp_kick type)
        seed: Random seed for reproducibility
        device: Torch device

    Returns:
        Initial points (n_points, 2)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Sample angles uniformly around the ring
    theta = torch.rand(n_points, device=device) * 2 * np.pi

    if perturbation_type == "radial":
        # Perturb radius by +/- delta_r (uniform distribution)
        r_perturbations = torch.empty(n_points, device=device).uniform_(-delta_r, delta_r)
        r = manifold_radius + r_perturbations

        # Convert to Cartesian
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        points = torch.stack([x, y], dim=-1)

    elif perturbation_type == "strong_gp_kick":
        # Start on manifold
        x = manifold_radius * torch.cos(theta)
        y = manifold_radius * torch.sin(theta)
        points = torch.stack([x, y], dim=-1)

        # Add random "kick" displacement
        kick = torch.randn(n_points, 2, device=device) * kick_magnitude
        points = points + kick

    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")

    return points


def rollout_dynamics(
    initial_points: Tensor,
    dynamics_fn: Callable[[Tensor], Tensor],
    n_steps: int,
) -> Tensor:
    """
    Roll out dynamics from initial points without external input.

    This tests the autonomous behavior of the learned dynamics - whether
    trajectories starting off-manifold return to the ring attractor.

    Args:
        initial_points: Starting states (n_points, d)
        dynamics_fn: Discrete dynamics x' = f(x) (returns next state, not velocity)
        n_steps: Number of steps to roll out

    Returns:
        Trajectories (n_points, n_steps + 1, d)
    """
    n_points, d = initial_points.shape
    device = initial_points.device

    trajectories = torch.zeros(n_points, n_steps + 1, d, device=device)
    trajectories[:, 0] = initial_points

    x = initial_points.clone()
    for t in range(n_steps):
        x = dynamics_fn(x)
        trajectories[:, t + 1] = x

    return trajectories


def compute_return_to_manifold_metrics(
    trajectories: Tensor,
    target_radius: float = 1.0,
    tolerance: float = 0.1,
) -> Dict[str, float]:
    """
    Compute metrics for return-to-manifold behavior.

    Measures how well the learned dynamics guide trajectories back to
    the ring attractor from off-manifold initial conditions.

    Args:
        trajectories: (n_points, n_steps, 2)
        target_radius: Target ring radius
        tolerance: Distance tolerance for "returned" classification

    Returns:
        Dictionary with metrics:
        - final_dist_to_manifold: Average final distance from ring
        - mean_return_time: Average steps to reach within tolerance
        - monotonic_approach_ratio: Fraction of steps moving toward ring
        - final_radii_std: Variance in final radii (should be low if all converge)
    """
    n_points, n_steps, _ = trajectories.shape

    # Compute radii over time
    radii = torch.linalg.norm(trajectories, dim=-1)  # (n_points, n_steps)

    # Distance from target manifold
    dist_to_manifold = torch.abs(radii - target_radius)  # (n_points, n_steps)

    # 1. Final distance to manifold (lower is better)
    final_dist = dist_to_manifold[:, -1].mean().item()

    # 2. Time to return (first time within tolerance)
    within_tolerance = dist_to_manifold < tolerance  # (n_points, n_steps)
    return_times = []
    for i in range(n_points):
        returned_indices = torch.where(within_tolerance[i])[0]
        if len(returned_indices) > 0:
            return_times.append(returned_indices[0].item())
        else:
            return_times.append(n_steps)  # Never returned
    mean_return_time = float(np.mean(return_times))

    # 3. Monotonic approach ratio (fraction of steps where distance decreases)
    # Higher is better - indicates consistent convergence
    dist_decreasing = (dist_to_manifold[:, 1:] < dist_to_manifold[:, :-1]).float()
    monotonic_ratio = dist_decreasing.mean().item()

    # 4. Final radii variance (should be low if all points converge to same radius)
    final_radii_std = radii[:, -1].std().item()

    return {
        "final_dist_to_manifold": final_dist,
        "mean_return_time": mean_return_time,
        "monotonic_approach_ratio": monotonic_ratio,
        "final_radii_std": final_radii_std,
    }


def compute_radial_attraction_quality(
    trajectories: Tensor,
    dynamics_fn: Callable[[Tensor], Tensor],
    target_radius: float = 1.0,
) -> Dict[str, float]:
    """
    Evaluate whether the vector field points toward the manifold.

    For a perfect ring attractor:
    - Inside ring (r < target): radial component should be positive (expansion)
    - Outside ring (r > target): radial component should be negative (contraction)

    Args:
        trajectories: (n_points, n_steps, 2)
        dynamics_fn: Discrete dynamics f(x) -> x'
        target_radius: Target ring radius

    Returns:
        Dictionary with metrics:
        - radial_alignment: Fraction of vectors pointing correct radial direction
        - radial_magnitude: Average magnitude of radial velocity component
    """
    n_points, n_steps, d = trajectories.shape

    # Sample a subset of points from trajectories (every 5th step to avoid correlation)
    sample_points = trajectories[:, ::5, :].reshape(-1, d)

    # Compute velocity direction from discrete dynamics
    # v ≈ f(x) - x (discrete velocity)
    with torch.no_grad():
        next_state = dynamics_fn(sample_points)
        velocity = next_state - sample_points

    # Compute radii and radial unit vectors
    radii = torch.linalg.norm(sample_points, dim=-1, keepdim=True)
    radial_unit = sample_points / (radii + 1e-8)

    # Radial component of velocity (dot product with radial direction)
    radial_component = (velocity * radial_unit).sum(dim=-1)  # (n_samples,)

    # Expected sign: positive if inside ring (need expansion), negative if outside
    expected_sign = torch.sign(target_radius - radii.squeeze())

    # Fraction with correct radial direction
    correct_direction = ((radial_component * expected_sign) > 0).float()
    radial_alignment = correct_direction.mean().item()

    # Average magnitude of radial component
    radial_magnitude = radial_component.abs().mean().item()

    return {
        "radial_alignment": radial_alignment,
        "radial_magnitude": radial_magnitude,
    }


def compute_attractor_smoothness(
    dynamics_fn: Callable[[Tensor], Tensor],
    n_angles: int = 100,
    target_radius: float = 1.0,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate smoothness of the learned attractor on the ring.

    A perfect ring attractor has:
    - Zero radial component on the manifold (purely tangential flow)
    - Constant tangential speed around the ring (or zero for fixed points)

    High variance in these quantities indicates a "bumpy" or irregular attractor.

    Args:
        dynamics_fn: Discrete dynamics f(x) -> x'
        n_angles: Number of sample points around the ring
        target_radius: Ring radius
        device: Torch device

    Returns:
        Dictionary with metrics:
        - on_manifold_tangential_mean: Mean tangential velocity
        - on_manifold_tangential_std: Std of tangential velocity (lower = smoother)
        - on_manifold_radial_mean: Mean radial velocity (should be ~0)
        - on_manifold_radial_std: Std of radial velocity (lower = cleaner attractor)
    """
    # Sample points uniformly on the target ring
    theta = torch.linspace(0, 2 * np.pi, n_angles + 1, device=device)[:-1]  # Exclude endpoint
    points = target_radius * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)

    # Compute velocity from discrete dynamics
    with torch.no_grad():
        next_state = dynamics_fn(points)
        velocity = next_state - points

    # Tangent and radial unit vectors at each point
    tangent_unit = torch.stack([-torch.sin(theta), torch.cos(theta)], dim=-1)
    radial_unit = points / target_radius

    # Project velocity onto tangent and radial directions
    tangential_component = (velocity * tangent_unit).sum(dim=-1)
    radial_component = (velocity * radial_unit).sum(dim=-1)

    return {
        "on_manifold_tangential_mean": tangential_component.mean().item(),
        "on_manifold_tangential_std": tangential_component.std().item(),
        "on_manifold_radial_mean": radial_component.mean().item(),
        "on_manifold_radial_std": radial_component.std().item(),
    }


def evaluate_forecasting(
    dynamics_fn: Callable[[Tensor], Tensor],
    forecast_cfg: dict,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Full forecasting evaluation pipeline.

    Runs all forecasting metrics for both initialization types.

    Args:
        dynamics_fn: Learned discrete dynamics f(x) -> x'
        forecast_cfg: Config with n_forecast_steps, perturbation_types, etc.
        device: Torch device

    Returns:
        Dictionary of all metrics, prefixed by perturbation type
    """
    all_metrics = {}

    # Convert OmegaConf to plain dict if needed
    if hasattr(forecast_cfg, "perturbation_types"):
        perturbation_types = list(forecast_cfg.perturbation_types)
    else:
        perturbation_types = forecast_cfg.get("perturbation_types", [])

    n_forecast_steps = forecast_cfg.get("n_forecast_steps", 50)
    n_eval_points = forecast_cfg.get("n_eval_points", 100)
    target_radius = forecast_cfg.get("target_radius", 1.0)

    for perturb_cfg in perturbation_types:
        # Handle both OmegaConf and dict
        if hasattr(perturb_cfg, "name"):
            perturb_name = perturb_cfg.name
            delta_r = getattr(perturb_cfg, "delta_r", 0.3)
            kick_magnitude = getattr(perturb_cfg, "kick_magnitude", 0.5)
        else:
            perturb_name = perturb_cfg.get("name", "unknown")
            delta_r = perturb_cfg.get("delta_r", 0.3)
            kick_magnitude = perturb_cfg.get("kick_magnitude", 0.5)

        # Initialize off-manifold points
        init_points = initialize_off_manifold_points(
            n_points=n_eval_points,
            manifold_radius=target_radius,
            perturbation_type=perturb_name,
            delta_r=delta_r,
            kick_magnitude=kick_magnitude,
            device=device,
        )

        # Roll out dynamics without external input
        with torch.no_grad():
            trajectories = rollout_dynamics(init_points, dynamics_fn, n_forecast_steps)

        # Compute all metrics
        return_metrics = compute_return_to_manifold_metrics(
            trajectories, target_radius=target_radius
        )
        attraction_metrics = compute_radial_attraction_quality(
            trajectories, dynamics_fn, target_radius=target_radius
        )

        # Prefix metrics with perturbation type
        for k, v in {**return_metrics, **attraction_metrics}.items():
            all_metrics[f"{perturb_name}_{k}"] = v

    # Add smoothness metrics (independent of perturbation type)
    smoothness_metrics = compute_attractor_smoothness(
        dynamics_fn, target_radius=target_radius, device=device
    )
    all_metrics.update(smoothness_metrics)

    return all_metrics


def evaluate_model_forecasting(
    model_path: str,
    forecast_cfg: dict,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Load a trained model and run forecasting evaluation.

    Args:
        model_path: Path to Lightning checkpoint
        forecast_cfg: Forecasting configuration
        device: Torch device

    Returns:
        Dictionary of forecasting metrics
    """
    from regularized_ssm import RegularizedLightningSSM

    # Load trained model
    lightning_model = RegularizedLightningSSM.load_from_checkpoint(
        model_path, map_location=device
    )
    lightning_model.eval()

    # Get the learned dynamics function
    # The dynamics are in ssm.dynamics_mod.mean_fn
    learned_dynamics = lightning_model.ssm.dynamics_mod.mean_fn
    learned_dynamics.to(device)

    return evaluate_forecasting(learned_dynamics, forecast_cfg, device)
