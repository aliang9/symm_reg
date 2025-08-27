import torch
from typing import Dict, Any, Callable, Union

from xfads.smoothers.lightning_trainers import LightningNonlinearSSM
from xfads.smoothers.nonlinear_smoother import LowRankNonlinearStateSpaceModel
from regularizers import LieDerivativeRegularizer, CurvatureRegularizer, RotationInvarianceRegularizer


class RegularizedSSM(LowRankNonlinearStateSpaceModel):
    """
    A State Space Model with symmetry regularization.

    Combines up to four loss components:
    1. ELBO loss (KL divergence - log likelihood)
    2. Lie derivative regularization for symmetry enforcement
    3. Optional curvature regularization for smooth trajectories
    4. Optional rotation invariance regularization for rotational symmetry
    """

    def __init__(
        self,
        dynamics_mod,
        likelihood_pdf,
        initial_c_pdf,
        backward_encoder,
        local_encoder,
        nl_filter,
        target_vector_field: Callable[[torch.Tensor], torch.Tensor] = None,
        lambda_lie: float = 1.0,
        lambda_curvature: float = 0.0,
        lambda_rotation: float = 0.0,
        lie_normalize: Union[bool, str] = False,
        curvature_order: int = 1,
        n_rotations: int = 1,
        device: str = "cpu",
    ):
        """
        Args:
            dynamics_mod: Dynamics module
            likelihood_pdf: Likelihood model
            initial_c_pdf: Initial condition prior
            backward_encoder: Backward encoder
            local_encoder: Local encoder
            nl_filter: Nonlinear filter
            target_vector_field: Target symmetry vector field g(x) (for Lie derivative)
            lambda_lie: Weight for Lie derivative regularization
            lambda_curvature: Weight for curvature regularization (0 = disabled)
            lambda_rotation: Weight for rotation invariance regularization (0 = disabled)
            lie_normalize: Normalization for Lie derivative ("yang", "new", or False)
            curvature_order: Order for curvature regularization
            n_rotations: Number of rotations to sample for rotation invariance regularization
            device: Device for computation
        """
        super().__init__(
            dynamics_mod,
            likelihood_pdf,
            initial_c_pdf,
            backward_encoder,
            local_encoder,
            nl_filter,
            device,
        )

        # Store regularization parameters
        self.lambda_lie = lambda_lie
        self.lambda_curvature = lambda_curvature
        self.lambda_rotation = lambda_rotation
        self.target_vector_field = target_vector_field

        # Initialize regularizers
        if lambda_lie > 0 and target_vector_field is not None:
            self.lie_regularizer = LieDerivativeRegularizer(
                self.dynamics_mod.mean_fn, target_vector_field, normalize=lie_normalize
            )
        else:
            self.lie_regularizer = None

        if lambda_curvature > 0:
            self.curvature_regularizer = CurvatureRegularizer(
                self.dynamics_mod.mean_fn, order=curvature_order
            )
        else:
            self.curvature_regularizer = None

        if lambda_rotation > 0:
            self.rotation_regularizer = RotationInvarianceRegularizer(
                self.dynamics_mod.mean_fn, n_rotations=n_rotations
            )
        else:
            self.rotation_regularizer = None

    def forward(
        self,
        y: torch.Tensor,
        n_samples: int,
        p_mask_y_in: float = 0.0,
        p_mask_apb: float = 0.0,
        p_mask_a: float = 0.0,
        p_mask_b: float = 0.0,
        get_P_s: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with regularization.

        Returns:
            total_loss: Combined loss (ELBO + regularization terms)
            z_s: Latent samples [n_samples, n_trials, n_time, n_latents]
            stats: Dictionary with loss components and other statistics
        """

        # ell = self.likelihood_pdf.get_ell(y, z_s).mean(dim=0)
        # loss = stats['kl'] - ell
        # loss = loss.sum(dim=-1).mean()
        # return loss, z_s, stats

        elbo_loss, z_s, stats = super().forward(
            y, n_samples, p_mask_y_in, p_mask_apb, p_mask_a, p_mask_b, get_P_s
        )
        reg_losses = self._compute_regularization_losses(z_s)

        # Combined loss
        total_loss = elbo_loss + reg_losses["total_reg_loss"]

        # Update stats
        stats.update({"elbo_loss": elbo_loss, "total_loss": total_loss, **reg_losses})

        return total_loss, z_s, stats

    def _compute_regularization_losses(
        self, z_s: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all regularization losses.

        Args:
            z_s: Latent samples [n_samples, n_trials, n_time, n_latents]

        Returns:
            Dictionary with regularization loss components
        """
        # Use mean of samples for regularization evaluation
        z_mean = z_s.mean(dim=0)  # [n_trials, n_time, n_latents]

        # Flatten to [n_trials * n_time, n_latents] for regularizer evaluation
        z_flat = z_mean.reshape(-1, z_mean.shape[-1])

        reg_losses = {}
        total_reg_loss = torch.tensor(0.0, device=z_flat.device)  
        
        # Lie derivative regularization
        if self.lambda_lie > 0 and self.lie_regularizer is not None:
            lie_loss = self.lie_regularizer.regularizer(z_flat)
            reg_losses["lie_loss"] = lie_loss
            total_reg_loss += self.lambda_lie * lie_loss
        else:
            reg_losses["lie_loss"] = torch.tensor(0.0, device=z_flat.device)

        # Curvature regularization
        if self.lambda_curvature > 0 and self.curvature_regularizer is not None:
            curvature_loss = self.curvature_regularizer.regularizer(z_flat)
            reg_losses["curvature_loss"] = curvature_loss
            total_reg_loss += self.lambda_curvature * curvature_loss
        else:
            reg_losses["curvature_loss"] = torch.tensor(0.0, device=z_flat.device)

        # Rotation invariance regularization
        if self.lambda_rotation > 0 and self.rotation_regularizer is not None:
            rotation_loss = self.rotation_regularizer.regularizer(z_flat)
            reg_losses["rotation_loss"] = rotation_loss
            total_reg_loss += self.lambda_rotation * rotation_loss
        else:
            reg_losses["rotation_loss"] = torch.tensor(0.0, device=z_flat.device)

        reg_losses["total_reg_loss"] = total_reg_loss

        return reg_losses

    def get_loss_components(
        self, y: torch.Tensor, n_samples: int, **kwargs
    ) -> Dict[str, float]:
        """
        Get individual loss components for monitoring.

        Returns:
            Dictionary with loss components as scalars
        """
        with torch.no_grad():
            _, _, stats = self.forward(y, n_samples, **kwargs)

        return {
            "elbo_loss": stats["elbo_loss"].item(),
            "lie_loss": stats["lie_loss"].item(),
            "curvature_loss": stats["curvature_loss"].item(),
            "rotation_loss": stats["rotation_loss"].item(),
            "total_reg_loss": stats["total_reg_loss"].item(),
            "total_loss": stats["total_loss"].item(),
        }

    def evaluate_regularizers_at_points(
        self, points: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate regularizers at specific points for visualization.

        Args:
            points: Points to evaluate [n_points, n_latents]

        Returns:
            Dictionary with regularizer values at each point
        """
        results = {}

        if self.lambda_lie > 0 and self.lie_regularizer is not None:
            results["lie_values"] = self.lie_regularizer.eval_regularizer(points)

        if self.lambda_curvature > 0 and self.curvature_regularizer is not None:
            results["curvature_values"] = self.curvature_regularizer.eval_regularizer(
                points
            )

        if self.lambda_rotation > 0 and self.rotation_regularizer is not None:
            results["rotation_values"] = self.rotation_regularizer.eval_regularizer(
                points
            )

        return results


class RegularizedLightningSSM(LightningNonlinearSSM):
    """
    Lightning wrapper for RegularizedSSM to work with existing training infrastructure.

    This is a minimal adapter to make RegularizedSSM work with the existing
    LightningNonlinearSSM training code.
    """

    def __init__(self, regularized_ssm, cfg):
        # Initialize with the regularized SSM instead of base SSM
        super().__init__(regularized_ssm, cfg)

    def training_step(self, batch, batch_idx):
        y = batch[0]
        loss, _, stats = self.ssm(y, self.cfg.n_samples)

        # Log individual components
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_elbo", stats["elbo_loss"], prog_bar=False)
        self.log("train_lie", stats["lie_loss"], prog_bar=False)
        if stats["curvature_loss"] > 0:
            self.log("train_curvature", stats["curvature_loss"], prog_bar=False)
        if stats["rotation_loss"] > 0:
            self.log("train_rotation", stats["rotation_loss"], prog_bar=False)
        self.log("train_kl", stats['kl'].sum(dim=-1).mean(), prog_bar=False)
        self.log("train_recon", stats['kl'].sum(dim=-1).mean() - stats['elbo_loss'], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        y = batch[0]
        loss, _, stats = self.ssm(y, self.cfg.n_samples)

        # Log individual components
        self.log("valid_loss", loss, prog_bar=True, sync_dist=True)
        self.log("valid_elbo", stats["elbo_loss"], prog_bar=False, sync_dist=True)
        self.log("valid_lie", stats["lie_loss"], prog_bar=False, sync_dist=True)
        if stats["curvature_loss"] > 0:
            self.log(
                "valid_curvature",
                stats["curvature_loss"],
                prog_bar=False,
                sync_dist=True,
            )
        if stats["rotation_loss"] > 0:
            self.log(
                "valid_rotation",
                stats["rotation_loss"],
                prog_bar=False,
                sync_dist=True,
            )
        self.log("valid_kl", stats['kl'].sum(dim=-1).mean(), prog_bar=False)
        self.log("valid_recon", stats['kl'].sum(dim=-1).mean() - stats['elbo_loss'], prog_bar=False)

        return loss


# Helper function to create a simple rotation vector field for testing
def create_rotation_vector_field(rotation_speed: float = 1.0):
    """
    Create a simple 2D rotation vector field: v(x) = rotation_speed * [-y, x]
    """

    def rotation_field(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            return rotation_speed * torch.tensor([-x[1], x[0]], device=x.device)
        else:
            return rotation_speed * torch.stack([-x[..., 1], x[..., 0]], dim=-1)

    return rotation_field


# Example usage and testing
if __name__ == "__main__":
    # Simple test

    # Create a simple 2D rotation field
    target_field = create_rotation_vector_field(rotation_speed=1.0)

    # Test the regularizer
    test_points = torch.randn(100, 2)

    print("RegularizedSSM implementation ready!")
    print(f"Test points shape: {test_points.shape}")
    print(f"Target field output shape: {target_field(test_points).shape}")
