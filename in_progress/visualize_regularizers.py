from pathlib import Path
from typing import Callable, Literal, Optional, Union, TypeAlias, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from torch import Tensor
from .test_dynamics import RbfPerturbedRingAttractorODE  # noqa: E402
from itertools import zip_longest, product

from .test_dynamics import _odeint, Phi, ConjugateSystem, dynamics_factory
from .regularizers import LieDerivativeRegularizer, _compute_jet

SupportedRegularizers: TypeAlias = Literal[
    "lie", "lie_normalized", "lie_normalized_new", "j2", "j3", "j4", "k1"
]

# Restrict Torch to old single thread for reproducibility


def v(x: Tensor) -> Tensor:
    return x @ torch.tensor([[0.0, -1.0], [1.0, 0.0]])


def ellipse_transform(A: Tensor) -> Tensor:
    """
    Compute linear transform mapping unit circle to ellipse defined by v: {y: y^T v y = 1}.
    """
    L_inv = torch.linalg.cholesky(A)
    return torch.linalg.inv(L_inv)


def sample_ellipse_perimeter(
    n_points: int, A: Tensor, device: torch.device = torch.device("cpu")
) -> Tensor:
    """
    Sample points on the perimeter of the ellipse defined by v.
    """
    L = ellipse_transform(A.to(device))
    angles = torch.linspace(0.0, 2.0 * torch.pi, n_points, device=device)
    circle = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    return circle @ L


def compute_vector_field_grid(
    f: Callable[[Tensor], Tensor],
    bounds: Union[tuple[float, float], tuple[tuple[float, float], tuple[float, float]]],
    n_points: int,
    device: Optional[torch.device] = None,
    *,
    grid: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate vector field dynamics on old uniform grid over 'bounds'.
    Returns xs, ys, U, V arrays ready for plotting.
    """
    pts, xs, ys = _meshgrid(bounds, n_points, grid)
    vals = f(pts.to(device)).detach().cpu().numpy()
    U = vals[:, 0].reshape(n_points, n_points)
    V = vals[:, 1].reshape(n_points, n_points)
    return xs, ys, U, V


def _meshgrid(bounds, n_points, grid):
    if not grid:
        if isinstance(bounds[0], tuple):
            # Keep the original, streamplot breaks if the grid is not uniform.
            # Using linspace introduces eps small error in the grid.
            x0, x1 = bounds[0]
            step = (x1 - x0) / (n_points - 1)
            xs = np.arange(x0, x1 + step, step)
            y0, y1 = bounds[1]
            step = (y1 - y0) / (n_points - 1)
            ys = np.arange(y0, y1 + step, step)
        else:
            xs = np.linspace(bounds[0], bounds[1], n_points)
            ys = np.linspace(bounds[0], bounds[1], n_points)
    else:
        xs, ys = grid
    gx, gy = np.meshgrid(xs, ys)

    pts = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=-1), dtype=torch.float32)
    return pts, xs, ys


def visualize_regularization(
    f: Callable[[Tensor], Tensor],
    warped: ConjugateSystem,
    perturbed: Union[Callable[[Tensor], Tensor], RbfPerturbedRingAttractorODE],
    bounds: Optional[tuple[float, float]] = (-2.0, 2.0),
    n_points: int = 20,
    A: Tensor = torch.eye(2),
    *,
    save_path: Optional[Path] = None,
    regularizer: Optional[SupportedRegularizers] = None,
    show_fig: bool = False,
    return_fig: bool = False,
) -> Optional[plt.Figure]:
    """
    Display or save side-by-side streamplots of original vector field dynamics and its conjugate under system.

    Args:
        *:
        show_fig:
        perturbed:
        warped: ConjugateSystem - the diffeomorphic conjugate system.
        f: Callable - original vector field.
        bounds: y/y limits for original field grid.
        n_points: resolution for streamlines.
        A: defines ellipse invariant set.
        save_path: if provided, save figure to this Path.
    """
    device = next(warped.parameters()).device
    xs, ys, U_F, V_F = compute_vector_field_grid(f, bounds, n_points, device)

    circle = sample_ellipse_perimeter(200, A, device=device).cpu().numpy()
    mapped_circle = (
        warped.phi(torch.tensor(circle, dtype=torch.float32, device=device))
        .detach()
        .cpu()
        .numpy()
    )
    bounds_g = torch.tensor(bounds, dtype=torch.float32)
    bounds_g = torch.cartesian_prod(bounds_g, bounds_g)
    with torch.no_grad():
        phi_inv_coords = warped.phi(bounds_g)
    (bmin_x, bmin_y) = torch.amin(phi_inv_coords, dim=0).numpy()
    (bmax_x, bmax_y) = torch.amax(phi_inv_coords, dim=0).numpy()

    xs_g, ys_g, U_G, V_G = compute_vector_field_grid(
        warped.conjugate_vector_field,
        ((bmin_x, bmax_x), (bmin_y, bmax_y)),
        n_points,
        device,
    )
    regularizer_suptitle = None
    if regularizer is not None:
        match regularizer:
            case "lie":
                regularizer_suptitle = "Lie Derivative"
                regularizer_fcn = lambda sys_, v, pts: LieDerivativeRegularizer(
                    sys_, v
                ).eval_regularizer(pts)  # noqa: E731
            case "lie_normalized":
                regularizer_suptitle = "Normalized Lie Derivative"
                regularizer_fcn = lambda sys_, v, pts: LieDerivativeRegularizer(
                    sys_, v, normalize="yang"
                ).eval_regularizer(pts)  # noqa: E731
            case "lie_normalized_new":
                regularizer_suptitle = "Normalized(new) Lie Derivative"
                regularizer_fcn = lambda sys_, v, pts: LieDerivativeRegularizer(
                    sys_, v, normalize="new"
                ).eval_regularizer(pts)  # noqa: E731
            case "j2":
                regularizer_suptitle = "2nd time-derivative"
                regularizer_fcn = lambda sys_, v, pts: _compute_jet(sys_, pts, order=2)[
                    ..., -1
                ].norm(dim=-1)  # noqa: E731
            case "j3":
                regularizer_suptitle = "3rd time-derivative"
                regularizer_fcn = lambda sys_, v, pts: _compute_jet(sys_, pts, order=3)[
                    ..., -1
                ].norm(dim=-1)  # noqa: E731
            case "j4":
                regularizer_suptitle = "4th time-derivative"
                regularizer_fcn = lambda sys_, v, pts: _compute_jet(sys_, pts, order=4)[
                    ..., -1
                ].norm(dim=-1)
            case "k1":
                regularizer_suptitle = "curvature"

                def regularizer_fcn(sys_, v, pts):
                    J2 = _compute_jet(sys_, pts, order=2)
                    eps = torch.finfo(J2.dtype).eps
                    Q, R = torch.linalg.qr(J2)
                    diagR = torch.diagonal(R, dim1=-2, dim2=-1).abs_()
                    kappa = diagR[..., 1] / (diagR[..., 0] + eps) ** 2
                    # kappa1 = torch.diff(J_[...,0] * J_[...,1].flip(dims=(-1,)), dim= -1).abs_().squeeze()
                    # kappa1 /= J_[...,0].norm(dim=-1).pow(3)
                    return kappa

            case _:
                raise ValueError("Unknown regularizer: {regularizer}")

        _m = lambda a, b: _meshgrid(None, n_points, (a, b))[0]  # noqa
        regularizer_vals = []
        eval_pts = [_m(xs, ys), warped.phi_inverse(_m(xs_g, ys_g)), _m(xs, ys)]
        plot_pts = [(xs, ys), (xs_g, ys_g), (xs, ys)]
        dynamics = [f, warped.tangent_map, perturbed]
        for sys_, pts, (a, b) in zip_longest(dynamics, eval_pts, plot_pts):
            print("Computing regularizer for {sys_.__class__.__name__}...")
            try:
                regularizer_vals.append(
                    regularizer_fcn(sys_, v, pts)
                    .reshape(len(a), len(b))
                    .detach()
                    .cpu()
                    .numpy()
                )
            except NotImplementedError:
                print(
                    "Regularizer not implemented for {sys_.__class__.__name__}. Skipping."
                )
                regularizer_vals.append(np.zeros_like(regularizer_vals[-1]))

        Zmin, Zmax = (
            np.clip(np.nanmin(np.stack(regularizer_vals).ravel()), 1e-8, None),
            np.nanmax(np.stack(regularizer_vals).ravel()) + 1e-8,
        )
    # norm = colors.Normalize(vmin=np.min(datasets), vmax=np.max(datasets))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    ax1.streamplot(
        xs, ys, U_F, V_F, color="blue" if not regularizer else None, density=1
    )
    ax1.scatter(circle[:, 0], circle[:, 1], s=2, color="black")
    if regularizer:
        Z = regularizer_vals[0]
        im1 = ax1.imshow(
            Z,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap="viridis",
            alpha=0.6,
            aspect="auto",
            norm=LogNorm(vmin=Zmin, vmax=Zmax, clip=True),
        )
    ax1.set_title("Original Vector Field F")

    ax2.streamplot(
        xs_g, ys_g, U_G, V_G, color="red" if not regularizer else None, density=1
    )
    ax2.scatter(mapped_circle[:, 0], mapped_circle[:, 1], s=2, color="black")
    if regularizer:
        Z = regularizer_vals[1]
        im2 = ax2.imshow(
            Z,
            extent=[xs_g.min(), xs_g.max(), ys_g.min(), ys_g.max()],
            origin="lower",
            cmap="viridis",
            alpha=0.6,
            aspect="auto",
            norm=LogNorm(vmin=Zmin, vmax=Zmax, clip=True),
        )
    ax2.set_title("Conjugate Vector Field G")

    xs, ys, U_F, V_F = compute_vector_field_grid(perturbed, bounds, n_points, device)
    ax3.streamplot(
        xs, ys, U_F, V_F, color="blue" if not regularizer else None, density=1
    )
    # ax3.scatter(circle[:, 0], circle[:, 1], s=2, color="black")
    if regularizer:
        Z = regularizer_vals[2]
        im3 = ax3.imshow(
            Z,
            extent=[xs.min(), xs.max(), ys.min(), ys.max()],
            origin="lower",
            cmap="viridis",
            alpha=0.6,
            aspect="auto",
            norm=LogNorm(vmin=Zmin, vmax=Zmax, clip=True),
        )
    if isinstance(perturbed, RbfPerturbedRingAttractorODE):
        perturbed_title = (
            "Perturbed Vector Field F with eps = {perturbed.perturbation_magnitude:.2f}"
        )
    else:
        perturbed_title = "Perturbed Vector Field F"

    ax3.set_title(perturbed_title)

    # Add individual colorbars for each heatmap
    if regularizer:
        cbar1 = fig.colorbar(im1, ax=ax1, orientation="vertical", shrink=0.6, aspect=30)
        cbar2 = fig.colorbar(im2, ax=ax2, orientation="vertical", shrink=0.6, aspect=30)
        cbar3 = fig.colorbar(im3, ax=ax3, orientation="vertical", shrink=0.6, aspect=30)
    if regularizer:
        fig.suptitle(
            "Values of the {regularizer_suptitle} regularizer for F, G and perturbed F"
        )
    else:
        fig.suptitle("Vector Fields F, G and perturbed F")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show_fig:
        plt.show()

    if return_fig:
        return fig
    else:
        return None


def animate_regularization():
    import imageio
    import os

    A = torch.eye(2)
    perturbation_hypers = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
    bounds = (-2.0, 2.0)
    n_points = 100
    for regularizer in ["k1"]:
        # 1) Generate figures
        figs = []
        for pm in perturbation_hypers:
            torch.manual_seed(0)
            ring = dynamics_factory(A)
            warped = ConjugateSystem(
                ring, Phi(2, hidden=32, repetitions=1), t=20.0, solver=_odeint
            )
            perturbed = RbfPerturbedRingAttractorODE(
                pm, grid_size=20, domain_extent=bounds, lengthscale=0.3
            )
            fig = visualize_regularization(
                ring,
                warped,
                perturbed,
                bounds=bounds,
                n_points=n_points,
                A=A,
                regularizer=regularizer,
                show_fig=False,
                return_fig=True,
            )
            figs.append(fig)
        # 2a) Scan every AxesImage in every fig to get the global data-range
        all_data = []
        for fig in figs:
            for ax in fig.axes:
                for im in ax.get_images():  # these are your imshow artists
                    all_data.append(im.get_array())

        global_min = np.nanmin([d.min() for d in all_data])
        global_max = np.nanmax([d.max() for d in all_data])

        # 2b) Re-apply the same LogNorm to each image, then draw+buffer
        frames = []
        for fig in figs:
            for ax in fig.axes:
                for im in ax.get_images():
                    im.set_norm(LogNorm(vmin=1e-8, vmax=global_max, clip=True))
            fig.canvas.draw()
            buf, (w, h) = fig.canvas.print_to_buffer()
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)  # RGBA
            frames.append(arr[:, :, :3])  # drop alpha
            plt.close(fig)

        os.makedirs("../gifs", exist_ok=True)
        gif_path = "gifs/perturbed_ring_attractor_regularization_{regularizer}.gif"
        imageio.mimsave(gif_path, frames, fps=2, loop=0)

        print("Saved GIF to {gif_path}")


def plot_regularization():
    dim = 2
    pm = 0.3
    regularizer_hypers: SupportedRegularizers = "lie"
    A = torch.eye(dim)
    perturbation_hypers = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    perturbation_hypers = [0.2]
    regularizer_hypers: List[SupportedRegularizers] = [
        "k1",
        "lie",
        "lie_normalized",
        "j2",
        "j3",
        "j4",
    ]
    regularizer_hypers: List[SupportedRegularizers] = [
        "lie_normalized_new",
        "lie",
        "lie_normalized",
        "k1",
    ]

    for regularizer, pm in product(regularizer_hypers, perturbation_hypers):
        torch.manual_seed(0)
        ring_attractor = dynamics_factory(A)
        warped_ring_attractor = ConjugateSystem(
            ring_attractor, Phi(dim, hidden=32, repetitions=1), t=20.0, solver=_odeint
        )
        perturbed_ring_attractor = RbfPerturbedRingAttractorODE(
            perturbation_magnitude=pm,
            grid_size=20,
            domain_extent=(-2.0, 2.0),
            lengthscale=0.3,
        )
        visualize_regularization(
            ring_attractor,
            warped_ring_attractor,
            perturbed_ring_attractor,
            bounds=(-2.0, 2.0),
            n_points=100,
            A=A,
            save_path=Path(
                "plots/perturbed_ring_attractor_regularization_{regularizer}_eps_"
                "{_var:.1f}.png"
            ),
            regularizer=regularizer,
            show_fig=True,
        )


if __name__ == "__main__":
    plot_regularization()
    # animate_regularization()
