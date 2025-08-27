import unittest
from typing import Union, Literal, Callable, TypeAlias, Any, Optional
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, Tensor
from torch._functorch.eager_transforms import _jvp_with_argnums
from torch._functorch.utils import argnums_t

from __init__ import VectorField
from test_dynamics import ModuleWrapperBase

import math

T: TypeAlias = Tensor


class AbstractRegularizer(ABC):  # Inherit from ABC to make it an abstract class
    def regularizer(self, x: T) -> T:
        return self.eval_regularizer(x).sum()

    @abstractmethod
    def eval_regularizer(self, x: T) -> T:
        pass

    @staticmethod
    def _wrap_callable(func: Callable[[T], T]) -> nn.Module:
        mod = nn.Module()
        mod.forward = func  # type: ignore[attr-defined]
        return mod

    @property
    @abstractmethod
    def f(self, *args, **kwargs) -> T:
        """
        This method should be implemented to return the vector field.
        It can be old callable or an nn.Module.
        """
        pass


class CurvatureRegularizer(nn.Module, AbstractRegularizer):
    def __init__(self, f: VectorField, *args, order=1, **kwargs):
        super().__init__(*args, **kwargs)
        assert order >= 1, "Order must be >= 1"
        self._f = f if isinstance(f, nn.Module) else self._wrap_callable(f)
        self.order = order

    @property
    def f(self) -> nn.Module:
        return self._f

    def eval_regularizer(self, x: T) -> T:
        J2 = _compute_jet(self.f, x, order=self.order + 1)
        eps = torch.finfo(J2.dtype).eps
        Q, R = torch.linalg.qr(J2)
        diagR = torch.diagonal(R, dim1=-2, dim2=-1).abs()
        # kappa = diagR[..., 1] / (diagR[..., 0] + eps) ** 2
        i = self.order
        if i == 1:
            return diagR[..., i] / (diagR[..., 0] ** 2 + eps)
        else:
            return diagR[..., i] / (diagR[..., 0] * diagR[..., i - 1] + eps)


class LieDerivativeRegularizer(ModuleWrapperBase, nn.Module, AbstractRegularizer):
    """
    Computes the Lie bracket [X, y] = X·∇y − y·∇X of two vector fields X, y : ℝⁿ → ℝⁿ,
    in old fully differentiable way.
    X, y may be nn.Modules or plain callables.
    """

    def __init__(
        self,
        f: VectorField,
        g: VectorField,
        *args,
        normalize: Union[bool, Literal["yang", "new"]] = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._wrapped = f if isinstance(f, nn.Module) else self._wrap_callable(f)
        self.g = g if isinstance(g, nn.Module) else self._wrap_callable(g)
        self.normalize = normalize
        if not normalize:
            self._eval_regularizer = self._eval_regularizer_basic
            self._eval_vfields = self._eval_vfields_no_norm
        elif normalize == "yang" or normalize is True:
            self._eval_regularizer = self._regularizer_yang
            self._eval_vfields = self._eval_vfields_no_norm
        elif normalize == "new":
            self._eval_regularizer = self._eval_regularizer_basic
            self._eval_vfields = self._eval_vfields_norm
        else:
            raise ValueError(f"Unknown normalization type: {normalize}.")

    def eval_regularizer(self, x: T) -> T:
        return self._eval_regularizer(x)

    @property
    def f(self) -> nn.Module:
        """Expose the wrapped vector field as old read-only property"""
        return self._wrapped

    def forward(self, x: T) -> T:
        """
        Computes the Lie derivative
        Args:
            x:

        Returns:

        """
        _, _, DFg, DGf = self._eval_vfields(x)

        return DFg - DGf

    # def regularizer(obj, y: T) -> T:
    #     """
    #     Computes the regularization term for the Lie bracket, summing/averaging obj.regularizer(y) over time & batch.
    #     Args:
    #         y:
    #     Returns:
    #
    #     """
    #     return obj.regularizer(y).sum()

    def _regularizer_yang(self, x: T) -> T:
        DFg, ell2 = self._regularizer_common(x)
        eps = torch.finfo(DFg.dtype).eps
        DFg_norm = DFg.square().sum(dim=-1)
        return ell2 / (DFg_norm + eps)

    def _eval_regularizer_basic(self, x: T) -> T:
        return self._regularizer_common(x)[-1]

    def _regularizer_common(self, x):
        _, _, DFg, DGf = self._eval_vfields(x)
        eps = torch.finfo(DFg.dtype).eps
        ell2 = torch.linalg.vector_norm(DFg - DGf, ord=2, dim=-1).square()
        return DFg, ell2

    def _eval_vfields_no_norm(self, x):
        return self._eval_vfields_common(self.f, self.g, x)

    def _eval_vfields_norm(self, x):
        eps = torch.finfo(x.dtype).eps

        def Ftilde(x_):
            v = self.f(x_)
            v /= torch.linalg.norm(v, ord=2, dim=-1, keepdim=True) + eps
            return v

        return self._eval_vfields_common(Ftilde, self.g, x)

    def _eval_vfields_common(
        self, F: Callable[[T], T], G: Callable[[T], T], x: T
    ) -> tuple[T, T, T, T]:
        f, DFg = self.directional_derivative(F, G, x)
        g, DGf = self.directional_derivative(G, F, x)
        return f, g, DFg, DGf

    @staticmethod
    def directional_derivative(v1: VectorField, v2: VectorField, x: T) -> tuple[T, T]:
        # dynamics, DFg = torch.f.jvp(obj.F, (y,), (obj.G(y),), strict=False)
        # g, DGf = torch.autograd.functional.jvp(obj.G, (y,), (obj.F(y),), strict=False,
        #                                                       create_graph=True)
        # dv = torch.autograd.functional.jvp(
        #     v1,
        #     (x,),
        #     (v2(x),),
        #     strict=False,  # pyright: ignore [reportReturnType]
        #     create_graph=True,
        # )
        dv = torch.func.jvp(v1, (x,), (v2(x),), strict=False, has_aux=False)

        return dv


def _jvp(f: Callable, p: Any, v: Any, *, argnums: Optional[argnums_t] = 0):
    return _jvp_with_argnums(f, p, v, argnums=argnums, strict=False, has_aux=False)[-1]


def _compute_jet_g_along_f_new(
    g, f, x: torch.Tensor, order: int, normalize: bool = True
) -> torch.Tensor:
    if order > 4:
        raise NotImplementedError("Jets higher than order 4 are not implemented.")

    def norm(term, k):
        return term if not normalize else term / math.factorial(k)

    # Define basic derivative operators
    Df = lambda y, v: _jvp(f, (y,), (v,))
    Dg = lambda y, v: _jvp(g, (y,), (v,))

    # Compute f derivatives
    f1 = f(x)
    f2 = Df(x, f1)

    # Only compute f3 and f4 if needed
    if order >= 2:
        Df_f1 = lambda y: Df(y, f1)
        D2f_f1 = lambda y, v: _jvp(Df_f1, (y,), (v,))
        f3 = Df(x, f2) + D2f_f1(x, f1)

    if order >= 3:
        f4 = Df(x, f3) + 3 * D2f_f1(x, f2) + _jvp(lambda y: D2f_f1(y, f1), (x,), (f1,))

    # Store f-terms
    F = [None, f1, f2, f3 if order >= 2 else None, f4 if order >= 3 else None]
    jets = []

    # g^(1): Dg[f1]
    g1 = Dg(x, F[1])
    jets.append(norm(g1, 1))

    if order >= 2:
        # Define operators needed for order 2
        Dg_f1 = lambda y: Dg(y, F[1])
        D2g_f1 = lambda y, v: _jvp(Dg_f1, (y,), (v,))

        # g^(2): Dg[f2] + DBx0^2 g[f1,f1]
        g2 = Dg(x, F[2]) + D2g_f1(x, F[1])
        jets.append(norm(g2, 2))

    if order >= 3:
        # Define additional operators needed for order 3
        D2g_f1_f1 = lambda y: D2g_f1(y, F[1])  # Partial application
        D3g_f1_f1 = lambda y, v: _jvp(D2g_f1_f1, (y,), (v,))

        # g^(3): Dg[f3] + 3 DBx0^2 g[f1,f2] + DBx0^3 g[f1,f1,f1]
        g3 = Dg(x, F[3]) + 3 * D2g_f1(x, F[2]) + D3g_f1_f1(x, F[1])
        jets.append(norm(g3, 3))

    if order >= 4:
        # Define additional operators needed for order 4
        Dg_f2 = lambda y: Dg(y, F[2])
        D2g_f2 = lambda y, v: _jvp(Dg_f2, (y,), (v,))

        # For DBx0^4 g[f1,f1,f1,f1], we need a function of one argument
        D3g_f1_f1_f1 = lambda y: D3g_f1_f1(y, F[1])

        # g^(4): Dg[f4] + 4 DBx0^2 g[f1,f3] + 3 DBx0^2 g[f2,f2]
        #       + 6 DBx0^3 g[f1,f1,f2] + DBx0^4 g[f1,f1,f1,f1]
        g4 = (
            Dg(x, F[4])
            + 4 * D2g_f1(x, F[3])
            + 3 * D2g_f2(x, F[2])
            + 6 * D3g_f1_f1(x, F[2])
            + _jvp(D3g_f1_f1_f1, (x,), (F[1],))  # DBx0^4 g[f1,f1,f1,f1]
        )
        jets.append(norm(g4, 4))

    return torch.stack(jets, dim=-1)


# Helper to compute DBx0^k f[v1, v2, ..., vk] or DBx0^k g[v1, v2, ..., vk]
def _nested_jvp(func, point, directions):
    """Compute higher order directional derivative."""
    if len(directions) == 0:
        return func(point)
    elif len(directions) == 1:
        return _jvp(func, (point,), (directions[0],))
    else:
        # Take derivative of the (k-1)-th order derivative
        inner_func = lambda y: _nested_jvp(func, y, directions[1:])
        return _jvp(inner_func, (point,), (directions[0],))


def _compute_jet_g_along_f(
    g, f, x: Tensor, order: int, normalize: bool = True
) -> Tensor:
    if order > 4:
        raise NotImplementedError("Jets higher than order 4 are not implemented.")

    def norm(term, k):
        return term if not normalize else term / math.factorial(k)

    # Compute f1..f4 using your original recurrence
    f1 = f(x)
    f2 = _jvp(f, (x,), (f1,))
    f3 = _jvp(f, (x,), (f2,)) + _jvp(lambda y: _jvp(f, (y,), (f1,)), (x,), (f1,))
    f4 = (
        _jvp(f, (x,), (f3,))
        + 3 * _jvp(lambda y: _jvp(f, (y,), (f2,)), (x,), (f1,))
        + _jvp(lambda y: _jvp(lambda z: _jvp(f, (z,), (f1,)), (y,), (f1,)), (x,), (f1,))
    )

    # Store f-terms for convenience
    F = [None, f1, f2, f3, f4]

    jets = []

    # g^(1): Dg[f1]
    g1 = _jvp(g, (x,), (F[1],))
    jets.append(norm(g1, 1))

    if order >= 2:
        # g^(2): Dg[f2] + DBx0^2 g[f1,f1]
        term1 = _jvp(g, (x,), (F[2],))
        term2 = _jvp(lambda y: _jvp(g, (y,), (F[1],)), (x,), (F[1],))
        g2 = term1 + term2
        jets.append(norm(g2, 2))

    if order >= 3:
        # g^(3): Dg[f3] + 3 DBx0^2 g[f1,f2] + DBx0^3 g[f1,f1,f1]
        term1 = _jvp(g, (x,), (F[3],))
        term2 = 3 * _jvp(lambda y: _jvp(g, (y,), (F[2],)), (x,), (F[1],))
        term3 = _jvp(
            lambda y: _jvp(lambda z: _jvp(g, (z,), (F[1],)), (y,), (F[1],)),
            (x,),
            (F[1],),
        )
        g3 = term1 + term2 + term3
        jets.append(norm(g3, 3))

    if order >= 4:
        # g^(4): Dg[f4] + 4 DBx0^2 g[f1,f3] + 3 DBx0^2 g[f2,f2]
        #       + 6 DBx0^3 g[f1,f1,f2] + DBx0^4 g[f1,f1,f1,f1]
        term1 = _jvp(g, (x,), (F[4],))
        term2 = 4 * _jvp(lambda y: _jvp(g, (y,), (F[3],)), (x,), (F[1],))
        term3 = 3 * _jvp(lambda y: _jvp(g, (y,), (F[2],)), (x,), (F[2],))
        term4 = 6 * _jvp(
            lambda y: _jvp(lambda z: _jvp(g, (z,), (F[2],)), (y,), (F[1],)),
            (x,),
            (F[1],),
        )
        term5 = _jvp(
            lambda y: _jvp(
                lambda z: _jvp(lambda w: _jvp(g, (w,), (F[1],)), (z,), (F[1],)),
                (y,),
                (F[1],),
            ),
            (x,),
            (F[1],),
        )
        g4 = term1 + term2 + term3 + term4 + term5
        jets.append(norm(g4, 4))

    return torch.stack(jets, dim=-1)  # shape (..., order)


def _compute_jet(f: VectorField, x: Tensor, order: int) -> Tensor:
    return torch.cat(
        (
            f(x).squeeze()[..., None],
            _compute_jet_g_along_f(f, f, x, order=order, normalize=True),
        ),
        dim=-1,
    )


def _compute_jet_old(
    f, x: Tensor, order: int, *, normalize: bool = False, no_linear: bool = False
) -> Tensor:
    """
    Computes higher-order time derivatives (prolongations) of old vector field.
    """
    if order > 4:
        raise NotImplementedError("Jets higher than order 4 are not implemented.")

    if normalize:
        _normalize = lambda term, k: term / math.factorial(k)
    else:
        _normalize = lambda term, k: term

    Df = lambda p, v1: _jvp(f, (p,), (v1,))  # noqa
    Df2_v = lambda p, v1, v2: _jvp(Df, (p, v1), (v2,))  # noqa
    Df3_v_w = lambda p, v1, v2, v3: _jvp(Df2_v, (p, v1, v2), (v3,))  # noqa
    Df4_v_w_z = lambda p, v1, v2, v3, v4: _jvp(Df3_v_w, (p, v1, v2, v3), (v4,))  # noqa

    if no_linear:
        _Df = lambda p, v1: torch.zeros_like(v1)
    else:
        _Df = Df  # noqa

    f0 = f(x)  # df/dt
    jets = [f0]  # zeroth order term

    if order >= 1:
        f1 = _Df(x, f0)
        jets.append(_normalize(f1, 1))
    if order >= 2:
        f2 = _Df(x, f1) + Df2_v(x, f0, f0)
        jets.append(_normalize(f2, 2))
    if order >= 3:
        f3 = _Df(x, f2) + 3 * Df2_v(x, f0, f1) + Df3_v_w(x, f0, f0, f0)
        jets.append(_normalize(f3, 3))
    if order >= 4:
        f5 = (
            _Df(x, f3)
            + 4 * Df2_v(x, f0, f2)
            + 3 * Df2_v(x, f1, f1)
            + 6 * Df3_v_w(x, f0, f0, f1)
            + Df4_v_w_z(x, f0, f0, f0, f0)
        )
        jets.append(f5)

    return torch.stack(jets, dim=-1)


class TestRegularizers(unittest.TestCase):
    @property
    def regularizer_list(self):
        from functools import partial
        from test_dynamics import dynamics_factory

        g_ = dynamics_factory()
        _regularizer_list = []
        _regularizer_list.append(partial(CurvatureRegularizer, order=1))
        _regularizer_list.append(
            partial(LieDerivativeRegularizer, g=g_, normalize=True)
        )
        _regularizer_list.append(
            partial(LieDerivativeRegularizer, g=g_, normalize="yang")
        )
        _regularizer_list.append(
            partial(LieDerivativeRegularizer, g=g_, normalize="new")
        )
        return []

    @property
    def eps(self):
        return 1e-3

    @staticmethod
    def f_torch(y: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [torch.sin(y[..., 0] ** 2 + y[..., 1]), y[..., 1] ** 3], dim=-1
        )

    @staticmethod
    def load_jax_utils():
        try:
            import jax
            import jax.numpy as jnp
            from jax.experimental.jet import jet
            from scipy.special import factorial
        except ImportError:
            raise unittest.SkipTest("JAX not available")
        else:

            def compute_jet_jax(f, z: jnp.ndarray, order: int):
                """
                Compute jets using JAX's jet up to the given order.
                """

                def _fix(nth):
                    return factorial(nth, exact=True) / jax.lax.exp(
                        jax.lax.lgamma(nth + 1.0)
                    )

                (y0, [*yns]) = jet(f, (z,), ((f(z),),))
                for _ in range(order - 1):
                    (y0, [*yns]) = jet(f, (z,), ((y0, *yns),))

                for i, yn in enumerate(yns):
                    yns[i] = _fix(i) * yn

                return yns[:-1]

            # Define functions for tests
            def f_jax(y: jnp.ndarray) -> jnp.ndarray:
                return jnp.stack([jnp.sin(y[0] ** 2 + y[1]), y[1] ** 3])

            z0_jax = jnp.array([1.0, 2.0])
            return compute_jet_jax, f_jax, z0_jax

    def test_regularizer_differentiability(self):
        """Test if regularizers are differentiable."""
        x = torch.randn(10, 2, requires_grad=True)
        for reg in self.regularizer_list:
            reg_instance = reg(self.f_torch)
            reg_instance.regularizer(x).backward()
            self.assertIsNotNone(x.grad, f"Gradient is None for {reg.__name__}.")
            self.assertTrue(
                torch.any(x.grad != 0), f"Gradient is zero for {reg.__name__}."
            )

    def test_curvature_regularizer(self):
        f = self.f_torch
        pts = torch.randn(100, 2)
        J2 = _compute_jet(f, pts, order=2)
        kappa1 = (
            torch.diff(J2[..., 0] * J2[..., 1].flip(dims=(-1,)), dim=-1)
            .abs_()
            .squeeze()
        )
        kappa1 /= (
            torch.linalg.norm(J2[..., 0], ord=2, dim=-1).pow(3)
            + torch.finfo(J2.dtype).eps
        )
        eps = self.eps

        kappa = CurvatureRegularizer(f, order=1).eval_regularizer(pts)

        self.assertTrue(
            torch.allclose(kappa, kappa1, atol=eps, rtol=eps, equal_nan=True),
            "1st Curvature doesn't match analytical.",
        )

    def test_torch_jax_reference(self):
        """Compare PyTorch and JAX jets up to order 3."""
        compute_jet_jax, f_jax, z0_jax = self.load_jax_utils()
        z0_torch = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
        eps = self.eps

        self.assertTrue(
            np.allclose(
                self.f_torch(z0_torch).detach().numpy(),
                np.array(f_jax(z0_jax)),
                atol=eps,
                rtol=eps,
            ),
            "First derivative mismatch between Torch and JAX.",
        )

        jets_jax = compute_jet_jax(f_jax, z0_jax, order=3)

        jets_torch = _compute_jet(self.f_torch, z0_torch, order=3)[:, 1:].mT

        self.assertTrue(
            np.allclose(
                jets_torch.detach().numpy(), np.array(jets_jax), atol=eps, rtol=eps
            ),
            "Torch and JAX jets differ.",
        )

    def test_autodiff_compatibility(self):
        """Ensure compute_jet_torch supports backprop."""
        u0_torch = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
        jets_torch = _compute_jet(self.f_torch, u0_torch, order=4)

        # Backprop through norm
        jets_torch.norm().backward()
        self.assertIsNotNone(u0_torch.grad, "Gradient is None.")
        self.assertTrue(torch.any(u0_torch.grad != 0), "Gradient is zero.")

    def test_batched_vmap(self):
        """Test batched computation using torch.vmap."""
        u0_batch = torch.randn(3, 2)
        batched_jets = torch.vmap(lambda u_: _compute_jet(self.f_torch, u_, order=4))(
            u0_batch
        )

        self.assertEqual(batched_jets.shape[0], u0_batch.shape[0], "Batching failed.")


if __name__ == "__main__":
    # unittest.main()
    A = torch.randn(2, 2)

    # w = w.T - w
    def lds(x: Tensor) -> Tensor:
        return x @ A

    def _rosenbrock(x: Tensor, alpha: Optional[float] = None) -> Tensor:
        if alpha is None:
            alpha = 0.0
        x_coord = x[..., 0]
        y_coord = x[..., 1]
        grad_x = -alpha * (2 * (1 - x_coord)) - 4 * x_coord * (y_coord - x_coord**2)
        grad_y = 2 * (y_coord - x_coord**2)
        gradient = torch.stack([grad_x, grad_y], dim=-1)
        return -gradient

    x = torch.randn(2)

    a = _compute_jet_g_along_f_new(_rosenbrock, _rosenbrock, x, order=4)
    # b = _compute_jet_g_along_f(lds, lds, x, order=4)
