import unittest
from typing import Union, Literal, Callable, TypeAlias
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn, Tensor
from torch._functorch.eager_transforms import _jvp_with_argnums

from __init__ import VectorField
T: TypeAlias = Tensor


class AbstractRegularizer(ABC):  # Inherit from ABC to make it an abstract class

    def regularizer(self, x:T) -> T:
        return self.eval_regularizer(x).sum()

    @abstractmethod
    def eval_regularizer(self, x:T) -> T:
        pass

    @staticmethod
    def _wrap_callable(func: Callable[[T], T]) -> nn.Module:
        mod = nn.Module()
        mod.forward = func  # type: ignore[attr-defined]
        return mod

    @property
    @abstractmethod
    def f(self,*args,**kwargs) -> T:
        """
        This method should be implemented to return the vector field.
        It can be a callable or an nn.Module.
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
    def eval_regularizer(self, x:T)->T:
        J2 = _compute_jet(self.f, x, order=self.order+1)
        eps = torch.finfo(J2.dtype).eps
        Q, R = torch.linalg.qr(J2)
        diagR = torch.diagonal(R, dim1=-2, dim2=-1).abs_()
        # kappa = diagR[..., 1] / (diagR[..., 0] + eps) ** 2
        i = self.order
        if i == 1:
            return diagR[..., i] / (diagR[..., 0]** 2 + eps)
        else:
            return diagR[..., i] / (diagR[..., 0] * diagR[..., i-1] + eps)



class LieDerivativeRegularizer(nn.Module, AbstractRegularizer):
    """
    Computes the Lie bracket [X, Y] = X·∇Y − Y·∇X of two vector fields X, Y : ℝⁿ → ℝⁿ,
    in a fully differentiable way.
    X, Y may be nn.Modules or plain callables.
    """

    def __init__(self, f: VectorField, g: VectorField, *args, normalize: Union[bool,Literal["yang","new"]] = False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._f = f if isinstance(f, nn.Module) else self._wrap_callable(f)
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

    def eval_regularizer(self, x:T) -> T:
        return self._eval_regularizer(x)

    @property
    def f(self) -> nn.Module:
        """Expose the wrapped vector field as a read-only property"""
        return self._f
    def forward(self, x: T) -> T:
        """
        Computes the Lie derivative
        Args:
            x:

        Returns:

        """
        _,_,DFg,DGf = self._eval_vfields(x)

        return DFg - DGf

    # def regularizer(self, x: T) -> T:
    #     """
    #     Computes the regularization term for the Lie bracket, summing/averaging self.regularizer(x) over time & batch.
    #     Args:
    #         x:
    #     Returns:
    #
    #     """
    #     return self.regularizer(x).sum()

    def _regularizer_yang(self, x: T) -> T:
        DFg, ell2 = self._regularizer_common(x)
        eps = torch.finfo(DFg.dtype).eps
        DFg_norm = DFg.square().sum(dim=-1)
        return ell2 / (DFg_norm + eps)

    def _eval_regularizer_basic(self, x: T) -> T:
        return  self._regularizer_common(x)[-1]

    def _regularizer_common(self, x):
        _, _, DFg, DGf = self._eval_vfields(x)
        eps = torch.finfo(DFg.dtype).eps
        ell2 = torch.norm(DFg - DGf, p=2, dim=-1).square()
        return DFg, ell2

    def _eval_vfields_no_norm(self, x):
        return self._eval_vfields_common(self.f, self.g, x)

    def _eval_vfields_norm(self, x):
        eps = torch.finfo(x.dtype).eps

        def Ftilde(x_):
            v = self.f(x_)
            v/=(self.f(x_).norm(dim=-1, keepdim=True) + eps)
            return v

        return self._eval_vfields_common(Ftilde, self.g, x)

    def _eval_vfields_common(self, F:Callable[[T], T], G:Callable[[T], T], x:T) -> tuple[T,T,T,T]:
        f, DFg = self.directional_derivative(F, G, x)
        g, DGf = self.directional_derivative(G, F, x)
        return f,g,DFg, DGf

    @staticmethod
    def directional_derivative(v1:VectorField, v2:VectorField, x:T) -> tuple[T,T]:
        # f, DFg = torch.func.jvp(self.F, (x,), (self.G(x),), strict=False)
        # g, DGf = torch.autograd.functional.jvp(self.G, (x,), (self.F(x),), strict=False,
        #                                                       create_graph=True)
        return torch.autograd.functional.jvp(v1, (x,), (v2(x),), strict=False,  # pyright: ignore [reportReturnType]
                                             create_graph=True)


def _jvp(f, p, v):
    return _jvp_with_argnums(f, p, v, argnums=0, strict=True, has_aux=False)[-1]


def _compute_jet(f, x: Tensor, order: int) -> Tensor:
    """
    Computes higher-order time derivatives (prolongations) of a vector field.
    """
    if order > 4:
        raise NotImplementedError("Jets higher than order 4 are not implemented.")

    j2 = lambda p, v: _jvp(f, (p,), (v,)) # noqa
    j3 = lambda p, v, w: _jvp(j2, (p, v), (w,)) # noqa
    j4 = lambda p, v, w, z: _jvp(j3, (p, v, w), (z,)) # noqa

    f1 = f(x)  # df/dt
    jets = [f1]

    if order >= 2:
        f2 = j2(x, f1) # d^2f/dt^2
        jets.append(f2)
    if order >= 3:
        f3 = j2(x, f2) + j3(x, f1, f1) # d^3f/dt^3
        jets.append(f3)
    if order >= 4:
        f4 = j2(x, f3) + 3 * j3(x, f1, f2) + j4(x, f1, f1, f1) # d^4f/dt^4
        jets.append(f4)

    return torch.stack(jets, dim=-1)


class TestRegularizers(unittest.TestCase):

    @property
    def regularizer_list(self):
        from functools import partial
        from test_dynamics import dynamics_factory
        g_ = dynamics_factory()
        _regularizer_list = []
        _regularizer_list.append(partial(CurvatureRegularizer, order=1))
        _regularizer_list.append(partial(LieDerivativeRegularizer,g=g_,normalize=True))
        _regularizer_list.append(partial(LieDerivativeRegularizer,g=g_,normalize="yang"))
        _regularizer_list.append(partial(LieDerivativeRegularizer,g=g_,normalize="new"))
        return []

    @property
    def eps(self):
        return 1e-3

    @staticmethod
    def f_torch(y: torch.Tensor) -> torch.Tensor:
        return torch.stack([torch.sin(y[...,0] ** 2 + y[...,1]), y[...,1] ** 3],dim=-1)

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
            self.assertTrue(torch.any(x.grad != 0), f"Gradient is zero for {reg.__name__}.")

    def test_curvature_regularizer(self):
        f = self.f_torch
        pts = torch.randn(100,2)
        J2 = _compute_jet(f, pts, order=2)
        kappa1 = torch.diff(J2[...,0] * J2[...,1].flip(dims=(-1,)), dim= -1).abs_().squeeze()
        kappa1 /= J2[...,0].norm(dim=-1).pow(3)
        eps = self.eps

        kappa = CurvatureRegularizer(f, order=1).eval_regularizer(pts)

        self.assertTrue(
            torch.allclose(
                kappa, kappa1, atol=eps, rtol=eps, equal_nan=True
            ),
            "1st Curvature doesn't match analytical.",
        )

    def test_torch_jax_reference(self):
        """Compare PyTorch and JAX jets up to order 3."""
        compute_jet_jax, f_jax, z0_jax = self.load_jax_utils()
        z0_torch = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
        eps = self.eps

        self.assertTrue(
            np.allclose(
                self.f_torch(z0_torch).detach().numpy(), np.array(f_jax(z0_jax)), atol=eps, rtol=eps
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
    unittest.main()