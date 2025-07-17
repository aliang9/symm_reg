import unittest

import numpy as np
import torch
from scipy.special import factorial
from torch import Tensor
from torch._functorch.eager_transforms import _jvp_with_argnums

# Helper for JVP in PyTorch
jvp_ = lambda f, p, v: _jvp_with_argnums(f, p, v, argnums=0, strict=True, has_aux=False)[-1]

def _f_torch(y: torch.Tensor) -> torch.Tensor:
	return torch.stack([torch.sin(y[0] ** 2 + y[1]), y[1] ** 3])

def compute_jet_torch(f, u: Tensor, order: int) -> Tensor:
	"""
	Compute jets of f at u up to the given order using PyTorch.
	Returns a tensor stack of jets (excluding the 0th jet).
	Supports up to order 4.
	"""
	if order > 4:
		raise NotImplementedError("Jets higher than order 4 are not implemented.")

	f1 = lambda p, v: jvp_(f, (p,), (v,))
	f2 = lambda p, v, w: jvp_(f1, (p, v), (w,))
	f3 = lambda p, v, w, z: jvp_(f2, (p, v, w), (z,))

	u1 = f(u)
	jets = list()

	if order >= 2:
		u2 = f1(u, u1)
		jets.append(u2)
	if order >= 3:
		u3 = f1(u, u2) + f2(u, u1, u1)
		jets.append(u3)
	if order >= 4:
		u4 = f1(u, u3) + 3 * f2(u, u1, u2) + f3(u, u1, u1, u1)
		jets.append(u4)

	return torch.stack(jets)



class TestJets(unittest.TestCase):

	@staticmethod
	def load_jax_utils():
		try:
			import jax
			import jax.numpy as jnp
			from jax.experimental.jet import jet
		except ImportError:
			raise unittest.SkipTest("JAX not available")
		else:
			def compute_jet_jax(f, z: jnp.ndarray, order: int):
				"""
				Compute jets using JAX's jet up to the given order.
				"""
				_fix = lambda nth: factorial(nth, exact=True) / jax.lax.exp(
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


	def test_torch_jax_reference(self):
		"""Compare PyTorch and JAX jets up to order 3."""
		compute_jet_jax, f_jax, z0_jax = self.load_jax_utils()

		jets_jax = compute_jet_jax(f_jax, z0_jax, order=3)

		eval_point = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
		jets_torch = compute_jet_torch(_f_torch, eval_point, order=3)

		eps = 1e-4
		self.assertTrue(
			np.allclose(jets_torch.detach().numpy(), np.array(jets_jax), atol=eps, rtol=1e-4),
			"Torch and JAX jets differ."
		)

	def test_autodiff_compatibility(self):
		"""Ensure compute_jet_torch supports backprop."""
		u0_torch = torch.tensor([1.0, 2.0], dtype=torch.float32, requires_grad=True)
		jets_torch = compute_jet_torch(_f_torch, u0_torch, order=4)

		# Backprop through norm
		jets_torch.norm().backward()
		self.assertIsNotNone(u0_torch.grad, "Gradient is None.")
		self.assertTrue(torch.any(u0_torch.grad != 0), "Gradient is zero.")

	def test_batched_vmap(self):
		"""Test batched computation using torch.vmap."""
		u0_batch = torch.randn(3, 2)
		batched_jets = torch.vmap(lambda u_: compute_jet_torch(_f_torch, u_, order=4))(u0_batch)

		self.assertEqual(batched_jets.shape[0], u0_batch.shape[0], "Batching failed.")


if __name__ == "__main__":
	unittest.main()