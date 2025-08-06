from typing import Callable, Dict, Tuple, Union

import torch
from torch import Tensor, nn as nn
from torch._functorch.eager_transforms import (  # noqa
    _jvp_with_argnums as jvp,
    _vjp_with_argnums as vjp,
)


def close_over_autogradfunc(
    diffeom: nn.Module,
) -> Tuple[Callable[[Tensor, Tensor], Tensor], Callable[[nn.Module], Tensor]]:
    flatten, unflatten = make_param_flattener(diffeom)
    if hasattr(diffeom, "named_parameters"):
        phi_inverse = lambda y, theta: torch.func.functional_call(
            diffeom, unflatten(theta), y, kwargs={"inverse": True}
        )
    else:
        phi_inverse = lambda x, theta: diffeom(x, kwargs={"inverse": True})

    class ConjugateVF(torch.autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def setup_context(ctx, inputs, output):
            out, vjp_fn = output
            y, theta = inputs
            ctx.vjp = vjp_fn
            # ctx.mark_non_differentiable(vjp_fn)
            ctx.save_for_backward(y, out, theta)
            ctx.save_for_forward(y, out, theta)
            ctx.set_materialize_grads(False)

        @staticmethod
        def forward(y, theta):
            out, vjp_fn = vjp(phi_inverse, y, theta, argnums=1, has_aux=False)
            return out, vjp_fn

        @staticmethod
        def backward(ctx, *gradients):
            dout = gradients[0]
            if dout is None:
                dtheta = None
            elif dout is not None and not torch.any(dout):
                dtheta = torch.zeros_like(ctx.saved_tensors[-1])
            else:
                dtheta = ctx.vjp(dout) if dout is not None and torch.any(dout) else dout

            return dout, dtheta

        @staticmethod
        def jvp(ctx, *tangents):
            y, out, theta = ctx.saved_tensors
            v_y: Union[Tensor, None] = tangents[0]
            v_theta: Union[Tensor, None] = tangents[1]
            if v_theta is not None and torch.any(v_theta):
                jv_theta = (
                    jvp(
                        phi_inverse, (y, theta), (v_theta,), argnums=(1,), has_aux=False
                    )
                )[-1]
            else:
                jv_theta = v_theta
            return v_y, jv_theta

    return ConjugateVF.apply, flatten


def make_param_flattener(obj: nn.Module):
    if not hasattr(obj, "named_parameters"):
        return lambda _: None, lambda _: {}

    param_dict = dict(obj.named_parameters())

    idx_map = {}
    idx = 0
    for name, p in param_dict.items():
        numel = p.numel()
        idx_map[name] = (idx, idx + numel, p.shape)
        idx += numel

    def flatten(obj_) -> Tensor:
        param_dict: Dict[str, Tensor] = dict(obj_.named_parameters())
        return torch.cat([p.view(-1) for p in param_dict.values()])

    def unflatten(flat_tensor: Tensor) -> Dict[str, Tensor]:
        out = {}
        for name, (start, end, shape) in idx_map.items():
            out[name] = flat_tensor[start:end].view(shape)
        return out

    return flatten, unflatten
