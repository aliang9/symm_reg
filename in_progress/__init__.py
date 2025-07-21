from typing import Union, Callable

from torch import nn, Tensor

VectorField = Union[nn.Module, Callable[[Tensor], Tensor]]
