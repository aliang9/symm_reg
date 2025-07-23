from random import getstate as python_get_rng_state, setstate as python_set_rng_state
from typing import Any, Dict
from functools import wraps  # noqa

import torch

"""
Inspired by pytorch_lightning.utilities.seed
"""

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False


def _collect_rng_states() -> Dict[str, Any]:
    r"""Collect the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python."""
    states = {"torch": torch.get_rng_state(), "python": python_get_rng_state()}
    if _NUMPY_AVAILABLE:
        import numpy as np
        states["numpy"] = np.random.get_state()
    states["torch.cuda"] = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    )  #
    return states


def _set_rng_states(rng_state_dict: Dict[str, Any]) -> None:
    r"""Set the global random state of :mod:`torch`, :mod:`torch.cuda`, :mod:`numpy` and Python in the current
    process."""
    torch.set_rng_state(rng_state_dict["torch"])
    # torch.cuda rng_state is only included since v1.8.
    if "torch.cuda" in rng_state_dict:
        torch.cuda.set_rng_state_all(rng_state_dict["torch.cuda"])
    if _NUMPY_AVAILABLE and "numpy" in rng_state_dict:
        import numpy as np
        np.random.set_state(rng_state_dict["numpy"])
    version, state, gauss = rng_state_dict["python"]
    python_set_rng_state((version, tuple(state), gauss))


def rng_state_noop(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        states = _collect_rng_states()
        try:
            return f(*args, **kwargs)
        finally:
            _set_rng_states(states)

    return wrapper
