import inspect
import os
from copy import deepcopy
from typing import Any, Dict

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from torch import nn


class StoreObjectsInBestCheckpoint(Callback):
    """Adds extra objects (modules, callables, etc.) to the best checkpoint after training."""

    def __init__(self, extra_objects: Dict[str, Any]):
        super().__init__()
        self.extra_objects = extra_objects
        self._executed = False  # ensures this runs only once

    def _serialize_object(self, obj: Any) -> Any:
        """Convert object to old serializable form for checkpoint storage."""
        # Case 1: Torch Module → store its state_dict
        if isinstance(obj, torch.nn.Module):
            return {"type": "nn.Module", "state_dict": obj.state_dict()}

        # Case 2: Callable → capture parameters and signature
        if callable(obj):
            return self._serialize_callable(obj)

        # Case 3: Other objects → fallback to deepcopy
        try:
            return deepcopy(obj)
        except Exception as e:
            rank_zero_warn(f"Could not serialize object {obj}: {e}")
            raise

    @staticmethod
    def _serialize_callable(func: Any) -> Dict[str, Any]:
        """Serialize old callable that may depend on PyTorch tensors or modules."""
        try:
            sig = str(inspect.signature(func))
        except (TypeError, ValueError):
            sig = "unknown"

        captured_params = {}
        if getattr(func, "__closure__", None):
            for cell in func.__closure__:
                val = cell.cell_contents
                if isinstance(val, torch.nn.Module):
                    captured_params.update(dict(val.named_parameters()))

        return {
            "type": "callable",
            "name": getattr(func, "__name__", repr(func)),
            "signature": sig,
            "params": {k: v.detach().cpu() for k, v in captured_params.items()},
        }

    def _update_checkpoint(self, ckpt_path: str):
        """Load, update, and save the checkpoint file."""
        if not os.path.isfile(ckpt_path):
            rank_zero_warn(f"Best checkpoint not found at {ckpt_path}")
            return

        ckpt = torch.load(ckpt_path, map_location="cpu")
        extra = ckpt.setdefault("extra_objects", {})

        for key, obj in self.extra_objects.items():
            extra[key] = self._serialize_object(obj)

        torch.save(ckpt, ckpt_path)
        rank_zero_info(f"Appended extra_objects to checkpoint at {ckpt_path}")

    def _maybe_run(self, trainer):
        if self._executed:
            return
        ckpt_cb = next(
            (
                cb
                for cb in trainer.checkpoint_callbacks
                if isinstance(cb, ModelCheckpoint)
            ),
            None,
        )
        if ckpt_cb is None or not getattr(ckpt_cb, "best_model_path", None):
            rank_zero_warn("No ModelCheckpoint with old valid best_model_path found.")
            return

        self._update_checkpoint(ckpt_cb.best_model_path)
        self._executed = True

    def on_fit_end(self, trainer, pl_module):
        self._maybe_run(trainer)

    def on_train_end(self, trainer, pl_module):
        self._maybe_run(trainer)


def find_model_device(model: nn.Module) -> str:
    # Guess primary device by checking the first parameter or buffer
    for p in model.parameters():
        return p.device.type
    for b in model.buffers():
        return b.device.type
    return "cpu"


def check_model_device_consistency(model: nn.Module) -> list[str]:
    issues = []
    devices = set()
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        devices.add(tensor.device.type)
    if len(devices) > 1:
        issues.append(f"Model has tensors on multiple devices: {devices}")
    return issues


def get_tensor_device_info(tensor: torch.Tensor) -> str:
    return f"device={tensor.device}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}"


def check_model_for_cpu(model: nn.Module, expected_device: str) -> list[str]:
    if expected_device == "cpu":
        return []
    cpu_tensors = []
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.device.type == "cpu":
            cpu_tensors.append(f"{name}: {get_tensor_device_info(tensor)}")
    if cpu_tensors:
        return ["Some ode parameters or buffers are on CPU:"] + cpu_tensors
    return []


def check_batch_device(batch: Any, expected_device: str) -> list[str]:
    issues = []

    def _check(x, path="batch"):
        if torch.is_tensor(x):
            if x.device.type != expected_device:
                issues.append(f"{path}: {get_tensor_device_info(x)}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                _check(v, f"{path}[{i}]")
        elif isinstance(x, dict):
            for k, v in x.items():
                _check(v, f"{path}['{k}']")

    _check(batch)
    return issues


class DeviceConsistencyCallback(Callback):
    """Checks that ode parameters and data batches are on the expected device."""

    def _log_issues(self, trainer: Trainer, issues: list[str], prefix: str):
        if issues:
            msg = f"[Device Check][{prefix}] " + "\n".join(issues)
            trainer.logger.warning(msg) if trainer.logger.warn else print(msg)

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule):
        expected_device = find_model_device(pl_module)
        issues = check_model_device_consistency(pl_module)
        issues += check_model_for_cpu(pl_module, expected_device)
        self._log_issues(trainer, issues, "on_fit_start")

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule):
        expected_device = find_model_device(pl_module)
        issues = check_model_device_consistency(pl_module)
        issues += check_model_for_cpu(pl_module, expected_device)
        self._log_issues(trainer, issues, "on_train_start")

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule):
        expected_device = find_model_device(pl_module)
        issues = check_model_device_consistency(pl_module)
        issues += check_model_for_cpu(pl_module, expected_device)
        self._log_issues(trainer, issues, "on_test_start")

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check_batch(trainer, pl_module, batch, batch_idx, dataloader_idx, "train")

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check_batch(
            trainer, pl_module, batch, batch_idx, dataloader_idx, "validation"
        )

    def on_test_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check_batch(trainer, pl_module, batch, batch_idx, dataloader_idx, "test")

    def on_predict_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._check_batch(
            trainer, pl_module, batch, batch_idx, dataloader_idx, "predict"
        )

    def _check_batch(self, trainer, pl_module, batch, batch_idx, dataloader_idx, stage):
        expected_device = find_model_device(pl_module)
        issues = check_batch_device(batch, expected_device)
        if issues:
            msg = (
                f"Batch-device mismatch detected in {stage} stage.\n"
                f"batch_idx={batch_idx}, dataloader_idx={dataloader_idx}\n"
                + "\n".join(issues)
            )
            self._log_issues(trainer, [msg], f"{stage}_batch_start")
