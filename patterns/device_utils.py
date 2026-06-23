#!/usr/bin/env python3
"""Reusable device helpers for CUDA GPU vs Ascend NPU PyTorch code."""

from __future__ import annotations

import os
import torch


def get_npu_device(index: int = 0) -> torch.device:
    """Return an Ascend NPU device and fail clearly if torch-npu is unavailable."""
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError(
            "Ascend NPU is not available. Check CANN, torch-npu, and "
            "ASCEND_VISIBLE_DEVICES/ASCEND_RT_VISIBLE_DEVICES."
        )
    return torch.device(f"npu:{index}")


def get_training_device(prefer: str = "npu", index: int = 0) -> torch.device:
    """Pick a training device with explicit NPU/CUDA/CPU modes."""
    prefer = prefer.lower()
    if prefer == "npu":
        return get_npu_device(index)
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available.")
        return torch.device(f"cuda:{index}")
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError("prefer must be one of: npu, cuda, cpu")


def move_batch_to_device(batch, device: torch.device):
    """Move tensors in a nested batch to the selected device."""
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(value, device) for value in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(value, device) for value in batch)
    return batch


def print_accelerator_hint(device: torch.device) -> None:
    """Print the matching system monitor command for students."""
    if device.type == "npu":
        visible = os.getenv("ASCEND_VISIBLE_DEVICES", "not set")
        print(f"Using Ascend NPU device: {device}; ASCEND_VISIBLE_DEVICES={visible}")
        print("Verify with: npu-smi info")
    elif device.type == "cuda":
        print(f"Using CUDA GPU device: {device}")
        print("Verify with: nvidia-smi")
    else:
        print("Using CPU")
