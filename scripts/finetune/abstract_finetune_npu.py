#!/usr/bin/env python3
"""Abstract finetuning implementation: GPU vs Ascend NPU."""

from __future__ import annotations

import torch
import torch.nn as nn

# =========================
# GPU version, for reference
# =========================
# device = torch.device("cuda:0")
# model = model.to(device)
# batch = {k: v.to(device) for k, v in batch.items()}
# Verify with: nvidia-smi

# =========================
# NPU version
# =========================
import torch_npu  # noqa: F401


def get_npu_device() -> torch.device:
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. Check CANN, torch-npu, and ASCEND_VISIBLE_DEVICES.")
    return torch.device("npu:0")


def move_batch_to_npu(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def finetune_one_step(model: nn.Module, batch: dict[str, torch.Tensor], optimizer: torch.optim.Optimizer):
    device = get_npu_device()
    model = model.to(device)
    batch = move_batch_to_npu(batch, device)

    outputs = model(**batch)
    loss = outputs.loss if hasattr(outputs, "loss") else outputs
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()


# Run command for real implementation:
# ASCEND_VISIBLE_DEVICES=0 MAX_STEPS=800 bash scripts/finetune/run_lora_finetune.sh
# Verify with: npu-smi info
