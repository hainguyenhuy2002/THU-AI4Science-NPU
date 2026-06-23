#!/usr/bin/env python3
"""Side-by-side minimal training code: CUDA GPU vs Ascend NPU.

Run on Ascend:
  ASCEND_VISIBLE_DEVICES=0 python patterns/cuda_to_npu_minimal.py --device npu

Run on CUDA:
  CUDA_VISIBLE_DEVICES=0 python patterns/cuda_to_npu_minimal.py --device cuda

The training logic is intentionally ordinary PyTorch. The important difference is
the device setup and importing torch_npu before using torch.device("npu:0").
"""

from __future__ import annotations

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import get_training_device, move_batch_to_device, print_accelerator_hint


def cuda_style_reference():
    """Reference only: this is what many students already write for GPU."""
    device = torch.device("cuda:0")
    model = nn.Linear(32, 2).to(device)
    x = torch.randn(8, 32).to(device)
    y = torch.randint(0, 2, (8,)).to(device)
    loss = F.cross_entropy(model(x), y)
    loss.backward()


def npu_style_reference():
    """Reference only: this is the equivalent code for Ascend NPU."""
    import torch_npu  # noqa: F401

    device = torch.device("npu:0")
    model = nn.Linear(32, 2).to(device)
    x = torch.randn(8, 32).to(device)
    y = torch.randint(0, 2, (8,)).to(device)
    loss = F.cross_entropy(model(x), y)
    loss.backward()


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hold-seconds", type=int, default=0, help="Keep process alive for npu-smi/nvidia-smi inspection.")
    args = parser.parse_args()

    device = get_training_device(args.device)
    print_accelerator_hint(device)

    model = TinyClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(1, args.steps + 1):
        batch = {
            "x": torch.randn(128, 32),
            "y": torch.randint(0, 2, (128,)),
        }
        batch = move_batch_to_device(batch, device)

        logits = model(batch["x"])
        loss = F.cross_entropy(logits, batch["y"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 5 == 0:
            print({"step": step, "loss": float(loss.detach().cpu()), "device": str(device)})

    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    if args.hold_seconds > 0:
        print(f"Holding process for {args.hold_seconds}s so system monitor can see device usage.")
        time.sleep(args.hold_seconds)


if __name__ == "__main__":
    main()
