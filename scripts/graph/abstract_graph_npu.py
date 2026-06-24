#!/usr/bin/env python3
"""Abstract graph-model implementation: GPU vs Ascend NPU."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# GPU version, for reference
# =========================
# device = torch.device("cuda:0")
# x = x.cuda()
# adj_norm = adj_norm.cuda()
# labels = labels.cuda()
# model = model.cuda()
# Verify with: nvidia-smi

# =========================
# NPU version
# =========================
import torch_npu  # noqa: F401


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x, adj_norm):
        h = F.relu(adj_norm @ self.fc1(x))
        return adj_norm @ self.fc2(h)


def train_graph_one_step(x, adj_norm, labels, model, optimizer):
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. Check CANN, torch-npu, and ASCEND_VISIBLE_DEVICES.")

    device = torch.device("npu:0")
    x = x.to(device)
    adj_norm = adj_norm.to(device)
    labels = labels.to(device)
    model = model.to(device)

    logits = model(x, adj_norm)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()


# Run command for real implementation:
# ASCEND_VISIBLE_DEVICES=0 EPOCHS=200 bash scripts/graph/run_gcn_training.sh
# Verify with: npu-smi info
