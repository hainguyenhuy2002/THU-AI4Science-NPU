#!/usr/bin/env python3
"""Template for adapting graph-model PyTorch code to Ascend NPU."""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from device_utils import get_training_device, print_accelerator_hint


class SimpleGraphConv(nn.Module):
    """Dense GCN layer: A_norm @ X @ W.

    Real projects can replace this with their graph library layer if that
    library supports torch-npu tensors. The device pattern is the same.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

    def forward(self, x, adj_norm):
        return adj_norm @ x @ self.weight


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, classes: int):
        super().__init__()
        self.conv1 = SimpleGraphConv(in_dim, hidden_dim)
        self.conv2 = SimpleGraphConv(hidden_dim, classes)

    def forward(self, x, adj_norm):
        h = F.relu(self.conv1(x, adj_norm))
        return self.conv2(h, adj_norm)


def make_toy_graph(nodes: int = 512, features: int = 32, classes: int = 3):
    labels = torch.randint(0, classes, (nodes,))
    x = torch.randn(nodes, features) + labels.float().unsqueeze(1) * 0.4
    adj = torch.eye(nodes)
    for i in range(nodes):
        same = torch.where(labels == labels[i])[0]
        neighbors = same[torch.randint(len(same), (4,))]
        adj[i, neighbors] = 1
    adj = torch.maximum(adj, adj.T)
    degree = adj.sum(1)
    d_inv_sqrt = degree.pow(-0.5)
    adj_norm = d_inv_sqrt[:, None] * adj * d_inv_sqrt[None, :]
    return x, adj_norm, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["npu", "cuda", "cpu"], default="npu")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    device = get_training_device(args.device)
    print_accelerator_hint(device)

    x, adj_norm, labels = make_toy_graph()

    # Key migration step:
    # CUDA code usually uses `.to("cuda")`; Ascend code uses `.to("npu")`.
    x = x.to(device)
    adj_norm = adj_norm.to(device)
    labels = labels.to(device)

    model = GCN(in_dim=x.shape[1], hidden_dim=64, classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    for epoch in range(1, args.epochs + 1):
        logits = model(x, adj_norm)
        loss = F.cross_entropy(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 5 == 0:
            acc = (logits.argmax(-1) == labels).float().mean()
            print({"epoch": epoch, "loss": float(loss.detach().cpu()), "acc": float(acc.cpu()), "device": str(device)})


if __name__ == "__main__":
    main()
