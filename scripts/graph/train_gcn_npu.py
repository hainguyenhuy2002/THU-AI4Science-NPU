#!/usr/bin/env python3
"""Train a small GCN on Ascend NPU using only PyTorch ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def require_npu() -> torch.device:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available. Check torch-npu and ASCEND_VISIBLE_DEVICES.")
    return torch.device("npu:0")


def make_synthetic_graph(nodes: int, features: int, classes: int, edges_per_node: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(classes, (nodes,), generator=generator)
    centers = torch.randn(classes, features, generator=generator)
    x = centers[labels] + 0.35 * torch.randn(nodes, features, generator=generator)

    adj = torch.eye(nodes)
    for i in range(nodes):
        same = torch.where(labels == labels[i])[0]
        diff = torch.where(labels != labels[i])[0]
        same_pick = same[torch.randint(len(same), (edges_per_node,), generator=generator)]
        diff_pick = diff[torch.randint(len(diff), (max(1, edges_per_node // 3),), generator=generator)]
        adj[i, same_pick] = 1
        adj[i, diff_pick] = 1
    adj = torch.maximum(adj, adj.T)
    deg = adj.sum(dim=1)
    norm = deg.pow(-0.5)
    adj_norm = norm[:, None] * adj * norm[None, :]

    idx = torch.randperm(nodes, generator=generator)
    train = idx[: int(0.6 * nodes)]
    val = idx[int(0.6 * nodes): int(0.8 * nodes)]
    test = idx[int(0.8 * nodes):]
    return x, adj_norm, labels, train, val, test


class GCN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = dropout

    def forward(self, x, adj):
        h = adj @ self.fc1(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        return adj @ self.fc2(h)


@torch.no_grad()
def accuracy(logits, labels, index) -> float:
    return float((logits[index].argmax(dim=-1) == labels[index]).float().mean().cpu())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=2500)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--edges-per-node", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/graph/gcn_metrics.jsonl")
    args = parser.parse_args()

    device = require_npu()
    x, adj, y, train_idx, val_idx, test_idx = make_synthetic_graph(
        args.nodes, args.features, args.classes, args.edges_per_node, args.seed
    )
    x, adj, y = x.to(device), adj.to(device), y.to(device)
    train_idx, val_idx, test_idx = train_idx.to(device), val_idx.to(device), test_idx.to(device)

    model = GCN(args.features, args.hidden, args.classes, dropout=0.35).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            logits = model(x, adj)
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "val_acc": accuracy(logits, y, val_idx),
                "test_acc": accuracy(logits, y, test_idx),
                "device": str(device),
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(row)

    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
