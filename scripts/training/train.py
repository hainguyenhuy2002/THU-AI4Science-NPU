#!/usr/bin/env python3
"""Train a small GCN on Ascend NPU using only PyTorch ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from exp_setup.graph_data import make_synthetic_graph


def require_npu() -> torch.device:
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        raise RuntimeError("Ascend NPU is not available. Check torch-npu and ASCEND_VISIBLE_DEVICES.")
    return torch.device("npu:0")


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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--nodes", type=int, default=1200)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--classes", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--edges-per-node", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/graph/gcn_metrics.jsonl")
    args = parser.parse_args()
    if args.nodes < 5 or min(args.epochs, args.features, args.classes, args.hidden, args.edges_per_node) < 1:
        parser.error("Use at least five nodes and positive values for epochs and graph dimensions")

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
    out.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, adj)
        loss = F.cross_entropy(logits[train_idx], y[train_idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                logits = model(x, adj)
            row = {
                "epoch": epoch,
                "loss": float(loss.detach().cpu()),
                "val_acc": accuracy(logits, y, val_idx),
                "test_acc": accuracy(logits, y, test_idx),
                "device": str(loss.device),
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(row)

    print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
