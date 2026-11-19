#!/usr/bin/env python3
"""Train a small GCN on Ascend NPU using only PyTorch ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from exp_setup.distributed import npu_run
from exp_setup.graph_data import make_synthetic_graph


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

    with npu_run() as (device, rank, world_size):
        train(args, device, rank, world_size)


def train(args, device, rank, world_size):
    x, adj, y, train_idx, val_idx, test_idx = make_synthetic_graph(
        args.nodes, args.features, args.classes, args.edges_per_node, args.seed
    )
    train_count = train_idx.numel()
    if train_count < world_size:
        raise ValueError("The graph needs at least one training node per rank.")
    x, adj, y = x.to(device), adj.to(device), y.to(device)
    local_idx = train_idx[rank::world_size].to(device)
    if rank == 0:
        val_idx, test_idx = val_idx.to(device), test_idx.to(device)

    gcn = GCN(args.features, args.hidden, args.classes, dropout=0.35).to(device)
    model = DistributedDataParallel(gcn, device_ids=[device.index], broadcast_buffers=False) if world_size > 1 else gcn
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)

    out = Path(args.output)
    if rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("", encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        model.train()
        logits = model(x, adj)
        # DDP averages rank gradients. Weight by the global node count so
        # uneven local splits still give each training node the same weight.
        loss = F.cross_entropy(logits[local_idx], y[local_idx], reduction="sum") * (world_size / train_count)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            mean_loss = loss.detach() / world_size
            if world_size > 1:
                dist.all_reduce(mean_loss)
            if rank != 0:
                continue
            gcn.eval()
            with torch.no_grad():
                logits = gcn(x, adj)
            row = {
                "epoch": epoch,
                "loss": mean_loss.item(),
                "val_acc": accuracy(logits, y, val_idx),
                "test_acc": accuracy(logits, y, test_idx),
                "device": str(loss.device),
                "world_size": world_size,
            }
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(row)

    if rank == 0:
        print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
