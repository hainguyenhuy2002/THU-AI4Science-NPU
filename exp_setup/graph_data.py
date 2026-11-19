"""Generate the small synthetic graph used by the training example."""

import torch


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
        adj[i, same_pick] = 1
        if len(diff):
            diff_pick = diff[torch.randint(len(diff), (max(1, edges_per_node // 3),), generator=generator)]
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
