"""Node2Vec helpers for MGTAB.

Uses PyG Node2Vec with a custom Numba biased-walk sampler.
This is necessary because the current pyg-lib backend only
supports uniform random walks.
"""

import numpy as np
import torch
from numba import njit
from torch_geometric.nn import Node2Vec


@njit
def _has_edge(rowptr, col, u, v):
    left = rowptr[u]
    right = rowptr[u + 1]

    while left < right:
        mid = (left + right) // 2

        if col[mid] < v:
            left = mid + 1
        else:
            right = mid

    return left < rowptr[u + 1] and col[left] == v


@njit
def biased_walks(rowptr, col, starts, length, p, q):
    walks = np.empty(
        (len(starts), length + 1),
        dtype=np.int64
    )

    for i in range(len(starts)):
        current = starts[i]
        previous = -1
        walks[i, 0] = current

        for step in range(length):
            begin = rowptr[current]
            end = rowptr[current + 1]

            if begin == end:
                next_node = current

            elif previous == -1:
                j = np.random.randint(begin, end)
                next_node = col[j]

            else:
                total = 0.0

                for j in range(begin, end):
                    candidate = col[j]

                    if candidate == previous:
                        weight = 1.0 / p
                    elif _has_edge(
                        rowptr, col, previous, candidate
                    ):
                        weight = 1.0
                    else:
                        weight = 1.0 / q

                    total += weight

                target = np.random.random() * total
                cumulative = 0.0
                next_node = col[end - 1]

                for j in range(begin, end):
                    candidate = col[j]

                    if candidate == previous:
                        weight = 1.0 / p
                    elif _has_edge(
                        rowptr, col, previous, candidate
                    ):
                        weight = 1.0
                    else:
                        weight = 1.0 / q

                    cumulative += weight

                    if target < cumulative:
                        next_node = candidate
                        break

            walks[i, step + 1] = next_node
            previous = current
            current = next_node

    return walks


@njit
def _seed_walks(seed):
    np.random.seed(seed)


def enable_biased_walks(model, seed=42):
    """Attach the custom biased sampler to PyG Node2Vec.

    Keep num_workers=0 so the sampler uses the same process.
    """
    rowptr = np.asarray(
        model.rowptr.cpu().numpy(),
        dtype=np.int64
    )
    col = np.asarray(
        model.col.cpu().numpy(),
        dtype=np.int64
    )

    _seed_walks(seed)

    def sampler(_rowptr, _col, batch, length, p, q):
        starts = batch.cpu().numpy().astype(
            np.int64, copy=False
        )

        rw = biased_walks(
            rowptr, col, starts, length, p, q
        )

        return torch.from_numpy(rw)

    model.random_walk_fn = sampler
    return model


def merged_edge_index(rel, num_nodes):
    """Merge all MGTAB relations into one undirected graph."""
    rows = []
    cols = []

    for src, dst in rel.values():
        rows.extend([src, dst])
        cols.extend([dst, src])

    src = np.concatenate(rows).astype(np.int64)
    dst = np.concatenate(cols).astype(np.int64)

    # Remove self-loops and duplicate directed edges.
    valid = src != dst
    keys = np.unique(
        src[valid] * num_nodes + dst[valid]
    )

    edges = np.stack([
        keys // num_nodes,
        keys % num_nodes
    ])

    return torch.from_numpy(edges).long()


def make_node2vec(
    edge_index,
    num_nodes,
    p=1.0,
    q=1.0,
    seed=42,
):
    """Create a 128-dimensional Node2Vec model."""
    if p <= 0 or q <= 0:
        raise ValueError("p and q must be positive")
    torch.manual_seed(seed)

    model = Node2Vec(
        edge_index=edge_index,
        embedding_dim=128,
        walk_length=80,
        context_size=10,
        walks_per_node=10,
        p=p,
        q=q,
        num_nodes=num_nodes,
        sparse=True,
    )

    return enable_biased_walks(model, seed=seed)


import os
import time


def train_node2vec(
    edge_index,
    num_nodes,
    p,
    q,
    output_dir,
    epochs=5,
    batch_size=256,
    lr=0.01,
    seed=42,
):
    """Train PyG Node2Vec with Numba biased walks.

    Saves checkpoints after each epoch; resume restores weights and optimizer,
    but not all random states, so interruption is not bitwise reproducible.
    Returns embeddings for all nodes.
    """
    if epochs < 1 or batch_size < 1 or lr <= 0:
        raise ValueError("epochs, batch_size, and lr must be positive")
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = make_node2vec(
        edge_index=edge_index,
        num_nodes=num_nodes,
        p=p,
        q=q,
        seed=seed,
    ).to(device)

    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=lr
    )

    tag = f"p{p}_q{q}"
    checkpoint = os.path.join(
        output_dir, f"node2vec_{tag}.pt"
    )

    start_epoch = 0
    history = []

    if os.path.exists(checkpoint):
        state = torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
        if (float(state["p"]), float(state["q"])) != (float(p), float(q)):
            raise ValueError("Checkpoint p/q mismatch")
        if state.get("batch_size") != batch_size or state.get("lr") != lr:
            raise ValueError("Checkpoint batch_size/lr mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"]
        history = state["history"]

        print(f"Resuming {tag} from epoch {start_epoch}")

    loader = model.loader(
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    for epoch in range(start_epoch, epochs):
        model.train()
        losses = []
        start = time.perf_counter()

        for pos_rw, neg_rw in loader:
            pos_rw = pos_rw.to(device)
            neg_rw = neg_rw.to(device)

            optimizer.zero_grad()
            loss = model.loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()

            losses.append(float(loss.item()))

        if device.type == "cuda":
            torch.cuda.synchronize()

        mean_loss = float(np.mean(losses))
        elapsed = time.perf_counter() - start

        history.append({
            "epoch": epoch + 1,
            "loss": mean_loss,
            "seconds": elapsed,
        })

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "history": history,
            "p": p,
            "q": q,
            "seed": seed,
            "batch_size": batch_size,
            "lr": lr,
        }, checkpoint)

        print(
            f"{tag} | epoch {epoch+1}/{epochs} | "
            f"loss={mean_loss:.4f} | "
            f"time={elapsed:.1f}s"
        )

    model.eval()
    with torch.no_grad():
        embeddings = model().detach().cpu().numpy()

    assert embeddings.shape == (num_nodes, 128)
    assert np.isfinite(embeddings).all()

    np.save(
        os.path.join(output_dir, f"node2vec_{tag}.npy"),
        embeddings,
    )

    return embeddings, history
