"""Shared MGTAB helpers: load, clean relations, fixed split, evaluate. Import this in every notebook."""
import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

DATA = "data/MGTAB"
RELATIONS = ["follow", "mention", "reply", "quote", "url", "hashtag"]
SEED = 42


def load(data=DATA):
    """Returns X (10199 x 788), y_bot, y_stance and the raw edges (edge_index, edge_type, edge_weight).
    X: cols 0-19 profile (already scaled by the authors), cols 20-787 mean LaBSE tweet embedding (not scaled)."""
    t = {n: torch.load(f"{data}/{n}.pt", weights_only=True).numpy() for n in
         ["features", "labels_bot", "labels_stance", "edge_index", "edge_type", "edge_weight"]}
    return t["features"], t["labels_bot"], t["labels_stance"], t["edge_index"], t["edge_type"], t["edge_weight"]


def relations(edge_index, edge_type, n):
    """Clean edges per relation: no self-loops, no duplicates. Follower and friend are merged into one
    follow relation (A follows B). Returns {name: (src, dst)}; url and hashtag are undirected (stored once)."""
    s, d = edge_index
    keys = {
        "follow": np.r_[s[edge_type == 1] * n + d[edge_type == 1], d[edge_type == 0] * n + s[edge_type == 0]],
        **{r: s[edge_type == k] * n + d[edge_type == k] for k, r in enumerate(RELATIONS[1:], start=2)},
    }
    out = {}
    for r, k in keys.items():
        k = np.unique(k[k // n != k % n])
        out[r] = (k // n, k % n)
    return out


def split(y, seed=SEED):
    """Stratified 70/15/15 split on the bot label. Everyone uses this one (seed 42)."""
    idx = np.arange(len(y))
    train, rest = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return train, val, test


def best_threshold(y_val, score_val):
    """Threshold that maximizes macro-F1 on the validation set."""
    cands = np.unique(score_val)
    f1s = [f1_score(y_val, score_val >= c, average="macro") for c in cands]
    return float(cands[int(np.argmax(f1s))])


def evaluate(y, score, threshold):
    """score = bot probability. Threshold must come from the validation set (best_threshold)."""
    order = np.argsort(-score)
    return {"pr_auc": average_precision_score(y, score), "roc_auc": roc_auc_score(y, score),
            "macro_f1": f1_score(y, score >= threshold, average="macro"),
            "p@100": float(y[order[:100]].mean()), "p@200": float(y[order[:200]].mean()),
            "threshold": float(threshold)}  # logged so the final test run reuses the same value


if __name__ == "__main__":  # self-check against the numbers in the proposal
    X, yb, ys, E, T, W = load()
    rel = relations(E, T, len(yb))
    assert X.shape == (10199, 788) and yb.sum() == 2748
    assert [len(rel[r][0]) for r in RELATIONS[:4]] == [425863, 110868, 22327, 20901]
    tr, va, te = split(yb)
    assert (len(tr), len(va), len(te)) == (7139, 1530, 1530) and yb[te].sum() == 412
    print("ok", {r: len(v[0]) for r, v in rel.items()})
