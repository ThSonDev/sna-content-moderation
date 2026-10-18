"""G-struct features (32 columns): 4 given profile counts + 28 computed per relation.
Uses no labels, so computing it on the whole graph is not leakage. Standardize with train statistics only."""
import os
import networkx as nx
import numpy as np

from mgtab import RELATIONS

PROFILE_COUNTS = {3: "followers", 5: "listed", 7: "friends", 14: "followers_friends_ratio"}  # cols of X, scaled
PROFILE_ONLY = [j for j in range(20) if j not in PROFILE_COUNTS]  # the 16 profile columns of group P
DIRECTED = ["follow", "mention", "reply", "quote"]


def _relation_features(name, src, dst, n):
    G = (nx.DiGraph if name in DIRECTED else nx.Graph)()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    U = G.to_undirected(as_view=True) if G.is_directed() else G
    cols = {}
    if G.is_directed():
        cols["in_degree"] = [G.in_degree(v) for v in range(n)]
        cols["out_degree"] = [G.out_degree(v) for v in range(n)]
    else:
        cols["degree"] = [G.degree(v) for v in range(n)]
    pr, cc, core = nx.pagerank(G), nx.clustering(U), nx.core_number(U)
    cols["pagerank"] = [pr[v] * n for v in range(n)]  # x n: 1.0 = average account
    cols["clustering"] = [cc[v] for v in range(n)]
    cols["kcore"] = [core[v] for v in range(n)]
    return {f"{name}_{k}": np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def struct_features(X, rel, cache="data/g_struct.npz"):
    """Returns (F, names): F is n x 32. Counts (degree, PageRank, k-core) are log1p-transformed because they are
    heavy-tailed. Cached to data/ (gitignored); delete the file to recompute (about 1.5 minutes)."""
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["F"], list(z["names"])
    n = len(X)
    cols = {f"profile_{v}": X[:, j].astype(np.float64) for j, v in PROFILE_COUNTS.items()}
    for r in RELATIONS:
        for k, v in _relation_features(r, *rel[r], n).items():
            cols[k] = v if k.endswith("clustering") else np.log1p(v)
    names = list(cols)
    F = np.column_stack([cols[k] for k in names])
    if cache:
        np.savez(cache, F=F, names=names)
    return F, names


if __name__ == "__main__":  # self-check: shape, no NaN, degrees match the cleaned edge counts
    from mgtab import load, relations
    X, y, _, E, T, _ = load()
    rel = relations(E, T, len(y))
    F, names = struct_features(X, rel, cache=None)
    assert F.shape == (len(y), 32) and not np.isnan(F).any(), F.shape
    assert np.expm1(F[:, names.index("follow_in_degree")]).round().sum() == len(rel["follow"][0])
    assert np.expm1(F[:, names.index("url_degree")]).round().sum() == 2 * len(rel["url"][0])
    print("ok", F.shape, names)
