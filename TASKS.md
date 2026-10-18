# Tasks: Progress Review 1 (slides only, Week 6)

Goal: show the problem formulation, working baselines and a first prototype. Validation results only.

| Who | Task | Output |
|---|---|---|
| **Sơn** | Repo, shared code and template. G-struct features per relation (degree, PageRank, clustering, k-core). Prototype P1 Fusion-LR (profile + tweet + G-struct). **Done.** | `features.py`, `notebooks/p1.ipynb`, `results/p1.csv` |
| **Đại** | Baselines: B0 majority, B1 content-only LR, B2a graph-structure LR (uses `features.py`), B3 label propagation (train labels only). | `notebooks/baselines.ipynb`, `results/baselines.csv` |
| **Minh** | Slides: formulation, data and relations, metrics, results table (baselines vs P1). Leakage check and one re-run of both notebooks. | slides |

B2a uses `struct_features` from `features.py` (standardize with train statistics, as in `p1.ipynb`).

Not in this round: P2 RGCN, B2b Node2Vec (both need PyTorch Geometric), ablations, fairness check.
Quyền, Phúc and Tuấn are free this round and take these later.
