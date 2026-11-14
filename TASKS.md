# Tasks: Progress Review 2 (Week 10)

Goal: act on the professor's feedback from Progress Review 1. Compare models fairly, measure what the graph really
adds, and find where we stand against published MGTAB methods. Validation results only.

**Why this round matters:** our P1 looked much better than the B1 baseline, but P1 had 4 profile counts (followers,
listed, friends, followers/friends ratio) that B1 did not. So part of the gain came from those counts, not from the
graph. A quick check on validation gave these PR-AUC values: P + T 0.823, adding the counts 0.878, adding the computed
graph features 0.915.

**Order:** environment first. Finish the error analysis and relation ablation **before** starting the RGCN, so they can
guide its design.

| Who | Task | Depends on | Output | Status |
|---|---|---|---|---|
| **Sơn** | **1. PyTorch environment for the team.** PyTorch Geometric and pyg-lib (Node2Vec needs it) in the setup. Tested locally; the Colab test is still to do. | None | `requirements.txt`, `install_pyg.py`, `notebooks/template.ipynb`, `README.md` | Done |
| | **2. Shared code.** `feature_groups` gives P, C (4 profile counts), T and G (28 computed graph features) separately. `evaluate` also returns accuracy. | None | `features.py`, `mgtab.py` | Done |
| | **3. Models.** Add Node2Vec (G-emb) to P1. | Đại (Node2Vec code) | `notebooks/p1.ipynb`, `results/`, etc. | |
| | **4. Models.** P2 RGCN and its MLP control (A3). | Minh (ablation, error analysis) | `notebooks/p2.ipynb`, `results/`, etc. | |
| **Đại** | **Finish all baselines:** fair baseline, graph-only without counts, B2b Node2Vec, B1 MLP, new model names. | Sơn (task 1) | `embeddings.py`, `notebooks/baselines.ipynb`, `results/baselines.csv`, etc. | Done |
| **Minh** | **Relation ablation (LR):** each of the 6 relations alone and left out. | None | `notebooks/relations.ipynb`, `results/relations.csv`, etc. | |
| | **Error analysis:** content-only vs. graph-only vs. fusion. | Đại (fair baseline settings) | `notebooks/errors.ipynb`, short findings for the slides, etc. | |
| **Phúc** | **Reproduce published methods:** official RGT, then BotHP, then the best one on our split. | None (own Colab setup) | `notebooks/reproduce.ipynb`, `results/reproduce.csv`, related notebooks and run logs, etc. | Done |
| **Tuấn** | **Literature table:** check published MGTAB results in the source papers. | None | `notes/sota-mgtab.md`, etc. | |
| | **Slides for Progress Review 2**, with every number verified. | Đại, Minh, Phúc, Sơn, Tuấn (literature table) | slides | |
| **Quyền** | **Document:** update the midterm review. | Đại, Minh, Phúc, Sơn, Tuấn (literature table) | `midterm-review-2.docx` | |

**Everyone who codes:** copy `notebooks/template.ipynb` to your output notebook, run the setup cells, and write your
code below them. See `notebooks/p1.ipynb` for a working example.

**Push everything your results depend on**, not only the files listed: extra notebooks, run logs, small result
files, and notes on anything unexpected. Keep cell outputs so others can check the numbers without rerunning.
Never push the MGTAB data. Write notes in the repo in English.

## Đại: finish all baselines

Context: B1 and B2a are not fair or clearly named today. "Content-only" B1 includes profile features but not the 4
counts, and "graph-only" B2a includes the counts. The professor asked for the counts in the baseline first.

1. Load the groups: `groups, g_names = feature_groups(X, rel)`. Standardize P, C and G with train statistics only.
2. **Fair baseline:** P + C + T with LR. Your result should be near PR-AUC 0.878.
3. **Graph-only without counts:** G alone with LR. Keep the old B2a as "G + counts" so the effect is visible.
4. **B2b:** Node2Vec on the merged graph, then LR. Settings: 128 dimensions, 10 walks of length 80, window 10,
   p and q in {0.5, 1, 2}, chosen on validation PR-AUC. Use `torch_geometric.nn.Node2Vec`.
   Put this code in `embeddings.py`, because Sơn reuses it for P1.
5. **B1 MLP:** same inputs as the fair baseline, 2 hidden layers, class weights. Grid: hidden {64, 128}, dropout
   {0.3, 0.5}, learning rate {0.005, 0.01}. Stop early on validation PR-AUC (up to 200 epochs).
6. **LR settings:** `class_weight="balanced"`, C in {0.01, 0.1, 1, 10} chosen on validation PR-AUC, threshold from
   `best_threshold`. This is `fusion_lr` in `p1.ipynb`.
7. Rerun P1 (P + C + T + G) and report its gain over the fair baseline. Give every model a name that says its features.

## Minh: relation ablation and error analysis

Context: the professor wants to know whether follow, reply, URL or hashtag contributes most, and why fusion wins or
fails. Sơn uses your findings to design the RGCN. This is the LR version; the RGCN version (A5) comes later.

**Relation ablation**

1. Each column in `g_names` starts with its relation, for example `follow_in_degree` or `url_degree`.
2. Start from P + C + T + G with the `fusion_lr` settings. Remove one relation's columns at a time (6 runs).
3. Then test P + C + T plus one relation at a time (6 runs). Use all 6 relations, including mention and quote.
4. Report PR-AUC, macro-F1, P@100 and P@200 for each run.

**Error analysis (validation accounts only)**

1. Train content-only (P + C + T), graph-only (G) and fusion (P + C + T + G). Use each model's own threshold.
2. Build two groups: fusion right and content-only wrong (network signal helped), and fusion wrong and content-only
   right (network noise hurt).
3. For each group: count bots and humans, and compare degrees per relation with the other accounts.
4. To see which relation drives a change, sum (LR coefficient × standardized value) per relation for each account.
5. Write 3–5 short findings for the RGCN design. Do not look at the test set.

## Phúc: reproduce published methods

Context: we have no strong published method to compare with. The MGTAB paper reports RGT at 92.1 accuracy and
90.4 macro-F1 (random 7:2:1 split, 5 seeds, follower + friend relations).

1. **Official RGT first.** Use `RGT.py` from https://github.com/GraphDetec/MGTAB unchanged, following its README.
   Results near 92.1 / 90.4 mean the setup matches the paper. Record the numbers and any differences.
2. **BotHP (KDD 2025):** https://github.com/Peien429/BotHP has an MGTAB script. Run the RGCN variant first, then RGT.
   It pins old packages (torch 1.13), so use a separate Colab; this install has not been tested yet.
   Check before you trust its numbers: it reports binary F1, and `dataset/raw/MGTAB/process.py` sets `test_mask=val_mask`.
3. **If BotHP will not install after a fair try, tell Sơn** and use `SHGN.py` from the official MGTAB repo instead.
4. **Run the best method on our split:** `split()` from `mgtab.py`, all 6 relations from `relations()`. Report
   `evaluate()` on validation (macro-F1, PR-AUC, accuracy).

Steps 1–2 use other random splits, which include our test accounts. Use them only to check published numbers, and
never to tune our models.

## Tuấn: literature table and slides

Context: the professor asked whether MGTAB has official results and what the newest strong methods are. A first
survey found these papers, but its numbers still need checking in the source.

**Literature table**

1. Check these papers: [MGTAB (RGT)](https://arxiv.org/abs/2301.01123), [BSG4Bot](https://arxiv.org/abs/2410.05356),
   [BotVA](https://www.ijcai.org/proceedings/2026/358), [BotHP](https://arxiv.org/abs/2506.00989),
   [RABot](https://arxiv.org/abs/2602.21749), [SeBot](https://arxiv.org/abs/2405.11225).
2. For each, record: venue and year, accuracy, F1, F1 type (macro or binary), split, number of runs, relations, code link.
   Write "not stated" when the paper does not say. Never guess.
3. Try to find these through the university library: MPS-Bot (WWW 2026), BotBR (SIGIR 2025), BotMGTD (IPM 2026),
   HEDGE (TIFS 2026). The first survey could not read their numbers.
4. Do not cite [RMNP](https://arxiv.org/abs/2503.09626) or [SAHG](https://arxiv.org/abs/2605.30166). Their authors withdrew them.

**Slides**

1. Start from the Progress Review 1 slides (ask Minh or Sơn for the file).
2. Answer the professor's questions: which approach wins, why, and what it means in practice. Include the fair
   comparison, relation ablation, error analysis and reproduced methods.
3. Check every number against `results/*.csv`. Label all results "validation, one split".
4. Do not rank published scores against ours unless the split and F1 type match.

## Quyền: document

Context: `midterm-review.docx` holds the proposal plus results so far, and needs to reflect this round. Get the file
from Sơn, save a copy as `midterm-review-2.docx`, and edit that copy in Word.

Keep it looking native: use Insert Caption for new tables and figures, Cross-reference for "Table N" and citations, and
the existing styles and list formats. Press Ctrl+A, then F9 when you finish, to update the numbers.

1. **Section 2 (Related Work):** add Tuấn's literature table and MGTAB's standard method (random 7:2:1, 5 seeds, macro-F1).
2. **Section 3 (Data):** MGTAB has no fixed split file, but its benchmark uses a standard method. Explain why we use our
   own split.
3. **Section 5 (Baselines):** new baseline definitions and names from Đại.
4. **Section 7 (Results):** new results table and the takeaways: which approach wins, why, and what it means in practice.
5. **Appendix 1 (Timeline):** error analysis and relation ablation now come before the RGCN. Update the owners.
6. Keep conclusions about bots only: a bot is not always harmful. Use real numbers from `results/*.csv` only.

## Rules for this round

- Report all our models on `split()`, and keep our test set locked.
- Published scores use different splits and F1 types. Do not rank them directly against ours unless the setup matches.
- Say "bot detection", not "harmful content detection".

# Tasks: Progress Review 1 (slides only, Week 6)

Goal: show the problem formulation, working baselines and a first prototype. Validation results only.

| Who | Task | Output | Status |
|---|---|---|---|
| **Sơn** | Repo, shared code and template. G-struct features per relation (degree, PageRank, clustering, k-core). Prototype P1 Fusion-LR (profile + tweet + G-struct). | `features.py`, `notebooks/p1.ipynb`, `results/p1.csv` | Done |
| **Đại** | Baselines: B0 majority, B1 content-only LR, B2a graph-structure LR (uses `features.py`), B3 label propagation (train labels only). | `notebooks/baselines.ipynb`, `results/baselines.csv` | Done |
| **Minh** | Slides: formulation, data and relations, metrics, results table (baselines vs P1). Leakage check and one re-run of both notebooks. | slides | |

B2a uses `struct_features` from `features.py` (standardize with train statistics, as in `p1.ipynb`).

Not in this round: P2 RGCN, B2b Node2Vec (both need PyTorch Geometric), ablations, fairness check.
Quyền, Phúc and Tuấn are free this round and take these later.
