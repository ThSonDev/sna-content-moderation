# sna-content-moderation

Course project, Social Networks (IS353, UIT). Topic: Using social network structure to support
content moderation. Task: Detect bot accounts on Twitter with account content + a multi-relation user graph.
Proposal: [`doc/new-proposal.pdf`](doc/new-proposal.pdf).

## Data

[MGTAB](https://github.com/GraphDetec/MGTAB) (Shi et al., 2023): 10,199 labeled accounts, 7 relation types.
License CC BY-NC-ND 4.0, so the data is **not** stored in this repo.
The notebook template downloads it for you. To do it by hand:
[download MGTAB.zip](https://drive.google.com/uc?export=download&id=1gbWNOoU1JB8RrTu2a5j9KMNVa9wX72Fe) (31 MB)
and extract it so the files are in `data/MGTAB/*.pt`.

## Start

**Colab:** open `notebooks/template.ipynb` (File → Open notebook → GitHub), save a copy, run the setup cells.

**Local:**
```bash
git clone https://github.com/ThSonDev/sna-content-moderation.git
cd sna-content-moderation
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter
```

## Files

| File | What it is |
|---|---|
| `mgtab.py` | Shared code: `load`, `relations` (cleaned), `split` (fixed 70/15/15), `best_threshold`, `evaluate` |
| `features.py` | G-struct: 32 graph features per account (`struct_features`), cached in `data/` |
| `notebooks/template.ipynb` | Copy this for your task: setup, data download and loading are done |
| `notebooks/p1.ipynb` | Prototype P1 Fusion-LR (P + T + G-struct) and the A2 check, validation results |
| `results/` | Small result tables (CSV) |
| `TASKS.md` | Who does what |

## Rules

- Use `split()` from `mgtab.py`. Do not make your own split.
- Fit scalers and models on train only. Report validation results; the test set stays locked.
- Tune the threshold on validation (`best_threshold`). Never use `stance` as a feature.
- One notebook per task in `notebooks/`. Commit with outputs cleared if they are large.
