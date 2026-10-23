# Level 3: cell-level CRISPR lineage correspondence

Does OT match metastatic cells to primary cells that are closer CRISPR relatives than the independent coupling would? Asked of the 54 Level 2 couplings: 9 cases × RNA-only / CNV-only / RNA+CNV × A (Standard OT, Balanced OT) / B (Our OT, UOT, rejection budget 0).

- `PLAN.md`: the design, fixed before any coupling was scored, plus one dated post-hoc section.
- `REPORT.md`: results and the answers to the three questions.

## Files

| File | What it does |
|---|---|
| `l3_core.py` | trees (project `Tree`, contracted depth), frozen leaf mapping, O(sources + depth) tree distances, per-cell status, matching quantities |
| `ground_truth.py` | method-independent status of every metastatic cell (evaluable / ambiguous / not_informative / not_evaluable), coverage and flags; reads no coupling |
| `score_level3.py` | per cell × strategy × method: coverage, D_OT, D_ind, ΔD, ΔD_norm, top-k, close-relative masses; one coupling in memory at a time; missing couplings reported |
| `summarize_level3.py` | cell → family / clone → mouse → across mice, 95% bootstrap intervals, A vs B paired on the same cells |
| `within_subclone.py` | post-hoc decomposition: the same quantities restricted to the cell's own frozen subclone |
| `plot_level3.py` | Figures A (lineage matching on a real clade), B (OT vs independent per family), C (rule-chosen success / failure cells) |
| `test_level3.py` | synthetic tests (14/14, `test_level3_result.txt`) |

## Reproduce

Run locally from the repository root. The couplings come from `level3_couplings.tar`, unpacked into the Level 2 download.

```bash
R=lineage_validation/level2_pairwise_ot/data/full_20261022_204810; L=lineage_validation/level3_cell_lineage
python $L/ground_truth.py --runs $R --out $L/results/ground_truth
python $L/score_level3.py --runs $R --out $L/results/scores
python $L/summarize_level3.py --scores $L/results/scores --ground-truth $L/results/ground_truth --out $L/results/summary
python $L/within_subclone.py --runs $R --out $L/results/within_subclone
python $L/plot_level3.py --runs $R --scores $L/results/scores --summary $L/results/summary --out $L/results/figures
```

## Results folder

- `ground_truth/`: `targets.tsv.gz` (every metastatic cell, its status and reason), `coverage.tsv`.
- `scores/`: `cells/<case>.tsv.gz` (cell × strategy × method), `run_status.tsv`.
- `summary/`: `family_summary.tsv` (with top-k), `site_summary.tsv`, `mouse_summary.tsv`, `overall.tsv`, `ab_paired.tsv`, `ab_overall.tsv`.
- `within_subclone/`: `cells/`, `family_summary.tsv`, `overall.tsv`.
- `figures/`: A (78 files), B, C (3 files).
