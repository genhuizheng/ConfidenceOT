# LARRY (Weinreb et al. 2020) in vitro: can RNA-OT recover true clonal fates? (plan)

**Status:** fixed 2026-11-12, after the data audit (`01_audit.py`) and before any OT was run. A later change is a dated
section at the end.

**Data:**
- GSE140802, `stateFate_inVitro`: 130,887 cells; Lin−Kit+Sca1+ (LSK) and Lin−Kit+Sca1− (LK) starting populations;
  days 2, 4, 6.
- 5,864 clones (each barcoded cell in exactly one clone); 1,816 clones have cells on day 2 and on day 4 / 6.
- **Counts:** the public matrix is total-count normalised. Every cell is a pure rescaling (normalised totals
  identical), and dividing by each cell's smallest non-zero value gives integers for 100% of entries. Those recovered
  integer counts are the input (median 2,249 UMIs per cell).

**Rules:** OT, preprocessing and calibration are not changed; only inputs and outputs are prepared. Nothing is tuned
after results.

## Units

4 units: starting population (LSK, LK) × target day (4, 6). Source = day-2 cells, target = day-4 or day-6 cells, same
starting population.

**Cells per side (input preparation):** at most 10,000 per side.
- All clonally informative cells go in first: day-2 cells whose clone has cells on the target day; target cells
  whose clone has cells on day 2.
- The side is then filled with randomly drawn other cells of that side (seed 20261112).
- The OT never sees clone labels. They are used only for scoring.
- The share of informative cells per side is reported.

## OT (unchanged pipeline)

**Pipeline:** `realdata_benchmark/v3_grid/worker.py`, frozen calibration v3 (within-side split null, `mnncovlocal`),
with the KP-Tracer fit arguments:
- M4-E and M4-R;
- acceptance 0.99;
- no extra cell cap;
- every cell in scope.

**Configurations:**
- Splatter L5 best `ranknm256-midrank|none|scale|cos|pca50`;
- logCPM + PCA `logcpm|none|noscale|sqeuc|pca30`.

The worker fits the 8 configurations of each of the two encoding keys; only these two are read.

**Run:** on TACC gg nodes, as the v3 grid.

## Scoring (fixed now)

### Truth

For a day-2 cell i of clone c, its sisters T_c are the target-side cells of clone c.

Only clones with ≥ 1 sister on the target side are scored. Fate classes use the authors' annotation; mature types
exclude "Undifferentiated".

### Metrics

1. **Clone mass.**
   - Definition: m_i = Σ_{j∈T_c} P(i, j) / Σ_j P(i, j), over retained day-2 cells.
   - Baselines: uniform |T_c| / n_target, and the independent coupling (each row distributed by the retained target
     marginal).
   - Reported:
     - median m_i;
     - median m_i / baseline;
     - share of cells with m_i > baseline;
     - top-k hit rate (≥ 1 sister among the cell's 10 highest-mass targets) against its chance level.
2. **Fate prediction.**
   - Predicted fate of i: its row-normalised coupling mass on target cell types (mature types only).
   - True fate: the cell-type composition of T_c (mature types only).
   - Reported:
     - total variation between the predicted and true distributions, against a label-permutation null (clone labels
       shuffled among scored day-2 cells, 100 draws, seed 20261112);
     - neutrophil vs monocyte: Spearman of the predicted and true neutrophil share among Neu + Mono, for clones with
       ≥ 2 Neu + Mono sisters, and the AUROC for clone majority Neu vs Mono. This is the authors' benchmark bifurcation.
3. **Gates.**
   - Rejected share per side and cell type.
   - Day-2 rejection for cells with vs without sisters on the target side. Clones that leave no sampled descendants
     are not a ground truth of absent fate.

### Comparison

L5 best vs logCPM + PCA; M4-E primary, M4-R reported; every unit separately.

## Reading

- One experiment.
- Fate is defined at clone level, with sister cells as the proxy for the day-2 cell's progeny.
- Clonal sampling is sparse, and the day-2 cell is not itself the ancestor of its sisters.
- Associations, interpretation and mechanism are kept apart.
