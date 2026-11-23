# ConfidenceOT, balanced vs unbalanced backbone over the v3 grid (Splatter N = 5,000; results 2026-11-23)

## What was run

The v3 grid protocol (`realdata_benchmark/v3_grid/README.md`, local report `realdata_benchmark/splatter_local/GRID_V3_REPORT.md`,
calibration `realdata_benchmark/splatter_local/CALIBRATION_V3.md`), unchanged, twice:
- UOT backbone: the existing run `v3_grid_20261104` (2220/2220 rows);
- balanced backbone: the same worklist with `--backbone balanced` (`v3_grid_bot_20261118`, jobs 1152224-1152227,
  2220/2220 rows SUCCESS, 0 FAILED). Only the backbone of the M4-E / M4-R fit differs.

Per unit (Splatter N = 5,000 per side, Splatter defaults, levels L1 depth / L2 + dropout / L3 + batch / L4 + perturbation /
L5 + imbalance, 4 pairs per level = the cases, replicates 1-3) and per configuration (296 = 37 encoding keys x PCA 30 / 50
x gene scaling no / yes x cost cos / sqeuc):
1. encode the joint counts: `logcpm` = log1p(1e4 x count / total); `ranknm{K}` = Top-K rank per cell, value 1 - r / K;
   `rank_value{K}` = the same after dividing each gene by its nonzero median; `rankhvg{H}` = ranks within the H
   top-variance log-CPM genes; tie rule production / expected / midrank;
2. 2,000 top-variance encoded genes (all H for rankhvg), optional gene scaling, PCA 30 or 50; unit rows for cos;
3. correction `within_side_split+mnncovlocal`: credible-anchor coverage gate; when open, each target cell is moved by
   the kernel average of the credible-anchor displacements; when closed, nothing moves;
4. fit `02_run_pair.py` (ConfidenceOT, M4-E and M4-R): cost = distance / median pair cost, calibrated by the within-side
   split null at acceptance 0.99, entropic backbone ε 0.1 (UOT λ 1, or balanced);
5. score M4-E with the effective readout of `summarize_local.py` (when one side retains no cell, the other side's
   retained cells count as unmatched).

## Metrics and selection rule (the protocol's own, not redefined)

- Case 1 (all populations shared): FRR, `case1_frr_eff`.
- Case 2 (one population lost or emerged): Rejection F1 on the side with the unmatched population, `case2_f1_eff`.
  This is the primary metric (Splatter F1 rule).
- Case 3 (disjoint populations): unmatched recall, `case3_recall_eff`.
- Selection: per level, the Pareto front on (Case 1 FRR low, Case 2 F1 high, Case 3 recall high); over L1-L5, the
  configurations on the front at the most levels, then mean Case 2 F1 (`summarize_local.py --stage grid`).

## Results

Case 2 F1 of the two backbones correlates 0.997 over the 296 configurations x 5 levels.

Leader of each per-level front (highest Case 2 F1 on the front; FRR / F1 / recall):

| Level | UOT backbone | Balanced backbone |
|---|---|---|
| L1 | rankhvg2000-production noscale cos pca50: 0.004 / 0.995 / 1.0 | ranknm512-production noscale cos pca30: 0.001 / 0.996 / 1.0 |
| L2 | rank_value2048-expected scale cos pca50: 0.004 / 0.995 / 1.0 | rank_value2048-expected noscale cos pca50: 0.004 / 0.994 / 1.0 |
| L3 | ranknm512-production noscale cos pca50: 0.002 / 0.996 / 1.0 | same: 0.002 / 0.996 / 1.0 |
| L4 | rank_value2048-production noscale cos pca50: 0.018 / 0.950 / 1.0 | same: 0.016 / 0.951 / 1.0 |
| L5 | ranknm256-midrank scale cos pca50: 0.146 / 0.675 / 1.0 | same: 0.211 / 0.627 / 1.0 |

The main configuration C1 (`ranknm256-midrank|none|scale|cos|pca50`, fixed by the task) is on the front only at L5, for
both backbones; at L1-L4 its F1 is 0.94-0.98 with FRR ≤ 0.026, the same for both backbones (within 0.007). It has the
highest mean Case 2 F1 over L1-L5 for both (UOT 0.902, balanced 0.894).

The over-L1-L5 rule does not single out one configuration at N = 5,000: with UOT the most fronts (2 of 5) are reached by
`ranknm256-expected|none|noscale|sqeuc|pca50` (mean F1 0.824), `ranknm512-production|none|noscale|cos|pca50` (0.801)
and several scale sqeuc configurations whose F1 is 0.02-0.05 (they reject almost nothing, so their FRR is lowest); with
the balanced backbone, `ranknm256-expected|none|scale|cos|pca50` (0.884) and the same kind of sqeuc configurations.

Backbone effect, per level (median over the 296 configurations of the three metrics, both backbones, in
`tables/DIFF_by_level_distance.tsv`):
- L1-L4: the same. Median Case 2 F1 cos 0.977 / 0.961 / 0.976 / 0.859 with UOT vs 0.981 / 0.961 / 0.977 / 0.864 balanced;
  FRR within 0.005; Case 3 recall unchanged.
- L5: the balanced backbone is worse: median Case 2 F1 cos 0.551 -> 0.522, sqeuc 0.464 -> 0.418; Case 1 FRR cos
  0.252 -> 0.315. A balanced plan has to move all mass, so the proportion difference pushes shared cells to poor
  partners and the gate rejects them.

The local N = 200 grid (GRID_V3_REPORT.md) put `ranknm512-expected|none|noscale|cos|pca30` first (4 of 5 fronts). At
N = 5,000 it is on no front with either backbone (UOT FRR / F1 at L4 0.071 / 0.826, at L5 0.369 / 0.391). It was called
C3 / "grid best" in this study after the local grid; that label is the local result only.

Tables: `results/ot_depth_benchmark_20261117/bot_uot_grid/tables/` (`PAIRED_bot_vs_uot_by_config_level.tsv`,
`DIFF_by_level_distance.tsv`, `NAMED_configs_bot_vs_uot.tsv`, `OVERALL_L1L5_mean_by_config.tsv`) and both grids' own
`GRID_SUMMARY_v3tacc.{md,tsv}` / `GRID_OVERALL_v3tacc.tsv`. Panels `figures/scatter_{metric}_{level}`.

## Exploratory metrics (user 2026-11-23; not the benchmark readout)

`explore_metrics_bot_uot_grid.py`, tables `bot_uot_grid/tables/exploratory/`. Only columns the grid already writes; the
whole-side threshold (>= 99% of a side rejected) is summarize_local.py's one_side_all rule. Cosine configurations,
median over configurations, UOT -> balanced:

| Level | E1 precision | E1 recall | E2 shared-pop. FRR | E3 F1 above reject-all | E5 Case 1/2 whole-side share | E6 F1 replicate SD |
|---|---|---|---|---|---|---|
| L1 | 0.961 -> 0.966 | 0.999 -> 0.999 | 0.014 -> 0.015 | 0.643 -> 0.647 | 0 -> 0 | 0.002 -> 0.002 |
| L2 | 0.952 -> 0.952 | 0.997 -> 0.997 | 0.020 -> 0.021 | 0.628 -> 0.628 | 0 -> 0 | 0.005 -> 0.005 |
| L3 | 0.962 -> 0.964 | 0.994 -> 0.995 | 0.012 -> 0.014 | 0.643 -> 0.644 | 0.012 -> 0.012 | 0.006 -> 0.006 |
| L4 | 0.807 -> 0.813 | 0.994 -> 0.994 | 0.091 -> 0.089 | 0.526 -> 0.531 | 0.028 -> 0.028 | 0.036 -> 0.039 |
| L5 | 0.450 -> 0.406 | 0.956 -> 0.950 | 0.271 -> 0.336 | 0.313 -> 0.283 | 0.044 -> 0.044 | 0.032 -> 0.030 |

- E1/E2: the L5 loss of the balanced backbone is a precision loss (more shared cells rejected); recall is unchanged.
- E4 Case 3 false asserted mass: 0 on every cosine configuration; sqeuclidean L1 / L2 0.093 -> 0.141 and 0.137 -> 0.157.
- E5: C1 has no Case 1/2 whole-side rejection at any level with either backbone.
- E6: C1 is least stable at L5 (F1 replicate SD 0.127 UOT, 0.084 balanced; ≤ 0.05 elsewhere).
- E7: the order of the 296 configurations by Case 2 F1 is the same under both backbones (Spearman 0.992-0.996 at
  L1-L4, 0.956 at L5).
- E8: M4-E and M4-R reject the same number of cells in only 16-32% of side x fit rows; they are different estimators.
- E9 per fit: M4-E converges in every fit (median 34 s UOT, 20 s balanced); M4-R cycles in 53% (UOT) / 58% (balanced)
  of fits (median 76 s / 103 s).
