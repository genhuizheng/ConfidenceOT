# L1-L5 preprocessing grid under frozen calibration v3 (2026-11-04)

**Setup.**
- Calibration: `FROZEN_CALIBRATION_v3.json` (`within_side_split+mnncovlocal`; see `CALIBRATION_V3.md`).
- Splatter: Splatter 1.30.0 defaults, `--pairing paired`, N = 200 per side, replicates 1-2.
- Method: M4-E at acceptance 0.99, with no rejection cap and no downsampling (DS none everywhere).
- Grid: every other dimension is kept (encoding, K / H, ties, scaling, cost, PCA 30 / 50), giving 297 configurations.
- Run size: 1,512 tasks, 11,823 fits, 47,520 scored rows, 0 failures, 2,975 s locally.

**Outputs** (gitignored) are in `results/splatter_local_20261103/`:
- `RESULTS_grid_v3.tsv`;
- `GRID_SUMMARY_v3.{tsv,md}`, which has the per-level Pareto front;
- `GRID_OVERALL_v3.tsv`.

**Readout.** These are the effective metrics of `summarize_local.py`:
- Case 1 FRR;
- Case 2 Rejection F1;
- Case 3 unmatched recall.

Selection uses the Pareto front on the three metrics, not Case 2 F1 alone.

## Overall

| Configuration | Fronts (of 5) | Case 1 FRR | Case 2 F1 | Case 3 recall | Worst-level F1 |
|---|---|---|---|---|---|
| ranknm512-expected \| noscale \| cos \| PCA30 | 4 | 0.015 | 0.859 | 0.948 | 0.645 |
| logcpm \| noscale \| cos \| PCA30 | 3 | 0.018 | 0.828 | 0.940 | 0.662 |
| ranknm512-production \| noscale \| cos \| PCA30 | 2 | 0.016 | 0.860 | 0.947 | 0.659 |
| Baseline: ranknm256-production \| noscale \| cos | 0 (rank 43) | 0.020 | 0.822 | 0.924 | 0.530 |

The ranknm512 tie rules (expected / production / midrank) differ by at most 0.002 in F1.

## Per level (best on the front, with the baseline beside it)

| Level | Best | Case 1 FRR | Case 2 F1 | Case 3 recall | Baseline FRR / F1 / recall |
|---|---|---|---|---|---|
| L1 depth | ranknm512 noscale cos PCA30 | 0.001 | 0.991 | 0.923 | 0.001 / 0.941 / 0.847 |
| L2 + dropout | ranknm512-expected / midrank noscale cos PCA30 | 0.000 | 0.888 | 0.818 | 0.001 / 0.874 / 0.774 |
| L3 + batch | logcpm noscale cos PCA30 | 0.002 | 0.893 | 1.000 | 0.004 / 0.869 / 1.000 |
| L4 + perturbation | ranknm256-expected / midrank noscale cos PCA30 | 0.001 | 0.901 | 1.000 | 0.001 / 0.895 / 1.000 |
| L5 + imbalance | ranknm1024-production scale cos PCA30 | 0.055 | 0.671 | 0.775 | 0.091 / 0.530 / 1.000 |

At L5 the best F1 costs Case 3 recall. Two alternatives at L5:
- logcpm reaches F1 0.662 with recall 0.972;
- ranknm512-production reaches F1 0.659 with recall 1.000.

## Factor means (Case 2 F1, all levels)

| Factor | Mean Case 2 F1 by value |
|---|---|
| Cost | cos 0.363; sqeuc 0.052 |
| Scaling | noscale 0.240; scale 0.175 |
| PCA | 30: 0.236; 50: 0.178 |
| Encoding | logcpm 0.336; ranknm 0.291; rank_value 0.142; rankhvg 0.140 |
| Ties | no difference |

**sqeuc is a cost-geometry failure, not a batch failure.** It rejects almost nothing even at L1, where there is no batch.
For example, ranknm256-production sqeuc PCA30 at L1 has these Case 2 rejected fractions:
- source-unmatched pairs: source side 0.035 against 39 cells truly to reject;
- target-unmatched pairs: target side 0.042 against 40.

## Batch-effect failure

- **Leading configurations:** the leaders and the baseline have no Case 1/2 whole-side rejection at any level.
- **L3 and L4:** these are no harder than L1-L2 for them. F1 is about 0.89 and FRR about 0.002.
- **Configurations that fail:** 247 of 297 fail at one or more batch levels, mostly sqeuc, rank_value, rankhvg and large-K
  ranknm. In 21 of them, Case 1/2 rejects a whole side at L3-L5.
- **L5 remains the weak level.** F1 is about 0.66 and Case 1 FRR is 0.055-0.09.

## Next

Take the leading configurations to the TACC N = 5000 run:
- ranknm512-expected noscale cos PCA30;
- logcpm noscale cos PCA30;
- the baseline, for reference.

Use GPU fits on gh after a CPU / GPU equivalence check.
