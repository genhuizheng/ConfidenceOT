# log1p + Pearson correlation over all genes (`log1p_noscale_pcc`), native benchmark

**Asked 2026-10-19** (PI): has log1p followed by Pearson correlation over all genes been tried as the OT geometry? It had not. The request was to run it in the simulation, report numbers only (no figure), and use no HVG.

## Configuration

- **Representation.** log(1 + count) on the raw counts, with no library-size scaling and no depth equalisation. Every common gene (10,000) is a coordinate: no HVG, no PCA, no gene z-score.
- **Cost.** Each cell is centred over genes and set to unit length, so the squared Euclidean cost is 2(1 − r), with r the Pearson correlation of two cells. The cost is divided by the median sampled cross-side pair, as for every arm.
- **Gate.** ConfidenceOT M4-E (and M4-R from the same run), ε 0.1, λ 1, rejection cost from the within-side split null at acceptance 0.99, 0.95 and 0.90.
- **Pairs.** The same 180 native pairs and un-equalised counts as the `raw` arm.
- **Code.** bdc9878 (arm), dd5373d (scorer fix), `benchmark/tacc/submit_log1p_pcc.sh`, `score_arms.slurm`, `58_summarise_arms.py`.
- **Jobs.**
  - OT: 1082230 (0.99), 1083330 (0.95) and 1082232 (0.90), all tasks completed, about 40 minutes each.
  - Scoring: 1083332 (0.99), 1083331 (0.95) and 1082233 (0.90).
- **Outputs.** `benchmark_native/runs_acceptance_<a>/log1p_noscale_pcc/` and `benchmark_native/metrics_log1p_pcc/acceptance_<a>/` (`summary_wide.tsv`, `pooled_<arm>.csv`).

The reference arms at 0.99 and 0.95 were scored in the same jobs with the same scorer. At 0.90 they come from `benchmark_metrics.csv`, through the same summary script. The scorer reproduces the previously reported values:

| Arm | Reported F1 | Reproduced F1 |
|---|---|---|
| ranknm256_noscale_ds_cos at 0.99 | 0.99 | 0.992 |
| ranknm256_noscale_cos at 0.99 | 0.98 | 0.983 |
| raw at 0.99 | 0.51 | 0.512 |
| ranknm256_noscale_ds_cos at 0.95 | 0.96 | 0.958 |
| ranknm256_noscale_ds_cos at 0.90 | 0.9174 | 0.9174 |

## Results (M4-E, mean over pairs)

**Directional F1** is the F1 of rejection on the side holding the unmatched cells, over lost, emerged and disjoint (135 pairs).

| Acceptance | log1p_noscale_pcc | ranknm256_noscale_ds_cos | ranknm256_noscale_cos | raw |
|---|---|---|---|---|
| 0.99 | **0.015** | 0.992 | 0.983 | 0.512 |
| 0.95 | **0.050** | 0.958 | 0.937 | 0.610 |
| 0.90 | **0.082** | 0.917 | 0.886 | 0.627 |

At 0.90, `logcpm_noscale_cos` gives 0.742.

**False asserted mass** is the cross-population mass between cells kept on both sides; the ideal is 0.

| Acceptance | Case | log1p_noscale_pcc | ds_cos | noscale_cos | raw |
|---|---|---|---|---|---|
| 0.99 | all_shared | 0.472 | 0.036 | 0.028 | 0.077 |
| 0.99 | populations_disjoint | 0.610 | 0.000 | 0.000 | 0.218 |
| 0.95 | all_shared | 0.455 | 0.029 | 0.021 | 0.059 |
| 0.95 | populations_disjoint | 0.580 | 0.000 | 0.000 | 0.099 |
| 0.90 | all_shared | 0.438 | 0.023 | 0.015 | 0.052 |
| 0.90 | populations_disjoint | 0.550 | 0.000 | 0.000 | 0.058 |

**Rejected fractions at 0.99** (L0 / L1 / L2; ground truth in brackets), log1p_noscale_pcc against ranknm256_noscale_cos:

| Readout | log1p_noscale_pcc | ranknm256_noscale_cos |
|---|---|---|
| all_shared source (0) | 0.010 / 0 / 0 | 0.005 / 0.002 / 0.004 |
| lost, unmatched source (1) | 0.014 / 0 / 0 | 1 / 1 / 1 |
| emerged, unmatched target (1) | 0.013 / 0.002 / 0.024 | 1 / 1 / 1.00 |
| disjoint source (1) | 0.017 / 0 / 0 | 1 / 1 / 1.00 |

At 0.90, log1p_noscale_pcc rejects about 10% of cells at L0, matched or not (all_shared source 0.099, lost unmatched source 0.128), and almost nothing at L1 and L2.

## Reading

- **The gate does not separate matched from unmatched cells.** Under this geometry it rejects at about the same low rate either way, at every acceptance. Lowering the acceptance raises the rate a little without making it discriminate.
- **The coupling mixes populations.** About 45% of the mass between kept cells crosses populations in `all_shared`, against 2–4% for the ranknm256 arms. So the geometry itself fails to pair cells, independently of the gate.
- **Unverified explanation.** Splatter's groups share one gene-mean backbone and differ only in a minority of DE genes. Pearson correlation over all 10,000 unscaled genes is then dominated by the shared backbone, so within-group and between-group correlations are nearly equal. HVG selection would concentrate on the DE genes.
- **How to check it without rerunning OT:** compare the 1 − r distributions of same-group and different-group cell pairs on a few conditions.
- **Scope.** This is one simulator. Splatter's DE structure decides how much signal sits outside the shared backbone, so the result is not a statement about real tissue.
