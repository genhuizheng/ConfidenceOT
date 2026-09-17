# Depth-treatment screen: experimental setup

Which preprocessing removes the sequencing-depth dependence from the
ConfidenceOT gate, measured against a construction whose correct answer is
known rather than against a cancer result.

Runner: `scripts/validate_depth_null_specificity.py`
Local triage: `scripts/screen_depth_configurations.sh`
Cluster: `scripts/tacc/screen_depth_configurations.slurm` (`--array=0-10`)

---

## 1. Simulation arms — the ground truth

One gene-mean vector is drawn per replicate and **shared by both sides**, so a
homogeneous arm is genuinely homogeneous. Every cell's relative expression
profile comes from the same distribution regardless of its depth, so any
dependence of the gate on depth is a specificity failure by construction.

| Arm | `depth_sigma` | Perturbed fraction | Correct answer |
|---|---:|---:|---|
| `homogeneous_depth_cv0` | 0.0 | 0.0 | reject nothing |
| `homogeneous_depth_cv_low` | 0.3 | 0.0 | reject nothing |
| `homogeneous_depth_cv_mid` | 0.6 | 0.0 | reject nothing |
| `homogeneous_depth_cv_high` | 0.9 | 0.0 | reject nothing |
| `perturbed_depth_cv0` | 0.0 | **0.2** | reject those 20%, and only those |

The four homogeneous arms differ **only** in how widely per-cell depth is
spread. The perturbed arm is the positive control at uniform depth: a genuine
subpopulation exists on the source side alone, so it separates sensitivity from
specificity. A method that flattens the depth effect by rejecting nothing at all
fails here and is caught.

`--depth-source` additionally enables two arms drawn from a real dataset's
observed depth distribution instead of a chosen sigma; not part of this screen.

---

## 2. Configurations under test

Three independent levers: the transform, whether reads are equalised, and the
cost geometry. `ds` = read equalisation on.

| # | Label | Transform | Equalise reads | Cost | Why it is in |
|---:|---|---|:-:|---|---|
| 0 | `logcpm` | log-CPM | — | Euclidean | baseline; what the project ran originally |
| 1 | `logcpm_ds` | log-CPM | **yes** | Euclidean | equalisation alone |
| 2 | `logcpm_cos` | log-CPM | — | **cosine** | the PI's suggestion, isolated |
| 3 | `rank256` | gene rank, top 256 | — | Euclidean | **the arm in the existing figure** |
| 4 | `rank256_ds` | gene rank, top 256 | **yes** | Euclidean | **the production cancer pipeline — never simulated before** |
| 5 | `rank256_ds_cos` | gene rank, top 256 | **yes** | **cosine** | production plus the PI's suggestion |
| 6 | `pearson_ds` | analytic Pearson residuals (ours) | **yes** | Euclidean | our residual implementation |
| 7 | `scanpy_pearson` | **scanpy** `normalize_pearson_residuals` | — | Euclidean | external reference for the residual |
| 8 | `scanpy_pearson_ds` | **scanpy** residuals | **yes** | Euclidean | same, plus equalisation |
| 9 | `sct` | **Seurat SCTransform v2** | — | Euclidean | field standard, external |
| 10 | `sct_ds` | **Seurat SCTransform v2** | **yes** | Euclidean | same, plus equalisation |

Rows 7–10 are external packages, so the transform comparison does not rest only
on our own code. They mirror the production tail exactly — stack both sides,
take top-variance genes, centre, scale per gene, joint PCA — so the transform is
the only thing that differs.

### What each lever does

- **gene rank**: divide each cell by its total, divide each gene by that gene's
  nonzero median over both sides stacked, rank within the cell, keep the top
  256 and discard the values.
- **read equalisation**: exact multivariate hypergeometric subsampling of every
  cell's reads to one shared total, the pooled 10th percentile. Cells already at
  or below it are left untouched, so the residual gradient matches the real runs.
  The same operation as `cancer_metastasis/27_downsample_counts.py`.
- **cosine**: L2-normalise each cell's PCA coordinates. For unit vectors
  `||a-b||^2 = 2 - 2cos`, so this is the cosine cost and it discards each cell's
  magnitude and nothing else.

---

## 3. Fixed parameters

| | Production (cluster) | Local triage |
|---|---:|---:|
| Cells per side | 1,500 | 600 |
| Genes | 4,000 | 2,500 |
| Joint HVG | 2,000 | 1,200 |
| PCs | 30 | 30 |
| Replicates per arm | 3 | 2 |
| Median depth | 10,000 | 10,000 |
| Dispersion | 2.0 | 2.0 |
| Perturbation | 1.0 log2 | 1.0 log2 |
| Calibration null | within-side split | within-side split |
| Within-side acceptance minimum | 0.90 | 0.90 |
| Source rejection cap | 0.85 | 0.85 |
| Target rejection cap | 0.00 | 0.00 |
| Cells for calibration | 2,000 | 600 |

The reduced local scale reproduces the production numbers: the `logcpm`
baseline gives a depth effect of 0.000 / 0.450 / 0.490 / 0.492 across the four
arms locally against 0.00 / 0.44 / 0.49 / 0.495 on the cluster, so the local
triage transfers.

---

## 4. Metrics, and what counts as passing

| Metric | Meaning | Passing |
|---|---|---|
| **`auc_total_counts`** | Mann–Whitney AUC of *pre-equalisation* depth against the retained set. The headline. | **0.5**; plotted as `abs(AUC - 0.5)`, so 0 |
| `spearman_decision_cost_total_counts` | Rank correlation of the decision cost with depth | 0 |
| `source_rejection_rate` | Fraction rejected | near 0 on homogeneous arms; near 0.2 on the perturbed arm |
| `sign_rule_retained_fraction` | What the calibrated rule alone would retain, before any cap | should equal the retained fraction; a gap means the cap is doing the work |
| **`perturbed_f1`**, `perturbed_recall`, `perturbed_precision` | Positive control | F1 must stay high; a configuration that kills the depth effect by rejecting nothing fails here |
| `calibration_valid`, `m4e_inference_valid` | Did calibration and inference succeed | true |

Depth is tested against each cell's **original** depth, not its post-equalisation
depth, which is near constant after correction and would make the test trivially
0.5. This is the same reason `27_downsample_counts.py` emits
`predownsample_depth.csv.gz`.

A configuration is only interesting if it clears **both** columns: depth effect
near 0 on all four homogeneous arms **and** the positive control intact.

---

## 5. Status

| | Where | State |
|---|---|---|
| Configurations 0–6 | this machine | running; `logcpm` done |
| Configurations 7–10 | cluster only | **untested** — local scanpy is broken against anndata 0.12 and there is no R here. Both were verified only to fail with a message naming the missing dependency. |
| All 11 at production scale | cluster | not yet submitted |

## 6. Prior expectation, written before the results

Recorded so the screen can contradict it.

- `rank256_ds` is the production pipeline and reaches AUC 0.493 on real prostate
  data, so it should be the one that clears all four arms. Its absence is why
  the existing figure showed ranking failing at high depth spread: that arm was
  ranking **alone**, which on the same real data gives 0.060.
- `cosine` removes magnitude only. Depth also acts through dropout and through
  transcriptome composition, neither of which is magnitude, so it should help
  partially and not close the high-spread arm on its own.
- `pearson_ds`, `scanpy_pearson*` and `sct*` are one family. Our analytic
  residual already failed, and the reason applies to all of them: a gene
  observed at zero keeps a large residual, so dropout survives the transform.
