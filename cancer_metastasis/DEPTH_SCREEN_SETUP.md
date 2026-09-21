# Depth-treatment screen: experimental setup

Which preprocessing removes the sequencing-depth dependence from the
ConfidenceOT gate, measured against a construction whose correct answer is
known rather than against a cancer result.

Runner: `scripts/validate_depth_null_specificity.py`
Local triage: `scripts/screen_depth_configurations.sh`
Cluster: `scripts/tacc/screen_depth_configurations.slurm` (`--array=0-12`)

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
| 9 | `sct` | **`sctransform::vst`, flavour v2** | — | Euclidean | field standard, external |
| 10 | `sct_ds` | **`sctransform::vst`, flavour v2** | **yes** | Euclidean | same, plus equalisation |
| 11 | `pearson_ds_cos` | analytic Pearson residuals (ours) | **yes** | **cosine** | added after 0–10: the two winning levers, combined |
| 12 | `sct_ds_cos` | **`sctransform::vst`, flavour v2** | **yes** | **cosine** | same combination on the best transform |

11 and 12 were added once 0–10 were in, because those results separated on two
different axes — see §7. They are appended rather than inserted, so array
indices 0–10 keep meaning what they meant.

The external rows call `sctransform::vst(vst.flavor = "v2", residual_type =
"pearson", min_cells = 1)` directly, not `Seurat::SCTransform`. Seurat's
function is a wrapper around that call, and its own dependency chain —
interactive and plotting packages — will not build in the container that was
first tried. Comparing against the package that implements the method is also
the more defensible reference.

The external rows are other people's implementations, so the transform
comparison does not rest only on our own code. They mirror the production tail
exactly — stack both sides, take top-variance genes, centre, scale per gene,
joint PCA — so the transform is the only thing that differs.

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
| Calibration grid size | 5 | 5 |
| Null calibration replicates | 5 | 5 |
| Null validation replicates | 5 | 5 |
| Entropic epsilon | 0.1 | 0.1 |
| lambda_a, lambda_b | 1.0, 1.0 | 1.0, 1.0 |
| Solver tolerance | 1e-4 | 1e-4 |
| Minimum detection rate | 0.0 (no gene filter) | 0.0 |
| Seed | 20260914 | 20260914 |

Cells per side, genes, HVG, PCs, replicates and calibration cells are passed
explicitly by `scripts/tacc/screen_depth_configurations.slurm`; everything else
above is the script's default. The per-replicate seed is
`seed + 7919 * replicate + hash(arm) % 10000`, so source and target of one
replicate share a generator and therefore share `gene_mean` — which is what
makes the homogeneous arms homogeneous.

The reduced local scale reproduces production on the baseline: `logcpm` gives a
depth effect of 0.000 / 0.450 / 0.490 / 0.492 across the four arms locally
against 0.000 / 0.462 / 0.490 / 0.494 on the cluster, so the local triage
transfers. (An earlier separate run, `sim_log_cpm_20260916`, gave 0.00 / 0.44 /
0.49 / 0.495; close, but it is a different run with no read-equalisation code
present, so it is not the comparison to quote.)

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
| Configurations 0–10 | cluster, production scale | **complete**, 3 replicates |
| Configurations 0–6 | this machine, reduced scale | complete, 2 replicates; used only as triage |
| Configurations 11–12 | cluster | submitted after 0–10 returned |
| Confirmation round, 10 replicates | cluster | for the configurations §7 leaves in contention |

R came from `$SCRATCH/conda_envs/infercnv_r`, which already had sctransform
0.4.3. The container route was abandoned: its image has compilers but not the
development headers, so 19 of Seurat's dependencies would not build and the
image is read-only.

`CONFIDENCEOT_REPLICATES` raises the replicate count and changes the output
label to `<label>_r<n>`, so a confirmation round cannot overwrite the run it
confirms.

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

---

## 7. Outcome, against the prior in §6

> **Superseded in part.** Everything below is the hand-built construction. The
> splatter round showed that read equalisation only helps there, because
> multivariate-hypergeometric downsampling is the exact inverse of how that
> construction creates depth variation, so this section's ranking of the
> equalised configurations does not transfer. See
> `SEQUENCING_DEPTH_RESOLUTION.md`. The cosine result below does transfer, and
> transfers more strongly.


Configurations 0-10, production scale, 3 replicates. Depth effect is
`abs(AUC - 0.5)` at depth CV 0.31 / 0.66 / 1.12; the CV-0 arm is 0.000 for
every configuration by construction.

| Configuration | 0.31 | 0.66 | 1.12 | worst, over replicates | control F1 | rejected (true 0.20) |
|---|---:|---:|---:|---|---:|---:|
| `logcpm` | 0.462 | 0.490 | 0.494 | 0.494 (0.494-0.495) | 0.887 | 0.249 |
| `logcpm_ds` | 0.046 | 0.190 | 0.258 | 0.258 (0.216-0.306) | 0.872 | 0.259 |
| `logcpm_cos` | 0.052 | 0.082 | 0.128 | 0.128 (0.111-0.142) | 0.952 | 0.220 |
| `rank256` | 0.020 | 0.138 | 0.363 | 0.363 (0.360-0.369) | 0.855 | 0.268 |
| `rank256_ds` | 0.022 | 0.053 | 0.127 | 0.127 (0.118-0.135) | 0.845 | 0.272 |
| `rank256_ds_cos` | 0.062 | 0.042 | **0.019** | 0.062 (0.030-0.083) | **0.945** | **0.223** |
| `pearson_ds` | 0.049 | 0.037 | 0.050 | 0.050 (0.025-0.055) | 0.852 | 0.269 |
| `scanpy_pearson` | 0.436 | 0.481 | 0.488 | 0.488 (0.486-0.492) | 0.867 | 0.261 |
| `scanpy_pearson_ds` | 0.044 | 0.031 | 0.055 | 0.055 (0.038-0.080) | 0.861 | 0.265 |
| `sct` | 0.031 | 0.071 | 0.066 | 0.071 (0.033-0.074) | 0.884 | 0.253 |
| `sct_ds` | **0.008** | **0.015** | 0.045 | 0.045 (0.015-0.054) | 0.870 | 0.260 |

### The prior in §6 was wrong in two of three places

1. **"`rank256_ds` should clear all four arms."** It does not: 0.127 at the
   highest depth spread. The production pipeline is not depth-clean in
   simulation.
2. **"`cosine` should help partially and not close the high-spread arm."** It
   closes it. `rank256_ds_cos` reaches 0.019 there, the lowest single value in
   the table, and it is the only configuration whose depth effect *falls* as
   depth spread rises.
3. **"the residual family is one family and should all fail, because a gene
   observed at zero keeps a large residual."** Right about the analytic form —
   scanpy's `normalize_pearson_residuals` alone gives 0.488, no better than log
   CPM — and wrong about `sctransform`, which alone gives 0.031 / 0.071 /
   0.066. They are not one family. The regularised negative-binomial fit does
   something a fixed-theta analytic residual does not, and that distinction is
   the substantive answer to the question about the SCT note.

### What the pass rule does not test

The rule is `worst depth effect <= 0.05 AND control F1 >= 0.60`, and §4 lists a
third criterion it never applies: the rejection *rate* on arms that contain
nothing to reject. Measured locally at reduced scale, the homogeneous arms are
rejected at:

| | median homogeneous rejection rate |
|---|---:|
| `logcpm`, `logcpm_ds`, `rank256`, `rank256_ds`, `pearson_ds` | 0.076 - 0.087 |
| `logcpm_cos` | 0.030 |
| `rank256_ds_cos` | 0.018 |

So every Euclidean configuration rejects 7-9% of a population with no
incompatible cell in it, and the cosine cost cuts that three- to fourfold. A
configuration can therefore pass both stated columns while still throwing away
8% of cells for no reason. This needs confirming at production scale before it
is relied on, and it is a second independent reason the cosine arms look
better than their depth column alone suggests.

### Where this leaves the choice

The 0.05 threshold falls inside the replicate range of all four leading
configurations: `sct_ds` 0.015-0.054, `pearson_ds` 0.025-0.055,
`scanpy_pearson_ds` 0.038-0.080, `rank256_ds_cos` 0.030-0.083. Which of them
"passes" is not currently a statement about the methods, so the specificity
ranking among them is undecided at 3 replicates.

The one difference that is clearly outside the noise is power. Both cosine
arms reach F1 0.945-0.952 and reject 22.0-22.3% against a true 20%, while
every other configuration reaches 0.845-0.887 and rejects 24.9-27.2%. Two
different transforms, one effect.

Hence configurations 11 and 12, and a 10-replicate confirmation round on the
few that are in contention. Three implementations of "residual transform plus
equalisation" landing within 0.010 of each other -- ours 0.050, scanpy's 0.055,
sctransform's 0.045 -- is the external corroboration the external packages were
added for, and it is reportable whichever configuration is finally chosen.
