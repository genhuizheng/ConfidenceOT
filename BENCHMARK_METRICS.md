# Benchmark metrics

What the nuisance benchmark returns as numbers, and what each number has to
read for the method to have passed. The figures state the *setting* — what is
generated and what the correct answer is; this file states the *scoring*.

Companion figures: `benchmark_results/nuisance/problem.png` (what the two
nuisances do to an embedding) and `benchmark_results/nuisance/benchmark.png`
(the pipeline the metrics are computed on).

## Notation

One pair is a primary sample `S` with `n` cells and a matched metastasis `T`
with `m` cells. A method returns

- a coupling `π ∈ R^{n×m}`, how much of primary cell `i` is carried onto
  metastatic cell `j`;
- a per-cell decision cost `c_i`, the continuous statistic the gate thresholds;
- a gate `ŷ_i ∈ {keep, reject}`, one call per primary cell.

The ground truth is `y_i ∈ {keep, reject}`. `x_i` is the cell's value of the
nuisance under test — `nCount` (total counts) in the depth arm, `nFeature`
(genes detected) in the gene-detection arm.

## Arm types

**Technical arms.** The biology is identical on both sides: same populations,
same proportions, nothing missing. Only the measurement differs. `y_i = keep`
for every cell, so every rejection is an error and the whole score is about
false positives.

**Biological arms.** A population is present on one side and absent from the
other (S1 extinction, S2 emergence). `y_i = reject` on exactly those cells.
Both kinds of error are possible, so the score is precision and recall.

Both are needed. A method that never rejects is perfect on the technical arms;
a method that rejects everything is perfect on biological recall. Either arm
alone defines a task that can be passed by doing nothing.

## Technical arms — the metrics

| metric | definition | target | column | emitted by |
|---|---|---|---|---|
| false rejection rate | `#{ŷ_i = reject} / n` | **0** | `source_rejection_rate` | pending the rebuilt benchmark |
| retained fraction under the sign rule | fraction with `c_i < c`, before any budget is enforced | **1** | `sign_rule_retained_fraction` | `cancer_metastasis/25_diagnose_gate_covariates.py` |
| nuisance AUC | Mann–Whitney AUC of `x_i` separating the two gate classes, **positive class = retained** | **0.5** | `auc_total_counts`, `auc_detected_genes` | `25_diagnose_gate_covariates.py` |
| cost–nuisance correlation | Spearman(`c_i`, `x_i`) over all primary cells | **0** | `spearman_decision_cost_total_counts`, `..._detected_genes` | same |
| coupling–nuisance correlation | Spearman(`x_i`, `x̄_i`) where `x̄_i = Σ_j π_ij x_j / Σ_j π_ij` is the nuisance barycentre of where cell `i` is sent | **0** | see note below | `cancer_metastasis/26_diagnose_pairing_quality.py` |
| depth-residual gate agreement | Jaccard between the gate and a same-size gate built on the cost with the nuisance ranks regressed out | **1** | `depth_residual_gate_jaccard` | `25_diagnose_gate_covariates.py` |

Note on the coupling metric: what is implemented today is the
reciprocal-dominant-edge and nearest-neighbour variant —
`spearman_reciprocal_edge_depth`, `spearman_nn_partner_depth` — which
thresholds `π` to one partner per cell. The barycentre form above uses all of
`π` and is the one to add; the edge form stays as the interpretable version.

### Why this many

Each one is defined where another one is blind.

- **The false rejection rate alone is uninformative while the budget cap is
  enforced.** The cap forces cells back in until the rejection rate equals it,
  so the metric reports the parameter rather than the method. Any cap value
  yields a rejection rate equal to that cap.
- **The nuisance AUC survives that**, because it is a property of *which*
  cells were rejected, not how many, and it is still defined when the count is
  pinned.
- **The cost–nuisance correlation is defined even when nothing is rejected at
  all**, because it is the pre-threshold version of the same question.
- **The coupling correlation is the only one that tests OT rather than the
  gate.** A method can have a clean gate and still be transporting along the
  nuisance, and that failure is invisible to every gate-level metric.
- **The depth-residual Jaccard separates "the nuisance correlates with the
  gate" from "the nuisance decides the gate"**: chance overlap for two sets of
  size `k` from `n` cells is about `k / (2n - k)`, so a value near that floor
  means removing the nuisance component selects different cells.

## Biological arms — the metrics

| metric | definition | target | column |
|---|---|---|---|
| recall | fraction of truly unmatched cells that were rejected | **1** | `perturbed_recall` |
| precision | fraction of rejected cells that are truly unmatched | **1** | `perturbed_precision` |
| F1 | harmonic mean of the two | **1** | `perturbed_f1` |
| separation AUC | AUC of the cost separating unmatched from matched cells | **1** | `auc_perturbed_rejected` |
| false rejection rate on matched cells | as in the technical arms, restricted to cells that do have a counterpart | **0** | derived from the gate and the mask |

## How the metrics are reported

Not as one number per method. Each metric is a **dose–response curve**:

- x axis: perturbation strength. `depth_sigma`, the sd of log depth, from 0 to
  0.9 in the depth arm; the spread of genes detected at fixed total counts in
  the gene-detection arm.
- one line per preprocessing: `logcpm`, `rank256`, `rank256_ds`,
  `rank256_ds_cos`.
- a method is invariant if its curve is **flat at the target**, not merely low
  at one setting.

This is also what answers the question of whether the read-equalisation step
earns its place: **the value of DS is the gap between the `rank256` and
`rank256_ds` curves**, and in the gene-detection arm that gap is expected to
be zero, because the totals DS divides by already agree.

## What the metrics currently read

Recorded in `CURRENT_WORK_SUMMARY_2026-09-13.md` §5. On the three cancer
datasets, with the rotation null and the budget cap in force:

| | GSE180661 | GSE181919 | GSE225857 |
|---|---:|---:|---:|
| `spearman_decision_cost_total_counts` | −0.330 | −0.593 | −0.568 |
| depth ratio, retained / rejected | 1.32× | 1.97× | 3.07× |

The transport plan pairs cells by depth at `rho = 0.64` on
reciprocal-dominant edges and 0.59 on nearest neighbours, in the retained and
rejected subsets alike. Removing the depth component changes 25–36% of the
retained set (`depth_residual_gate_jaccard` 0.47–0.60 against a chance floor
of 0.08–0.18).

So the metrics are not decorative: three of them currently fail on real data,
and they are the acceptance test for the calibration change described in the
session plan (replacing the cross-side rotation null with a within-side
split, and reporting the budget instead of enforcing it).

## What invalidates a reading

- **A cap in force pins the false rejection rate.** Read it only with the
  budget reported rather than enforced, or read the other metrics instead.
- **Downsampled counts make the stored depth near constant**, so
  `auc_total_counts` against it is trivially 0.5 and proves nothing. The
  diagnostics must join `predownsample_depth.csv.gz` and test the gate against
  each cell's *original* depth; `depth_residual_gate_jaccard` already prefers
  `predownsample_total_counts` when the column is present.
- **A dataset with no `MT-` genes** has a constant zero mitochondrial
  percentage, so any QC claim resting on that threshold does not hold there
  (GSE225857).
- **Two nuisances varied at once** makes a failure unattributable. The
  gene-detection arm holds total counts fixed for exactly this reason.
