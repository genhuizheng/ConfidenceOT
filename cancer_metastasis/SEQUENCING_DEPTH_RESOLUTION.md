# Sequencing depth: what is settled, what is not

The standing answer to "is the depth confound solved?". The screen's design and
its prespecified expectations are in `DEPTH_SCREEN_SETUP.md`; this is the
conclusion drawn from it, and the caveats that come with it.

Short version: **the depth effect can be reduced three- to fivefold by the
cosine cost, and that reduction is the only one that holds across two
independent simulators. Nothing measured so far removes the effect.** The
treatment that looked best for a week — read equalisation — turns out to be
favoured by an artefact of our own simulation.

---

## 1. The problem, on real data

Established by `cancer_metastasis/25_diagnose_gate_covariates.py` and
`26_diagnose_pairing_quality.py` before any simulation existed.

| | GSE180661 (ovarian) | GSE181919 (head & neck) | GSE225857 (colorectal) |
|---|---:|---:|---:|
| rho(decision cost, depth) | −0.33 | −0.59 | −0.57 |
| retained : rejected depth | 1.32× | 1.97× | 3.07× |
| `auc_total_counts` | 0.67 | — | 0.79 |

The gate itself is a faithful threshold on `decision_cost` (AUC 0.000,
sign-rule concordance 0.997), so this is not a solver defect: the cost geometry
carries depth. Removing the depth component changes 25–36% of the retained set,
Jaccard 0.47–0.60 against a chance level of 0.08–0.18.

Two mechanisms, roughly equal:

- a **partial depth axis** in the joint PCA, `max_abs_spearman_pc_depth` = 0.60,
- **depth-graded dispersion**, rho(depth, nearest-neighbour distance) = −0.41.

That split is why no single correction was expected to suffice, and it is the
reason the cosine cost was worth testing at all: cosine removes magnitude,
which is one of the two.

The gene-level consequence is the headline number for anyone reading the old
results: on prostate, the ranked free-cost arm gave **996 significant genes
raw, and 3 with read equalisation**. Ninety-nine point seven percent of that
signal was depth.

---

## 2. The simulation, and the trap in it

`scripts/validate_depth_null_specificity.py` builds data whose correct answer
is known: four arms differing **only** in how widely per-cell depth is spread,
containing no incompatible cell, so any dependence of the gate on depth is a
specificity failure by construction. A fifth arm plants a real 20%
subpopulation as the positive control.

On that construction, read equalisation looked decisive:

| transform | alone | + equalisation |
|---|---:|---:|
| log CPM | 0.494 | 0.258 |
| rank-value | 0.363 | 0.127 |
| scanpy Pearson residual | 0.488 | **0.055** |
| sctransform v2 | 0.071 | **0.045** |

(worst-arm depth effect, `abs(AUC − 0.5)`, production scale, 3 replicates)

**It is an artefact of the construction.** In that simulation
`counts | depth ~ Multinomial(depth, p)` with `p` drawn independently of depth,
and multivariate-hypergeometric downsampling of a multinomial draw to a common
total is again `Multinomial(D, p)`. Read equalisation is the **exact inverse**
of the mechanism that simulation uses to create depth variation. The null was
therefore grading a correction against the same assumption it was built on.

This is a limitation of that construction and belongs in the paper as one.

---

## 3. What splatter says

The same arms rebuilt with `splatter::splatSimulateGroups`, depth set by the
package's own `lib.loc` and `lib.scale` — the same lognormal parameterisation,
verified against the source and measured at 5,000 cells (realised
sd(log depth) 0.3012 / 0.6024 / 0.9036 for `lib.scale` 0.3 / 0.6 / 0.9, where
the standard error is 0.009).

Splatter applies its biological coefficient of variation **after**
library-size scaling, and the BCV falls with the mean, so the relative profile
itself depends on depth. Equalising the totals cannot undo that; it only
discards reads. The sign flips:

| transform | alone | + equalisation |
|---|---:|---:|
| log CPM | 0.490 | 0.448 |
| rank-value | 0.153 | **0.219** |
| scanpy Pearson residual | 0.074 | **0.430** |
| sctransform v2 | 0.386 | **0.458** |

### The cosine cost is what transfers

Every configuration improves under it on splatter, by large factors:

| | hand-built | splatter | splatter, N=5000 |
|---|---:|---:|---:|
| log CPM + eq. → **+ cosine** | 0.258 → 0.128 | 0.448 → **0.083** | 0.409 → **0.089** |
| rank + eq. → **+ cosine** | 0.127 → 0.062 | 0.219 → **0.031** | 0.183 → 0.122 |
| Pearson + eq. → **+ cosine** | 0.050 → 0.063 | 0.428 → **0.051** | 0.403 → 0.300 |
| sctransform + eq. → **+ cosine** | 0.045 → 0.036 | 0.458 → **0.112** | 0.452 → 0.221 |

It also costs nothing in power. Both cosine arms reach control F1 0.945–0.958
and reject 21.4–22.3% against a true 20%, where every non-cosine configuration
reaches 0.845–0.887 and rejects 24.9–27.2%. Two different transforms, one
effect, so it is not noise.

### The recommendation, and it is the simple one

`log CPM + cosine` is the most stable configuration across all three settings —
worst arm **0.083 / 0.128 / 0.089**, control F1 0.945–0.958 — and it is the
simplest: no read equalisation, no rank encoding, one change to the cost.
`rank-value + equalisation + cosine` reaches a lower best case (0.031 on
splatter at N=1500, the only configuration to clear 0.05 there) but is less
consistent (0.122 at N=5000).

**Nothing clears the prespecified 0.05 on every arm of both simulators.** The
honest claim is a three- to fivefold reduction, not a solution.

---

## 4. What is not settled

### 4a. The two external residuals swap places

Used alone, without equalisation:

| | hand-built | splatter |
|---|---:|---:|
| scanpy Pearson residual | 0.488 | **0.074** |
| sctransform v2 | **0.071** | 0.386 |

A clean reversal in both directions. It is explicable in principle — a fixed
theta analytic residual against a regularised negative-binomial fit — but a
coincidence this tidy should not enter a paper unchecked.

Two checks are running or queued:

1. **Array index 13, `pearson`** — our own analytic residual with no
   equalisation, the one empty cell in the design. Every other family has an
   "alone" arm; ours did not, so there was no way to ask whether our analytic
   residual behaves like scanpy's. If it does, the analytic-versus-fitted
   distinction is real. If it does not, an implementation is doing something
   unexpected and the swap is not a property of the transforms.
2. **A transform-level diagnostic** measuring depth dependence in the residual
   matrix itself — rho(depth, PC1..PC3), row mean, row norm — on both
   constructions, so a real difference between the residuals separates from
   something the HVG selection, the PCA or the calibration does downstream.

### 4b. The pass rule ignores a criterion it lists

`DEPTH_SCREEN_SETUP.md` §4 lists the rejection *rate* on arms containing
nothing to reject, and the rule never applies it. Measured locally at reduced
scale:

| | median homogeneous rejection rate |
|---|---:|
| every Euclidean configuration | 0.076 – 0.087 |
| log CPM + cosine | 0.030 |
| rank + eq. + cosine | 0.018 |

Every Euclidean configuration discards 7–9% of a population with no
incompatible cell in it, and the cosine cost cuts that three- to fourfold. So a
configuration can pass both stated columns while throwing away 8% of cells for
no reason — a second, independent reason the cosine arms look better than their
depth column alone says. **Unconfirmed at production scale**, and whether it
joins the pass rule has to be decided before the outstanding configurations are
read, not after.

### 4c. Nobody has located the real data on the ladder

The four arms are a dose–response ladder at `depth_sigma` 0 / 0.3 / 0.6 / 0.9,
realised CV 0 / 0.31 / 0.66 / 1.12, p90/p10 read ratios 1.0 / 2.2 / 4.6 / 10.1.
The values were **chosen, not fitted** — the code says so.

The 1.32× / 1.97× / 3.07× figures above are ratios between two gate groups, not
distribution widths, so they do not place a dataset on this ladder. The
per-dataset `sd(log depth)` is directly computable from the stored
`cell_confidence.csv` files and has not been computed. It decides the
recommendation: `sctransform + equalisation` is strongest at the low-spread end
on the hand-built construction, `rank + eq. + cosine` at the high-spread end on
splatter.

### 4d. Three replicates cannot separate the leaders

The 0.05 threshold falls inside the replicate range of all four leading
configurations: `sct_ds` 0.015–0.054, `pearson_ds` 0.025–0.055,
`scanpy_pearson_ds` 0.038–0.080, `rank256_ds_cos` 0.030–0.083. Which of them
"passes" is not currently a statement about the methods.
`CONFIDENCEOT_REPLICATES=10` exists for a confirmation round on the few in
contention, and writes to its own label so it cannot overwrite what it confirms.

---

## 5. Consequences beyond the screen

- **The production cancer pipeline is not depth-clean.** `rank-value +
  equalisation` is what the cancer workflow runs, and it leaves 0.127 at the
  highest depth spread on the hand-built construction and 0.219 on splatter.
  Any conclusion resting on the current gate inherits that.
- **`calibration_valid` carries no information.** False in 584 of 585
  production runs, driven entirely by `m4r_validation_clean`, which is gate
  cycle detection in held-out M4-R validation fits. Cycling reproduces at 2, 5,
  10 and 30 dimensions and does not resolve with 200 outer iterations, while
  cost selection succeeded 585/585. If that flag appears in a table or is used
  as a filter it will reject everything.
- **The mouse-embryo study is on the older calibration**, pinned deliberately.
  If depth handling changes, whether to re-run it is a live question; the data
  and 43 finished pairs are still on TACC.

---

## 6. What would close this out

1. Read the production homogeneous rejection rates, and decide whether §4b
   joins the pass rule — **before** reading index 13.
2. Compute `sd(log depth)` per real dataset and mark which arm is the relevant
   one.
3. Land index 13 on both constructions and resolve §4a.
4. Ten-replicate confirmation on `logcpm_cos`, `rank256_ds_cos`, `sct_ds_cos`,
   `pearson_ds` — the four that are actually in contention.
5. Fix the configuration, write it down, and only then re-run the cancer
   analysis.
