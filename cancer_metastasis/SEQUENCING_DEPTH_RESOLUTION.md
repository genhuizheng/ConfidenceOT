# Sequencing depth: what is settled, what is not

The standing answer to "is the depth confound solved?". The screen's design and
its prespecified expectations are in `DEPTH_SCREEN_SETUP.md`; this is the
conclusion drawn from it, and the caveats that come with it.

Short version: **the depth effect can be reduced three- to fivefold by the
cosine cost, and that reduction is the only one that holds across two
independent simulators. Nothing measured so far removes the effect.** The
treatment that looked best for a week — read equalisation — turns out to be
favoured by an artefact of our own simulation.

**And on real data it is not resolved.** The chosen configuration ran on all
four datasets on 2026-09-21 (§8). A third of ovarian pairs and half of prostate
pairs still exceed the bound, the pooled statistic the acceptance test first
named hid that by averaging effects in both directions, and all four datasets
sit at or above the hardest rung the screen ever measured. No differential
expression from these runs.

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
| rank + eq. → **+ cosine** | 0.127 → 0.062 | 0.219 → **0.026** | 0.183 → not measurable |
| Pearson + eq. → **+ cosine** | 0.050 → 0.063 | 0.428 → **0.051** | 0.403 → not measurable |
| sctransform + eq. → **+ cosine** | 0.045 → 0.036 | 0.458 → **0.112** | 0.452 → not measurable |

"Not measurable" is the point of §4b, not a gap: at N=5000 those three reject
one to five cells on arms that should reject none, and an AUC on five cells has
a standard error of 0.13. The numbers 0.122, 0.300 and 0.221 previously written
here were noise, and are withdrawn.

It also costs nothing in power. Both cosine arms reach control F1 0.945–0.958
and reject 21.4–22.3% against a true 20%, where every non-cosine configuration
reaches 0.845–0.887 and rejects 24.9–27.2%. Two different transforms, one
effect, so it is not noise.

### The recommendation

Read on the column that does not break when a method succeeds — the
false-rejection rate of §4b — the three **equalisation + cosine**
configurations lead, and by a wide margin:

| | false reject, N=1500 | false reject, N=5000 | readable depth effect | control F1 |
|---|---:|---:|---:|---:|
| `rank + eq. + cosine` | 0.021 | **0.001** | 0.026 – 0.062 | 0.945 – 0.967 |
| `sctransform + eq. + cosine` | 0.021 | **0.000** | 0.036 – 0.112 | 0.940 – 0.947 |
| `Pearson + eq. + cosine` | 0.020 | **0.001** | 0.051 – 0.063 | 0.924 – 0.954 |
| `log CPM + cosine` | 0.057 | 0.032 | 0.083 – 0.128 | 0.945 – 0.958 |
| everything Euclidean | 0.071 – 0.098 | 0.076 – 0.098 | 0.045 – 0.494 | 0.816 – 0.887 |

They reject essentially nothing where nothing should be rejected *and* hold the
highest control F1 in the table, so this is not a gate that has gone quiet:
it rejects about 21% where the truth is 20%.

`log CPM + cosine` remains the simplest — no equalisation, no rank encoding,
one change to the cost — and it is the one whose depth effect stays readable
everywhere, but it false-rejects two to thirty times more.

**This qualifies §2 rather than reversing it.** Equalisation's apparent benefit
*to the depth AUC* was an artefact of the hand-built construction. Its
contribution *to the false-rejection rate*, in combination with cosine, is
visible on both simulators. But the design cannot attribute it: there is no
`logcpm_ds_cos` arm, so "equalisation helps the rate" is not separated from
"the transform helps the rate". One run would settle that, and it should be run
before the configuration is fixed.

**Nothing clears the prespecified 0.05 on every readable arm of both
simulators.** The honest claim is a large reduction in false rejection and a
three- to fivefold reduction in the depth effect, not a solution.

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

### 4b. Resolved, and it is the cleanest result in the screen

`DEPTH_SCREEN_SETUP.md` §4 lists the rejection *rate* on arms containing
nothing to reject, and the pass rule never applied it. Measured at production
scale across all 39 configurations, two simulators and two sizes, it separates
**perfectly**:

| | median false-rejection rate |
|---|---|
| all 27 Euclidean configurations | **0.071 – 0.098** |
| all 12 cosine configurations | **0.000 – 0.057** |

No overlap anywhere. That is a cleaner separation than the depth-effect column
itself achieves, and it does not depend on which simulator produced the counts.
Within the cosine arms the three that also equalise reach 0.019–0.021 at
N=1500 and **0.000–0.001** at N=5000 — one to five cells out of five thousand,
which is the correct answer.

So every Euclidean configuration discards 7–10% of a population containing
nothing to discard, and the cosine cost cuts that by four- to a hundredfold.

Two consequences:

1. **The depth AUC breaks exactly when a configuration succeeds.** It is an AUC
   of depth against the retained/rejected label, so its standard error is about
   `sqrt(1/(12k))` in `k` rejected cells: 0.17 at one cell, 0.05 at twenty. A
   method that correctly rejects almost nothing therefore reports a *large*
   depth effect made entirely of noise. Checked against the data: grouped by
   how many cells each arm rejected, the observed replicate spread tracks that
   standard error to within a few thousandths (0.152 observed against 0.167
   predicted below ten cells; 0.022 against 0.024 above a hundred). The
   collector now withholds an arm below twenty rejected cells and reports
   `unread` when no arm clears it.
2. **The false-rejection rate should carry the decision, not the AUC.** It is
   directly interpretable, it has no such failure mode, and §4 listed it from
   the start. That is a change to how the prespecified rule is *read*, not a
   change to the threshold, and it is being written down before index 13's
   result is looked at.

One gap this exposes: the design has no `logcpm_ds_cos` arm, so within the
cosine configurations "equalisation helps the rate" cannot be separated from
"the transform helps the rate". One more run would settle it.

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

1. ~~Read the production homogeneous rejection rates~~ — done, §4b, and it
   is the cleanest separation in the screen. The decision it forces, recorded
   before index 13 was looked at: the false-rejection rate carries the reading,
   the depth AUC is secondary and is withheld where it is not estimable.
2. ~~Compute `sd(log depth)` per real dataset and mark which arm is the
   relevant one~~ — done, §8. All four sit at or above the highest rung, and
   every one of them spans more than a rung across its own samples, so none is
   located by a single number.
3. Land index 13 on both constructions and resolve §4a.
4. Ten-replicate confirmation on `logcpm_cos`, `rank256_ds_cos`, `sct_ds_cos`,
   `pearson_ds` — the four that are actually in contention.
5. Add `logcpm_ds_cos`, the arm that separates "equalisation helps the
   false-rejection rate" from "the transform does".
6. ~~Fix the configuration, write it down, and only then re-run the cancer
   analysis~~ — done, §7, ahead of items 2 to 5 and saying so. Items 2 to 5
   remain open and can still change the ranking among the cosine
   configurations; none of them can restore a Euclidean one.

---

## 7. The configuration fixed for the real-data re-run

Written on 2026-09-21, before the run, so the comparison cannot be chosen after
seeing it. This is item 6 of §6 executed with items 2–5 still open, which is a
decision and not an oversight; what it rests on and what it does not is below.

### The configuration

    rank256_ds_cos

That string *is* the configuration. `confidenceot.Preprocessing` prints it for
rank encoding at 256 genes per cell, read equalisation, and the cosine cost, and
parses it back — so the same name reaches the pair runner (`--preprocessing`),
the rank-cut audit, the output directory and the run record. Setting the three
axes separately is still possible and is no longer how the chain does it: three
variables that can disagree is how a rank cut gets set while the transform stays
at its default, producing a run with an ignored cut and no name.

The equalisation happens at the file level, in
`27_downsample_counts.py --target-quantile 0.10`, so that the OT, the
pseudobulk, the differential expression and the scoring all read one matrix. The
`_ds` in the label is therefore *recorded* by the run, not applied by it; without
that, a depth-equalised cosine rank run would record itself as `rank256_cos` and
be indistinguishable from one on untouched counts.

Submitted by `cancer_metastasis/submit_rank_cosine_chain.sh`, one dataset per
invocation, into `ot_<accession>_rank256_ds_cos_<stamp>`. Every existing output
root is untouched, so the two configurations are read side by side rather than
one overwriting the other.

Four datasets: GSE180661 (ovarian), GSE271675 (prostate), GSE225857
(colorectal), GSE181919 (head & neck). The first two already have equalised
counts and reuse them; re-running the quantile on a different cell set would
change the target depth as well, and two runs would then differ in two ways at
once. The last two are equalised for the first time by the same chain.

### Why this one

`rank + eq. + cosine` leads the false-rejection column at both sizes (0.021 and
0.001) and holds the highest control F1 in the screen (0.945–0.967). The
simpler `log CPM + cosine` was the alternative and was rejected on that one
column: it false-rejects 0.057 and 0.032, two to thirty times more. On a method
whose entire output is a partition, the rate at which it rejects cells that
should not be rejected is the quantity that decides whether the partition means
anything.

### What it does not rest on

- **Three replicates.** §4d: the tolerance falls inside the replicate scatter
  for the contenders. The ten-replicate round (§6 item 4) has not run. The
  ranking among the three cosine configurations is therefore not established;
  the separation of cosine from Euclidean is.
- **Equalisation is carried unattributed.** §3: there is no `logcpm_ds_cos`
  arm, so "equalisation helps the rate" is not separated from "the transform
  does". It is kept because it is cheap, decoupled, and preserves integer
  counts, not because the screen established it. Should item 5 show the
  transform carries the effect, the equalisation stage drops out and the OT
  configuration is unchanged.
- **§4a is unresolved.** The two external residual methods swap places between
  the simulators. Neither is in this configuration, so it does not bear on it.

### One more thing changed, and it changed a number

The screen's read equalisation drew over each cell's full gene vector; the
production stage draws over its nonzero genes. The two are the same distribution
— a colour with zero balls contributes nothing — but they consume the random
generator differently, so they were not the same operation, and the screen's
conclusion only transfers if they are. They are now both the production form,
which is also the only one possible on a real matrix: densifying 150,000 cells
by 30,000 genes to subsample them is not an option.

That moves the simulated arms where equalisation does something. The arms where
it does not — `perturbed_depth_cv0`, at zero depth spread — are unchanged, which
is the consistency check. The equalised objects already on disk are unaffected:
the production draw is bit-identical to the one that wrote them, pinned by
`tests/test_preprocessing.py`.

Past screen runs were already unreproducible for a separate reason: the per-arm
seed was `abs(hash(arm)) % 10_000`, and Python randomises string hashing per
process, so no two invocations ever ran the same configuration. It is a stable
digest now. The table above survives that — it is an average over replicates and
arms, and the cosine/Euclidean separation has no overlap to flip — but the
individual numbers in it cannot be re-derived exactly, which is the same
limitation §4d already states.

### The new risk, which is specific to real data

`rank_value` keeps each cell's top 256 genes, and `rank_value_encode` keeps
`min(256, detected)` of them. A cell detecting fewer than 256 genes is encoded
on fewer coordinates, which is the low-content artefact the fixed cut exists to
prevent. Equalised counts are where that is least safe: every cell has been
subsampled to the 10th percentile of pooled depth. The screen ran at 256 on
simulated cells with 4,000 genes and near-uniform detection, which is no
evidence at all about these cells — and GSE225857's rejected cells already sat
at a median of 4,243 counts before any equalisation.

`cancer_metastasis/tools/audit_rank_top_n.py` measures this on the equalised
manifest, over the gene intersection the run uses and after the same annotation
scoping, and the chain makes it a **gate**: more than 1% of any pair's cells
below the cut and the OT is never submitted. It reports the largest cut that
would clear the tolerance, so a failure is actionable rather than terminal.
Under the cosine cost a short vector's smaller norm is normalised away, so the
magnitude half of the artefact goes; the support difference does not.

### Acceptance test, prespecified

Run by the chain's last stage, against each cell's **original** depth from
`predownsample_depth.csv.gz` — after equalisation the stored depth is nearly
constant, so testing the gate against it would be trivially 0.5 and prove
nothing.

| statistic | current | acceptance |
|---|---:|---:|
| `auc_predownsample_total_counts` | 0.67 (ovarian), 0.79 (colorectal) | \|AUC − 0.5\| ≤ 0.10, **median over pairs** |
| `spearman_decision_cost_predownsample_total_counts` | −0.33 to −0.59 | \|rho\| ≤ 0.20 |
| `depth_residual_gate_jaccard` | 0.47–0.60 | ≥ 0.75 |
| source rejection rate | — | reported, not constrained |

**Two corrections to that first row, both made after it was written and before
the numbers were read.** It said `auc ≤ 0.60`, which is one-sided: an AUC of
0.35 clears it while tracking depth exactly as hard in the other direction, so
the criterion has to be the distance from the null. And it said nothing about
*which* AUC, which the tooling then took to mean the pooled one — see §8, where
that choice turned out to hide the answer entirely. It is the median of the
per-pair deviations on any dataset with at least six pairs, and the pooled
value only below that, where there is no distribution to form.

0.60 rather than 0.55 because the screen's own best readable depth effect was
0.026–0.062 on simulated data with a known answer, and no real dataset has yet
been placed on that depth ladder (§4c). A result between 0.60 and 0.67 is a
partial reduction and should be reported as one, not as a pass.

The rejection rate is deliberately unconstrained. §5.8.4 of
`CURRENT_WORK_SUMMARY_2026-09-13.md` records that the retained fraction is
close to inversely proportional to how far apart the two sides are, so a target
rate would be a target for the wrong quantity.

**No differential expression from these runs until the three statistics above
are read.** The prostate arm is the precedent: 996 significant genes raw, 3
after equalisation. A DEG computed before the acceptance test is a number that
cannot be withdrawn once it has been seen.

### What the rank-cut audit actually found, 2026-09-21

All four datasets cleared a cut of 256 with no cell encoded short. The margins
are the interesting part:

| dataset | cells equalised | pairs | target depth | largest cut every cell fills |
|---|---:|---:|---:|---:|
| GSE180661 ovarian | — (reused) | 20 | 3,119 | **282** |
| GSE271675 prostate | — (reused) | 23 | — | 501 |
| GSE225857 colorectal | 23,954 | 5 | 2,455 | (see log) |
| GSE181919 head & neck | 810 | 4 | 4,933 | 981 |

Two things follow.

**Ovarian clears 256 by 26 genes.** The pipeline's own default is
`--rank-top-n 512`, and at 512 that dataset would have been encoded short — some
cells on fewer coordinates than others, which is the low-content artefact the
fixed cut exists to prevent. The audit was worth the job number.

**Prostate's floor of 501 is a QC threshold, not a property of the data.** Every
one of its pairs reports `detected_min` of exactly 501, because the object was
prepared with `--minimum-detected-genes 500`. Anyone reading `safe_top_n = 501`
as "this data supports a cut of 500" would be reading the filter.

### The acceptance test is not estimable on GSE181919, and I should have checked

§7 prescribes one threshold — `auc_predownsample_total_counts ≤ 0.60` — for all
four datasets. It was written without checking that any of them could carry it.
GSE181919 cannot.

That dataset has **810 malignant cells in total**, across four pairs of 89, 112,
182 and 427 cells counting both sides. The depth AUC is an AUC of depth against
the retained/rejected label, so its standard error runs about `sqrt(1/(12k))` in
`k` rejected cells. Splitting each pair evenly and assuming a 20% rejection rate:

| pair | cells | source | rejected `k` | s.e.(AUC) | 0.10 expressed in s.e. |
|---|---:|---:|---:|---:|---:|
| P22 | 182 | 91 | 18 | 0.068 | 1.5 |
| P38 | 112 | 56 | 11 | 0.086 | 1.2 |
| P46 | 427 | 214 | 43 | 0.044 | 2.3 |
| P59 | 89 | 44 | 9 | 0.097 | 1.0 |

The distance between the null and the acceptance threshold is one to two
standard errors. Three of the four pairs cannot distinguish a pass from a
failure, and a per-pair verdict on them would be a coin toss wearing a
threshold.

**Amendment, recorded before the numbers exist.** For GSE181919 the acceptance
test is evaluated on the gate **pooled across pairs** — 810 cells, about 81
rejected, s.e. 0.032, which does resolve 0.10 — and the per-pair statistics are
reported without a verdict. This is not a free move:
`25_diagnose_gate_covariates.py` aggregates per-pair statistics by their median
and tests them with a Wilcoxon that needs at least six values, so on four pairs
it produces neither a usable median nor a p-value. The pooled figure has to be
computed separately when the results land; the per-pair file it writes carries
what is needed.

The same arithmetic passes the other three: GSE225857 has 23,954 cells over five
pairs, and ovarian and prostate have 20 and 23 pairs. The threshold stands for
them unchanged.

**What this does not license.** Pooling is the right estimator for a small
dataset and it is also the one that hides heterogeneity between pairs, which is
the thing the per-pair view exists to show. A pooled GSE181919 result at 0.55
means "this dataset, taken together, still tracks depth a little"; it does not
mean every pair does, and the four per-pair numbers should be printed beside it
so a reader can see the spread they are not being asked to interpret.

---

## 8. What the run found: the depth confound is not resolved

All four datasets ran to completion on 2026-09-21 under `rank256_ds_cos`: 127
pairs, every job `COMPLETED 0:0`, the rank cut held on all four (§7), and the
configuration reached the pipeline by name rather than by three variables.

**The answer is no, and the statistic in §7 as first written would have said
yes.**

### Read on the per-pair deviations

| | pairs | pooled AUC | median per-pair \|AUC − 0.5\| | pairs over 0.10 | worst pair | retained |
|---|---:|---:|---:|---:|---:|---:|
| GSE180661 ovarian | 94 | **0.497** | 0.061 | **34%** | 0.417 | 32% |
| GSE271675 prostate | 24 | 0.562 | 0.101 | **50%** | 0.429 | 11% |
| GSE225857 colorectal | 5 | 0.776 | 0.231 | 80% | 0.362 | 71% |
| GSE181919 head & neck | 4 | 0.536 | 0.031 | 25% | 0.309 | 54% |

Ovarian pools to 0.497 — as clean a null as one could ask for — while its 92
estimable pairs run from **0.18 to 0.92**. The pooled figure averages a gate
favouring deep cells in one pair against a gate favouring shallow cells in
another, and that average is not a measurement of whether the gate tracks
depth. Prostate is worse: pooled 0.562, but the *mean* of its per-pair AUCs is
0.437, so the two disagree about the direction.

Colorectal is the one dataset where nothing cancels — its five pairs read 0.52,
0.63, 0.73, 0.74, 0.86, all on the same side — and it is also the one that fails
by any reading, with rho(cost, depth) = −0.41 and retained cells 2.2× deeper
than rejected ones.

Head and neck is reported without a verdict: 810 cells over four pairs, for the
reason recorded in §7 before the numbers existed.

### Two defects in my own measurement, both found by reading the result

The standard error was `sqrt(1/(12k))` with `k` the rejected count. That
approximates the exact Mann-Whitney null error only when `k` is the *smaller*
class, and a gate rejecting 89% of cells makes it the larger one: the reported
error was 3.0× too small on prostate and 1.8× on ovarian. It is
`sqrt((n1+n2+1)/(12·n1·n2))` now. No verdict moved, but the estimability check
depends on it.

And the acceptance criterion was one-sided and silent about which AUC. Both are
corrected in §7 with the reasoning; the second is what let the pooled statistic
be used at all.

### The retention rates are implausible

Against the stated expectation that no more than about 15% of primary malignant
cells carry metastatic potential:

- prostate retains **11%** — on target,
- ovarian **32%**,
- head and neck **54%**,
- colorectal **71%**, and two of its five pairs retain 88% and 71%.

This reverses an earlier reading of mine. I had called prostate's 89% rejection
rate the "returns almost nothing" pathology of §5.8.4; against a ≤15% prior it
is the expected magnitude, and the datasets in trouble are the two that retain
most. Colorectal both retains 71% *and* has the worst depth signal, with its
retained cells 1.22× deeper in the two pairs measured so far — so a large part
of what that 71% selects is depth.

The bound is reported and not tested. It is an expectation about the biology,
not something these data established, and choosing a configuration because it
reproduces the expected rate would be selecting on the answer, which §5.1 of
`CURRENT_WORK_SUMMARY_2026-09-13.md` sets as a standard for exactly this
situation.

### The real data is harder than anything the screen measured

The screen's arms sit at `sd(log depth)` of 0.0, 0.3, 0.6 and 0.9, and every
conclusion in §3 is stated per rung. Measured on the equalisation step's own
pre-equalisation depths (§4c, now closed):

| | pooled | per-sample range | nearest rung |
|---|---:|---|---|
| GSE180661 | 0.759 | 0.50 – **1.41** | high (0.9) |
| GSE271675 | 0.916 | 0.60 – **1.30** | high (0.9) |
| GSE225857 | 0.682 | 0.49 – 0.96 | mid (0.6) |
| GSE181919 | 0.782 | 0.38 – 1.07 | high (0.9) |

Two things follow. Every dataset sits at or above the screen's **hardest** rung,
so §3's numbers at "moderate spread" never described this data. And every
dataset's samples span **more than one rung**, several reaching past 0.9 into a
regime the screen never tested at all — a pair is two samples seen jointly, so
those datasets are not located by any single number and the per-sample rows are
the ones to read.

### What this does and does not change

It does not overturn §3. Cosine still separated from Euclidean on both
simulators, and nothing here restores a Euclidean configuration. What it
removes is the inference from "best of the screened configurations" to
"sufficient on real data": the best screened configuration leaves a third to a
half of pairs over the bound, in a depth regime harder than the screen ever
reached.

It also means **no differential expression from these runs**, which §7 already
required and which now has a measured reason rather than a precautionary one.

### What would move this

1. The dose-response, which the tooling now computes: per-pair depth deviation
   against that pair's own `sd(log depth)`. A correlation would say the residual
   effect is a property of how hard the data is, and the ladder predicts it; no
   correlation would say the cosine cost is not addressing the cause and
   something else is.
2. The per-pair UMAPs, for the pairs at the extremes. If a 0.92-AUC pair's
   retained set is visibly the deep lobe of its own embedding, that is the
   mechanism; if it is not, the AUC is picking up something the geometry does
   not show.
3. A screen arm above 0.9, because three of four datasets sit there and it has
   never been measured.
4. The retention rate needs its own treatment, separate from depth. A gate
   keeping 71% of primary cells is making a different error from one that
   tracks depth, and no configuration tested so far addresses it.
