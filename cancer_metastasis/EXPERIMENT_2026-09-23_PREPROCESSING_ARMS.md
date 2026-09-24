# Three preprocessing arms on the four real datasets, 2026-09-23

What the simulation established, what is running on the real data, and what
counts as which answer. Sections 1 to 3 are results already in hand, all of
them from simulation. Sections 4 to 6 were written and committed **before any
real-data arm returned**, which is the only thing that makes a prespecified
reading worth anything. Section 7 is the empty slot they go into.

The standard section 5.1 of the project summary sets applies here as it does to
the DEG: **a configuration must not be chosen because it produces a more
interesting biological result.** Every readout below is a covariate statistic
or a detection rate against known ground truth. None is a gene list, and no
differential expression is run from any of these arms.

Full derivations and caveats: `SEQUENCING_DEPTH_RESOLUTION.md` section 9. The
screen's design and its original expectations: `DEPTH_SCREEN_SETUP.md`.

---

## 1. The three arms

All three share one gene-key resolution, one pair list per dataset, one
calibration null (`within_side_split`), one budget (source 0.85, target 0.00)
and one evaluation. They differ only in the preprocessing configuration, named
by the label `confidenceot.Preprocessing` prints and parses back.

| arm | label | what it adds | why it exists |
|---|---|---|---|
| **primary** | `rank256_ds_cos` | rank 256, read equalisation, cosine | the shipped configuration, see 2f |
| **sensitivity** | `rank256_rg-genes_ds_cos` | + detected-gene regress-out | 2b: the primary does not act on detection breadth. Reported beside the primary, not instead of it |
| no-equalisation | `rank256_cos` | − read equalisation | 2c below. **Demoted after 2c and 2e**: see the note under section 4 |

The baseline is not re-run. Its 2026-09-21 outputs are the comparison.

---

## 2. What the simulation established

### 2a. The method comparison

Splatter counts, three sizes, anomaly scenarios S1-S5, directional F1.

**No batch effect**

| method | N=1000 | N=5000 | N=10000 |
|---|---:|---:|---:|
| ConfidenceOT | **0.935** | **0.957** | **0.971** |
| Partial OT | 0.884 | 0.893 | 0.905 |
| Vanilla UOT | 0.311 | 0.334 | 0.348 |
| Traditional OT | 0.000 | 0.000 | 0.000 |

**Mild batch effect**

| method | N=1000 | N=5000 | N=10000 |
|---|---:|---:|---:|
| ConfidenceOT | **0.765** | **0.852** | **0.843** |
| Partial OT | 0.698 | 0.790 | 0.776 |
| Vanilla UOT | 0.236 | 0.256 | 0.260 |
| Traditional OT | 0.000 | 0.000 | 0.000 |

Traditional OT scores exactly zero because it is balanced: it transports every
cell and so rejects none, by construction rather than by failure.

Specificity at N=10000 is where the comparison is decided. Each method's
rejection signal on the planted anomaly, against its rejection on populations
containing nothing to reject:

| method | planted anomaly | worst unaffected population |
|---|---:|---:|
| ConfidenceOT | 0.97 | **0.01** |
| Partial OT | 0.96 | 0.09 |
| Vanilla UOT | 0.47 | 0.21 |
| Traditional OT | 0.00 | 0.00 |

Partial OT matches ConfidenceOT on detection and is nine times worse on false
rejection, which is the whole point of the comparison: recovering the planted
group is easy, and not rejecting everything else is not.

### 2b. The preprocessing ablation

Twelve configurations, one evaluation pipeline, 1,500 cells per side, 4,000
genes, 2,000 HVG, 30 PCs, 3 replicates, 11 arms.
`benchmark_results/ablation_20260923/screen_ablation.csv`.

`gate/*` and `axis/*` are `|AUC − 0.5|` and the leading-axis correlation
against the named covariate, lower is better. `F1 *` is power, higher is
better. The `split` columns are the fixed-count breadth arm.

| configuration | F1 clean | F1 deep | F1 breadth | F1 breadth split | gate/counts | gate/genes | axis/counts | axis/genes | axis/genes split |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| log CPM | 0.880 | 0.249 | 0.447 | 0.217 | 0.495 | 0.439 | 0.933 | 0.907 | 0.926 |
| rank | 0.856 | 0.718 | 0.791 | 0.532 | 0.365 | 0.053 | 0.974 | 0.084 | 0.726 |
| log CPM + DS | 0.880 | 0.803 | 0.447 | 0.217 | 0.291 | 0.439 | 0.200 | 0.907 | 0.926 |
| log CPM + cos | 0.943 | 0.684 | 0.780 | 0.245 | 0.115 | 0.183 | 0.945 | 0.890 | 0.864 |
| rank + DS | 0.856 | 0.802 | 0.791 | 0.532 | 0.157 | 0.053 | 0.072 | 0.084 | 0.726 |
| rank + cos | 0.963 | 0.837 | 0.950 | 0.651 | **0.014** | 0.043 | 0.969 | 0.085 | 0.730 |
| log CPM + DS + cos | 0.943 | 0.930 | 0.780 | 0.245 | 0.053 | 0.183 | 0.198 | 0.890 | 0.864 |
| **rank + DS + cos** | 0.963 | 0.934 | 0.950 | 0.651 | 0.057 | 0.043 | **0.073** | **0.085** | 0.730 |
| rank + DS + cos, dr10 | 0.946 | 0.952 | 0.951 | 0.671 | 0.039 | 0.030 | 0.054 | 0.143 | 0.700 |
| rank + DS + cos, dr25 | 0.949 | 0.945 | 0.935 | 0.604 | 0.063 | 0.002 | 0.059 | 0.079 | 0.698 |
| rank + DS + cos + rg | 0.948 | 0.934 | 0.950 | **0.686** | 0.037 | 0.034 | 0.066 | 0.011 | **0.012** |
| log CPM + cos + rg | 0.936 | 0.895 | 0.904 | 0.264 | 0.051 | 0.039 | 0.295 | 0.164 | 0.042 |

**Each treatment controls one column and no other.**

| treatment | column | without | with |
|---|---|---:|---:|
| rank encoding | axis vs detected genes | 0.890-0.907 | 0.084-0.085 |
| read equalisation | axis vs total counts | 0.933-0.974 | 0.072-0.200 |
| cosine cost | F1 at sd(log depth) 0.9 | 0.249-0.803 | 0.684-0.934 |

There is no interaction. Adding DS leaves the detected-gene axis unchanged to
three decimals, and adding rank leaves the total-count axis at 0.933-0.974
either way. The full model is a composition of three separate fixes, not a
redundancy.

**The column none of them controls.** On the fixed-count breadth arm -- 3,119
counts for every cell, detected genes spanning 296 to 1,111 -- the full model
leaves the leading axis correlated with detected genes at **0.730**, against
0.085 on the depth arms. It removes that axis when breadth rides on depth and
not when breadth moves on its own. Per-replicate: 0.724, 0.730, 0.732.

The detection-rate floor, the other candidate, does not fix it: 0.700 at 10%
and 0.698 at 25%. Detected-gene regress-out does: **0.012** (replicates 0.007,
0.012, 0.013), non-overlapping, at no measurable cost in power.

This is what connects to the real data. Section 8 of the resolution document
measured the leading axes against detected genes at **rho 0.645** on the four
datasets. The ablation says that is not a residual the configuration failed to
reach; it is a covariate the configuration does not act on once totals are
equalised.

### 2c. The arm where depth carries signal

Every arm above draws depth independently of cell identity, so read
equalisation cannot destroy anything in any of them: it was being measured
somewhere it could not lose. That is the objection the resolution document
makes against regress-out, and it had not been applied to equalisation.

`perturbed_depth_informative` fixes that. sd(log depth) 0.6 underneath, and the
planted 20% drawn **twice as deep** as everything else, source side only, so
the cue equalisation deletes is perfectly correlated with the right answer. Its
control, `homogeneous_depth_informative`, makes the same 20% of source cells
twice as deep while expressing identically, so the correct answer there is to
reject nothing. Results in `<depth_screen>/*_informative/`, three replicates,
ranges in brackets.

| configuration | power, informative arm | control rejection |
|---|---|---|
| log CPM | 0.146 (0.089-0.182) | 0.089 (0.083-0.093) |
| rank | 0.847 (0.833-0.858) | 0.094 (0.084-0.107) |
| rank + DS | 0.850 (0.844-0.858) | 0.085 (0.078-0.093) |
| log CPM + cos | 0.868 (0.844-0.897) | 0.045 (0.039-0.049) |
| **rank + cos** | 0.927 (0.908-0.940) | 0.034 (0.028-0.044) |
| log CPM + DS + cos | 0.952 (0.946-0.962) | 0.020 (0.015-0.023) |
| **rank + DS + cos** | 0.949 (0.938-0.960) | 0.023 (0.020-0.026) |
| rank + DS + cos + rg | 0.951 (0.942-0.958) | 0.022 (0.017-0.025) |

**The arm was built to give the no-equalisation case its best chance, and the
no-equalisation case lost it.** Equalisation is ahead on both columns in every
configuration.

How firm each half is. On power the rank comparison is marginal: 0.908-0.940
against 0.938-0.960 overlap by 0.002, so the direction is consistent across all
three replicates but the ranges effectively touch. On log CPM the same
comparison separates cleanly, by 0.049. On the control the rank comparison
separates outright. Read together: equalisation is clearly more specific and
not less powerful, and the hypothesis that it destroys biological signal is not
supported by the arm built to detect exactly that.

The mechanism is legible from 2b: depth organising the representation, 0.969 on
the leading axis, costs more than depth is worth as a cue, even with its value
as a cue made as large as this construction allows. Without the cosine cost the
stage makes no difference here at all, 0.847 against 0.850, so the interaction
is with the cost rather than with the encoding.

### 2d. What all of this does not measure

- **Whether real depth or breadth differences between cell states are
  biological or technical.** Every arm assumes one or the other by
  construction. 2c assumes biological, which is why its result is a price and
  not a verdict. The breadth arms give breadth no biology, which is why
  regress-out's zero cost there is guaranteed rather than measured.
- **What equalisation costs the differential expression.** There is no
  differential expression in the simulation, and `27_downsample_counts.py`
  rewrites the h5ads, so the pseudobulk, the DEG and the UCell scoring would
  read the reduced counts too. 2c justifies equalising **for the
  representation and the gate** and says nothing about the file-level
  application. **Section 2e settles that by design rather than by
  measurement.**

### 2e. The decision: equalised counts for the gate, raw counts for the DEG

Taken 2026-09-23. Read equalisation stays, because 2b and 2c establish what it
does and the arm built to make it lose did not. The one cost the simulation
could not price -- what downsampling does to the differential expression --
is removed rather than measured: **the expression stages read the original
count matrices.**

No code change is needed.
`21_prepare_four_state_malignant_pseudobulk.py` already takes `manifest_csv`
and `gate_root` as separate arguments, and joins gate labels to cells by
`observation_id` against `obs_names`, raising `KeyError` on any identifier it
cannot find. So the original manifest with an equalised-counts gate root is a
supported invocation that fails loudly if the two ever stop lining up, rather
than a new path that has to be trusted.

It is also more correct in one respect that has nothing to do with depth. The
equalised h5ads hold only the analysed subset, so the `*_nonmalignant` state
is empty in them; the original matrices carry the other cell types and that
state means something again.

**The precondition, and it is not optional.** A gate computed on equalised
counts is only safe to combine with raw-count expression if the gate itself is
depth-neutral. Otherwise raw counts *amplify* a residual depth preference
instead of neutralising it -- and the residual is real: the retained and
rejected groups differed in depth by 1.32x, 1.97x and 3.07x in the
pre-equalisation runs, and section 8 of the resolution document found a third
of ovarian pairs and half of prostate pairs still over the bound after it.

So the order is fixed:

1. gate on equalised counts
2. `25_diagnose_gate_covariates.py`, per pair, against each cell's **original**
   depth
3. raw-count expression **only for the pairs that clear**
   `|auc_predownsample_total_counts - 0.5| <= 0.10`

A pair that fails step 2 is not carried into step 3 with a caveat attached. It
is excluded, and the count of exclusions is reported, because a depth-confounded
gate read at full depth is the exact failure this whole round exists to avoid.

### 2f. The shipped configuration: `rank256_ds_cos`, with regress-out as sensitivity

Taken 2026-09-23, after 7a returned. **The primary analysis uses
`rank256_ds_cos` on equalised counts with the expression stages on raw counts
(2e). Detected-gene regress-out is not adopted as the default.**

The reason is not caution for its own sake. Regress-out removes a covariate
that is **partly biological on real data**. The calibration round established
this: a purely technical detection breadth, simulated at wider-than-real
spread, reaches an axis correlation of only 0.229, far short of the 0.645
observed. Read equalisation preserves each cell's proportions, so a fivefold
spread in detected genes at a fixed 3,119 counts means the underlying profiles
differ in how many genes they express at all. That is transcriptome
complexity, and it is biology.

So part of the improvement regress-out shows is the removal of real signal.
Section 5.1's standard -- a setting must not be chosen because it produces a
more interesting result -- applies just as directly to choosing a setting
because it produces a cleaner covariate statistic. 7a shows the regression
reaches its target covariate on real data. It does not show that removing the
covariate is right, and nothing available can show that.

**What this choice costs, stated rather than hidden.** The shipped
configuration leaves the leading axis correlated with detected genes at 0.422
on head and neck and 0.645 across datasets. Any differential expression drawn
from it is open to exactly that objection, and the objection is a fair one.

**The answer to it is the sensitivity arm, which is what the four
`rank256_rg-genes_ds_cos` runs now are.** Not a candidate default -- a direct
response to a known limitation:

| | configuration |
|---|---|
| primary | `rank256_ds_cos` gate on equalised counts, expression on raw counts |
| sensitivity | the same gate with detected genes regressed out of the components |

If a conclusion holds under both, the covariate was not driving it, which is
stronger evidence than either arm produces alone. If it does not hold, the
conclusion was never publishable. The arms therefore run to completion and are
reported together; neither is dropped once the other is read.

---

## 3. Figure 2

`benchmark_results/figure2/`, composite plus each panel as PNG and PDF.

| panel | file | what it shows |
|---|---|---|
| (a) | `panel_a_f1_no_batch` | 2a, no batch effect, F1 against N |
| (b) | `panel_b_f1_mild_batch` | 2a, mild batch effect |
| (c) | `panel_c_specificity` | 2a, planted anomaly against worst unaffected population, N=10000 |
| (d) | `panel_d_runtime` | seconds per OT fit against N, with the O(N²) reference |
| (e) | `panel_e_validation` | how the ground truth is built |
| (f) | `panel_f_ablation` | 2b, three metric families per configuration |
| supp. | `supplementary_ablation_all` | the full ablation table, all columns |

Panel (e) is the one that makes the rest readable, and it is not a result. Two
rows: both sides of a pair drawn independently from one gene-mean vector, so in
the homogeneous row no source cell lacks a counterpart and the correct
rejection rate is **zero by design**; in the perturbed row a second population
is added to the source only, so the correct answer is exactly that set. The
nuisance -- depth spread, and detection breadth at fixed counts -- is drawn
once, below both rows, because it is applied to **both sides**. Applying it to
one side would make it a batch effect between samples, which is a different
problem with its own literature.

Panel (f) currently carries three columns: axis against counts, axis against
genes, and power at high depth spread. The informative arm of 2c is not in it,
because those runs live in a different output root from the ablation.

---

## 4. What was submitted

Submitted 2026-09-23 from `cancer_metastasis/submit_rank_cosine_chain.sh`, one
chain per dataset: J1 equalise (skipped or reused), J2 rank-cut audit, J3 OT
array, J4 gate and pairing diagnostics, each `afterok` the previous.

### `rank256_rg-genes_ds_cos` — complete

| dataset | accession | audit | OT array | diagnose |
|---|---|---|---|---|
| ovarian | GSE180661 | 1019183 | 1019184 `[0-7]` | 1019185 |
| prostate | GSE271675 | 1019186 | 1019187 `[0-7]` | 1019188 |
| colorectal | GSE225857 | 1019189 | 1019190 `[0-3]` | 1019191 |
| headneck | GSE181919 | 1019192 | 1019193 `[0-3]` | 1019194 |

All four reuse the existing equalised counts rather than rebuilding them.
Re-running the depth quantile on a different cell set would change the target
depth, and the two arms would then differ in two ways at once.

### `rank256_cos` — incomplete, `QOSMaxSubmitJobPerUserLimit`

| dataset | state |
|---|---|
| colorectal | submitted: 1019197, 1019198 `[0-3]`, 1019199 |
| ovarian | audit 1019195 submitted, OT and diagnostics refused — **orphan** |
| prostate | audit 1019196 submitted, OT and diagnostics refused — **orphan** |
| headneck | nothing submitted; the audit itself was refused |

The two orphan audits write a `rank_top_n_audit.csv` into an OT directory that
has no OT in it. That file reads afterwards as though the chain's gate had been
cleared, so they are to be cancelled rather than left to finish, and the three
missing chains resubmitted once the queue drains.

This is a defect in the chain, not in the cluster: J3 failing after J2 has been
accepted leaves half a chain and no rollback, and when J2 itself is refused
`set -e` kills the script instead. Recorded so the resubmission is not mistaken
for a fresh start.

The refusals are arithmetic, not chance. The `gg` queue allows **40 submitted
jobs per user**, and an array counts as its task count rather than as one job.
The four complete chains occupy 32 (8 + 8 + 4 + 4 array tasks, plus four
audits and four diagnostics). The next chain's audit takes it to 33, and
ovarian's eight-task array would reach 41; the same for prostate. Colorectal's
four-task array reaches 39, and its diagnostic lands on exactly 40, which is
why that one dataset went through. Headneck's audit would have been 41.

The three outstanding chains need 26 slots between them: ovarian 10, prostate
10, headneck 6. If they are submitted at all, it is one dataset at a time as
the arrays drain, smallest first, checking headroom with
`squeue -u $USER -h -r | wc -l` -- `-r` expands array tasks into separate rows,
which is how the limit counts them.

### Why `rank256_cos` is demoted rather than completed

It was submitted for two reasons and has lost both. The measurable one --
whether equalisation destroys signal -- was answered in 2c, in the arm built
to let it lose. The practical one -- differential expression at full depth --
is answered by 2e, which gets it without dropping the stage.

What is left is 6c, a descriptive agreement statistic that selects no arm,
because real data has no ground truth to score either configuration against.
That is not worth 26 job slots while the breadth arm is still running.

Colorectal is already submitted and costs nothing more, so it runs and stands
as a single-dataset check. The two orphan audits are cancelled. The other
three chains are submitted only if colorectal shows the two arms disagreeing
far more than the simulation suggests they should.

---

## 5. Open before any result is read

### 5a. Pair alignment between the equalised and original manifests

**This is the one that can invalidate the comparison silently.** The
equalisation arms read `pair_manifest_downsampled.csv`; the no-equalisation arm
reads the original `pair_manifest_*_eligible.csv`. The array index maps to a
manifest row, so if the two files differ in row count or row order, index 0 is
a different pair in the two arms and nothing reports it.

**Resolved 2026-09-23, and the first version of this section was wrong about
the mechanism.** An array task is not a pair: each task iterates several pairs
from the manifest. Ovarian's eight-task array produces 94 per-pair `SUCCESS`
markers, and colorectal's four-task array covers all five of its pairs, which
is what the baseline's five markers show. So `OT_ARRAY=0-3` against a five-pair
manifest was never the defect it looked like, and the index does not map to a
manifest row at all.

The underlying concern survives in a weaker form. The two arms still read
different manifests, so they must still list the same pairs -- but the check is
now direct, and better than comparing manifests: **compare the per-pair output
directory names across the arms**, which is what was actually run rather than
what was listed. Counts must match too, at 94, 24, 5 and 4 for ovarian,
prostate, colorectal and headneck.

Headneck cleared this on completion: four markers in the regress-out arm
against four in the baseline.

### 5b. Cell-set identity

Both paths select on author malignant labels with QC thresholds at `0/0/100`,
so no cell is filtered by either and the cell sets should match. The one place
they can diverge is `CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000`: the cap
is deterministic per sample, but the equalised h5ads need not store cells in
the original order, so a pair with more than 10,000 cells per side could be
capped to a different subset. Compare `source_n` and `target_n` across arms;
where they differ, that pair is not comparable.

---

## 6. Prespecified reading

Fixed before any arm returned. Three readouts, in order. A later readout is not
interpreted if an earlier one fails.

### 6a. The axis — the breadth arm's own test

`max_abs_spearman_pc_detected_genes`, per pair, from
**`tools/audit_depth_axis.py`**.

**The chain does not produce this.** J4 runs
`25_diagnose_gate_covariates.py`, which scores the *gate* against covariates,
and `26_diagnose_pairing_quality.py`. Neither computes the representation's
leading axis, so the primary readout -- the one every later readout is
conditioned on -- needs a separate invocation against the stored joint PCA
coordinates that `CONFIDENCEOT_SAVE_PAIRING_EDGES=1` writes:

    python cancer_metastasis/tools/audit_depth_axis.py \
      --dataset <name>=<ot_root> --dataset <name>=<other_ot_root> \
      --predownsample-depth <equalised>/predownsample_depth.csv.gz \
      --out <axis_comparison.csv>

Recorded here because a diagnostics job that reports `COMPLETED` invites the
assumption that the diagnostics are complete, and for 6a they are not.

| | |
|---|---|
| baseline, observed | **per dataset**; 0.645 was the across-dataset figure and is not any one dataset's reference |
| simulation says regress-out gives | 0.012 |
| **pass** | the regress-out arm's distribution falls substantially towards zero |
| **fail** | it does not move |

Read per dataset against that dataset's own baseline. An earlier draft named
0.645 as "the observed baseline", which is the across-dataset number from
section 8 of the resolution document; head and neck's own baseline is 0.422.

A failure here means the regression is not reaching the real covariate, and
**nothing else about that arm is interpretable** — not its gate, not its
retention, not its agreement with the baseline.

### 6b. The gate, per pair and never pooled

`|auc_predownsample_total_counts − 0.5|` and the detected-gene equivalent, read
as the **per-pair distribution** for any dataset with at least 6 pairs.

Pooling is what hid the answer last time: ovarian pooled to 0.497 while its
pairs ran 0.18 to 0.92. The pooled number is not reported here.

The informative comparison is which way 6a and 6b move together:

| axis (6a) | per-pair gate deviations | reading |
|---|---|---|
| falls | fall | breadth was driving the gate; regress-out addresses it |
| **falls** | **unchanged** | breadth was not driving the gate; the residual is something no arm here touches |
| does not fall | any | 6a failed; see above |

The second row is a real possible outcome and would be the most informative
one. It must not be reported as a failure of the arm — the arm would have done
exactly what it claims and shown the claim was about the wrong covariate.

### 6c. Withdrawn 2026-09-24

This was the Jaccard agreement between the equalisation and no-equalisation
arms. It was written down as "descriptive, not a test" whose outcomes "select
no arm on their own", which is a readout that cannot change what is reported --
so it does not belong in a prespecified reading, and keeping it would have cost
the three unsubmitted `rank256_cos` chains 26 job slots to produce.

Recorded rather than deleted, because the reason it was withdrawn applies to
the next check that gets proposed: checks were accumulating faster than they
were being retired. The ones that remain are 6a, which conditions everything
after it, 6b, which sets the per-pair exclusion rule the differential
expression depends on, and 6d, which is one number.

Also dropped for the same reason: the same-patient primary-versus-primary
control, which can only state the limit more precisely than the partial
Spearman of 0.79 already does, and the subcluster-uniformity test, which was
demoted to affecting how a result is described rather than whether it can be
drawn.

**The cross-patient mismatched manifest was dropped, then reinstated on
2026-09-24 with an argument about calibrating a retention floor, and both moves
were wasted: it was built and run on 2026-09-15.** See
`28_build_mismatched_manifest.py`, `29_compare_matched_mismatched.py` and
`30_validate_similarity_readout.py`. Searching the repository before proposing
an experiment would have cost less than either decision.

What those runs found, recorded in `30`'s docstring:

| | |
|---|---|
| retained fraction, matched to mismatched | **0.351 to 0.000**, 94 pairs, signed-rank p = 8e-17 |
| cost ordering preserved under the swap | rho = 0.378 |
| retention vs pseudobulk correlation | **partial Spearman 0.79**, partialling out cell count; 0.785 within solid lesions alone |
| same patient, different lesion | 64% of one patient's primary cells change label |
| SPECTRUM-OV-003, same 1,332 cells | 0.420 to 0.032 on naming a different lesion |
| across patients, one lesion each | 0.000 to 0.879 |

**The floor this control was reinstated to measure does not exist.** Mismatched
retention is 0.000, not 40%, so the gate carries no generic
"cells-of-this-cancer-resemble-cells-of-this-cancer" component and no
downstream contrast is diluted by one. The whole argument of 2026-09-24 was for
measuring a quantity already measured at zero.

**And the finding that matters is a different one, which the floor argument
obscured.** Retention is not merely similarity-tracking in the aggregate: the
*same primary cells* get a different answer depending on which lesion of the
*same patient* is on the other side. 64% label churn, and one patient moving
0.420 to 0.032 over an identical cell set. A property that changes when you
point at a different lesion of one tumour is not a property of the cells, so it
cannot be metastatic competence -- and no control, floor or regression settles
that better than this already does.

`30`'s prespecified reading was three-way, and the outcome was *supported*: the
correlation is strong across the union of matched and mismatched arms, with the
mismatched pairs extending the similarity range downward continuously rather
than forming a separate cluster. The `min_cell_n` confound the script names for
itself was tested and is what the "partial" in 0.79 controls for.

This is the origin of the framing in `DEG_PRESPECIFICATION_2026-09-15.md`,
written the same day: "This is **not** a test of metastatic competence." That
sentence is not a caveat someone added for safety. It is the conclusion of
these three scripts.

**Nothing to run.** The regression proposed on 2026-09-24 -- matched and
mismatched on one similarity axis, testing whether matched pairs sit above the
line -- is what `30` already implements and already answered.

---

## 6. Prespecified reading

Fixed before any arm returned. Three readouts, in order. A later readout is not
interpreted if an earlier one fails.

### 6a. The axis — the breadth arm's own test

`max_abs_spearman_pc_detected_genes`, per pair, from
**`tools/audit_depth_axis.py`**.

**The chain does not produce this.** J4 runs
`25_diagnose_gate_covariates.py`, which scores the *gate* against covariates,
and `26_diagnose_pairing_quality.py`. Neither computes the representation's
leading axis, so the primary readout -- the one every later readout is
conditioned on -- needs a separate invocation against the stored joint PCA
coordinates that `CONFIDENCEOT_SAVE_PAIRING_EDGES=1` writes:

    python cancer_metastasis/tools/audit_depth_axis.py \
      --dataset <name>=<ot_root> --dataset <name>=<other_ot_root> \
      --predownsample-depth <equalised>/predownsample_depth.csv.gz \
      --out <axis_comparison.csv>

Recorded here because a diagnostics job that reports `COMPLETED` invites the
assumption that the diagnostics are complete, and for 6a they are not.

| | |
|---|---|
| baseline, observed | **per dataset**; 0.645 was the across-dataset figure and is not any one dataset's reference |
| simulation says regress-out gives | 0.012 |
| **pass** | the regress-out arm's distribution falls substantially towards zero |
| **fail** | it does not move |

Read per dataset against that dataset's own baseline. An earlier draft named
0.645 as "the observed baseline", which is the across-dataset number from
section 8 of the resolution document; head and neck's own baseline is 0.422.

A failure here means the regression is not reaching the real covariate, and
**nothing else about that arm is interpretable** — not its gate, not its
retention, not its agreement with the baseline.

### 6b. The gate, per pair and never pooled

`|auc_predownsample_total_counts − 0.5|` and the detected-gene equivalent, read
as the **per-pair distribution** for any dataset with at least 6 pairs.

Pooling is what hid the answer last time: ovarian pooled to 0.497 while its
pairs ran 0.18 to 0.92. The pooled number is not reported here.

The informative comparison is which way 6a and 6b move together:

| axis (6a) | per-pair gate deviations | reading |
|---|---|---|
| falls | fall | breadth was driving the gate; regress-out addresses it |
| **falls** | **unchanged** | breadth was not driving the gate; the residual is something no arm here touches |
| does not fall | any | 6a failed; see above |

The second row is a real possible outcome and would be the most informative
one. It must not be reported as a failure of the arm — the arm would have done
exactly what it claims and shown the claim was about the wrong covariate.

### 6c. Withdrawn 2026-09-24

This was the Jaccard agreement between the equalisation and no-equalisation
arms. It was written down as "descriptive, not a test" whose outcomes "select
no arm on their own", which is a readout that cannot change what is reported --
so it does not belong in a prespecified reading, and keeping it would have cost
the three unsubmitted `rank256_cos` chains 26 job slots to produce.

Recorded rather than deleted, because the reason it was withdrawn applies to
the next check that gets proposed: checks were accumulating faster than they
were being retired. The ones that remain are 6a, which conditions everything
after it, 6b, which sets the per-pair exclusion rule the differential
expression depends on, and 6d, which is one number.

Also dropped for the same reason: the same-patient primary-versus-primary
control, which can only state the limit more precisely than the partial
Spearman of 0.79 already does, and the subcluster-uniformity test, which was
demoted to affecting how a result is described rather than whether it can be
drawn.

**The cross-patient mismatched manifest was dropped with them and should not
have been.** It was judged as a significance test -- do matched and mismatched
pairs separate -- which is trivially yes and proves nothing, and that judgement
was right about the wrong question. Its use is to calibrate a scale, not to
test a hypothesis.

Nobody knows what retention fraction is correct. Section 8 lists retention as a
defect separate from depth that no configuration tested so far addresses, and a
gate keeping 71% of primary cells has no reference to be read against. The
mismatched pairing supplies one, because its true answer is near-total
rejection: if mismatched pairs still retain 40%, then most of a matched pair's
retained set is a floor made of "cells of this cancer resemble cells of this
cancer", and every downstream contrast is diluted by that much. If they fall to
5%, the matched retention is mostly pairing information.

That floor is quantitative, and the partial Spearman of 0.79 does not give it.
0.79 says retention tracks similarity; it does not say whether retention goes
to zero as similarity does, or plateaus.

Scope it as a calibration rather than a survey: a dozen or so mismatched pairs
from one dataset give the distribution. No code changes --
`01_build_pair_manifest.py` emits the same columns, so shuffling which
metastasis accompanies which primary is a manifest edit and the rest of the
pipeline runs unmodified. Random pairing and per-patient retention are the same
measurement under different names.

### 6d. Retention, reported and not used to choose

Retention rates are a defect separate from depth — a gate keeping 71% of
primary cells is making a different error from one that tracks depth — and no
configuration tested so far addresses it. Report it for all three arms. **If it
is unchanged, that stays open**, and it is not evidence against any arm here.

---

## 7. Real-data results

Filled as arms return, in the order of section 6.

### 7a. Head and neck, GSE181919 — 6a passes

Complete 2026-09-23, four pairs, 810 malignant cells. Medians over pairs from
`tools/audit_depth_axis.py`.

| | pre-equalisation depth axis | **detected-genes axis** | equalised-depth control |
|---|---:|---:|---:|
| `rank256_ds_cos` | 0.282 | **0.422** | 0.190 |
| `rank256_rg-genes_ds_cos` | 0.288 | **0.109** | 0.172 |

**6a passes, and the shape of the pass is what makes it credible.** The
detected-gene axis falls by a factor of 3.9 while the depth axis and the
equalisation control do not move at all. The regression reached its target
covariate on real data and left the neighbouring one alone, which is the
outcome a spurious improvement would not produce.

Per pair: P38 0.513 → 0.099, P22 0.344 → 0.049, P46 0.409 → 0.118, and P59
0.434 → 0.336. Three fall hard; P59 barely moves and is recorded as such.

### 7b. Head and neck — 6b is not estimable here, as was already established

Retention runs 0.579, 0.938, 0.394 and 0.732 in the baseline, so the
second pair rejects about 6% of roughly 56 source cells. The per-pair gate AUCs
printed for this dataset are a handful of cells wide, and the apparent
worsening on that pair, `auc_n_genes_by_counts` 0.117 → 0.252, is noise of that
width rather than a result.

This was not a surprise: section 4b of the resolution document established on
2026-09-21 that GSE181919 cannot carry a per-pair acceptance test -- three of
its four pairs sit one to two standard errors from the threshold -- and
recorded the amendment **before any number existed**, that this dataset is
evaluated on the **pooled** gate. That amendment governs here, and it is the
one place where section 6b's "never pooled" does not apply.

### 7d. Ovarian differential expression, GSE180661, 2026-09-24

The first differential expression this project has run that clears its own
checks. It is also, on its face, a result about the cell cycle and nothing
else.

**What ran.** The original manifest held 243 rows across 84 patients spanning
all four datasets; trimming to the pairs the gate root supplies left 94 rows
and 29 patients, and the trim is what kept `21_` from raising on the first
non-ovarian patient it met. Four pairs were dropped for
`calibration_m4e_inference_valid`, which is the prespecification's own
calibration filter, taking two patients with them. 27 patients reached the fit.

**The contrast.** 25,492 genes tested, 1,228 at FDR 0.05, and at the
prespecified two-fold floor 138 genes -- 90 of them outside the OT feature
set. Every one of the 90 is higher in the **retained** cells; not one is
higher in the rejected. At the looser 1.4-fold floor it is 208 against 11.

**What the genes are.** KIF20A, PLK1, PIF1, DLGAP5, CDCA3, GAS2L3, FAM83D,
KIF2C, DEPDC1, GTSE1, CCNA2, CKAP2L, at 4 to 5.7-fold and FDR around 1e-27.
A mitotic signature, unmixed.

**All four checks pass.**

| | |
|---|---|
| Disqualifier 1, keratin/SPRR/S100A in the top 30 | **0** |
| Disqualifier 2, effects tracking the retained fraction | median abs rho 0.170, max 0.529, 11 genes flagged |
| Disqualifier 3, leave-one-patient-out over 27 folds | worst Jaccard **0.871**, does not fire |
| survival across folds | **115 of 138** survive every fold, 0 survive none |

Disqualifier 1 clearing is the preprocessing round's return: keratin and SPRR
led the rejected group of all three datasets before the depth work, and they
are now absent from the top 30 entirely.

**The size-factor explanation for the one-sidedness is refuted.** A pseudobulk
dominated by a few genes would inflate its size factor and push everything
else down, producing exactly this asymmetry. It is not what happened: the top
gene takes 0.0287 of the rejected library against 0.0259 of the retained, and
the top twenty 0.234 against 0.218. Neither side is dominated. The asymmetry
is real, and its mechanism is that retained cells share one coherent state
while rejected cells are heterogeneous and average to no shared programme.

**EMT is absent, and this was checked on the all-gene table.** The non-OT
validation table excludes the features the representation used, which is most
of the EMT and epithelial sets; on the all-gene table 26 of 26 EMT genes and
10 of 10 epithelial genes are present. No EMT gene is significant and none
exceeds 1.21-fold. Among the epithelial genes only CDH1 reaches significance,
at 1.16-fold and on the rejected side -- an order of magnitude below the
mitotic effects and far below the two-fold floor.

So what the gate separates is cycling from non-cycling, not invasive from
non-invasive.

**What that does and does not license.** Proliferation is required for a
metastasis to grow once it has arrived, so retained cells resembling a
proliferative lesion is a coherent reading. It is not evidence that they were
the cells able to leave: dissemination is associated with EMT and with
cell-cycle exit, and neither appears here. A metastasis that has already
colonised is proliferative because it is growing, and primary cells that
resemble it may simply be the primary's own cycling fraction.

Proliferation is also the most common confounder in single-cell differential
expression -- any two subsets differing in cycling fraction produce this --
so finding it is not by itself evidence of anything.

**The number that separates the two readings is not yet in.**
`36_proliferation_correspondence.py` asks whether the retained-minus-rejected
proliferation gap tracks the partner lesion's own proliferation. The gap is
+1.32 in log1p CPM at the median and positive in 26 of 27 patients, which is
suggestive of a property of the primary alone but not decisive, since ovarian
metastases may all be proliferative. Also outstanding: the pathway-level view,
where the GSEA results are keyed on a `pathway` column rather than `Name`.

### 7c. Remaining

| | baseline `rank256_ds_cos` | breadth `rank256_rg-genes_ds_cos` | no-equalisation `rank256_cos` |
|---|---|---|---|
| ovarian, 94 pairs | done | *running* | not submitted, see section 4 |
| prostate, 24 pairs | done | *running* | not submitted |
| colorectal, 5 pairs | done | *running* | *running* |
| headneck, 4 pairs | done | **7a, 7b** | not submitted |

Ovarian and prostate are the datasets that decide this. Head and neck is the
smallest of the four and the only one whose acceptance test was already known
not to be estimable, so 7a is a licence to keep reading rather than a result
about the configuration.

---

## 8. Out of scope until this closes

**No differential expression from any of the three arms**, which was already
required as a precaution and now has a measured reason. The same applies to
GSEA, UCell scoring and any pseudobulk contrast.

Nothing here is a claim about metastatic competence. The retained fraction was
shown on 2026-09-15 to track primary-metastasis transcriptional similarity
(partial Spearman 0.79), and the per-pair UMAPs on 2026-09-21 showed retention
following sample overlap rather than any property of the cells.

The controls this project still lacks, unchanged by this round and not
submitted here: the cross-patient mismatched manifest, and the same-patient
primary-versus-primary comparison that 16 of 29 GSE180661 patients make
possible. Either would separate a patient-specific result from a technical
overlap, and neither exists yet.
