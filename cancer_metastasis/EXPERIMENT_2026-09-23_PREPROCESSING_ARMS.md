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
| baseline | `rank256_ds_cos` | rank 256, read equalisation, cosine | the configuration run on 2026-09-21, section 8 of the resolution document |
| breadth | `rank256_rg-genes_ds_cos` | + detected-gene regress-out | 2b below: the baseline does not act on detection breadth at all |
| no-equalisation | `rank256_cos` | − read equalisation | 2c below: the stage was argued for on evidence that could not price it |

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
  rewrites the h5ads, so the pseudobulk, the DEG and the UCell scoring read the
  reduced counts too. 2c justifies equalising **for the representation and the
  gate**. It says nothing about equalising **at the file level**, which is a
  separate decision currently made on a consistency argument: one matrix, so no
  stage can disagree with another about which counts it used.
- **The design neither arm tests**: equalised counts for the OT, original
  counts for everything downstream.

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

---

## 5. Open before any result is read

### 5a. Pair alignment between the equalised and original manifests

**This is the one that can invalidate the comparison silently.** The
equalisation arms read `pair_manifest_downsampled.csv`; the no-equalisation arm
reads the original `pair_manifest_*_eligible.csv`. The array index maps to a
manifest row, so if the two files differ in row count or row order, index 0 is
a different pair in the two arms and nothing reports it.

One dataset already looks suspect. The colorectal source manifest was resolved
by search to a file with **5 pairs**, while `OT_ARRAY` for colorectal is
hardwired to `0-3`. Either the equalised manifest also has 5 and one pair has
never been run in either arm, or it has 4 and the indices do not correspond.

Required before reading anything: confirm for all four datasets that the two
manifests list the same pairs in the same order, comparing pair identity and
not just count. If any dataset fails, its no-equalisation chain is cancelled
and the arm rebuilt against a manifest that aligns.

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
`25_diagnose_gate_covariates.py`.

| | |
|---|---|
| baseline, observed | 0.645 |
| simulation says regress-out gives | 0.012 |
| **pass** | the regress-out arm's distribution falls substantially towards zero |
| **fail** | it does not move |

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

### 6c. Agreement between the equalisation and no-equalisation arms

Per pair, the Jaccard index of the retained sets from `rank256_ds_cos` and
`rank256_cos`, against the chance level for those two retention rates.

| | reading |
|---|---|
| high agreement | equalisation is not changing which cells are called; its file-level cost buys little and is worth revisiting |
| low agreement | the stage materially decides the answer, and 2c's simulated verdict is carrying real weight it was never tested for on real data |

Descriptive, not a test. Neither outcome selects an arm on its own, because the
simulation cannot say what fraction of a real depth difference is biological.

### 6d. Retention, reported and not used to choose

Retention rates are a defect separate from depth — a gate keeping 71% of
primary cells is making a different error from one that tracks depth — and no
configuration tested so far addresses it. Report it for all three arms. **If it
is unchanged, that stays open**, and it is not evidence against any arm here.

---

## 7. Real-data results

*Empty. To be filled from the runs in section 4, read strictly in the order of
section 6, and only after section 5 is closed.*

| | baseline `rank256_ds_cos` | breadth `rank256_rg-genes_ds_cos` | no-equalisation `rank256_cos` |
|---|---|---|---|
| 6a axis vs detected genes | 0.645 | *pending* | *pending* |
| 6b per-pair gate deviation | 1/3 ovarian, 1/2 prostate over bound | *pending* | *pending* |
| 6c Jaccard against baseline | — | *pending* | *pending* |
| 6d retention | *see section 8 of the resolution document* | *pending* | *pending* |

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
