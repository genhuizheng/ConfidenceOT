# Three preprocessing arms on the four real datasets, 2026-09-23

Written at submission, before any of the three has returned. The simulation
evidence behind each arm is in `SEQUENCING_DEPTH_RESOLUTION.md` section 9; this
file is the record of what was run, what each arm is for, and what counts as
which answer.

The standard section 5.1 of the project summary sets applies here as it does to
the DEG: **a configuration must not be chosen because it produces a more
interesting biological result.** The readouts below are covariate statistics.
None of them is a gene list, and no differential expression is run from any of
these arms.

---

## 1. The three arms

All three share one gene-key resolution, one pair list per dataset, one
calibration null (`within_side_split`), one budget (source 0.85, target 0.00)
and one evaluation. They differ only in the preprocessing configuration, named
by the label `confidenceot.Preprocessing` prints and parses back.

| arm | label | what it adds | why it exists |
|---|---|---|---|
| baseline | `rank256_ds_cos` | rank 256, read equalisation, cosine | the configuration run on 2026-09-21, section 8 |
| breadth | `rank256_rg-genes_ds_cos` | + detected-gene regress-out | section 9c: the baseline does not act on detection breadth at all |
| no-equalisation | `rank256_cos` | − read equalisation | section 9b: the stage was argued for on evidence that could not price it |

The baseline is not re-run. Its outputs from 2026-09-21 are the comparison.

### Why the breadth arm

Section 9c is the finding this round turns on. On the fixed-count breadth arm
-- total counts held at 3,119 while the detected-gene count varies -- the
baseline leaves the representation's leading axis correlated with detected
genes at **0.729** (replicates 0.724, 0.730, 0.732), against 0.085 on the depth
arms. It removes that axis when breadth rides on depth and not when breadth
moves on its own, and equalisation puts the real data in exactly the second
regime: every cell at 3,119 counts, detected genes spanning 282 to 1,400.

Section 8 measured the leading axes against detected genes at **rho 0.645** on
the real data. The ablation says that is not a residual the baseline failed to
reach but a covariate it does not act on.

Regress-out takes the simulated axis to **0.011** (0.007, 0.012, 0.013),
non-overlapping across replicates, with no measurable cost in power on any arm
including the one where depth carries signal (0.951 against the baseline's
0.949).

The detection-rate floor, which was the other candidate, does not work: 0.701
at 10% and 0.697 at 25% against 0.729.

### Why the no-equalisation arm

Two rounds, and the second reversed the first.

Section 9b first argued the stage could not be justified, because every arm in
the screen draws depth independently of cell identity -- so downsampling reads
cannot destroy signal anywhere in it, and equalisation was being measured
somewhere it could not lose. That is the objection section 9d makes against
regress-out, and it had not been applied to equalisation.

Section 9f built the arm where it can lose: the planted 20% drawn twice as deep
as everything else, source side only, so the cue equalisation deletes is
perfectly correlated with the right answer. **Equalisation won anyway**, on
both columns and in every configuration:

| configuration | power on the informative arm | control rejection |
|---|---|---|
| `rank256_cos` | 0.927 (0.908-0.940) | 0.034 (0.028-0.044) |
| `rank256_ds_cos` | 0.949 (0.938-0.960) | 0.023 (0.020-0.026) |
| `logcpm_cos` | 0.868 (0.844-0.897) | 0.045 (0.039-0.049) |
| `logcpm_ds_cos` | 0.952 (0.946-0.962) | 0.020 (0.015-0.023) |

On power the rank comparison is marginal -- the ranges overlap by 0.002 -- so
that half is a consistent direction rather than a separation. On the control it
separates outright, and on log CPM the power comparison separates by 0.049.
Read together: equalisation is clearly more specific and not less powerful, and
the hypothesis that it destroys biological signal is not supported by the arm
built to detect it.

So this arm is **not** here because equalisation is suspected of being wrong.
It is here for two things the simulation cannot supply:

1. **Whether the simulated verdict transfers.** The informative arm assumes the
   depth difference between cell states is biological. On real data some of it
   is technical -- a sample sequenced deeper -- and no simulation can say which
   fraction. A real-data comparison can at least show whether the two arms
   agree about which cells are retained.
2. **The cost the simulation has no way to measure.** There is no differential
   expression in the simulation, and `27_downsample_counts.py` rewrites the
   h5ads, so the pseudobulk, the DEG and the UCell scoring all read the reduced
   counts. Section 9f justifies equalising *for the representation and the
   gate*; it says nothing about equalising *at the file level*. This arm is the
   only one whose downstream stages run at full depth.

---

## 2. What was submitted

Submitted 2026-09-23 from `cancer_metastasis/submit_rank_cosine_chain.sh`, one
chain per dataset: J1 equalise (skipped or reused), J2 rank-cut audit, J3 OT
array, J4 gate and pairing diagnostics, each `afterok` the previous.

### `rank256_rg-genes_ds_cos` -- complete

| dataset | accession | audit | OT array | diagnose |
|---|---|---|---|---|
| ovarian | GSE180661 | 1019183 | 1019184 `[0-7]` | 1019185 |
| prostate | GSE271675 | 1019186 | 1019187 `[0-7]` | 1019188 |
| colorectal | GSE225857 | 1019189 | 1019190 `[0-3]` | 1019191 |
| headneck | GSE181919 | 1019192 | 1019193 `[0-3]` | 1019194 |

All four reuse the existing equalised counts rather than rebuilding them.
Re-running the depth quantile on a different cell set would change the target
depth, and the two arms would then differ in two ways at once.

### `rank256_cos` -- incomplete, `QOSMaxSubmitJobPerUserLimit`

| dataset | state |
|---|---|
| colorectal | submitted: 1019197, 1019198 `[0-3]`, 1019199 |
| ovarian | audit 1019195 submitted, OT and diagnostics refused -- **orphan** |
| prostate | audit 1019196 submitted, OT and diagnostics refused -- **orphan** |
| headneck | nothing submitted; the audit itself was refused |

The two orphan audits write a `rank_top_n_audit.csv` into an OT directory that
has no OT in it. That file reads afterwards as though the chain's gate had been
cleared, so they are to be cancelled rather than left to finish, and the three
missing chains resubmitted once the queue drains.

This is a defect in the chain, not in the cluster: J3 failing after J2 has been
accepted leaves half a chain and no rollback, and when J2 itself is refused
`set -e` kills the script instead. Recorded here so the resubmission is not
mistaken for a fresh start.

---

## 3. Open before any result is read

### 3a. Pair alignment between the equalised and original manifests

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
and the arm is rebuilt against a manifest that aligns.

### 3b. Cell-set identity

Both paths select on author malignant labels with QC thresholds at
`0/0/100`, so no cell is filtered by either. The cell sets should match. The
one place they can diverge is
`CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=10000`: the cap is deterministic per
sample, but the equalised h5ads need not store cells in the original order, so
a pair with more than 10,000 cells per side could be capped to a different
subset. Compare `source_n` and `target_n` across arms; where they differ, that
pair is not comparable.

---

## 4. Prespecified reading

Written before the runs return. Three readouts, in order. A later readout is
not interpreted if an earlier one fails.

### 4a. The axis -- the breadth arm's own test

`max_abs_spearman_pc_detected_genes`, per pair, from
`25_diagnose_gate_covariates.py`.

| | |
|---|---|
| baseline, observed | 0.645 (section 8) |
| simulation says regress-out gives | 0.011 |
| **pass** | the regress-out arm's distribution falls substantially towards zero |
| **fail** | it does not move |

A failure here means the regression is not reaching the real covariate, and
**nothing else about that arm is interpretable** -- not its gate, not its
retention, not its agreement with the baseline.

### 4b. The gate, per pair and never pooled

`|auc_predownsample_total_counts - 0.5|` and the detected-gene equivalent, read
as the **per-pair distribution** for any dataset with at least 6 pairs.

Pooling is what section 8 caught hiding the answer: ovarian pooled to 0.497
while its pairs ran 0.18 to 0.92. The pooled number is not reported here.

The informative comparison is which way 4a and 4b move together:

| axis (4a) | per-pair gate deviations | reading |
|---|---|---|
| falls | fall | breadth was driving the gate; regress-out addresses it |
| **falls** | **unchanged** | breadth was not driving the gate; the residual is something no arm here touches |
| does not fall | any | 4a failed; see above |

The second row is a real possible outcome and would be the most informative
one. It must not be reported as a failure of the arm -- the arm would have done
exactly what it claims and shown the claim was about the wrong covariate.

### 4c. Agreement between the equalisation and no-equalisation arms

Per pair, the Jaccard index of the retained sets from `rank256_ds_cos` and
`rank256_cos`, against the chance level for those two retention rates.

| | reading |
|---|---|
| high agreement | equalisation is not changing which cells are called; its file-level cost buys little and is worth revisiting |
| low agreement | the stage materially decides the answer, and section 9f's simulated verdict is carrying real weight it was never tested for on real data |

This is a descriptive comparison, not a test. Neither outcome selects an arm on
its own, because the simulation cannot say what fraction of a real depth
difference is biological, and this document does not pretend otherwise.

### 4d. Retention, reported and not used to choose

Section 8 recorded retention rates as a defect separate from depth -- a gate
keeping 71% of primary cells is making a different error from one that tracks
depth -- and no configuration tested so far addresses it. Report it for all
three arms. **If it is unchanged, that stays open**, and it is not evidence
against any arm here.

---

## 5. What this cannot settle

- **Whether real depth or breadth differences between cell states are
  biological or technical.** Every simulation arm assumes one or the other by
  construction. Section 9f assumes biological, which is why its result is a
  price rather than a verdict; section 9d notes that the breadth arms give
  breadth no biology, which is why regress-out's zero cost there is guaranteed
  rather than measured.
- **Whether equalisation belongs at the file level.** The arms differ in the
  whole stage, so the design where the OT sees equalised counts and everything
  downstream sees original ones is not among them and remains untested.
- **Anything about metastatic competence.** The retained fraction was shown on
  2026-09-15 to track primary-metastasis transcriptional similarity (partial
  Spearman 0.79), and the per-pair UMAPs on 2026-09-21 showed retention
  following sample overlap rather than any property of the cells.

---

## 6. Out of scope until this closes

**No differential expression from any of the three arms**, which section 7
required as a precaution and section 8 now supports with a measurement. The
same applies to GSEA, UCell scoring and any pseudobulk contrast.

The controls this project still lacks, unchanged by this round and not
submitted here: the cross-patient mismatched manifest, and the same-patient
primary-versus-primary comparison that 16 of 29 GSE180661 patients make
possible. Either would separate a patient-specific result from a technical
overlap, and neither exists yet.
