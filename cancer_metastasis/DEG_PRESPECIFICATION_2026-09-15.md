# Prespecified reading of the GSE180661 retained-vs-rejected DEG

Written and committed before the differential expression is run. Section 5.1 of
the project summary sets the standard this enforces: a setting must not be
chosen because it yields more metastasis-related genes.

## The question

What distinguishes primary malignant cells that sit closer to the matched
metastasis than the primary's own internal spread, from those that do not?

That is the criterion the gate actually applies. The threshold is
`c = within-side spread / cross-side distance` and the cost is
`d(cell, metastasis)² / cross-side distance`, so the cross-side scale cancels
and `cost < c` reduces to

    d(primary cell, metastasis) < typical within-primary distance

expressed in units of each patient's own primary heterogeneity. The rule is
identical in every patient; only the fraction of cells passing it differs,
because the tumours differ in how alike the two sides are.

This is **not** a test of metastatic competence. The retained fraction was
shown on 2026-09-15 to be a readout of primary--metastasis transcriptional
similarity (partial Spearman 0.79 against pseudobulk correlation, 0.785 within
solid lesions alone), so any claim about which cells will metastasise is out of
scope for this analysis.

## Configuration, fixed

| | |
|---|---|
| Contrast | `primary_rejected_vs_primary_retained`, the only one |
| Gate | `ot_downsampled_free_20260914`, unconstrained source rejection |
| Calibration null | `within_side_split` |
| Lesion | one per patient, the largest by malignant QC-passing cells |
| Depth | both sides subsampled to 3,119 counts |
| Calibration filter | M4-E feasible cost, clean fit, valid certificate |
| Minimum cells | 20 per patient per status |
| Design | paired, patient as the blocking factor |

The bounded gate is deliberately not used: it carries a rejection interval we
chose from prior expectation, and the primary analysis must not.

## Amendment, 2026-09-24, recorded before the differential expression is run

The configuration table above predates the preprocessing round of 2026-09-21
to 09-23. The gate it names was produced under a configuration since
superseded, so the table is amended rather than reinterpreted. The original
rows stay as written; what follows replaces them where they conflict. Nothing
in **Thresholds**, **Disqualifiers** or **What a positive result is** changes.

| | was | is |
|---|---|---|
| Gate | `ot_downsampled_free_20260914` | `ot_<accession>_rank256_ds_cos_20260921` |
| Depth | both sides subsampled to 3,119 counts | **gate** on equalised counts; **expression on raw counts** |

**Why the expression stage reads raw counts.** The equalisation stage is
justified for the representation and the gate and was never priced for what it
costs the differential expression, because the simulation contains no
differential expression to price it with. Reading the original matrices removes
that cost instead of waiting on a measurement of it. It needs no new code:
`21_prepare_four_state_malignant_pseudobulk.py` takes the manifest and the gate
root as separate arguments and joins cells by identifier, raising on anything
it cannot find. Section 2e of `EXPERIMENT_2026-09-23_PREPROCESSING_ARMS.md`.

**The precondition, which is a new exclusion rule.** A gate computed on
equalised counts is only safe to read raw counts against if the gate is itself
depth-neutral; otherwise full depth amplifies a residual preference rather than
neutralising it. A pair enters the pseudobulk **only** if it clears
`|auc_predownsample_total_counts - 0.5| <= 0.10`. Pairs that fail are excluded
and counted, not carried with a caveat. GSE181919 is evaluated pooled rather
than per pair, per the amendment recorded in section 4b of
`SEQUENCING_DEPTH_RESOLUTION.md` on 2026-09-21, because three of its four pairs
sit one to two standard errors from the threshold.

**One number in Disqualifier 1 is stale.** The
`auc_predownsample_total_counts` of 0.587 quoted there was measured on the
superseded configuration. The disqualifier stands; the figure to test against
is what the 2026-09-21 gate reports per pair, which the exclusion rule already
reads.

**The regress-out gate is held in reserve, not run alongside.** The shipped
configuration leaves the leading axis correlated with detected genes at 0.422
on head and neck and 0.645 across datasets, which is a fair objection to any
list drawn from it. The `rank256_rg-genes_ds_cos` runs exist and are the answer
to it. They are **not** a mandatory second pass: Disqualifier 2 already tests
the same failure directly on the gene list, which is cheaper and closer to the
claim than re-running the pipeline. The reserve gate is brought out **only** if
a disqualifier fires or a reviewer raises the axis, and then the question is
whether the conclusion survives it. Running both by default would double every
downstream stage to answer an objection that may never be raised, and this
document's own standard is that a check must be able to change what is
reported.

**What the framing is, restated because it has been re-derived twice.** The
gate is a geometric filter: primary cells closer to the matched metastasis than
to their own primary's internal spread. Selecting on similarity is the intended
mechanism, not a defect -- a cell with metastatic potential should resemble the
metastasis. What similarity does **not** license is comparing the retained
*fraction* across patients, because that fraction tracks the two samples'
pseudobulk correlation at partial Spearman 0.79. The within-pair selection is
the object of this analysis; the between-pair fraction is a pairing-quality
readout and is reported as such. Disqualifier 2 is the operational form of that
distinction and is unchanged.

## Thresholds

Significant means `FDR < 0.05` **and** `|log2FC| >= 1.0`.

The fold-change floor is not decoration. The GSE180661 UCell result that
motivated this audit was delta 0.005 at p = 1.9e-8 -- a minute effect with an
enormous p-value, which is the signature of a systematic artefact rather than
biology. A 2-fold floor excludes that class of result. `13_run_paired_pydeseq2.py`
also emits 0.5; that output is exploratory and is not the claim.

## Disqualifiers

Any one of these means the gene list cannot be interpreted, whatever it
contains.

1. **Keratin/SPRR domination.** These families appeared in the rejected group of
   all three datasets before depth correction. If they still lead the ranked
   list, the residual depth signal that survived downsampling
   (`auc_predownsample_total_counts` 0.587) is still driving the result.

2. **Similarity tracking.** If the per-gene effect sizes correlate strongly with
   each patient's retained fraction, the DEG is reading how alike the two
   tumours are rather than what distinguishes the two groups of cells. The
   retained fraction ranges 0.000 to 0.879 across patients and is itself a
   similarity measure, so this is the specific way this analysis can fool
   itself.

3. **Patient dominance.** If removing any single patient changes the
   significant set by more than half, the result rests on one patient rather
   than on 29.

## What a positive result is

Genes passing both thresholds, surviving all three disqualifiers, and coherent
as a set -- a recognisable programme rather than a list. Given the similarity
finding, the honest framing of any such programme is "what makes a primary cell
resemble the matched metastasis", and any further claim needs its own evidence.

A null result -- nothing passes -- is reportable and is not a failure of the
analysis. The mismatched control and the similarity validation already stand on
their own.
