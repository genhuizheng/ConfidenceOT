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
| Minimum cells | 20 per patient per status | **10** per patient per status |
| Pair exclusion | *(added 2026-09-24, then withdrawn)* | none; all pairs enter |

**Why the cell floor moves to 10.** Retention runs from 0.37 to 0.95 across
pairs, so a high-retention patient has very few rejected cells: 200 primary
cells at 95% retention leaves 10. A floor of 20 drops exactly those patients,
and they are the ones the patient count depends on -- `31_audit_deg_disqualifiers.py`
raises below 6 patients, and disqualifier 3 is the check that decides whether a
result is reported. Lowering the floor is the *include-more* direction, which
is the less suspicious one, but it is a change to a prespecified parameter and
so is recorded here before any result exists.

What it costs: a pseudobulk summed over 10 cells is noisier than one over 20.
DESeq2 estimates dispersion per gene and is built for that, and the floor
applies per patient per status rather than to the fit as a whole, so the effect
is on the weight a marginal patient carries rather than on the validity of the
model. `13_run_paired_pydeseq2.py` is the only place this threshold acts:
`21_prepare_four_state_malignant_pseudobulk.py` emits a contrast whenever both
states are non-empty, so nothing upstream needs changing.

**Why there is no pair exclusion.** A depth-based per-pair exclusion was added
to this amendment on 2026-09-24 and withdrawn the same day. It was
data-dependent selection layered on an already prespecified analysis, and the
three result-level disqualifiers below already test the same failure directly
on the gene list, which is closer to the claim. All pairs enter the pseudobulk.
The per-pair gate statistics remain available in
`gate_covariate_pair_statistics.csv` and are reported as context: of the pairs
with a readable statistic, 61 of 92 ovarian, 12 of 24 prostate, 1 of 5
colorectal and 3 of 4 head and neck clear `|auc - 0.5| <= 0.10`.

**Why the expression stage reads raw counts.** The equalisation stage is
justified for the representation and the gate and was never priced for what it
costs the differential expression, because the simulation contains no
differential expression to price it with. Reading the original matrices removes
that cost instead of waiting on a measurement of it. It needs no new code:
`21_prepare_four_state_malignant_pseudobulk.py` takes the manifest and the gate
root as separate arguments and joins cells by identifier, raising on anything
it cannot find. Section 2e of `EXPERIMENT_2026-09-23_PREPROCESSING_ARMS.md`.

### Two methods, scoped by dataset size, 2026-09-24

The same contrast is estimated two ways. They are not a primary and a
sensitivity: they answer different questions, and the deliverable is which
genes and pathways recur under both.

**Method A -- paired pseudobulk.** One lesion per patient, one pseudobulk per
patient per status, `~patient_id + comparison_status`, floor 10 cells per
status. Asks: is there a **common** effect across patients. Implemented, needs
only a submitter: `21_prepare_four_state_malignant_pseudobulk.py` ->
`13_run_paired_pydeseq2.py` -> `15_run_pydeseq2_gseapy.py`.

**Method B -- per-patient, then combined.** Within each patient, cell-level
retained vs rejected. Record the effect size and the per-patient gene list,
then combine three ways: **gene overlap**, **direction consistency** (the
fraction of patients agreeing on sign), and **meta-analysis** across patients.
Asks: how **consistent** is the effect, and how many patients carry it.

Method B is what makes patient support a reported column rather than something
recovered afterwards, which is why it and the leave-one-patient-out refit
together replace the pair-level filter this document withdrew.

**Scope by dataset.** Method B needs enough patients for overlap and a
meta-analysis to mean anything.

| dataset | pairs | Method A | Method B |
|---|---:|---|---|
| GSE180661 ovarian | 92 | yes, primary | **yes, reported in full** |
| GSE271675 prostate | 24 | yes, primary | exploratory only; no overlap requirement |
| GSE225857 colorectal | 5 | yes | no |
| GSE181919 head and neck | 4 | yes | no |

For the two small datasets, note that `31_audit_deg_disqualifiers.py` raises
below 6 patients, so no patient-dominance verdict is computable for them at
all. That is a property of the datasets, not a failure of the run, and is
reported as such.

**What has to be written.** Method A is complete. Method B is not:

1. Per-patient primary DE does not exist. `11_run_robust_target_deg.py` has the
   right core -- a Scanpy Wilcoxon on a rejected-versus-retained label with BH
   correction and tie correction -- but it is bound to the **target** side in two
   places (`side == "target"` when reading the gate, and `load_exact_side` on the
   target paths) and is built around the cap-robust target states, which the
   primary side has no analogue for. A new script is cleaner than a `--side`
   flag. It must read counts from the **original** matrices and labels from the
   equalised gate root, the same split Method A uses.
2. The combination step needs a driver.
   `12_meta_analyze_robust_target_deg.py` already implements
   `meta_table(patient_effects, minimum_patients)` -- a patient-level Wilcoxon
   meta-analysis with BH adjustment -- but is wired to `11_`'s output layout.
3. Leave-one-patient-out refit for Method A. Each patient dropped in turn,
   `13_run_paired_pydeseq2.py` refitted, and the per-gene survival profile
   recorded. This is the literal Disqualifier 3 and exists nowhere.
4. The cross-method comparison. Which genes clear Method A, which clear Method
   B's meta-analysis, in how many patients individually, and with what
   direction agreement -- and the same at pathway level from the GSEA outputs.
   This is the deliverable, and it is the one piece whose output format is not
   already fixed by an existing script.

### Which cells are malignant, decided 2026-09-24

**The malignant compartment stays the deposit's own label. It does not move to
the collection's uniform inferCNV call.**

The collection now carries `malignant`, one inferCNV rule applied to every
deposit, and ConfidenceOT can read it -- `02_run_pair.py` and the pseudobulk
both take `--malignant-column`, the two modes refuse to run together, and each
run records which rule chose its compartment. It is available and it is not
the default.

Measured on the 92 ovarian pairs by
`37_compare_malignant_definitions.py`: of 215,158 cells the deposit labels
`Ovarian.cancer.cell`, the uniform call names 135,328, adds **none**, and
leaves 79,830 out, with a further 38,521 `undetermined`. Jaccard 0.629, and
135,328/215,158 is 62.9% -- the handoff's own 62.6% recall figure, reproduced
on this dataset. The uniform call is a strict subset.

**Neither definition has DNA behind it.** Section 3.1 of the collection's
progress record, written 2026-09-24 after reading each paper's Methods, says
GSE180661's label is CellAssign on marker genes, the same tier as GSE225857,
and that no deposit in the collection has DNA behind its single-cell
malignancy labels. The premise that SPECTRUM was WGS-anchored survived in that
document for weeks before the Methods were read. So the 79,830 cells the
uniform call omits are not false positives being removed; they are cells two
expression-based methods disagree about, and the validation behind the 62.6%
figure was itself against another expression-based label.

**The reason to prefer the deposit's label here is specific to what this
analysis found.** inferCNV's sensitivity depends on a cell having enough
expression to resolve a copy-number shift across the genome, so its call is
biased toward cells with more RNA and more detected genes. The handoff says
the missing third is "the cells whose CNV signal was weakest". This contrast's
result is that retained cells are proliferating -- cells with more RNA and more
detected genes. Adopting a compartment whose selection criterion correlates
with the outcome variable would add bias in the direction of the finding while
buying uniformity that does not bring the definition closer to truth, since
neither side has any. CellAssign is expression-based too, but it rests on a
handful of lineage markers rather than on genome-wide detection, so its
dependence on depth is weaker and less directly aligned with the outcome.

**What this costs, stated.** The malignant compartment is defined by one rule
for ovarian, three labels for prostate and eleven for colorectal, so the
malignancy method is not held constant across cancer types. That is a real
limitation of any cross-cancer comparison drawn here and belongs in Methods.
The prostate files come from a different source again and predate the uniform
call, so they could not use it even if the rest did.

### Inclusion is closed, 2026-09-24

**There is no pair-level inclusion criterion. Every pair with a computable gate
enters the pseudobulk.**

The following are **diagnostics**. They are computed, reported and available for
interpretation. None of them filters a pair in or out:

- `auc_predownsample_total_counts` -- the gate against each cell's original depth
- `auc_n_genes_by_counts` -- the gate against detected genes, the covariate the
  preprocessing does not act on
- the per-pair estimability of either, i.e. whether that pair has enough
  rejected cells for the standard error to resolve the quantity at all

This is recorded as closed because it was re-opened twice in one day: a depth
exclusion was added, withdrawn, and a detected-gene exclusion was then
proposed. Each version was data-dependent selection layered onto an already
prespecified analysis -- choosing which pairs to analyse using a statistic
computed from the analysis itself -- and each would have been decided by
whoever looked at the numbers last. The prespecified design does not have a
pair-level filter, and adding one now is not a refinement of it.

**Where the concern is discharged instead: at the patient level, after the
fit.** The depth concern is real; section 8 measured a third of ovarian and half
of prostate pairs over the depth bound. It is answered by asking whether the
result depends on particular patients, which is a question about the result
rather than a pre-emptive edit to its input:

1. **Leave-one-patient-out refit.** Drop each patient in turn and refit. A gene
   whose significance survives every leave-one-out fit does not rest on one
   patient. This is the literal form of Disqualifier 3 and **is not implemented
   anywhere in the repository** -- `31_audit_deg_disqualifiers.py` computes a
   per-patient sign-agreement proxy from CPM and its own docstring says it
   cannot replace the refit. It is new code.
2. **Patient-level meta-analysis.** Compute the contrast within each patient,
   then combine across patients with a minimum-patients requirement, so every
   gene carries the number of patients supporting it as a reported quantity
   rather than as something recovered afterwards.
   `12_meta_analyze_robust_target_deg.py` already implements this pattern for
   another contrast; `meta_table(patient_effects, minimum_patients)` is the
   reusable part.

Between them these subsume what a depth filter was meant to protect against. A
systematic depth artefact spread across a third of patients is exactly what the
original three disqualifiers could miss -- Disqualifier 3 catches single-patient
dominance, not a shared artefact -- and it is what a leave-one-out profile and a
patient-support count make visible.

**For context, not for filtering.** Of the pairs with a readable statistic, 61
of 92 ovarian, 12 of 24 prostate, 1 of 5 colorectal and 3 of 4 head and neck
clear `|auc_predownsample_total_counts - 0.5| <= 0.10`. Those numbers are
reported beside the result. They do not select it.

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
