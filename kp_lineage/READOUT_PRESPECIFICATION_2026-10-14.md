# KP-Tracer readout, fixed before any gate is seen (2026-10-14)

Status: **proposed, awaiting the user's confirmation.** It was committed before the OT run was submitted. Any later change is a new dated section, never an edit of this one.

## What was run

- **Data:** KP-Tracer (Yang, Jones et al. 2022), `raw/X` integer counts.
- **Cells:** malignant means `not is_normal`, the authors' label.
- **Pairs:** `kp_lineage/01_prepare_kptracer_pairs.py` pairs every metastasis with every primary tumour of its mouse. That gives 131 pairs, of which 104 are eligible (at least 20 malignant cells per side).
- **Depth:** `27_downsample_counts.py` with the default QC (1000 counts, 500 genes, 20% mitochondrial), to the pooled 10th percentile.
- **Gate:**
  - representation `ranknm256_noscale_ds_cos`;
  - M4-E gate, within-side split null, acceptance 0.99;
  - no per-side cap (`--max-observed-cells-per-side 0`).
- **Truth:** the authors' `MetFamily` only. A pair is `same_met_family` when the two tumours share it.

## Primary readout

- **Per pair:** the source retained fraction, i.e. the share of the primary tumour's analysed malignant cells that the M4-E gate retains.
- **Who enters:** every metastasis that has at least one eligible same-family primary and at least one eligible other-family primary in its mouse. That is 31 metastases from 8 mice.
- **Per metastasis:** `d` = the same-family retained fraction minus the mean of its other-family retained fractions.
- **Test:** a one-sided Wilcoxon signed-rank test of `d > 0` across these metastases. The test is supported in the expected direction if the median of `d` is above 0 and p < 0.05.

## Reported alongside, never instead

- **Per mouse:** in how many of the 8 mice the median `d` is above 0. Metastases in one mouse share their primaries, so they are not independent.
- **Target side:** the same contrast on the target retained fraction, i.e. the share of the metastasis's cells retained.
- **Every pair:** every pair's values, including the 9 metastases whose mouse has a single primary (3465, 3724) and the 4 whose family is not a sampled primary (3460_Lkb1_N1 and 3467_Lkb1_N1/N2/Rib). These are reported descriptively. Nothing is dropped, filtered or substituted.
- **A pair that fails:** reported as missing, with its error. It does not enter the test.

## Not part of this readout

- Any clade or subclone label derived from the trees.
- The cross-mouse mismatched control (`28_build_mismatched_manifest.py`). It is the next step and gets its own dated section.
- Any change to the configuration after the gates are seen.
