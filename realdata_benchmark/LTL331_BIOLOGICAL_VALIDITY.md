# LTL331 (GSE297328) biological validity of the frozen run

Scope: the frozen LTL331 run (`ranknm256_noscale_ds_cos`, `logcpm` = library size 1e4 + log1p; 7 adjacent pairs; M4-E /
M4-R; acceptance 0.99; job 1111186) audited locally on 2026-10-31. Nothing in the frozen outputs was changed; every new
fit is in its own directory. The configuration benchmark (Rank-K, Rank-HVG, DS levels, tie rules) is reported
separately (`PDX_GATE_COUPLING_METRICS.tsv`, `RANK_K_HVG_RESULTS.tsv`, `DOWNSAMPLING_ROBUSTNESS.tsv`).

Sources: `results/ltl331_nepc/analysis_20261031/` — `gate_summary/` (05_), `validity_audit/` (07_), `deg_gsea/` and
`deg_gsea_equalised_counts/` (06_), `composition_sensitivity/` and `composition_size_control/` (08_); and
`results/realdata_benchmark_20261031/DOWNSAMPLING_INFORMATION_LOSS.tsv` (02_).
Labels: O = observed result, T = technical association, B = biological inference.

## 1. Run integrity and calibration

- O. Conversion: all 51,726 author cells, counts reproduce the authors' `.raw` to 8.9e-16; 50,492 cells after QC;
  P10 target 2,817 UMIs. Both arms 7 / 7 pairs. The cost of every pair was rebuilt locally from the code at the run's
  commit and equals `run.json` (scale, median, max).
- O. M4-E inference validity (formal statements only on valid pairs):

  | pair | ds_cos | logcpm |
  |---|---|---|
  | 0 -> 2 wk | valid | **invalid** |
  | 2 -> 4 wk | **invalid** | **invalid** |
  | 4 -> 8 wk | valid | valid |
  | 8 -> 12 wk | valid | **invalid** |
  | 12 -> 16 wk | valid | valid |
  | 16 -> 20 wk | valid | valid |
  | 20 -> 22 wk | valid | valid |

- O. M4-R: 14 / 14 fits did not converge: the outer loop reached its 30-iteration cap with a gate cycle of length 2;
  its calibration is invalid in every fit (`m4r_failure_mode.tsv`). Not used for any statement; no solver default was
  changed.

## 2. NE preferential retention at 16 -> 20 weeks

- O. 16-week source cells retained (M4-E): NE 0.68 vs PRAD / Basal / Other 0.05 / 0.06 / 0.11 (ds_cos); NE 0.32 vs 0.00
  (logcpm). NE minus non-NE: 0.63 (cell bootstrap 95% CI 0.56-0.69; ds_cos), 0.32 (0.25-0.39; logcpm). Cells are the
  units (one tumour per time point): intervals describe cell sampling in this tumour, not biological replication.
- T. Depth: NE cells were sequenced shallower here (16 wk median 4,611 UMIs vs PRAD 10,249). Within depth quintiles
  NE stays the most retained type in every quintile (ds_cos rejected fraction NE 0.21-0.65 vs 0.83-0.99 for the others); the
  Mantel-Haenszel odds ratio over depth quintiles is 22 (15-32; ds_cos) and 537 (134-2,160; logcpm, near
  separation). Depth does not explain the contrast, but see T-items in section 4.
- T. OT-free RNA similarity: the k = 10 mean cost of a 16-week cell to the 20-week cells (the cost OT used) predicts
  the OT gate almost perfectly (AUC 0.98 ds_cos, 0.94 logcpm), and NE cells are closer to the 20-week cells (AUC
  0.83 ds_cos). A kNN gate without OT, calibrated like the within-side null, also retains NE preferentially (0.87 vs
  0.43-0.57 ds_cos; 0.40 vs 0.21-0.32 logcpm). The direction is therefore already in plain expression similarity;
  OT sharpens it. A logistic model with NE, log depth and the kNN cost keeps an NE odds ratio of 42 (17-103; ds_cos),
  but that coefficient depends on the linear form of the kNN term and is not read as an effect beyond similarity.
- T. Target composition: the 20-week target is 90% NE (1,198 / 1,328). With every non-NE target cell kept and the NE
  target cells subsampled, NE minus non-NE source retention falls with target NE abundance: ds_cos 0.38 (50% NE),
  0.23 (20%), 0.16 (0% NE); logcpm 0.27, 0.05, 0.03. A size-matched control (same target sizes, cells drawn at
  random, ~90% NE) keeps it at ds_cos 0.57 / 0.51 / 0.48 and logcpm 0.67. The preference is therefore largely a
  consequence of the target being NE-rich. Smaller targets also raise retention for every type and often make the
  calibration invalid: ds_cos invalid in 7 / 7 NE-subsampled and 6 / 9 size-control fits; logcpm in 2 / 7 and 4 / 9.
  These sensitivity fits are therefore descriptive, not formal results.
- B (supported, descriptive): 16-week NE cells are, of the four annotated types, the ones transcriptionally closest to
  the 20-week relapse, which is mostly NE; this holds in both representations and at matched depth.
- B (not supported): that 16-week NE cells are the ancestors of the relapse, that PRAD cells did not contribute, or any
  fate / transdifferentiation path. Gates and couplings measure similarity between two samples of one tumour taken
  at different times; there is no lineage information in these data.

## 3. Other pairs

- O. NE minus non-NE source retention at the other valid pairs: 12 -> 16 wk 0.19 (0.12-0.24; ds_cos) and 0.09
  (0.04-0.13; logcpm); 20 -> 22 wk -0.03 (ds_cos) and 0.22 (logcpm): the two arms disagree in sign at 20 -> 22 and
  at 0 -> 2 (target side). Agreement in direction holds at 12 -> 16 and 16 -> 20 on both sides.
- O. Latent time (authors' velocity): later latent time goes with retention within types, most in NE (rho of latent
  time with the signed margin -0.43 to -0.63 at 8 -> 20 wk; ds_cos). Cell cycle: G1 cells are rejected more than S /
  G2M at most pairs. Both are associations within one tumour and share the depth caveat below.

## 4. Technical findings that bound every statement

- T. Depth: in ds_cos, cells that were deeper before equalisation are rejected more (AUC 0.64-0.77 in most pairs,
  also within each cell type); in logcpm shallower cells are rejected more. Equalisation does not remove the depth
  association; it reverses its sign.
- T. P10 equalisation removes 42-74% of each cell's detected genes; a cell's top-256 genes after equalisation share
  42-50% with its own top-256 on the original counts, and two equalisation draws of the same cell share 31-42%
  (`DOWNSAMPLING_INFORMATION_LOSS.tsv`). Shallow cells lose less and are encoded more stably (16 wk NE 0.50 vs PRAD
  0.38 draw-to-draw), a technical route by which NE cells can look more self-consistent.
- T. Rank ties: at 2,817 UMIs the count at rank 256 is typically 2, shared by 230-270 genes, so about 70-80 of a cell's
  256 slots (28-31%) are filled by tie-breaking. Production breaks ties by the stored order of the joint sparse
  matrix, an implementation-defined permutation of the genes (not alphabetical); replacing it by the expected encoding
  changes the equalised representation's PCA subspace to 0.92 similarity (0.99 on original counts).
- T. Arms: the two representations agree on cell types (type-level Spearman 0.8-1.0 at 12 -> 16 and 16 -> 20) but not
  on cells (kappa -0.09 to 0.35). Individual cell calls are not stable across representations; type-level patterns at
  those two pairs are.
- T. Coupling: rows of the M4-E coupling are nearly uniform (median normalised entropy 0.84-1.00); the kernel
  exp(-C / 0.1) is itself flat because the scaled cosine costs vary by little relative to epsilon. Annotation-level
  transport at 16 -> 20 is within 1.02-1.06x of the abundance-preserving permutation null (z up to 6 only because
  cells are many); earlier pairs show a modest NE -> NE excess (1.5-4x; largest 12 -> 16 wk). The coupling is
  uninformative at the cell level and weakly informative at the annotation level; no cell-to-cell ancestry or fate is
  inferred from it.

## 5. DEG / GSEA (exploratory)

- T. The pre-specified primary DEG (original counts, library-size normalised, log1p, cell-level Wilcoxon, rejected vs
  retained within type) is dominated by a global shift: per unit the median Wilcoxon score runs from -4.75 to +3.65 and
  the positive-gene share from 1% to 95%, in the direction of each arm's depth association (Spearman with the unit's
  depth AUC 0.92 logcpm, 0.67 ds_cos). gseapy returns NES = +-1.000 with p = 1 when the permutation null has no ES of
  the observed sign (38 / 3,882 rows): those rows are not estimates.
- T. On the same cells' equalised counts the shift largely disappears (median score -0.36 to +0.20; positive share
  0.17-0.64; association with depth AUC 0.36 and 0.00) and NES correlate with the primary at 0.79-0.84 (sign
  agreement 0.78-0.84).
- Audit rule (post hoc, set after the score distributions were seen): a unit is read only if, on equalised counts,
  |median score| <= 0.1 and the positive-gene share is within 0.4-0.6. 26 / 40 (logcpm) and 28 / 38 (ds_cos) units
  pass.
- O (exploratory). 16 -> 20 wk, ds_cos, units that pass: 16-week Basal and Other cells without a supported relapse
  match (606 / 437 rejected against 37 / 54 retained) are lower in E2F / G2M / MYC / OXPHOS and higher in TNFa-NFkB,
  inflammatory, IL6-JAK-STAT3, TGFb, EMT and androgen response than the retained ones; among 20-week NE cells, those
  without a 16-week match are higher in MYC targets / OXPHOS / mTORC1. Few retained cells, one tumour, cell-level
  tests: hypotheses only. The 16-week NE and PRAD units fail the audit and are not read.

## 6. Classification

| supported (descriptive, this tumour) | technical confounding | exploratory | unsupported |
|---|---|---|---|
| 16-wk NE cells are the annotated type most similar to the NE-dominated relapse, at matched depth and in both arms; 12 -> 16 wk NE also preferentially retained in both arms | depth association of the gate (sign set by the representation); P10 information loss and draw-to-draw instability; tie-decided rank slots; target composition and size; cell-level disagreement between arms; DEG global shift | pathway differences of retained vs rejected cells; latent-time and cell-cycle associations; annotation-level NE -> NE excess at earlier pairs | NE cells as relapse ancestors; PRAD non-contribution; transdifferentiation paths; cell-to-cell fate from couplings; anything from M4-R |
