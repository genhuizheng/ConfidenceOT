# Lineage ground-truth validation of ConfidenceOT: KP-Tracer and Quinn 2021

Status: **DRAFT, not frozen.** Freezing is a commit that changes this line to `FROZEN <date> <commit>`. After freezing, any change is a new dated section marked post hoc.

**Decisions taken by the user (2026-10-21):**
- KP-Tracer A and C2 are the pre-specified tests; everything else is exploratory.
- Acceptance 0.99 only, with no cell cap.
- Quinn target cells whose clone is absent from LL get a separate rejection test.
- Freeze before any gate is compared with lineage.

The audit behind every count here is `AUDIT_SUMMARY.md` (job 1087596).

## 0. Question and leakage rule

Can ConfidenceOT, fitted on RNA and/or CNV only, recover lineage-related matches? We test this at three levels:
- **site**: which primary or metastatic site is related;
- **clone / family**;
- **cell**: where a metastatic cell's coupling mass goes.

**Leakage rule:** lineage information never enters an OT fit, representation, QC, gate or calibration. This covers MetFamily, LineageGroup, trees, character matrices and the authors' metastatic scores. The OT input files carry only expression, the site (`sample_id`, the tumour or tissue name) and the malignant flag. Lineage labels are joined to OT outputs only after every run has finished.

## 1. Units and cells

- **Unit.** All work is within one mouse; the mouse is the independent unit.
- **Sites.**
  - KP-Tracer: tumours; primary = tumour id starting with T (the README rule, as in `01_prepare_kptracer_pairs.py`).
  - Quinn: the authors' `sampleID`; LL (`30kLL` in 30k) = injection site = the only primary.
- **Cells per site.**
  - KP-Tracer: the authors' malignant cells (`is_normal` false).
  - Quinn: cells with an author meta row, so that their site is known. Cells without a meta row are in no side.
- **Eligibility.** A site needs ≥ 20 such cells after the project's QC; this was fixed before the audit. LM0 (pre-injection) is not a site.

## 2. OT runs (all through the existing pipeline)

| Design | Source | Target | Used by | Datasets |
|---|---|---|---|---|
| D_A | all eligible primaries of the mouse, pooled | one eligible metastasis (one run per metastasis) | A, C2 | KP-Tracer |
| D_B | one eligible primary | all eligible metastases of the mouse, pooled (one run per primary) | B, C1 | KP-Tracer (mice with ≥ 2 eligible metastases), Quinn (LL) |
| D_C3 | all eligible primaries, pooled | all eligible metastases, pooled | C3 | KP-Tracer (≥ 2 and ≥ 2) |

- **Pooling.** A pooled side is one H5AD whose cells share one `sample_id`, with the original site in obs `site`. The pooled cells are exactly the union of the sites' cells.
- **Representation and gate.** `ranknm256_noscale_ds_cos`; ConfidenceOT M4-E (primary) and M4-R (reported); within-side split null; acceptance 0.99; ε = 0.1, λ = 1.
- **Cells used.** `--max-observed-cells-per-side 0`, so every cell is used. Couplings are saved (`--save-coupling`), and so are the joint coordinates (`--save-pairing-edges`).
- **Depth equalisation.**
  - KP-Tracer pooled sides are built from the existing equalised per-tumour files, the same files the 104-pair runs used.
  - Quinn per-site files go once through `27_downsample_counts.py` at its defaults, then are pooled.
- **Five strategies, one run each:**
  - `rna`;
  - `cnv`: the infercnvpy profile; KP-Tracer reference = the authors' normal cells, Quinn = none;
  - `rna_cnv`: cost = [C_rna/med(C_rna) + C_cnv/med(C_cnv)] / med. RNA is 2 − 2cos on unit PCs; CNV is squared Euclidean on CNV PCs. This was verified in the code;
  - `cnv_then_rna` and `rna_then_cnv`: step 2 is fitted only on the cells step 1 retained on both sides, so both steps reject.
- **Failures.** A run that fails is listed, never substituted.

## 3. Ground-truth labels (validation only)

### KP-Tracer
- **Site truth.** Primary s is related to metastasis M iff M's MetFamily equals s's tumour name.
  - Metastases with MetFamily `Unk` (3467) or with a family whose primary is not an eligible site (3460 N1) have no site truth. They are run but not scored in A/C2; they are listed.
- **Cell truth for a target cell j in metastasis M.**
  - The correct source group is the related primary's cells. Every other primary is a different lineage (an independent tumour).
  - Every malignant cell of a scored metastasis is **matchable**, because its family's primary is in the pooled source.
- **Tree relatedness.**
  - L_ij is the contracted MRCA depth, on a tree that holds both cells: the family tree `<family>_Fam`, else `<mouse>_<genotype>_All`. Per-tumour trees are never used across tumours.
  - Cells on no such tree have no tree label.

### Quinn 2021
- **Clone c(i)** = the authors' LineageGroup.
- **Label class of a target cell j:**
  - matchable: ≥ 1 LL cell of clone c(j);
  - unobserved_in_source: labelled, but no LL cell of c(j). This is not a confirmed negative;
  - ambiguous: no label (such cells are in no side, §1).
- **Correct source cells** for a matchable j: the LL cells of clone c(j).
- **Tree relatedness** within a clone, from that clone's tree.
  - Tree file: `*_tree_hybrid_priors.alleleThresh.processed.txt`; other files for the same clone are listed, not used.
  - Leaf names are matched to cell names as written; in 10k, as `<sampleID>.<leaf name>` (the rule that joins its meta table).
  - Cells of different clones have no tree relation.
- **Direction.** Neither dataset gives true migration direction (same-time sampling; Quinn reports multidirectional seeding). Results are stated as "same family / same clone / tree-related", never as "came from".

## 4. Coupling quantities

- **Inputs.** P = the M4-E coupling as saved: every analysed cell, rows and columns in `cell_confidence.csv` order. Retained and rejected flags come from `cell_confidence.csv`. Marginals: a = P1, b = Pᵀ1.
- **Relative matching weight:**
  - q_ij = P_ij·1[i retained]·1[j retained] / Σ_k P_kj·1[k retained];
  - q_ij = 0 when the denominator is 0 or j is rejected.
- **Source-group weight** of target j: W_j(g) = Σ_{i∈g} q_ij. The group g is the primary site (KP-Tracer) or the clone (Quinn).

## 5. Metrics

### Site level (A, B)

- r_s = fraction of source site s's cells that M4-E retains. For target sites: t_s = fraction retained.
- **A (KP-Tracer), per scored metastasis M:**
  - retention enrichment e_M = mean r(related) − mean r(unrelated);
  - Top-1: the related primary has the highest r (ties scored as the expected value of a random order);
  - pairwise AUROC of r for related vs unrelated primaries (ties ½).
  - Also reported: the coupling-mass share of each primary.
- **B (KP-Tracer), per primary s:**
  - with both related and unrelated metastases (3457 T1, T4; 3460 T1): the same three metrics on t;
  - with no related metastasis: the mean rejected fraction of all target cells (a negative control; high is correct).
- **B (Quinn).** Spearman, per mouse, between t_X and f_X. f_X is the fraction of X's labelled cells whose clone is observed in LL. No AUROC: no binary site truth.

### Cell level (C1, C2, C3)

For each scored target cell j with a correct group g*(j) (KP-Tracer: the related primary; Quinn: clone c(j) in LL):

| Name | Definition | Reference |
|---|---|---|
| correct-group weight w_j | W_j(g*) (0 if j is rejected) | chance π_j = share of source cells in g*; independent coupling a ⊗ b on retained cells |
| end-to-end score | mean over matchable j of w_j (rejected j count 0) | mean π_j |
| conditional score | mean over matchable **retained** j of w_j | mean over the same j of the a ⊗ b share Σ_{i∈g*} a_i / Σ a_i (retained i) |
| Top-1 / Top-k origin accuracy | g*(j) is the group with the largest W_j (k = 1; Quinn also k = 3); end-to-end (rejected = wrong) and conditional | the group with the largest a ⊗ b share |
| correct-group coupling mass | Σ_j Σ_{i∈g*(j)} P_ij / Σ_ij P_ij over retained cells | a ⊗ b |
| tree agreement | ΔS_norm and ΔR exactly as frozen in `kp_lineage/validation_v3/LEVEL3_VALIDATION_DESIGN.md`, on cells sharing a tree (§3) | a ⊗ b; target permutation |
| coverage | retained target fraction and retained source fraction, reported with every score; coverage–accuracy curve: target cells ordered by gate score r = −margin, Top-1 accuracy among the top fraction | — |

- **Quinn rejection test.** Rejection rate of unobserved_in_source vs matchable target cells, and the AUROC of the gate score between them, per target site. This tests "no counterpart should be rejected". These cells are excluded from the accuracy metrics.
- **Nulls and controls:**
  - independent coupling a ⊗ b;
  - group-label shuffle: permute g among source cells preserving group sizes (Quinn: within LL; KP-Tracer: across the pooled primaries), 1,000 permutations, seed 20261021;
  - baselines from the saved coordinates: balanced OT and kNN (k = 15), as in the Level 3 design;
  - clone size: report w by clone-size bin, with π_j beside it;
  - depth: Spearman of the gate score with pre-equalisation depth, per site;
  - retained fraction beside every score.

## 6. Pre-specified tests (KP-Tracer, strategy `rna`, M4-E)

- **T1 (A, site level).** Mouse mean of e_M over scored metastases.
  - One-sided Wilcoxon signed-rank, H1 > 0. 8 mice have scored metastases (3451, 3454, 3457, 3460, 3466, 3508, 3513, 3515; 31 metastases), so the smallest p is 1/256.
  - "Supported" if the mean of mouse means > 0 and p < 0.05.
- **T2 (C2, cell level).** Mouse mean of (end-to-end score − mean π_j) over the same metastases. Same test and rule.
- **Reported with both:**
  - the mean and median of mouse means;
  - 95% bootstrap interval over mice (10,000, seed 20261021);
  - mice > 0;
  - leave one mouse out;
  - Top-1 and AUROC (T1), conditional score and Top-1 (T2).
- **Everything else is exploratory, with no p values pooled over cells:**
  - the other four strategies (paired differences from `rna` per mouse);
  - M4-R;
  - B, C1, C3;
  - tree agreement;
  - all of Quinn;
  - baselines.

## 7. Not applicable (and why)

| Experiment | Where | Reason |
|---|---|---|
| A, C2 | KP-Tracer 3465, 3724; all Quinn mice | one eligible primary (Quinn: one injection site) |
| B, C1 | KP-Tracer 3466, 3513 | one eligible metastasis |
| C3 | KP-Tracer 3465, 3466, 3513, 3724; all Quinn | needs ≥ 2 primaries and ≥ 2 metastases |
| site AUROC in B | KP-Tracer except 3457, 3460; all Quinn | no primary with both related and unrelated metastases; Quinn has no binary site truth |
| any scoring | KP-Tracer 3467, 3460 N1 | no family truth |
| tree agreement | cells on no shared tree (e.g. metastases of 3451, 3465, 3466, 3467; 3454 L1; Quinn 100k clone 1) | no tree holds both cells |

## 8. Outputs

All under `/scratch/10119/ghzheng/OT_project/lineage_validation/`; nothing existing is overwritten.

- `inputs/`: pooled H5ADs and manifests.
- `runs/<dataset>/<strategy>/...`: the pipeline's own layout.
- `labels/`: cell, site, family or clone, tree leaf.
- `scores/`: per run, target cell and site.
- `summary/`: mouse and dataset tables.
- `figures/`: per mouse:
  - A/B retention heatmaps and rankings;
  - C1–C3 coupling heatmaps and clone/primary-level coupling matrices;
  - lineage-distance comparisons;
  - the five-strategy comparison;
  - a fixed joint UMAP (`48_reference_umap.py`) coloured by site, lineage label and retain/reject/exclude.

DEG/GSEA, through the existing downstream, only describes retained and rejected cells; it is never ground truth.
