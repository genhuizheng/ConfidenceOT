# CRISPR lineage (KP-Tracer, Quinn 2021): what exists, and the real-data benchmark on the v3 grid

Status: **DRAFT 2026-11-08, for the user's review.** It consolidates the 34 earlier Markdown files under
`lineage_validation/` and `kp_lineage/` (list in §7) and defines the benchmark of §4. No benchmark score has been
computed. Once the user confirms §4, it is frozen: any later change is a dated section at the end, never an edit.

## 1. Data

| | KP-Tracer (Yang, Jones et al. 2022) | Quinn 2021 (GSE161363) |
|---|---|---|
| What | Cas9 lineage-traced KP mouse lung adenocarcinoma: primary tumours (`T*`) and metastases (other suffixes) per mouse | A549 xenograft, left lung (LL) → other tissues, 4 mice (5k, 10k, 30k, 100k); LM0 not used |
| Source | Zenodo 5847462 `KPTracer-Data.tar.gz` (md5 353d18a618f60e975089f2d278acaa88) | GEO GSE161363, converted on TACC |
| Counts | `adata_processed.combined.h5ad` `raw/X`: 104,197 cells, 27,998 genes | converted H5ADs |
| Malignant | `not is_normal` (authors' label) | every cell (A549) |
| Lineage truth | `MetFamily` per tumour (`_Unk` = unknown); Cassiopeia trees per tumour and `_Fam` / `_All`; character matrices | `LineageGroup` (clone) per cell; one tree per clone (116 of 148 clones) |
| Trees | no branch lengths; unary nodes; polytomies up to 8,362 children (3724_NT_T1), 6,519 (3724_NT_All), 1,033 (3508_Apc_All) | `*_tree_hybrid_priors.alleleThresh.processed.txt` |

- **KP pairs** (`kp_lineage/01_prepare_kptracer_pairs.py`): every metastasis × every primary of its mouse. 131 pairs,
  **104 eligible** (≥ 20 malignant cells per side), 40 of them same-family, 11 mice with eligible pairs.
- **L1 test set:** every metastasis with at least one same-family and one other-family eligible primary in its mouse:
  **31 metastases in 8 mice** (3451, 3454, 3457, 3460, 3466, 3508, 3513, 3515). Mice 3465 and 3724 have one primary;
  3460_Lkb1_N1 and 3467's metastases have no sampled same-family primary.
- **Subclones** (`within_family_subclones/SUBCLONE_RULES.md`, fixed a545f1c, 2026-10-22): from the authors' trees only.
  - A cell is reliable if its missing share is ≤ 0.5.
  - A branch is supported by a marker edit: ≥ 50% of its cells non-missing at the character, ≥ 80% of those carry the
    state, ≤ 20% elsewhere.
  - A level-1 subclone is a supported child with ≥ 20 reliable cells at the first split; level 2 is the same rule inside
    each level-1 subclone.
  - Eligible families: 26 in 9 cases (KP 3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1; Quinn 100k 4 clones, 10k 3,
    30k 2, 5k 12).
- **Known confound:** mouse 3724 carries most of the fine-grained signal, and there both the gate score and the tree
  depth follow sequencing depth (Spearman −0.45 to −0.53 in L1-L3, +0.63 in S1; D1 post hoc).

## 2. What has been done (verified numbers; earlier report errors corrected)

| # | Analysis | Representation, method | Result |
|---|---|---|---|
| 1 | **L1 tumour-level readout**, 104 pairs (2026-10-14/15) | `ranknm256_noscale_ds_cos` and `raw`, M4-E, acceptance 0.99 | d = same-family − mean other-family source retained fraction. ds arm: median d 0.086, mean 0.144, 27/3/1 (+/−/0) of 31; one-sided Wilcoxon p 5.0e-7 (exact; 7.5e-6 normal approximation); median d > 0 in 8/8 mice; mouse-mean Wilcoxon p 0.0039. raw arm: median 0.006, p 0.017, 5/8 mice. Target side (ds): median 0.032, 5/8 mice |
| 2 | **v2 E1**: shared-ancestry depth s, retained vs rejected source cells (Cliff's δ), 26 pairs / 7 mice (frozen prereg b0817f1) | ds arm, M4-E | mean of mouse means 0.070 (CI −0.028 to 0.209), 4/7 mice, p 0.289: **inconclusive**. Without 3724: 0.007. E1b 0.083, p 0.039 on Cassiopeia trees, 0.049, p 0.344 on NJ trees |
| 3 | **Frozen protocol T1 / T2** (VALIDATION_PROTOCOL.md, 508609f): pooled primaries → one metastasis, 31 runs in 8 mice | `rna` = ranknm256_noscale_ds_cos, M4-E | T1 (e_M, site retention) 0.270, Holm p 0.0234; T2 (E2E − chance) 0.078, Holm p 0.0234; both **supported**. But T2 is about 90% gate: COND − COND_ind = 0.0066 (CI −0.0013 to 0.0155); asserted precision equals the independent coupling's to ≤ 0.016. 3508 and 3513 carry it. 5 of the 8 3508 runs have `calibration_m4e_inference_valid` False. All 204 baseline runs FAILED (separability check); depth control missing on single-tumour sides. The results exist only in the TACC export (`EXP/summary/tests.tsv`); no committed report stated them before this file |
| 4 | **Within-family subclones**, 4 local families (2026-10-22) | protocol couplings | Subclone coupling follows the retained source composition (deviation 0.01-0.07; argmax = independent argmax in 89-100%); OT never beats the size rule. The 22-family TACC run never happened; superseded by Level 2 |
| 5 | **Level 2**: subclone identification, 9 cases / 26 families × RNA, CNV, RNA+CNV (2026-10-22) | A = logCPM PCA 30 + balanced OT; B = ranknm256_noscale_ds_cos + UOT with rejection budget 0 (**not** an M4-E gate) | Balanced accuracy − chance, median −0.024 to +0.029. Above chance only where sites are seeded by different branches: KP 3724 T1 level-2 BA 0.93-0.99; 3515 L1 site recall 0.85-0.93; Quinn 100k LG4 CNV 0.856. Rare subclones (π < 0.10) almost never recovered (median recall 0). A ≈ B |
| 6 | **Level 3**: tree distance of matched primaries, same 54 couplings | as Level 2 | ΔD_norm = (D_ind − D_OT) / (D_ind − d_min); 39,849 evaluable metastatic cells (the report's "47,000+" is wrong). ΔD > 0 in 22-25 of 26 families, 7-8 of 8 mice, but median mouse ΔD_norm only 0.019-0.029; 3724 0.32-0.36, and it is subclone-level (within-subclone 0.001-0.018) |
| 7 | **validation_v3 Level 3** (ΔS_norm on MRCA depth; design frozen 7fe6202) | — | design and a 12/12 coupling-reproduction pilot only; metric code (stage F) never written |
| 8 | **v3 grid on TACC** (2026-11-04/05) | 296 configurations, calibration v3, M4-E and M4-R | **KP complete**: 4,181 rows (113 units × 37 keys) × 8 = 33,448 fits, 0 failed; collect done 2026-11-05. Gates and both couplings saved for every fit. **Not scored yet** (this file, §4) |

Other work, not part of the benchmark:
- isCORAL audit (2026-10-31): unsupervised, but not reproducible end to end; not used.
- CRISPR-site input feasibility (2026-10-31): held-out intBC design possible for 3454, 3457 T1, 3508 and 3515;
  marginal for 3724 and 3457 T4 (about 3 intBCs). Out of scope after the user's 2026-11-01 RNA-only decision.
- Lineage × site application (Phase 0-1 done; 2-6 not started); local lineage figures (done).

**What the earlier work says about the signal:**
- Family level (L1): the gate separates families.
- Within a family (L2/L3): the coupling mostly follows source composition. Real fine-grained signal is confined to
  3724 (subclone level, depth-confounded) and a few Quinn clones.

## 3. Definitions reused (exact)

- **Retained fraction of a side:** 1 − `source_final_rejection_rate` (or `target_`) of the M4-E row of
  `pair_metrics.csv`. A cell is retained when `signed_rejection_margin` < 0.
- **L1 d** (READOUT_PRESPECIFICATION_2026-10-14): per metastasis, d = mean same-family source retained fraction − mean
  other-family source retained fraction, over completed pairs.
- **Retained-block coupling (L2, L3):** P with every rejected source row and rejected target column set to 0. Row
  mass m_i = Σ_j P_ij; column mass c_j = Σ_i P_ij; q_ij = P_ij / c_j for c_j > 0.
- **L2** (level2_pairwise_ot/PLAN.md, scorer `score_pairwise.py`):
  - S_gj = Σ_{i∈g} q_ij and S̃_gj = S_gj / Σ_{g∈F} S_gj, where F includes `family unlabelled`.
  - π_g = Σ_{i∈g} m_i / Σ_{i∈F} m_i and E_gj = S̃_gj / π_g. E is defined only for candidates: π_g ≥ 0.01 and ≥ 5 source
    cells.
  - Prediction: argmax of E over the candidates.
  - Balanced accuracy = mean recall over the true subclones.
  - Chance = 1 / number of candidates.
  - A family-level is not evaluable with < 2 true subclones or < 2 candidates.
- **L3** (level3_cell_lineage/PLAN.md, `l3_core.py`):
  - d(i, j) = depth(i) + depth(j) − 2·depth(LCA), counted in branching steps.
  - S_j = the reliable primary leaves of j's family tree.
  - D_OT = Σ_{i∈S_j} q̃_ij d(i, j) with q̃ = q / Σ_{S_j} q; D_ind = Σ_{i∈S_j} π̃_i d(i, j) with π̃_i = m_i / Σ_{S_j} m.
  - ΔD_norm = (D_ind − D_OT) / (D_ind − d_min).
  - Status: `not_evaluable` (not a reliable leaf, or |S_j| < 5); `not_informative` (every source at the same distance);
    `ambiguous` (the nearest set holds ≥ 50% of S_j); `evaluable`.

## 4. The benchmark (to confirm)

**Question:** with the v3 grid fixed, which RNA preprocessing gives ConfidenceOT the best agreement with CRISPR
lineage? CRISPR is used only to score: never in a cost, the calibration or a choice of configuration.

**Fits** (already on TACC, `/scratch/10119/ghzheng/OT_project/v3_grid_20261104/kp/`; nothing new is fitted):

| Item | Value |
|---|---|
| Units | `UNITS.tsv` of the lineage factorial: 104 L1 pairs (`L1__<pair_id>`) and 9 L2/L3 cases (`L23__<case_id>`: KP 3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1; Quinn 100k, 10k, 30k, 5k) |
| Configurations | 296 = 37 encoding keys × PCA {30, 50} × scaling {none, z-score} × cost {cos, sqeuc}; no depth equalisation |
| Calibration | v3: within-side split null + `mnncovlocal`, acceptance 0.99, no rejection cap |
| Method | M4-E primary; M4-R saved, not used to choose |
| Saved per fit | `cell_confidence.csv.gz`, `pair_metrics.csv`, `run.json`, `coupling_m4e.npz` / `coupling_m4r.npz` (float32, `source_ids`, `target_ids`); collected in `FIT_METRICS_v3tacc.tsv.gz` and `CELL_GATES_v3tacc/` |

**Metrics** (one primary per level; secondaries are reported beside it, never combined):

| Level | Data | Primary | Secondary |
|---|---|---|---|
| L1 family | 104 pairs; 31 metastases in 8 mice | mean over mice of the per-mouse mean d | mice with mean d > 0; median d over metastases; target-side d |
| L2 subclone | 9 cases, M4-E retained block | balanced accuracy − chance per evaluable family-level, mean over mice of the per-mouse mean | target coverage (retained share); size-rule accuracy; rare-subclone recall; the all-cells variant (rejected target = unassigned) |
| L3 cell | 9 cases, M4-E retained block | ΔD_norm, median over evaluable cells within a family, then mean over mice | coverage; within-subclone ΔD_norm |

- At L2/L3 a Quinn experiment counts as one mouse.
- Flags are columns, never filters: `calibration_m4e_inference_valid`, convergence, and the gate score's depth
  correlation.

**Reading** (fixed before any score is seen):
1. **Per level:** rank the 296 configurations by the primary metric.
2. **Best set:** every configuration whose mouse-bootstrap 95% interval for its difference from the top configuration
   contains 0 (10,000 resamples of mice). The whole set is reported, not one winner.
3. **Leave-one-mouse-out:** best-set membership with each mouse left out. Membership that depends on 3724 is flagged.
4. **Comparison with Splatter:** for every configuration, its lineage rank at each level beside its Splatter v3 rank
   (L5 Case 2 F1), and in particular for Splatter L5 best (`ranknm256-midrank|none|scale|cos|pca50`), the production
   baseline (`ranknm256-production|none|noscale|cos|pca30`) and logCPM + PCA only (`logcpm|none|noscale|sqeuc|pca30`).
   Lineage checks the Splatter choice; it is not a second selection. An empty overlap is reported as it is.
5. **Publication:** all 296 × 3 levels are published, with their flags.
6. **Rules:** no composite score and no PASS / FAIL.

**Code needed** (versioned, tested locally before any TACC job):
- `score_l1.py`: from `FIT_METRICS_v3tacc.tsv.gz` and `UNITS.tsv`; small inputs, so it runs locally.
- `score_l23.py`: on TACC, one task per case. It reads each configuration's `coupling_m4e.npz` and gates, builds the
  retained block, and calls the existing L2 / L3 functions (`score_pairwise.score_one`, `l3_core.ground_truth`,
  `l3_core.cell_metrics`).
  - Sites come from Level 2's `cells.tsv.gz`, matched by cell id.
  - The couplings stay on TACC: about 117 GB for M4-E alone, 3724 is about 93 GB of that.
- `summary.py`: best set, leave-one-mouse-out and the Splatter comparison.
- **Local test:** the small case 3457 T4 (about 24k entries per coupling), downloaded for all 296 configurations.

**Differences from `preprocessing_factorial_20261103/PLAN.md`** (never confirmed; this section replaces it):
- 296 configurations at PCA 30 and 50 with no depth equalisation, instead of 592 at PCA 50 with DS
  {none, P10, P25, P50}.
- Calibration v3 (`+mnncovlocal`) instead of the within-side split null alone.
- No reproduction check of the earlier `ranknm256_noscale_ds_cos` gates: that configuration is not in the v3 grid.
- L1 is read as the mean of mouse means with a best set, not by the readout's Wilcoxon over 31 metastases. The readout
  script also dropped d = 0 values, which its prespecification never stated.

**Limits known before scoring:**
- **L1:** 8 mice. 3508 and 3513 carry most of the T2 effect.
- **L2/L3, evaluability:** few evaluable family-levels (17 at subclone level 1 and 12 at level 2 in the earlier run),
  and 14 of 29 have a true subclone with < 20 cells.
- **L2/L3, input selection:** the inputs were chosen by lineage (`subclone_structure_eligible`), so they are not
  lineage-blind.
- **3724:** depth-confounded.

**To decide (user):**
1. Baselines: none (as the plan), or add balanced OT on logCPM + PCA for the same units. Adding it needs new TACC fits.
2. Quinn: include its 4 cases at L2/L3 (default), or KP only.
3. Whether L1 also reports the prespecified one-sided Wilcoxon over the 31 metastases per configuration (descriptive),
   beside the mean of mouse means.

## 5. Where things are

| What | Path |
|---|---|
| KP pair preparation, L1 readout | `kp_lineage/01_prepare_kptracer_pairs.py`, `kp_lineage/02_kptracer_readout.py` |
| Frozen protocol code (T1/T2) | `lineage_validation/00_audit.py` … `06_figures.py`, `submit_runs.sh`, `submit_score.sh` |
| Subclone truth | `lineage_validation/within_family_subclones/` (`define_subclones.py`, `output/feasibility.tsv`, `output/cells/`) |
| L2 / L3 code and inputs | `lineage_validation/level2_pairwise_ot/` (`inputs/cases.tsv`, `family_cells.tsv.gz`, `subclones.tsv`, `score_pairwise.py`); `lineage_validation/level3_cell_lineage/` (`l3_core.py`, `score_level3.py`) |
| Units and factorial code | `lineage_validation/preprocessing_factorial_20261103/` (`01_build_units.py`, `10_worker.py`) |
| v3 grid code | `realdata_benchmark/v3_grid/` (`worker.py`, `collect.py`, `tacc/submit_v3.sh`) |
| Local exports | `results/kp_tracer_lineage_validation/export/` (trees, character matrices, v2 gates); `lineage_validation/local_lineage_visualization/data/` (protocol summary, couplings of 3508, 3515, Quinn 10k); `lineage_validation/level2_pairwise_ot/data/full_20261022_204810/` |
| TACC | KP `/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/`; protocol `.../lineage_validation/`; v3 grid `.../v3_grid_20261104/kp/`; units `.../lineage_preprocessing_factorial_20261103/UNITS.tsv` |

## 6. Naming that collides in the older files

- **"Level 3"** means two different things:
  - validation_v3: ΔS_norm on the MRCA root depth (a similarity);
  - `level3_cell_lineage`: ΔD_norm on tree distance (a distance).
  §4 uses the second.
- **"L1/L2":**
  - this file: levels;
  - subclone files: subclone levels;
  - KP: liver-metastasis names (e.g. 3724 L1-L3).
- **"A/B":**
  - protocol: designs (site D_A / D_B);
  - Level 2: methods (balanced OT / UOT with budget 0);
  - CRISPR-site audit: designs.
- **"Site":**
  - anatomical site (protocol);
  - CRISPR target site (2026-10-31 site audit).
- **Level 2 "B" is not a ConfidenceOT M4-E gate result.**

## 7. The 34 files it consolidated

**2026-11-08, user:** the 8 frozen or fixed documents below are kept as they are; the other 26 were deleted. They are
recoverable from git: every one of them is in the commit before the deletion (the 7 that were untracked were committed
first for this purpose).

Frozen or fixed documents, kept (definitions above are copied from them):
- `lineage_validation/VALIDATION_PROTOCOL.md` (frozen 508609f).
- `kp_lineage/validation_v2/VALIDATION_V2_PREREGISTRATION.md` (frozen b0817f1).
- `kp_lineage/validation_v3/LEVEL3_VALIDATION_DESIGN.md` and `METRIC_TAXONOMY_AND_FEASIBILITY.md` (frozen 7fe6202).
- `lineage_validation/within_family_subclones/SUBCLONE_RULES.md` (fixed a545f1c).
- `lineage_validation/level2_pairwise_ot/PLAN.md` and `level3_cell_lineage/PLAN.md` (fixed 2026-10-22).
- `lineage_validation/lineage_site_application/GROUND_TRUTH_DEFINITIONS.md` (fixed 2026-10-22).

Reports, audits, plans and indexes, deleted (summarised in §2):
- `lineage_validation/`:
  - `AUDIT_SUMMARY.md`;
  - `level2_pairwise_ot/REPORT.md`, `level2_pairwise_ot/strategy_check/README.md`;
  - `level3_cell_lineage/README.md`, `REPORT.md`;
  - `lineage_site_application/PLAN.md`;
  - `local_lineage_visualization/README.md`;
  - `within_family_subclones/README.md`, `AUDIT_2026-10-22.md`, `audit/AUDIT_REPORT.md`;
  - `preprocessing_factorial_20261103/PLAN.md`;
  - `iscoral_extension_20261031/`: `EXISTING_LINEAGE_VALIDATION_AUDIT.md`, `ISCORAL_LITERATURE_AUDIT.md`,
    `ISCORAL_METHOD_REPRODUCIBILITY.md` (untracked);
  - `lineage_benchmark_v2_20261031/`: `BASELINE_COMPATIBILITY.md`, `CRISPR_SITE_FEASIBILITY_AUDIT.md`,
    `INPUT_SETTING_AUDIT.md`, `PREVIOUS_EXPERIMENT_AUDIT.md` (untracked).
- `kp_lineage/`:
  - `READOUT_PRESPECIFICATION_2026-10-14.md` (status never changed from "proposed");
  - `tumour_level/DECK_20261015_v3_SLIDES.md`, `TUMOUR_LEVEL_REPRESENTATION_AUDIT.md`;
  - `validation_v2/D1_RESULTS_2026-10-15.md`, `LINEAGE_DATA_AUDIT.md`, `VALIDATION_V2_EXECUTION_PLAN.md`;
  - `validation_v3/LINEAGE_VALIDATION_LITERATURE_REVIEW.md`, `TACC_EXECUTION_PLAN.md`.

Errors in those files that this one corrects:
- **Level 3 REPORT:** "47,000+ evaluable" should be 39,849; its "low tree resolution" flag on 3724 is not set in the
  table.
- **Level 2 REPORT:** "own-clone mass about 0.3" includes two KP families at 1.0 (Quinn alone 0.20 / 0.20); the
  rare-subclone counts and coverage medians are evaluable-only.
- **Within-family README:** its 0.25 share for 3515 L1.2 should be 0.228.
- **v2 and v3 execution plans:** their stage tables were never updated.
