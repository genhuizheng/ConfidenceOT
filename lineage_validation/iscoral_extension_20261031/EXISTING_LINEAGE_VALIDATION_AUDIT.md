# Audit of the existing ConfidenceOT lineage validation (KP-Tracer, Quinn 2021)

Status: **2026-10-31, repo HEAD 43993165ed990635f0db7693913191cdd99d50e7 (4399316), read-only.** No existing file was changed, no OT was run, no TACC job was submitted. Every number marked "verified" was recomputed from the named table with `D:\miniconda\python.exe` (pandas, scipy). Numbers marked "report only" could not be recomputed from a local table.

## Path abbreviations

All paths are relative to `D:\Xia_lab\CellOT\`. "untracked" means the folder is a local download that git does not track.

| Abbreviation | Path | In git? |
|---|---|---|
| EXP | `lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export/` (TACC export, provenance commit e83c93d) | untracked |
| EXP/summary | EXP`summary/`, a byte copy of TACC `/scratch/10119/ghzheng/OT_project/lineage_validation/summary/` (`05_aggregate.py`; its `provenance.json` gives commit 8f4e855, protocol FROZEN 508609f) | untracked |
| L2 | `lineage_validation/level2_pairwise_ot/` (`results/` tracked; `data/` = 8.1 GB untracked download) | partly |
| L3 | `lineage_validation/level3_cell_lineage/results/` | `summary/` tracked; `ground_truth/`, `scores/`, `within_subclone/` untracked |
| WFS | `lineage_validation/within_family_subclones/` (`output/` untracked, `audit/` tracked) | partly |
| LSM | `lineage_validation/lineage_site_application/lineage_site_matrices/` | untracked |
| KPX | `results/kp_tracer_lineage_validation/export/` (validation_v2 export, job 1074570) | untracked |

---

## 1. Inventory of completed analyses

### 1.0 Overview

| # | Analysis | Status | Representation(s) | OT method(s) | Rejection? | Couplings | Unit of aggregation | Uncertainty |
|---|---|---|---|---|---|---|---|---|
| 1a | Tumour-level pairwise readout (`kp_lineage/READOUT_PRESPECIFICATION_2026-10-14.md`) | done (104 pairs, jobs on TACC) | ranknm256_noscale_ds_cos (primary), raw | ConfidenceOT M4-E, acceptance 0.99 | yes | not saved in the 104 runs; recomputed for 50 pairs in 1d | metastasis (31) in the pre-specified test; mouse reported beside it | Wilcoxon over 31 metastases |
| 1b | validation_v2 D1 / E1 (`kp_lineage/validation_v2/`) | done (job 1074570) | same two arms | M4-E gate (retained vs rejected), no coupling | yes | none | mouse (7) | mouse bootstrap, Wilcoxon on mouse means |
| 1c | Frozen protocol (`lineage_validation/VALIDATION_PROTOCOL.md`), incl. **T1/T2** | **done**: 340 runs × M4-E/M4-R, all scored | rna = ranknm256_noscale_ds_cos; cnv = infercnvpy; rna_cnv; cnv_then_rna; rna_then_cnv | ConfidenceOT M4-E (scored) and M4-R (reported) | yes | TACC; local only for 3508, 3515, Quinn 10k | mouse | mouse bootstrap, Wilcoxon + Holm, LOMO |
| 1d | validation_v3 Level 3 (`kp_lineage/validation_v3/`) | **not completed**: design frozen (7fe6202), couplings for 50 pairs recomputed on TACC, stage F code never written | ranknm256_noscale_ds_cos, raw | M4-E, M4-R | yes | TACC only | (mouse, planned) | none computed |
| 2 | Within-family subclones (`WFS`) | definitions done for every tree; OT classification only for 4 of 26 eligible families (local couplings); the TACC run for the other 22 is not recorded anywhere | protocol couplings (5 strategies) | M4-E, M4-R | yes | protocol couplings | family | none |
| 3 | Level 2 subclone matching (`L2`) | done (54 runs) | A: logcpm_noscale on raw counts; B: ranknm256_noscale_ds_cos on equalised counts; strategies rna, cnv, rna_cnv | A: entropic **balanced** OT; B: ConfidenceOT **UOT backbone with rejection budget 0** | **no** | 54 `coupling.npz`, local and TACC | family × level (no mouse aggregation) | none ("no test statistic", `L2/PLAN.md`) |
| 4 | Level 3 cell-level tree distance (`L3`) | done (54 couplings of item 3) | as item 3 | as item 3 | **no** | reuses item 3 | family, mouse counts reported | 95% bootstrap **over cells** per family |
| 5 | Lineage × site matrices (`LSM`) | Phase 0 (archive) and Phase 1 (matrices) done; Phases 2–6 not started | none (annotation only) | none | — | — | mouse | — |
| 6 | Tumour-level exploratory ranking (`kp_lineage/tumour_level/`) | done, exploratory | ranknm256_noscale_ds_cos, raw | M4-E | yes | — | mouse bootstrap | exploratory |

### 1.1 Level 1a: tumour-level pairwise readout (KP-Tracer only)

- **Question.** For each metastasis, is the source retained fraction of its same-family primary higher than that of its other-family primaries?
- **Pre-specification.** `kp_lineage/READOUT_PRESPECIFICATION_2026-10-14.md`, committed before the gates existed (e6579be, d61cddc).
- **Runs.** 131 primary–metastasis pairs; 104 eligible (≥ 20 malignant cells per side) and completed, 27 ineligible; 12 mice in the manifest, 11 with eligible pairs (3519 has 3 pairs, none eligible).
  - Verified from KPX`readout_tumour_level/pairs_ranknm256_noscale_ds_cos.csv`, column `status`: `completed` 104, `ineligible` 27.
- **Representation and method.**
  - ranknm256_noscale_ds_cos (primary arm) and raw (comparison arm);
  - ConfidenceOT M4-E, within-side split null, acceptance 0.99, no cell cap.
  - No CNV arm: `kp_lineage/tumour_level/TUMOUR_LEVEL_REPRESENTATION_AUDIT.md` §1 (report only).
- **Headline, verified** from KPX`readout_tumour_level/metastasis_contrast_source_<label>.csv`, rows `enters_test == True` (31 metastases, 8 mice), column `d`:

| Arm | Side | Median d | Mean d | d > 0 / < 0 / = 0 | Wilcoxon over 31 metastases (one-sided) | Mice with median d > 0 | Wilcoxon on 8 mouse means (my recomputation) |
|---|---|---|---|---|---|---|---|
| ranknm256_noscale_ds_cos | source (pre-specified) | 0.086 | 0.144 | 27 / 3 / 1 | statistic 443; p = 5.0e-7 (exact) or 7.5e-6 (normal approximation) | 8 / 8 | p = 0.0039 |
| ranknm256_noscale_ds_cos | target | 0.032 | 0.071 | 17 / 7 / 7 | p = 0.032 | 5 / 8 | p = 0.055 |
| raw | source | 0.006 | 0.012 | 18 / 11 / 2 | p = 0.017 | 5 / 8 | p = 0.074 |

- **About the p value.** The reported p = 4.99e-7 (`readout_summary.json`) is reproduced only with scipy's exact method. The local scipy default (normal approximation) gives 7.5e-6. The statistic, 443, is identical.
- **Unit.** The pre-specified test treats the 31 metastases as replicates, although metastases of one mouse share their primaries. The mouse-level result (8/8 mice, p = 0.0039 on mouse means) is my own recomputation; it is not a pre-specified quantity.

### 1.2 Level 1b: validation_v2 D1 (E1, E1b)

- **Question.** In same-family pairs, do retained primary cells have deeper shared ancestry s with the metastasis than rejected primary cells?
  - E1 = Cliff's δ; E1b = Spearman of −margin with s.
- **Coverage.**
  - 40 same-family pairs, 28 on a tree.
  - Primary arm: 26 evaluable pairs in 7 mice. Raw arm: 21 pairs in 6 mice.
- **Verified** from KPX`lineage/summary.json` (`labels.ranknm256_noscale_ds_cos.E1`):
  - mean of mouse means 0.0703, median 0.0393, 4 of 7 mice > 0;
  - mouse means: 3454 −0.005, 3457 0.047, 3460 −0.064, 3508 0.039, 3513 −0.075, 3515 0.097, 3724 0.452.
- **Report only** (`kp_lineage/validation_v2/D1_RESULTS_2026-10-15.md`): one-sided p = 0.289, so inconclusive. I did not recompute it from `cell_scores.csv.gz`.
- **Reported post hoc:** in 3724 the gate score and s both follow sequencing depth (ρ −0.45 to −0.53 in L1–L3, +0.63 in S1).
- **Couplings.** None; this analysis uses gates only.

### 1.3 Level 1c: frozen protocol, five strategies (KP-Tracer and Quinn)

- **Runs.** 68 pairs × 5 strategies = 340 runs, each scored for M4-E and M4-R: 680 rows, all `scored`.
  - Verified from EXP/summary`run_status.tsv` (`status`) and `runs.tsv` (rows by `block`).
  - Per method: kp_D_A 175 = 35 pairs × 5; kp_D_B 110 = 22 × 5; kp_D_C3 35 = 7 × 5; quinn_D_B 20 = 4 × 5.
  - This matches the expected run list of protocol §3.3.
- **Representations.**
  - `rna` = ranknm256_noscale_ds_cos.
  - `cnv` = infercnvpy: KP-Tracer `infercnvpy_mouse` with the authors' normal cells as reference; Quinn `infercnvpy_experiment` with no reference.
  - `rna_cnv`, `cnv_then_rna`, `rna_then_cnv`.
  - Source: protocol §4 and `lineage_validation/submit_runs.sh` (report and code).
- **Method.** ConfidenceOT M4-E (scored) and M4-R (reported), within-side acceptance 0.99, `--save-coupling`.
- **Baselines (protocol §10: balanced entropic OT ε = 0.1 and kNN) do not exist.**
  - EXP/summary`baseline_status.tsv`: 204 of 204 (68 pairs × rna / cnv / rna_cnv) are `FAILED`, and `baselines.tsv` is empty.
  - The failure reason is not in the local table. `04_baselines.py:103–105` writes FAILED when the cost rebuild fails the separability check.
  - **So there is no balanced-OT result on the protocol's cells.** The references are only the analytic chance CH and the independent coupling (ρa)⊗b.
- **Couplings.**
  - All on TACC under `/scratch/10119/ghzheng/OT_project/lineage_validation/runs/acc0.99/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos/<pair>/` (protocol §13).
  - Local: 110 `coupling_m4e.npz` + 110 `coupling_m4r.npz` for KP 3508, KP 3515 and Quinn 10k, under EXP`couplings/`.
- **Unit.** The mouse (protocol §11).
- **T1 / T2 and the exploratory results:** §2 below.
- **Quinn (exploratory, 4 mice, one run each).** Verified from EXP/summary`runs.tsv`, `block == quinn_D_B`, `method == M4-E`, `strategy == rna`:

| Mouse | LL source cells | Target cells | Clones in LL | Unmatched_observed targets | COND | COND_ind | Top-1 COND / chance | Target rejection AUROC (unobserved clone) | Spearman t_X vs f_X |
|---|---|---|---|---|---|---|---|---|---|
| 100k | 1,746 | 4,012 | 29 | 103 | 0.516 | 0.252 | 0.748 / 0.541 | 0.545 | 0.40 |
| 10k | 381 | 2,136 | 12 | 856 | 0.408 | 0.243 | 0.558 / 0.391 | 0.953 | 0.39 |
| 30k | 640 | 16,535 | 3 | 132 | 0.892 | 0.882 | 0.947 / 0.948 | 0.712 | 0.71 |
| 5k | 1,995 | 35,817 | 73 | 1,587 | 0.173 | 0.140 | 0.248 / 0.203 | 0.518 | −0.10 |

- Across the 5 strategies, `COND_minus_ind` is > 0 in 4/4 mice. Its mouse mean is 0.12 (rna) to 0.18 (cnv, rna_then_cnv). Source: EXP/summary`exploratory.tsv`, `block == quinn_D_B`, `metric == COND_minus_ind`.
- **Who drives the Quinn signal.** It comes from 100k and 10k; in 30k and 5k COND − COND_ind is 0.01–0.04.
  - **30k** is effectively one clone: LG1 is 15,548 of 16,535 target cells, with source share 0.933 (EXP`ot/scores/M4-E/rna/quinn_D_B/30k__D_B__LL__metastases/groups.tsv`).
  - **100k:** the largest contributor is LG1 (2,113 target cells; E2E 0.684 against source share 0.416). LG1 has no tree.
- **The 10k rejection AUROC of 0.953** is the XM tissue: 856 of its 857 cells belong to clones not observed in LL (`AUDIT_SUMMARY.md`; 856 unmatched_observed in `runs.tsv`).
- **Protocol tree agreement (§9, M4-E gated couplings, exploratory).** Verified from EXP/summary`exploratory.tsv`, `metric == tree_dS_norm`, `strategy == rna`:
  - kp_D_A: 6 mice, mean −0.002 (CI −0.037 to 0.031);
  - kp_D_C3: 5 mice, mean 0.022;
  - kp_D_B: 6 mice, mean 0.153, carried by 3724 (0.819) with the other mice ≤ 0.038;
  - Quinn: 4 mice, mean 0.025.

### 1.4 Level 1d: validation_v3 Level 3 (pairwise couplings vs tree)

- **Design.** `kp_lineage/validation_v3/LEVEL3_VALIDATION_DESIGN.md`, FROZEN 7fe6202.
- **Stages** (`kp_lineage/validation_v3/TACC_EXECUTION_PLAN.md`):
  - A (audit) done;
  - B (pilot) passed: 12/12 exact reproductions;
  - D (couplings for 50 shared-tree pairs × 2 labels × M4-E/M4-R, jobs 1075134 / 1075135): reported done in `lineage_validation/AUDIT_SUMMARY.md` ("Couplings: 50 pairs × 2 labels, reproduction passed") and in `quinn2021/LINEAGE_OT_VALIDATION_DISCUSSION.md` §3;
  - F (metrics): "code not yet written". Protocol §0 confirms it was not run.
- **No local output of stage D or F exists.** None is under `results/`.

### 1.5 Within-family subclones (WFS)

- **Definitions** (`SUBCLONE_RULES.md`, frozen a545f1c). Authors' trees and character matrices only:
  - reliable cells have ≤ 50% missing characters;
  - a level-1 subclone is a branch with ≥ 20 reliable cells at the family's first split with ≥ 2 such branches, carrying a marker edit (80/20 rule).
- **Feasibility, verified** from WFS`output/feasibility.tsv` (160 rows, column `subclone_structure_eligible`):
  - KP-Tracer: 12 families on a `_Fam` / `_All` tree in 7 mice; 5 eligible (3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1).
  - Quinn: 148 clones; 21 eligible (100k 4, 10k 3, 30k 2, 5k 12); 32 have no tree file.
  - Level-1 subclones over the eligible families: KP 11 (3454 2, 3457 4, 3515 2, 3724 3); Quinn 107 (100k 11, 10k 6, 30k 16, 5k 74).
- **OT classification** (M4-E, protocol couplings): only 4 of 26 eligible families, the ones whose couplings are local: KP 3515 T1 and Quinn 10k LG2, LG3, LG5 (WFS`output/classification_local/`).
  - `README.md` says the other 22 "run on TACC with the same code (`submit_tacc.sh`)". No output, commit or report records that run, and `audit/AUDIT_REPORT.md` says "No TACC job was submitted". **Not done.**
- **Mass audit** (WFS`audit/AUDIT_REPORT.md`, tables `audit/*.tsv`): 220 local runs = 22 pairs × 5 strategies × M4-E/M4-R.
  - The mass calculation is correct (rank-one gated block, log-residual ≤ 8e-13).
  - Defects A–C were in the outputs only, and were fixed.
  - Conclusion (report only): within a family the subclone-level coupling equals the retained source composition to within 0.01–0.07 ("H3"). In 3508, rejection selects families, not subclones.

### 1.6 Level 2: subclone matching, A "Standard OT" vs B "Our OT" (L2)

**Methods, verified from code** (`L2/methods.py`, `L2/results/diagnostics.tsv`):

| | A_standard_OT | B_our_OT |
|---|---|---|
| Counts | raw | depth-equalised (27_) |
| RNA representation | `logcpm_noscale` (1e4 → log1p → 2,000 HVG → PCA 30, no scaling) | `ranknm256_noscale_ds_cos` |
| Solver | `traditional_ot.balanced.balanced_ot(threshold=1e-9, max_iterations=10000)`: **entropic balanced OT, uniform marginals**, ε = 0.1 | `ConfidenceOT(backbone='uot', variant='exact', lambda_a=1, lambda_b=1, tolerance=1e-4)`, ε = 0.1 |
| Rejection | none | budgets 0 → bounds (0, 0): **no cell gated**. `source_retained = target_retained = 1.0` in all 27 B runs; KL-relaxed marginals; total mass 0.64–0.73 |
| Gate (M4-E / M4-R) | — | **no gate is applied.** `variant='exact'` is the M4-E solver, but with bounds (0, 0) nothing is rejected. M4-R was not run |

- **What A vs B compares.** For `rna` and `rna_cnv`, A and B differ in counts, representation **and** solver together. Only `cnv` (the same CNV PCA for both) isolates the solver. Balanced OT on ranknm256_noscale_ds_cos coordinates was removed by the user before any coupling (`L2/PLAN.md`, "simplification").
- **Design.** One plan per family: primary → pooled metastatic sites.
  - KP: 5 cases (families);
  - Quinn: one case per mouse, LL → all tissues, with the mouse's eligible clones;
  - 9 cases, 26 families, 3 strategies × A/B = 54 runs, all `ok` (`L2/results/run_status.tsv`).
- **Couplings.** 54 `coupling.npz` local (`L2/data/full_20261022_204810/{full,full_cnv,full_rna_cnv}/<case>/<method>/`, 8.1 GB) and on TACC (`<root>/level2_pairwise_ot/...`).
- **Reproducibility.** The local rescore (`data/.../scores_local/`) equals the committed `results/` to ≤ 3e-16 on every numeric column of all 9 tables (my comparison).

**Evaluability, verified** from `L2/results/comparison.tsv` (`evaluation_status`): L1 17 evaluable of 26 family-levels; L2 12 of 21.
- KP evaluable: L1 = 3515, 3724 (2 mice); L2 = 3454, 3515, 3724 (3 mice).
- Quinn evaluable: L1 = 15 clones (100k 4, 10k 2, 30k 2, 5k 7); L2 = 9 (100k 2, 30k 2, 5k 5; 10k none).

**Headline, verified** from `L2/results/comparison.tsv`. Evaluable rows; families above chance means `balanced_accuracy_enrichment__<method>` − `chance` > 0.

| Strategy | Level | n | A above chance | B above chance | Median A − chance | Median B − chance | Median B − A | B > A / A > B |
|---|---|---|---|---|---|---|---|---|
| rna | 1 | 17 | 10 | 8 | +0.029 | −0.020 | −0.010 | 4 / 13 |
| rna | 2 | 12 | 6 | 5 | +0.001 | −0.024 | −0.006 | 3 / 9 |
| cnv | 1 | 17 | 6 | 8 | −0.014 | −0.012 | +0.002 | 11 / 5 |
| cnv | 2 | 12 | 5 | 5 | −0.017 | −0.017 | −0.000 | 4 / 6 |
| rna_cnv | 1 | 17 | 8 | 8 | −0.003 | −0.006 | −0.004 | 6 / 11 |
| rna_cnv | 2 | 12 | 8 | 7 | +0.017 | +0.004 | −0.000 | 5 / 6 |

**Cases with signal, verified** (`comparison.tsv`, `per_site.tsv`):
- **KP 3724 T1, L2** (2 true subclones, 6,206 scored cells). Balanced accuracy A / B:
  - rna 0.993 / 0.968;
  - cnv 0.927 / 0.947;
  - rna_cnv 0.983 / 0.977.
  - At L1, which has 3 true classes with the smallest of 3 cells, only 0.62–0.66.
- **KP 3515 T1, L1** (A / B): rna 0.690 / 0.596.
  - Site L1 recall 0.85–0.93.
  - N1 L1.2 (12 cells) recall 0.67–0.92.
  - The other sites are near 0.5.
- **Quinn 100k LG4, L1** (52 scored cells; smallest true subclone 20). A / B:
  - cnv 0.856 / 0.856;
  - rna 0.506 / 0.653;
  - rna_cnv 0.750 (B).
- **Rare subclones** (π < 0.10, evaluable family-levels): 107–109, mean recall 0.11–0.13, median 0; 55 not candidates; 5 (A) / 6 (B) with recall > 0.5. This matches `REPORT.md` only when restricted to evaluable family-levels; over all family-levels it is 123–125 and 69.
- **Small classes.** In 14 of 29 evaluable family-levels the smallest true subclone has < 20 scored cells (`per_subclone.tsv`, minimum `target_cells_scored`).
  - Examples: 3454 L2 = 1 cell; 3515 L2 = 3; 3724 L1 = 3; 30k LG2 L2 = 6.

### 1.7 Level 3: cell-level tree distance (L3)

- **Question.** Does OT put a metastatic cell's mass on primary cells at a smaller tree distance than the independent coupling with the method's own marginals?
  - ΔD_norm = (D_ind − D_OT) / (D_ind − d_min).
- **Inputs.** The 54 Level 2 couplings, so there is **no rejection, no M4-E / M4-R gate**, and A is balanced OT.
- **Trees.**
  - KP: `_All` (3454, 3724) or `_Fam` (3457 T1 / T4, 3515 T1).
  - Quinn: `m<mouse>_lg<c>_tree_hybrid_priors.alleleThresh.processed.txt`.
  - Contracted unary nodes; no branch lengths.
- **Evaluable cells, verified** from L3`ground_truth/targets.tsv.gz` (`status`) and `coverage.tsv` (`n_evaluable`):
  - **39,849 evaluable**, 26,863 not evaluable;
  - 47,793 metastatic cells lie in the 26 families (`metastatic_cells_in_family`).
- **Headline, verified** from L3`summary/family_summary.tsv` (26 families per strategy × method) and `mouse_summary.tsv` (8 mice):

| Strategy | Method | Families ΔD > 0 | Cell-bootstrap CI > 0 / < 0 | Mice ΔD > 0 | Median mouse ΔD_norm | Median family D_OT / D_ind / d_min |
|---|---|---|---|---|---|---|
| rna | A | 23/26 | 15 / 0 | 8/8 | 0.029 | 11.24 / 11.35 / 5.29 |
| rna | B | 22/26 | 16 / 1 | 7/8 | 0.025 | 11.23 / 11.34 |
| cnv | A | 24/26 | 13 / 1 | 8/8 | 0.021 | 11.24 / 11.35 |
| cnv | B | 24/26 | 14 / 1 | 8/8 | 0.019 | 11.21 / 11.34 |
| rna_cnv | A | 25/26 | 15 / 1 | 8/8 | 0.029 | 11.27 / 11.35 |
| rna_cnv | B | 22/26 | 18 / 0 | 8/8 | 0.025 | 11.22 / 11.34 |

- **KP 3724.**
  - ΔD_norm: rna 0.359 (A) / 0.32 (B), cnv 0.32 / 0.34.
  - Top-1 in the tree-nearest set: 9.4% vs 2.9%.
  - Every other family is −0.032 to 0.137.
- **Within the subclone** (L3`within_subclone/family_summary.tsv`), 3724's level-1 ΔD_norm is 0.001–0.018. Its effect is the subclone-level split: c_OT 0.85–0.91 vs c_ind 0.53.
- **A vs B** (L3`summary/ab_overall.tsv`): families B higher / A higher are rna 9/17, cnv 16/10, rna_cnv 10/16. Mice B higher: 2, 3, 2 of 8.
- **Uncertainty.** The interval is a bootstrap over cells (1,000 draws; `summarize_level3.py:8`, `L3/PLAN.md` line 72). Cells of one family are not independent, so these intervals are not inferential across mice.

### 1.8 Lineage × site matrices (LSM)

- **Content.** Authors' annotations only: per mouse, lineage (family / clone) × site counts and fractions, sharing classes, and subclone × site tables for the eligible families.
  - Verified from LSM`coverage_mice.tsv`: 11 KP mice (47,549 malignant cells) and 4 Quinn mice.
- **Phase 2** (cell-level gate validation over every mouse) is not started. It needs a TACC export that was "not submitted without the user's go-ahead" (`lineage_site_application/PLAN.md`).

---

## 2. The frozen pre-specified test (T1 / T2, KP D_A, strategy rna, M4-E)

**It was run.**
- Result: EXP/summary`tests.tsv` (with `companions.tsv` and `lomo.tsv`).
- The report text is EXP/summary`REPORT.md`, a TACC file downloaded with the export.
- **No committed report in the repo states T1/T2.**
- The project memory note ("runs + scoring code written, not yet run on TACC") is out of date.

**Re-derivation, verified.**
- Filter: EXP/summary`runs.tsv`, `block == kp_D_A`, `strategy == rna`, `method == M4-E`, `site_truth == True`. That gives 31 runs in 8 mice (3451 5, 3454 2, 3457 3, 3460 5, 3466 1, 3508 8, 3513 1, 3515 6).
- Then the per-mouse mean of `e_M` (T1) or `E2E_minus_CH` (T2), and scipy `wilcoxon(alternative='greater')`.

| Test | Mean of mouse means | Median | Mice > 0 | Wilcoxon p | Holm p | 95% bootstrap (report) | Verdict |
|---|---|---|---|---|---|---|---|
| T1 (e_M) | 0.2696 | 0.1915 | 6/8 | 0.0195 | 0.0234 | 0.101–0.456 | supported |
| T2 (E2E − CH) | 0.0783 | 0.0210 | 7/8 | 0.0117 | 0.0234 | 0.011–0.158 | supported |

- **Mouse values.**
  - T1: 3451 0.375, 3454 0.248, 3457 0.135, 3460 −0.001, 3466 −0.003, 3508 0.702, 3513 0.623, 3515 0.078.
  - T2: 3451 0.021, 3454 0.036, 3457 0.009, 3460 0.003, 3466 −0.004, 3508 0.297, 3513 0.245, 3515 0.021.
- **Companions** (verified, same filter):
  - site Top-1 0.688;
  - site AUROC 0.729;
  - COND − COND_ind 0.0066 (bootstrap −0.0013 to 0.0155, 5/8 mice > 0);
  - conditional Top-1 0.703;
  - cov_T 0.955;
  - cov_S 0.657.
- **Leave-one-mouse-out** (EXP/summary`lomo.tsv`): T1 p 0.016–0.039, T2 p 0.008–0.023. Every LOMO stays below 0.05 unadjusted.

**What carries T2, my decomposition from the same rows.**
- E2E − CH = (E2E_ind − CH) + (E2E − E2E_ind). The first term is what the gate alone gives under an independent coupling on the retained cells; the second is the pairing.
- **The gate dominates.**
  - 3508: 0.275 of 0.297 is the gate term;
  - 3513: 0.232 of 0.245;
  - every other mouse: |pairing term| ≤ 0.023.
- **Asserted precision equals the independent-coupling precision** to ≤ 0.016 in all 31 runs (`asserted_precision` − `asserted_precision_ind`, mean 0.001).
- So the supported T2 says the M4-E **gate** rejects cells of unrelated primaries. It does not show that the coupling pairs metastatic cells with related cells beyond that.
- **Exploratory strategies** (EXP/summary`exploratory.tsv`, kp_D_A, M4-E). Mouse-mean e_M: rna 0.270, cnv 0.104, rna_cnv 0.197, cnv_then_rna 0.288, rna_then_cnv 0.283.
  - E2E − CH: 0.078 / 0.041 / 0.065 / 0.093 / 0.085.
  - M4-R is similar: e_M 0.252 rna.

---

## 3. Data inventory

Per-mouse numbers are in `DATASET_FEASIBILITY.tsv`; the definitions of its columns are below.
- **n_cells_rna.**
  - KP: malignant cells in the prepared per-tumour RNA files, summed over the mouse's tumours (KPX`readout_tumour_level/pairs_ranknm256_noscale_ds_cos.csv`, `source/target_malignant_n`). The authors' normal cells, about 29,759 per `TUMOUR_LEVEL_REPRESENTATION_AUDIT.md` (report only), are not counted.
  - Quinn: rows of the converted per-experiment obs (EXP`meta/Quinn2021_obs_<m>.csv.gz`), all A549.
- **n_cells_with_lineage.**
  - KP: cells with a MetFamily (KPX`metadata/kptracer_meta_subset.csv.gz`, every row `TS_Present` and `RNA_Present`; 47,551 rows, 47,549 malignant), plus leaves of the Cassiopeia trees (KPX`audit/trees.tsv`, `kind == cassiopeia`).
  - Quinn: cells with a LineageGroup (obs `LineageGroup` non-missing).
- **Trees.**
  - Neither dataset has branch lengths: KP `edges_with_length` is 0 for all 169 trees in KPX`audit/trees.tsv`; the Quinn trees are newick with character-state internal labels and no `:length`, read locally.
  - KP: Cassiopeia plus NJ trees; per-family `_Fam` or `_All` trees exist only for 3454, 3457, 3460, 3508, 3513, 3515, 3724. 3451, 3465, 3466 and 3467 have per-primary trees only, so their metastases are on no tree.
  - Quinn: one tree per clone. 116 of 148 clones have one (5k 82/100, 10k 11/13, 30k 4/4, 100k 19/31; 100k LG1, the largest, has none).
  - Polytomies are large: KP 3724 `_All` has max 6,519 children; 3508 `_All` 1,033; Quinn 30k LG1 556 (L3`ground_truth/coverage.tsv`, `tree_max_children`).
- **CNV.**
  - KP: infercnvpy 0.6.1 per mouse (11 mice), reference = the authors' `is_normal` cells (`kp_lineage/cnv/01_build_infercnvpy_units.py`, `submit_infercnvpy_kp.sh`).
  - Quinn: one unit per experiment, `--no-reference`, so the reference is the mean of all A549 cells (`quinn2021/submit_infercnvpy.sh`). Profiles are relative between A549 subclones only.
  - **Outputs on TACC only.** No local reference-cell or window table was found. The window counts (2,269 KP; 2,944–2,995 Quinn) are in `quinn2021/LINEAGE_OT_VALIDATION_DISCUSSION.md` only.
- **Raw counts (TACC only).**
  - KP: `/scratch/10119/ghzheng/OT_project/data/kp_tracer/KPTracer-Data` → `kp_tracer_confidenceot/prepared/h5ad/<tumour>.h5ad`, equalised `kp_tracer_confidenceot/downsampled/`.
  - Quinn: `/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363/converted` (from EXP`provenance.json` and `submit_*.sh`).
  - Local copies: authors' metadata, the trees and character matrices of KP 3508 / 3515 and of every Quinn mouse, plus the 10 KP family trees in KPX`trees/cassiopeia/`. No count matrix is local.
- **Quinn LM0** (pre-injection, 8,726 cells per `LINEAGE_OT_VALIDATION_DISCUSSION.md`; report only). There is no local obs table and it is used in no analysis.

---

## 4. Coverage, sample sizes and what drives each result

- **Independent units.**
  - KP-Tracer: 11 mice with eligible pairs; 8 enter T1/T2.
  - Quinn: 4 mice, one run each per strategy, so the smallest one-sided Wilcoxon p would be 1/16. No Quinn test exists.
  - Level 2 / 3: 26 families in 8 mice (KP 3454, 3457, 3515, 3724; Quinn 100k, 10k, 30k, 5k). **12 of the 26 families are clones of one mouse (Quinn 5k).** "22–25 of 26 families" is therefore dominated by 4 Quinn mice (21 families).
- **T1 / T2.**
  - Two mice have one metastasis each (3466, 3513).
  - 3508 (8 metastases, 750 target cells) and 3513 (1 metastasis, 239 cells) supply most of T2: mouse values 0.297 and 0.245 against ≤ 0.036 elsewhere.
  - 3460 and 3466 are ≈ 0 because their related primary holds 86% (3460) or 50% (3466) of the source cells and almost nothing is rejected (cov_S 0.98, 1.00).
- **KP 3724** is one mouse, one primary and one family, yet:
  - it carries the D1 E1 mean (without it 0.007), with a depth confound;
  - it is the only Level 2 family with balanced accuracy > 0.9;
  - it holds the largest Level 3 effect (ΔD_norm 0.36, subclone-level) and 6,215 of the 39,849 Level 3 evaluable cells;
  - it drives the kp_D_B tree ΔS_norm mean (0.82 vs ≤ 0.04).
  - Its tree has a 6,519-child polytomy.
- **KP 3515.**
  - Level 2 L1 signal: 60 L1-site cells and 12 N1 L1.2 cells.
  - Within-family "weak exception": 6–12 L1.2 cells.
- **Quinn 100k LG4.** 52 scored Level 2 cells (smallest true subclone 20); the CNV-only result is 0.856.
- **Quinn evaluable Level 3 sources** are small: 20–34 comparable LL cells in 5k LG6, LG11–LG13, LG15, LG26 and 10k LG5 (`coverage.tsv`, `sources_comparable`).
- **Not evaluable in Level 3:**
  - 18,919 cells are in no family / clone tree of their case;
  - 7,944 have character missing share > 0.5.
  - High missingness: Quinn 30k LG1 (34% of leaves unreliable; 5,268 of 15,548 cells not evaluable) and 5k LG1.

---

## 5. Inconsistencies and leakage risks

### 5.1 Reports vs tables

1. **Level 3 "47,000+ evaluable" (L3 `REPORT.md`).** The tables give 39,849 evaluable. 47,793 is the number of metastatic cells in the families (39,849 + 7,944). **Not reproduced.**
2. **Level 3 "Flags. Low tree resolution: KP 3724" (REPORT).** `ground_truth/coverage.tsv` has `flag_low_tree_resolution == False` for all 26 families, 3724 included. The prose and the flag column disagree.
3. **Level 2 "Quinn: about 0.3 at L1" own-clone mass share, "0.306 vs 0.326" (REPORT).** These are medians over the 17 evaluable L1 family-levels **including 2 KP families at 1.0**. Quinn-only evaluable medians are 0.20 (A) / 0.20 (B) (`summary.tsv`, `mean_family_mass_share`).
4. **Level 2 rare-subclone counts (107–109, 55 non-candidates)** hold only for evaluable family-levels. The report does not say so.
5. **Level 2 coverage medians 0.81 / 0.41** hold for evaluable family-levels. Over all family-levels they are 0.77 / 0.37.
6. **Level 1a p = 5.0e-7** depends on scipy's exact Wilcoxon; the normal approximation gives 7.5e-6.
7. **Quinn per-tissue counts in `quinn2021/LINEAGE_OT_VALIDATION_DISCUSSION.md` §2 do not match the converted obs.**
   - Example: 5k M1 13,475 vs 12,473 labelled, LL 2,235 vs 1,995; 100k LL 1,826 vs 1,755. The discussion's tissue columns do not sum to its own annotated totals (5k: 40,969 vs 37,814).
   - The totals (37,814, 17,189, 2,541, 5,806) do match.
   - The table predates the 10k join fix (4a46f5a).
8. **WFS `README.md` describes a TACC run of 22 families as the next step.** It was never run; the analysis was superseded by Level 2.
9. **Protocol depth control (§10) is missing on single-tumour sides.** In the local per-run tables (EXP`ot/scores/M4-E/rna/*/*/depth.tsv`), `with_depth == 0` for every kp_D_A target (14 runs) and kp_D_B source (5 runs).
   - The site key of a single-tumour side is the category (`metastasis` / `primary`), the same defect that `cac2ee9` fixed in Level 2.
   - Pooled sides are joined correctly.
   - The KP depth check therefore exists only on the pooled side.
10. **Protocol baselines** (balanced OT, kNN): 204/204 FAILED with no reason in the summary, and no report mentions it.
11. **Within-family audit:** several M4-R fits did not converge in the outer loop (e.g. 3515 D_C3 rna) and are used as saved (`audit/AUDIT_REPORT.md`, Part A #9). Report only; I did not check every run's `outer_converged`.

### 5.2 Leakage and analyst degrees of freedom already present

1. **Lineage selects the Level 2 / 3 OT inputs.**
   - Families enter only if `subclone_structure_eligible` (needs a tree split with marker edits).
   - KP target sites enter only with ≥ 20 reliable family cells carrying an L1 label (`L2/make_inputs.py` docstring). Left out: 3454 Rib (10) and K1 (1); 3457 L2 (15).
   - Each KP case pairs a primary only with its own MetFamily's metastases, so relatedness is given, not inferred.
   - Cells within a site are all its malignant cells, not only labelled ones.
   - This is conditioning on lineage at case level. It is stated in PLAN.md, but no Level 2 / 3 number is lineage-blind.
2. **The subclone rules were fixed after OT output had been seen on trees.**
   - `SUBCLONE_RULES.md` (a545f1c, 2026-10-22 16:06) came 25 min after the figures of c15be74 (15:41): KP 3508 / 3515 trees and Quinn 10k LG1 / LG2 and 5k LG3, overlaid with five-strategy gates and OT-predicted sources.
   - Its claim "fixed before any subclone-level OT quantity is computed" is literally true, but the rules were not blind to OT behaviour on those families.
3. **Level 2's design followed the within-family results.**
   - The design was written after the 4-family classification and audit had shown size dominance (`L2/PLAN.md`, "Why no rejection here").
   - The strategy set was widened to cnv / rna_cnv (c52bfed, 20:47) after the Quinn RNA-only couplings existed: the local coupling files are dated 20:11–20:26. Whether any were scored first is not recorded.
4. **The Level 3 plan came after the Level 2 results.**
   - PLAN.md (f532d01, 21:24) was written 9 min after the Level 2 results commit (7919953, 21:15).
   - The within-subclone decomposition (361a7c8, 22:51) is labelled post hoc. Its "defined before computed" cannot be checked from git: the results commit is at 23:07.
5. **The T1 hypothesis was seen before the freeze.**
   - Protocol §0 states that Level 1a (supported) and E1 (inconclusive) were seen. T1 re-tests the Level 1a hypothesis on pooled designs of the same mice, cells and gate configuration, so T1 is not independent confirmation of Level 1a.
   - Acceptance 0.99 was chosen on the simulation benchmark with truth in view (project note `acceptance-sweep-2026-10-06`), not on lineage.
6. **Stated exception (protocol §2):** single KP tumour sides reuse equalised files whose obs carry `met_family`. `02_run_pair.py` does not read it (report only; not re-checked here).
7. **Depth.**
   - D1 shows that in 3724 both the gate score and lineage depth s track sequencing depth after equalisation.
   - The protocol's depth control is missing on single-tumour sides (5.1 #9).
   - Any 3724-driven result (D1, Level 2, Level 3, kp_D_B tree) carries this confound.
