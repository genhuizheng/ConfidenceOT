# Previous lineage experiments on KP-Tracer and Quinn 2021: per-experiment audit

Status: **2026-10-31, repo HEAD 43993165ed990635f0db7693913191cdd99d50e7, read-only.** No existing file was changed and no OT was run.

**Companion files.**
- Here: `PREVIOUS_RUN_INVENTORY.tsv` (1,599 rows, one per run unit), `PREVIOUS_RESULTS_SUMMARY.tsv` (717 recomputed numbers, with report-only ones marked) and `REUSABLE_OUTPUTS.tsv` (17 output locations).
- `../iscoral_extension_20261031/`: `EXISTING_LINEAGE_VALIDATION_AUDIT.md` (detailed audit, inconsistencies, leakage) and `DATASET_FEASIBILITY.tsv`.

**Path abbreviations.** EXP = `lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export/`; L2 = `lineage_validation/level2_pairwise_ot/`; L3 = `lineage_validation/level3_cell_lineage/results/`; KPX = `results/kp_tracer_lineage_validation/export/`; TACC = `/scratch/10119/ghzheng/OT_project/`.

**Cost notation, used throughout.** "cos" = squared Euclidean on unit-norm PCs, which equals 2 − 2cos. Every cost is divided by the median random source–target pair cost (`median_pair_scale`). The CNV block is squared Euclidean on CNV PCs, not row-normalised, although `run.json` field `cost` says "cosine" (`cancer_metastasis/02_run_pair.py:705–725`). `rna_cnv` is **RNA + CNV**, the two blocks concatenated after each is scaled to median pair cost 1. It never contains site.

---

## 1. Experiments, 12 points each

### E1. L1a: tumour-level pairwise readout (KP-Tracer)

1. **Data.** 11 mice. Every metastasis × every primary of its mouse: 131 pairs, 104 run (≥ 20 malignant cells per side), 27 ineligible. One tumour per side.
2. **OT method.** ConfidenceOT M4-E (scored) and M4-R (saved), KL-UOT backbone.
3. **Features.** RNA only.
4. **Preprocessing.**
   - `ranknm256_noscale_ds_cos` (primary): equalised counts.
   - `raw` (comparison): original counts, no normalisation, 2,000 HVG, z-scored genes, PCA 30. It differs from the primary in five settings at once (`kp_lineage/tumour_level/TUMOUR_LEVEL_REPRESENTATION_AUDIT.md` §2).
5. **Cost.** cos / median (primary); squared Euclidean / median (raw).
6. **Rejection and calibration.** within_side_split null, acceptance 0.99, no cell cap, ε 0.1, λ 1.
7. **Completion.** 104/104 per label.
8. **Convergence and validity** (KPX `config/ot_runs/<label>/<pair>/run.json`, `methods[]`).
   - M4-E outer-converged in 104/104 runs in both labels.
   - M4-R outer-converged in 31/104 (ranknm) and 103/104 (raw); gate cycle in 73/104 ranknm M4-R runs.
   - `calibration_m4e_inference_valid` is True in only 54/104 (ranknm) and 61/104 (raw) pairs.
9. **Metric.** L1 only. d = same-family − other-family source retained fraction, over 31 metastases in 8 mice.
   - Median 0.086; d > 0 in 27/31; exact Wilcoxon p = 5.0e-7 over metastases.
   - 8/8 mice have median d > 0; Wilcoxon on mouse means p = 0.0039 (my recomputation).
   - The raw arm: median 0.006.
10. **Baseline and uncertainty.** Null 0. The pre-specified unit is the metastasis, which is pseudo-replicated within mice.
11. **Outputs.**
    - Local: KPX `readout_tumour_level/`, `tables/pair_gates.csv`, run.json.
    - TACC: `kp_tracer_confidenceot/ot/kptracer/<label>/<pair>/`. These runs did not save couplings.
12. **Reusable without rerun.** For L1 rescoring, yes. For coupling-based work, only through the v3 recomputed couplings of E4 (50 pairs, TACC).

### E2. L1b: validation_v2 D1 / E1 (KP-Tracer)

1. **Data.** 40 same-family pairs; 28 tree-covered; evaluable in 26 pairs / 7 mice (ranknm) and 21 / 6 (raw).
2. **OT method.** No new OT. It uses the M4-E gates of E1.
3. **Features.** RNA (E1's two labels).
4. **Preprocessing.** As E1.
5. **Cost.** As E1.
6. **Rejection and calibration.** As E1.
7. **Completion.** Done (job 1074570).
8. **Validity.** It inherits E1's flags.
   - Tree-rule ties are heavy: the modal s holds a median 62% of a pair's source cells.
9. **Metric** (cell level: retained vs rejected primary cells, by shared-ancestry depth s on the authors' tree).
   - E1 Cliff's δ: mean of mouse means 0.070, 4/7 mice > 0 (KPX `lineage/summary.json`).
   - p = 0.289, inconclusive (report only).
   - It is carried by 3724 (0.452), where the gate score and s both follow sequencing depth.
10. **Baseline and uncertainty.** Null 0; mouse bootstrap.
11. **Outputs.** KPX `lineage/`, `tables/cell_lineage.csv.gz`; TACC `kp_tracer_confidenceot/validation_v2/`.
12. **Reusable.** Yes, as gate-vs-tree tables.

### E3. L1c: frozen protocol (`lineage_validation/VALIDATION_PROTOCOL.md`, FROZEN 508609f)

1. **Data.** 68 pairs:
   - KP: D_A pooled primaries → one metastasis (35), D_B one primary → pooled metastases (22), D_C3 pooled → pooled (7);
   - Quinn: LL → pooled tissues (4).
2. **OT method.** ConfidenceOT M4-E (scored) and M4-R (reported).
3. **Features.** rna, cnv, rna_cnv (RNA + CNV), cnv_then_rna, rna_then_cnv.
4. **Preprocessing.**
   - rna: `ranknm256_noscale_ds_cos`.
   - cnv: infercnvpy PCs (30). KP `infercnvpy_mouse` with reference = the authors' normal cells; Quinn `infercnvpy_experiment` with no reference.
5. **Cost.** As in the notation above.
6. **Rejection and calibration.** within_side_split, acceptance 0.99, budget cap 0.95 per side, no cell cap; two-step strategies refit on the step-1-retained cells.
7. **Completion.** 340 runs × 2 methods, all scored (EXP `summary/run_status.tsv`).
8. **Convergence and validity** (local for 125 runs: KP 3508, 3515; Quinn 5k, 10k, 30k, 100k; EXP `ot/gates/.../pair_metrics.csv`). The other 9 KP mice are TACC only.
   - M4-E outer-converged in 125/125.
   - M4-R outer-converged in 51/125; gate cycles are common.
   - `calibration_m4e_inference_valid` is True in only 86/125 (rna 13/25, cnv 19/25).
   - **5 of the 8 3508 D_A rna M4-E runs, the mouse that drives T1/T2, are False.**
9. **Metrics** (EXP `summary/runs.tsv`).
   - **Pre-specified, KP D_A rna M4-E, 31 runs / 8 mice:**
     - T1 e_M: mean of mouse means 0.270, Wilcoxon p 0.0195, Holm 0.0234;
     - T2 E2E − CH: 0.078, p 0.0117, Holm 0.0234;
     - both "supported".
   - **What carries T2.** It is ~90% the gate term (E2E_ind − CH): 3508 0.275 of 0.297, 3513 0.232 of 0.245. COND − COND_ind is 0.0066 (CI −0.001 to 0.015).
   - **Quinn (exploratory).** COND − COND_ind > 0 in 4/4 mice; 0.26 (100k) and 0.16 (10k) for rna, 0.01–0.03 in 30k and 5k.
   - **Tree subset ΔS_norm ≈ 0** for D_A (−0.002) and D_C3 (0.022). D_B is 0.15, from 3724 alone (0.82).
10. **Baseline and uncertainty.**
    - References: analytic chance CH and the independent coupling (ρa)⊗b.
    - Label shuffle: 1,000 draws.
    - Mouse bootstrap (10,000) and LOMO.
    - **The pre-specified balanced-OT and kNN baselines failed in 204/204 runs** (EXP `summary/baseline_status.tsv`; `baselines.tsv` empty).
11. **Outputs.**
    - Local: EXP summary, scores, gates (125 runs), couplings (110 M4-E + 110 M4-R for 3508, 3515, 10k).
    - TACC: `lineage_validation/runs/acc0.99/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos/<pair>/` (couplings, gates, joint PCA coordinates).
12. **Reusable.**
    - Yes: every run is complete, with couplings saved.
    - Depth control (`depth.tsv`) needs correction on single-tumour sides: `with_depth` is 0.
    - Baselines must be redone from the saved coordinates.

### E4. L1d: validation_v3 pairwise couplings (KP-Tracer)

1. **Data.** 50 shared-tree pairs in 7 mice.
2. **OT method.** ConfidenceOT M4-E and M4-R, rerun at the original calibrated cost.
3. **Features.** RNA.
4. **Preprocessing.** ranknm256_noscale_ds_cos and raw.
5. **Cost.** As E1.
6. **Rejection and calibration.** As E1; the pilot reproduced exactly (12/12).
7. **Completion.** Couplings done per report (jobs 1075134 / 1075135). **Stage F (Level 3 metrics) was never written.**
8. **Validity.** TACC only.
9. **Metrics.** None computed.
10. **Baseline and uncertainty.** Planned EOT, balanced OT and kNN; none run.
11. **Outputs.** TACC `kp_tracer_confidenceot/validation_v3/coupling/`; nothing local.
12. **Reusable.** Yes, for a pairwise L3 without rerunning OT.

### E5. Within-family subclone classification (WFS)

1. **Data.** Subclones defined for 160 families / clones (`SUBCLONE_RULES.md`, frozen a545f1c); 26 eligible. Classified only 4: KP 3515 T1 (D_B T1 → pooled metastases) and Quinn 10k LG2, LG3, LG5 (LL → pooled).
2. **OT method.** No new OT. It uses the protocol M4-E / M4-R couplings, gate-masked.
3. **Features.** Five strategies.
4. **Preprocessing.** As E3.
5. **Cost.** As E3.
6. **Rejection and calibration.** As E3.
7. **Completion.** 4 families × 10 runs. The other 22 families were never run: the inventory lists 220 "not classified" rows.
8. **Validity.** The mass audit passed on 220 local runs. Output defects A–C were fixed (c193dfb); `classification_local_v1/` is the pre-fix version.
9. **Metric** (L2, level-1 accuracy, M4-E). OT never beats the size rule:
   - 3515: 0.92–0.99 vs size rule 0.99–0.995;
   - 10k LG2: 0.80–0.84 vs 0.81–0.86;
   - LG3: 0.79–0.83 vs 0.82–0.85;
   - LG5: 0.71–0.75 vs 0.73–0.80.
10. **Baseline and uncertainty.** Size rule and chance by mass; no interval.
11. **Outputs.** `within_family_subclones/output/` (definitions, classification) and `audit/`.
12. **Reusable.** The subclone definitions: yes, as L2 ground truth for any method.

### E6. Level 2: subclone matching, "A Standard OT" vs "B Our OT" (L2)

1. **Data.** 9 cases, 26 families:
   - KP 3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1: primary → its family's metastases pooled;
   - Quinn 100k, 10k, 30k, 5k: LL → all tissues pooled.
2. **OT method** (`L2/methods.py`).
   - **A** = entropic balanced OT (`traditional_ot.balanced.balanced_ot`), uniform marginals, ε 0.1.
   - **B** = `ConfidenceOT(backbone='uot', variant='exact', λ=1)` with rejection budget 0, so bounds (0, 0): KL-relaxed UOT with **no gate**. No cell is rejected (retained 1.0 / 1.0 in all 27 B runs; total mass 0.64–0.73).
   - **B is not a ConfidenceOT M4-E / M4-R result.**
3. **Features.** rna, cnv (the same CNV PCA for A and B), rna_cnv (RNA + CNV).
4. **Preprocessing.**
   - A: `logcpm_noscale` on raw counts.
   - B: `ranknm256_noscale_ds_cos` on equalised counts.
   - So in rna and rna_cnv, counts, representation and solver all differ together. Only cnv isolates the solver.
5. **Cost.** A: squared Euclidean / median. B: cos / median (rna), squared Euclidean / median (cnv, rna_cnv).
6. **Rejection and calibration.** None for either method.
7. **Completion.** 54/54 ok.
8. **Convergence.** All converged (`L2/results/diagnostics.tsv`).
9. **Metric.** Balanced accuracy of the enrichment argmax.
   - Evaluable: L1 17 / L2 12 family-levels.
   - Median (BA − chance): −0.024 to +0.029 per strategy and method.
   - Families above chance at L1: rna A 10/17, B 8/17; cnv 6 vs 8; rna_cnv 8 vs 8.
   - Strong cases:
     - 3724 L2: 0.93–0.99;
     - Quinn 100k LG4 L1, cnv: 0.856, on 52 cells.
10. **Baseline and uncertainty.** Chance = 1 / candidates and the size rule. **No uncertainty and no mouse aggregation.** 14/29 family-levels have a smallest true subclone of fewer than 20 cells.
11. **Outputs.**
    - Couplings: local `L2/data/full_20261022_204810/` (8.1 GB) and TACC.
    - Tables: `L2/results/` (tracked).
12. **Reusable.**
    - Yes, as a balanced-OT and no-rejection-UOT reference on fixed cells.
    - Not as a ConfidenceOT gate result.

### E7. Level 3: cell-level tree distance (L3)

1. **Data.** The same 54 couplings as E6: 26 families, 8 mice, 39,849 evaluable metastatic cells.
2. **OT method.** Those of E6: balanced OT and UOT with budget 0, so there is no rejection.
3. **Features.** As E6.
4. **Preprocessing.** As E6.
5. **Cost.** As E6.
6. **Rejection and calibration.** None.
7. **Completion.** 54/54 scored.
8. **Validity.** Column masses match Level 2 to 1e-11 (report).
9. **Metric** (ΔD_norm vs the independent coupling with the method's own marginals).
   - ΔD > 0 in 22–25/26 families and 7–8/8 mice.
   - Median mouse ΔD_norm 0.019–0.029.
   - 3724: 0.32–0.36, of which within-subclone only 0.001–0.018.
10. **Baseline and uncertainty.** Independent coupling. 95% bootstrap **over cells**, which are not independent; no inference over mice.
11. **Outputs.** L3 `ground_truth/`, `scores/`, `summary/`, `within_subclone/`.
12. **Reusable.** Yes, especially `ground_truth/targets.tsv.gz` (cell status, comparable primaries, d_min) for any new method.

### E8. Lineage × site matrices

1. **Data.** All 11 KP mice and 4 Quinn mice.
2. **OT method.** None.
3. **Features.** Annotations only.
4. **Preprocessing.** —
5. **Cost.** —
6. **Rejection and calibration.** —
7. **Completion.** Phase 1 done; Phases 2–6 not started.
8. **Validity.** —
9. **Metrics.** Counts and fractions only.
10. **Baseline and uncertainty.** —
11. **Outputs.** `lineage_validation/lineage_site_application/lineage_site_matrices/` (also archived on F:).
12. **Reusable.** Yes, as ground truth.

---

## 2. Why RNA, CNV and RNA + CNV were limited on fine-grained lineage, and which cases were strong

### What limits the result

1. **The coupling inside a family follows the source composition, not the CRISPR branch.**
   - In all 4 classifiable families, the mean within-family weight on each primary subclone equals that subclone's retained-mass share to within 0.01–0.07, and the raw argmax equals the independent-coupling argmax for 89–100% of cells (`within_family_subclones/audit/AUDIT_REPORT.md`).
   - Cell-level profiles are far from independent: their total variation from the share is 0.48–0.69. But the cells they concentrate on are spread over subclones in proportion to subclone size.
2. **The gate resolves families, not subclones.**
   - In 3508 the M4-E gate keeps 2.7% of the unrelated T1 and 81% of T2 (audit). T2 is ~90% this gate term (§1, E3).
   - Within 3515 T1 both subclones are retained at the same rate (0.29–0.34 for RNA-type strategies).
3. **Expression differences between sister subclones are small next to within-subclone variation.**
   - Level 3's top-1 / 5 / 10 partners sit at about the independent expected distance (11.2–11.3 vs D_ind 11.34).
   - The within-subclone ΔD_norm is ~1–2% of the achievable improvement.
4. **CNV is a weak independent axis in these data.**
   - It is inferred from the same RNA.
   - The Quinn profiles have no reference, so they are relative between A549 subclones only.
   - KP tumours carry few large CNVs (`TUMOUR_LEVEL_REPRESENTATION_AUDIT.md` §3, report).
   - The cnv gate retains almost every source cell: KP D_A cov_S 0.91 vs 0.66 for rna. That halves the site signal (e_M 0.10 vs 0.27).
   - Where CNV helps, it is a few cases: Quinn 100k LG4 0.856 on 52 cells; Quinn COND − COND_ind 0.17 for cnv vs 0.12 for rna, over 4 mice.
5. **Resolution of the truth.**
   - KP has no cell-level label within a family.
   - The trees have no branch lengths and huge polytomies (3724: 6,519 children; 3508: 1,033).
   - D1's modal s holds 62% of the cells.
   - Many subclones are tiny: 14/29 evaluable family-levels have a minority class under 20 cells, and 55 rare subclones can never be candidates.
6. **Identity under balanced OT.** The column-mass-weighted mean of the subclone weight equals the source share (L2 `PLAN.md`), so a one-subclone target can never show enrichment. Pooling targets only partly lifts this.
7. **Depth.** In 3724, the strongest family, both the gate score and lineage depth follow sequencing depth after equalisation (D1 post hoc).

### Strong case studies

- **KP 3724 T1.** S1 is seeded by L1.1 and L1–L3 by L1.2. Level 2 L2 balanced accuracy is 0.93–0.99 for every strategy and both methods; Level 3 ΔD_norm is 0.32–0.36. Both are subclone-level separation of whole sites. One mouse.
- **KP 3508 and 3513 (family-level gate).** T1 e_M 0.70 and 0.62; T2 0.30 and 0.25. 3513 is a single metastasis, and 5 of 8 3508 runs have M4-E calibration-inference validity False.
- **Quinn 100k and 10k (clone level).**
  - COND − COND_ind 0.26 and 0.16 (rna) to 0.40 and 0.27 (rna_then_cnv); label-shuffle p 0.001.
  - 10k XM: 856 of 857 cells belong to clones unobserved in LL, and the target rejection AUROC is 0.95.
- **KP 3515 T1.** Weak: site L1 recall 0.85–0.93 and 12 N1 L1.2 cells; it never beats the size rule.

---

## 3. Site metadata and other OT methods: checks (a)–(c)

**(a) Was site ever a feature or part of a site-aware cost? No.**
- **`cancer_metastasis/02_run_pair.py`.**
  - The cost is `squared_euclidean(source_pca, target_pca) / scale` (line 725), on expression PCs, CNV PCs or their concatenation (lines 705–720).
  - The obs columns `malignant`, `sample_type`, `site`, `tissue` and `lesion` are only copied into the output table (`confidence_frame`, lines 217–219).
  - No batch or site key exists in the file.
- **`lineage_validation/01_build_inputs.py`.** It writes `site` into the pooled obs (lines 65, 76, 177), but the protocol states that OT reads only `sample_id`, `malignant` and, for output tables, `cell_type` / `site` (protocol §2).
- **`level2_pairwise_ot/run_pairwise.py`.** Site only selects and labels cells (lines 96–142).
- **`kp_lineage`** contains no site term in any cost.
- **Lineage did not enter the cost either, but it did choose inputs.** Level 2 families and sites were chosen by CRISPR composition (see the companion audit, §5.2).

**(b) What site metadata exists, and is it constant within a side?**
- **KP-Tracer.**
  - `Tumor` (the site; T* = primary lung tumour, others = metastasis), `SubTumor` (sequenced piece), `is_normal`, `MetFamily`.
  - The obs `site` of the prepared files is only the category (primary / metastasis).
- **Quinn.**
  - `sampleID` = tissue; LL (or 30kLL) = left lung = the injection site, the only "primary".
  - All cells are A549, with no normal cells.
- **One site per side** in L1a pairwise (E1, E2) and in the v3 couplings (E4). There, a site feature would be **constant within each side and carry no cell-level information**: it would add the same constant to every cost entry.
- **Several sites per side** in the pooled designs:
  - protocol D_A source (2–5 primaries), D_B target (2–8 metastases), D_C3 both, Quinn target (5–6 tissues);
  - Level 2 / 3 targets (pooled metastatic sites).
  - There a site label varies within a side. For KP D_A it would be near-equivalent to MetFamily (each primary = one family), i.e. **lineage leakage**. For pooled targets it identifies the metastasis.
  - Within one site, every cell of a KP tumour has the same MetFamily, so site carries no within-family information.
- **The source of Level 2 and of the Quinn designs is always one site** (one primary, or LL), so site is constant on the source side there.

**(c) LineageOT, CloneOT, moscot, moslin, UOT or balanced-OT runs on these datasets?**
- **LineageOT, CloneOT, moslin:** none. A repo-wide grep finds them only in `kp_lineage/validation_v3/` literature and design documents. moscot appears only in `mouse_embryo/` (MOSTA).
- **Balanced OT:**
  - Level 2 / 3 "A Standard OT": 27 runs, entropic, ε 0.1, uniform marginals.
  - The protocol §10 balanced-OT baseline: 204 runs, **all FAILED**.
  - The v3 stage-F balanced-OT / EOT baselines: never run.
- **Plain UOT without gating:** Level 2 / 3 "B Our OT" (27 runs, ConfidenceOT KL-UOT with budget 0).
- **Every other KP / Quinn run** is ConfidenceOT M4-E / M4-R with rejection.
- **No partial OT or comparator-benchmark method** (UOT with u > 0.5, IC-POT) was run on KP-Tracer or Quinn.
