# Lineage ground-truth validation: audit summary (2026-10-21)

Source: `lineage_validation/00_audit.py` (commit 3ec25b4), job 1087596, output `/scratch/10119/ghzheng/OT_project/lineage_validation/audit/`. Read only: no OT was fitted and no gate was read against lineage. Rules fixed in the script before the run:
- a site is eligible with >= 20 usable cells;
- a lineage is "observed" at a site with >= 1 labelled cell (sensitivity: >= 5).

Code facts are from a read of the repo; file:line references are in the session notes.

## 1. Data availability

### KP-Tracer (11 mice with eligible pairs)

| Mouse | Eligible primaries | Eligible metastases | Metastasis families (authors' MetFamily) | Trees |
|---|---|---|---|---|
| 3451 | T1 T2 T3 | K1 L1 PF S1 S2 | all T1 | one tree per primary only (no family tree): **metastases on no tree** |
| 3454 | T1 T2 T3 | L1 N1 | both T1 | `_All` (1,849 leaves) + per primary; L1 on no tree (1,491 cells), N1 on `_All` |
| 3457 | T1–T5 | K1 L1 N2 | K1, N2 → T1; L1 → T4 | T1_Fam, T4_Fam, per primary |
| 3460 | T1 T2 | N1–N6 | N2–N6 → T1; N1 → its own family (no sampled primary) | T1_Fam, per primary; N1 on no tree |
| 3465 | T1 | DN1 K1 K3 L1 Spl1 | all T1 | per primary only: metastases on no tree |
| 3466 | T1 T2 | N1 | T2 | per primary only: N1 on no tree |
| 3467 | T1 T2 | N1 N2 Rib | **`3467_Lkb1_Unk`** (unknown) | per primary only |
| 3508 | T1 T2 | N1–N6 PF Rib | all T2 | `_All` (4,798), T2_Fam, per primary |
| 3513 | T1–T5 | N1 | T1 | T1_Fam, per primary |
| 3515 | T1 T2 T3 | K1–K3 L1 L2 N1 | all T1 | `_All` (2,209), T1_Fam, per primary |
| 3724 | T1 | L1 L2 L3 S1 | all T1 | `_All` (21,108), per primary |

- **Cell-level truth.** Different primaries are independent tumours (separate Cre events), so cells of different primaries are different lineages. That part is solid.
- **No per-cell clone label within a family.** Within a family, relatedness comes only from the tree (MRCA depth), and only where a family or `_All` tree holds both sides.
- **Cross-primary tree distances** need an `_All` tree: 3454, 3508, 3515 (and 3724, which has one primary).
- **Existing results:**
  - OT ranknm256_noscale_ds_cos and raw: 104 pairs each.
  - Couplings: 50 pairs × 2 labels, reproduction passed.
  - infercnvpy: 11 mice.

### Quinn 2021 (4 mice + LM0)

| Mouse | Tissues (eligible) | Clones | Unlabelled cells | Target cells whose clone is unobserved in LL (>= 1 LL cell rule) |
|---|---|---|---|---|
| 5k | LL, Liv, M1, M2, RE, RW | 100 | 5,609 | Liv 83, M1 596, M2 237, RE 449, RW 222 |
| 30k | 30kLL, 30kBL, 30kLN, 30kM1, 30kM2, 30kRL | **4** | 5,171 | 30kM1 132, others 0 |
| 10k | LL, M1, M2, NE, NW, SE, XM (BL 4 and SW 5 ineligible) | 13 | 5,677 | **XM 856 of 857**, others 0–1 |
| 100k | LL, MS, NE, NW, SE, SW | 31 | 1,484 | MS 21, NE 2, NW 33, SE 47, SW 0 |
| LM0 | pre-injection | — | 0 | LG_Present: Engrafted 731, NotEngrafted 7,995 |

- Each annotated cell carries a clone (LineageGroup) and a tissue (`sampleID`). Unlabelled cells have neither (in 30k and 100k their barcodes carry no tissue).
- Trees: one file per clone; the coverage of tree leaves on cell names is still to be read (`tree_coverage.tsv`, Quinn rows).
- Existing results: converted H5ADs, infercnvpy for 5 experiments (no reference). No OT run yet.

## 2. Feasibility matrix

| Experiment | KP-Tracer | Quinn 2021 |
|---|---|---|
| **A** pooled primaries → one metastasis (site level) | 9 mice (3451 3454 3457 3460 3466 3467 3508 3513 3515); **binary truth in 8** (3467's family is Unk): 31 metastases with related and unrelated primaries | **N/A**: one injection site per mouse |
| **B** one primary → pooled metastases (site level) | 9 mice have ≥ 2 metastases; a primary has both related and unrelated metastases only in **3457 (T1, T4) and 3460 (T1)**. Elsewhere every metastasis is related to one primary and unrelated to the others, so B is a negative control per unrelated primary (all its metastases should be rejected), not a ranking | 4 mice; **no binary site truth** (every tissue shares clones with LL), so site level is "retention vs shared-clone fraction", not AUROC |
| **C1** one primary → pooled metastases (cell level) | same runs as B; truth: family (same family or not) for every malignant cell; tree depth only where a Fam/All tree holds both | same runs as B; truth: clone per cell; tree depth within clone |
| **C2** pooled primaries → one metastasis (cell level) | same runs as A; truth as C1; cross-primary tree depth only in 3454, 3508, 3515 | **N/A** |
| **C3** pooled primaries → pooled metastases | 7 mice (3451 3454 3457 3460 3467 3508 3515); truth in 6 | **N/A** |

So the OT runs are three designs per mouse (A/C2, B/C1, C3), each under five strategies, at acceptance 0.99. C1–C3 score the couplings of those same runs.

## 3. Reuse plan

| Need | Reuse | New |
|---|---|---|
| OT fit, five strategies | `02_run_pair.py`, `tacc_cnv_rna_strategies.slurm` (env `STRAT_*`), two-step via `--input-gate-root` (step-1-rejected cells removed before step 2) | manifests for the pooled designs; `--max-observed-cells-per-side 0` (no cap, as KP-Tracer used) |
| `rna_cnv` = equal-weight normalised cost | verified: cost = [C_rna/med + C_cnv/med] / med; RNA is cosine (2 − 2cos), CNV is squared Euclidean | state the CNV geometry in the report (`run.json` says only "cosine") |
| Pooled sides | — | 02_ selects one `sample_id` per side, so a prep script writes pooled H5ADs (raw and equalised) sharing one sample_id, the original site kept in obs |
| Depth equalisation | `27_downsample_counts.py` (KP-Tracer already equalised) | Quinn: run 27_ once; add `malignant` = all A549 and `sample_id` to its prepared files |
| CNV | infercnvpy units, keyed `<dataset>/<mouse>`, cells by name | none (pooled sides read the same unit) |
| Couplings, gate scores | `coupling_m4e.npz` (dense, all analysed cells incl. rejected, rows/cols = cell_confidence order); `cell_confidence.csv` | marginals computed from T; q_ij = P_ij / Σ_k P_kj with rejected cells set to 0 |
| Trees | `validation_v2/04_lineage_concordance.Tree`, `validation_v3/01_audit_v3.lca_distribution` | Quinn per-clone tree reader (format still to be confirmed) |
| Level 3 metrics (ΔS_norm, ΔR, nulls, baselines) | frozen design, `LEVEL3_VALIDATION_DESIGN.md` | analysis code (stage F) has not been written; write it once for both datasets |
| Fixed joint UMAP | `48_reference_umap.py` on a gates root laid out `<root>/<strategy>/arm_*/<block>/<label>` | gates root for the new runs |
| DEG/GSEA (explanation only) | existing downstream | none |

## 4. Key limitations

1. **KP-Tracer has no per-cell clone label within a family.** Cell-level "origin" stops at tumour/family; within a family only tree relatedness exists, and only where a Fam/All tree holds both sides. The metastases of 3451, 3465, 3466 and 3467, 3454 L1 and 3460 N1 are on no tree.
2. **Same clone ≠ origin ≠ direction.** Both datasets sample at one time point. Quinn reports multidirectional seeding, so a clone shared by LL and a tissue does not show that the cells came from LL.
3. **Few true negatives in Quinn.** Most target cells' clones are present in LL. The exceptions are 10k XM (856/857 unobserved), 5k (several hundred per tissue) and 30k M1 (132). 30k has only 4 clones.
4. **"Unobserved in source" is not a confirmed negative.** A clone may exist at a site but not be sampled.
5. **Mouse is the unit.** Quinn has 4 mice (smallest one-sided Wilcoxon p = 1/16); KP-Tracer A has 8 with truth. Report effect sizes per mouse; never treat cells as replicates.
6. **Clone sizes are very uneven** (one clone dominates in several Quinn mice). Clone size goes in as a covariate; clone-label shuffle and the independent coupling are the nulls.
7. **Quinn unlabelled cells** (5k 5,609; 30k 5,171; 10k 5,677; 100k 1,484) stay in the OT fits but are never scored.
