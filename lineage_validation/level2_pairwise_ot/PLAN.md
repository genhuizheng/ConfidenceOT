# Level 2: within-family subclone identification with pair-wise Balanced OT, two preprocessings

Status: **fixed 2026-10-22, before any Level 2 coupling was computed.**
- An exploratory biological application, separate from the frozen protocol (`../VALIDATION_PROTOCOL.md`, unchanged) and from the earlier subclone analysis and its audit (kept as they are).
- Subclones are the frozen CRISPR subclones of `../within_family_subclones/SUBCLONE_RULES.md` (a545f1c), recomputed by the same code.
- Nothing here is tuned on CRISPR labels. A later change is a dated section at the end, marked post hoc.

## Why no rejection here

Level 1 already tells which primary and metastasis are related (same CRISPR family / clone). Level 2 asks, inside one related pair, whether OT matches metastatic cells to the primary subclone that shares their CRISPR branch. So every eligible malignant cell of both sides is matched, and no gate is used.

## Solver (verified, `verify_zero_rejection.py`, `verify_zero_rejection.tsv`)

**Zero rejection does not by itself give Balanced OT.**
- In `ConfidenceOT`, `source_rejection_bounds = target_rejection_bounds = (0, 0)`, or the legacy `rejection_budget = 0` (mapped to `(0, 0)` by `resolve_rejection_bounds`, `src/confidenceot/_cpu_uot.py:263`), keeps every cell. The result no longer depends on the rejection cost.
- But with `backbone="uot"` (the only backbone `02_run_pair.py` uses, `:754`, `:801`) the plan is still KL-relaxed UOT (λ = 1). On the test cost its total mass is 0.82 and the largest row-marginal deviation is 62% of 1/n.
- `ConfidenceOT(backbone="balanced", bounds (0, 0))` is identical entry for entry to `traditional_ot.balanced.balanced_ot` (max difference 0, marginal residual 5e-15).

**Level 2 uses `balanced_ot` directly:**
- entropic balanced OT, uniform marginals;
- ε = 0.1;
- cost = squared Euclidean between coordinates, divided by `median_pair_scale` (the `02_run_pair.py` rule);
- threshold 1e-9, at most 10,000 iterations;
- convergence and marginal residuals recorded for every run.

## Pairs and cells

- **A pair** is one CRISPR family's primary and one of its metastatic sites:
  - KP-Tracer: `<family primary tumour>` → a metastasis with `MetFamily` = that family;
  - Quinn: LL → one tissue (within-clone analysis for every clone with subclones).
- **Cells.** All malignant cells of the two sites that passed the project's QC: the cells of the depth-equalised files (`27_downsample_counts.py`). Both preprocessings use exactly these cells.
- **Eligible pair.** The family/clone is eligible under SUBCLONE_RULES.md §5, and the metastatic site has ≥ 20 reliable metastatic cells with a level-1 subclone label.
- **Pilot** (fixed now, chosen by CRISPR composition only, before any Level 2 coupling exists):
  - KP 3457 T1 → K1;
  - 3515 T1 → L2;
  - 3515 T1 → K2;
  - 3724 T1 → S1;
  - 3724 T1 → L2;
  - Quinn 10k LL → M2.
  - These cover metastasis-associated branches with few primary cells (3457 L1.1.2; 3515 L1.1.2), two metastases on different primary branches (3724 S1 vs L2) and a Quinn mouse with several clones.

## Preprocessings (same cells, same solver, same ε and cost-scale rule)

| Arm | Label (`Preprocessing.from_label`) | Counts | Steps |
|---|---|---|---|
| A, standard BOT baseline | `logcpm_noscale` | raw | library size 1e4 → log1p → 2,000 joint HVGs → PCA 30 (centred, no per-gene scaling) → squared Euclidean |
| B, project preprocessing | `ranknm256_noscale_ds_cos` | depth-equalised (27_) | rank encoding without gene-median division, top 256 per cell → 2,000 joint HVGs → PCA 30, no per-gene scaling → unit rows (cosine / angular geometry) |

- n_hvg, n_pcs and the seed are `02_run_pair.py`'s defaults (2,000, 30, 20260831).
- `logcpm` with per-gene scaling is the scanpy-standard variant. It is not run, so the comparison stays at two arms, as asked.

## Metrics (per pair, arm, level L1 and L2)

**Weights.** For a target cell j and a source subclone g of its family / clone:
- q_ij = P_ij / Σ_k P_kj;
- S_gj = Σ_{i∈g} q_ij;
- within-family S̃_gj = S_gj / Σ_{g∈F} S_gj, where F also contains `family unlabelled`.
- The source marginal share π_g^F = Σ_{i∈g} a_i / Σ_{i∈F} a_i. Under Balanced OT with uniform a this is the cell share, and it is the independent coupling's S̃.
- Enrichment E_gj = S̃_gj / π_g^F, left undefined when π_g^F < 0.01 or g has < 5 source cells.

**Scored cells.** Target cells with a level-1 (or level-2) label whose subclone has ≥ 1 source cell. Cells without a label, or whose subclone has no source cell (`unmatched_observed`), are counted, not scored.

**Reported:**
1. **Own-subclone enrichment.** For each true subclone t: mean S̃_tj, π_t^F and E_t = mean S̃_tj / π_t^F; and the cell-level mean of log2(S̃_{t(j)} j / π_{t(j)}).
2. **Classification.**
   - Enrichment argmax: argmax_g E_gj over defined g. This is the primary rule, because raw mass follows the source marginals.
   - Raw-mass argmax (reported for comparison).
   - For each: accuracy, balanced accuracy (mean per-true-subclone recall), and the confusion matrix.
3. **Rare subclones.** Recall of the enrichment argmax for true subclones with π_t^F < 0.10.
4. **References.** The independent coupling (S̃ = π, E = 1 for every g), whose balanced accuracy for the enrichment argmax is the chance level 1 / (number of candidate subclones); and the size rule (always the largest subclone).
5. **A − B per pair:** own-subclone log-enrichment and balanced accuracy.

No test statistic; pairs are not pooled over cells. The decision to extend from the pilot to every eligible pair is taken on whether the two arms run cleanly (convergence, residuals) and on the size of the A–B differences, reported with the pilot.

**Level 3 (not in this step).** The pair-wise couplings are saved, so cell-level lineage correspondence (tree distance of matched cells) can reuse them.
