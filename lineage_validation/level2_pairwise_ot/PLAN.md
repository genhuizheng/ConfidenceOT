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

**Solvers used in Level 2** (amended 2026-10-22 by the user, before any coupling: "ranknm256_noscale_ds_cos → Our OT", "我们的OT应该是UOT"):
- **BOT**: `traditional_ot.balanced.balanced_ot`, entropic balanced OT with uniform marginals, threshold 1e-9 and at most 10,000 iterations.
- **Our OT (UOT, no rejection)**: `ConfidenceOT(backbone="uot", variant="exact", source_rejection_bounds=(0, 0), target_rejection_bounds=(0, 0))`. This is the M4-E backbone with λ_a = λ_b = 1, tolerance 1e-4 and device cpu, all `02_run_pair.py` defaults. Its marginals are KL-relaxed, not fixed; the verification above shows it keeps every cell.
- **Common to both**:
  - ε = 0.1;
  - cost = squared Euclidean between coordinates, divided by `median_pair_scale` (the `02_run_pair.py` rule and seeds);
  - convergence, total mass and row/column marginal deviations recorded for every run.

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

## Arms (same cells, same ε and cost-scale rule)

| Arm | Label (`Preprocessing.from_label`) | Counts | Representation | Solver |
|---|---|---|---|---|
| **A, standard BOT baseline** | `logcpm_noscale` | raw | library size 1e4 → log1p → 2,000 joint HVGs → PCA 30 (centred, no per-gene scaling) → squared Euclidean | BOT |
| **B, our pipeline** | `ranknm256_noscale_ds_cos` | depth-equalised (27_) | rank encoding without gene-median division, top 256 per cell → 2,000 joint HVGs → PCA 30, no per-gene scaling → unit rows (cosine / angular geometry) | our UOT, no rejection |
| A×UOT (decomposition) | `logcpm_noscale` | raw | as A | our UOT, no rejection |
| B×BOT (decomposition) | `ranknm256_noscale_ds_cos` | depth-equalised | as B | BOT |

- A vs B is the comparison asked for. It changes the preprocessing and the solver together.
- The two decomposition runs reuse the same two cost matrices. They tell how much of A − B comes from the representation (A vs B×BOT, A×UOT vs B) and how much from the solver (A vs A×UOT, B×BOT vs B). They are reported beside A vs B, never instead of it.
- n_hvg, n_pcs and the seed are `02_run_pair.py`'s defaults (2,000, 30, 20260831). The seed index is the pair's row in `pilot_pairs.tsv`, the same for every arm.

## Metrics (per pair, arm, level L1 and L2)

**Weights.** For a target cell j and a source subclone g of its family / clone:
- q_ij = P_ij / Σ_k P_kj;
- S_gj = Σ_{i∈g} q_ij;
- within-family S̃_gj = S_gj / Σ_{g∈F} S_gj, where F also contains `family unlabelled`.
- The source marginal share π_g^F = Σ_{i∈g} m_i / Σ_{i∈F} m_i, with m_i = Σ_j P_ij the plan's realised row mass. Under BOT m_i = 1/n, so π is the cell share. Under UOT it is the mass the plan actually sends, which is the share the independent coupling with the same marginals would give (the retained-effective-mass reference of the audit). The cell-count share is reported beside it.
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
5. **A − B per pair:** own-subclone log-enrichment and balanced accuracy, with the four-run decomposition.

No test statistic; pairs are not pooled over cells. The decision to extend from the pilot to every eligible pair is taken on whether the two arms run cleanly (convergence, residuals) and on the size of the A–B differences, reported with the pilot.

**Level 3 (not in this step).** The pair-wise couplings are saved, so cell-level lineage correspondence (tree distance of matched cells) can reuse them.

## 2026-10-22, before any real coupling: what a single-pair plan can and cannot show

Found while testing the pipeline on synthetic counts (`test_pipeline.py`, `test_pipeline_result.txt`); no real pair has been run.

**Identity.** For any plan P with column masses c_j, the column-mass-weighted mean over target cells of S_gj equals the plan's own source share π_g:

Σ_j c_j S_gj = Σ_j Σ_{i∈g} P_ij = row mass of g.

- Under BOT c_j = 1/m, so the plain mean over target cells of S_g is exactly the source cell share. If every scored target cell belongs to one subclone t (a metastasis made of one subclone at that level), the mean own-subclone enrichment is exactly 1 for every preprocessing. Below 1 in log mean by Jensen. The enrichment-argmax then reads noise. On a synthetic monoclonal target the balanced accuracy was 0.34 (chance 0.5).
- Under our UOT the same identity holds with the realised row masses. Against the realised share the mean own enrichment is again ≈ 1 (1.17 on the same synthetic case). Against the cell-count share it is not (1.65), because the relaxed marginals let some source cells send less mass, i.e. the soft rejection is where the signal sits.
- On a mixed target (40 A + 40 B) both solvers separate the subclones: balanced accuracy 0.86 BOT, 0.89 UOT. Enrichment can only show *which target cells* differ from the others, never that the whole site differs from the primary's composition.

**Consequence for the pilot.** At the scored level almost every pilot metastasis is one subclone:
- 3457 K1 L2: 137 / 137 L1.1.2;
- 3515 L2: 241 / 241 L1.1.2;
- 3515 K2: 360 / 362 L1.1.1;
- 3724 S1: 3,779 / 3,789 L1.1;
- 3724 L2: 1,510 / 1,512 L1.2;
- at L1, 3515 L2 / K2 and 3457 K1 are 100% L1.1.

Quinn 10k M2 is mixed (LG2 73 L1.1 / 19 L1.2).

On those KP pairs arm A (BOT) is at its null by construction. A − B would measure the solver's marginal relaxation, not the preprocessing. The design is not run until the user decides how to proceed.

## 2026-10-22, decision (user, before any real coupling): one plan from the primary to all its family's metastases

Chosen by the user after the identity above ("Pool the family's metastases").

- **Source.** All eligible malignant cells of the family's primary.
- **Target.** All eligible metastatic sites of the family, in one plan; still no rejection, the same four runs.
  - KP-Tracer: every metastatic site with ≥ 20 reliable family cells carrying a level-1 subclone label.
  - Quinn: the lineage-validation metastases side (every tissue with ≥ 20 cells), because every tissue holds cells of every clone.
  - The rule reads CRISPR composition only (`make_inputs.py`).
- **Why it helps.** The sites now compete for each source cell's mass, so a site can differ from the primary's composition even under BOT. The identity still fixes the mean over *all* target cells, but no longer that of one site.
- **Level 3.** A primary → site block is a column slice of the plan; `cells.tsv.gz` keeps the order.
- **Pilot** (replaces the six pairs above):
  - KP 3457 T1 → {K1, N2};
  - 3515 T1 → {K1, K2, K3, L1, L2, N1};
  - 3724 T1 → {L1, L2, L3, S1};
  - Quinn 10k LL → all tissues, clones LG2, LG3, LG5.
  - No site was left out by the ≥ 20 rule.
- **3457 is known in advance to be uninformative.** Its scored metastatic cells are one subclone at both levels: L1.1 at L1, L1.1.2 at L2 in K1 and N2. The scorer flags such a family / level (`single_true_class`) and it is reported as not evaluable, not as a result.
- **Scoring additions.** A per-site table (`per_site.tsv`): recall and own log-enrichment per site and true subclone.
