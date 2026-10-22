# Level 2: within-family subclone identification, Standard OT workflow vs Our OT workflow

Status: **fixed 2026-10-22, before any Level 2 coupling was computed.** Simplified by the user the same day, still before any coupling, to exactly two OT runs per test case.
- An exploratory biological application, separate from the frozen protocol (`../VALIDATION_PROTOCOL.md`, unchanged) and from the earlier subclone analysis and its audit (kept as they are). Level 1 is not re-run.
- Subclones are the frozen CRISPR subclones of `../within_family_subclones/SUBCLONE_RULES.md` (a545f1c), recomputed by the same code.
- Nothing here is tuned on CRISPR labels. A later change is a dated section at the end, marked post hoc.
- The question is a direct comparison of two whole workflows, not a preprocessing × solver factor analysis. No other OT method is run.

## Why no rejection here

Level 1 already tells which primary and metastases are related (same CRISPR family / clone). Level 2 asks, inside one related family, whether OT matches metastatic cells to the primary subclone that shares their CRISPR branch. So every eligible malignant cell is matched, and no gate is used.

## The two methods

| | Method A — Standard OT baseline | Method B — Our OT |
|---|---|---|
| Counts | raw | depth-equalised (`27_downsample_counts.py`) |
| Preprocessing (`Preprocessing.from_label`) | `logcpm_noscale`: library size 1e4 → log1p → 2,000 joint HVGs → PCA 30 (centred, no per-gene scaling) | `ranknm256_noscale_ds_cos`: rank encoding without gene-median division, top 256 per cell → 2,000 joint HVGs → PCA 30, no per-gene scaling → unit rows (cosine) |
| Cost | squared Euclidean / `median_pair_scale` | the same |
| Solver | `traditional_ot.balanced.balanced_ot`: entropic Balanced OT, uniform marginals, threshold 1e-9, ≤ 10,000 iterations | `ConfidenceOT(backbone="uot", variant="exact")` (M4-E): our KL-relaxed UOT, λ_a = λ_b = 1, tolerance 1e-4, device cpu |
| Rejection | none | `source_rejection_budget = target_rejection_budget = 0`: no cell is gated (checked on every run). The KL-relaxed marginals are kept: this is **not** Balanced OT |
| ε | 0.1 | 0.1 |

- Shared by both methods:
  - the same cells (the eligible malignant cells of the equalised files, the same ids in the raw files);
  - the same primary and pooled metastatic sites;
  - the same seeds as `02_run_pair.py` (representation seed 20260831 + index, scale seed 20260831 + index × 104729; the index is the test case's row in `inputs/pilot_pairs.tsv`);
  - the same CRISPR ground truth.
- Convergence, total mass, row / column marginal deviations and gate retention are recorded for every run.

### Verification that budget 0 is not Balanced OT (`verify_zero_rejection.py`, `verify_zero_rejection.tsv`)

- `rejection_budget = 0` is mapped to the bounds (0, 0) by `resolve_rejection_bounds` (`src/confidenceot/_cpu_uot.py:263`) and keeps every cell. The plan is then independent of the rejection cost, and identical to passing the bounds (0, 0) directly.
- With `backbone="uot"` the plan stays KL-relaxed UOT. On the test cost its total mass is 0.82 and the largest row-marginal deviation is 62% of 1/n.
- Only `backbone="balanced"` with (0, 0) equals `balanced_ot` (marginal residual 5e-15). Level 2 does not use it; Method A uses `balanced_ot` itself.

## Test cases (pilot)

One plan per family, from the primary to all the family's metastatic sites pooled (user decision 2026-10-22, section below):
- **Source.** All eligible malignant cells of the family's primary.
- **Target.** All eligible metastatic sites of the family:
  - KP-Tracer: every metastatic site with ≥ 20 reliable family cells carrying a level-1 subclone label;
  - Quinn: the lineage-validation metastases side (every tissue with ≥ 20 cells), because every tissue holds cells of every clone.
  - A site without an equalised file, or with < 20 malignant cells in it, is left out and recorded (`sites.json`).
  - The rule reads CRISPR composition only (`make_inputs.py`).
- **Pilot:**
  - KP 3457 T1 → {K1, N2};
  - 3515 T1 → {K1, K2, K3, L1, L2, N1};
  - 3724 T1 → {L1, L2, L3, S1};
  - Quinn 10k LL → all tissues, clones LG2, LG3, LG5.
- **3457 is known in advance to be uninformative.** Its scored metastatic cells are one subclone at both levels. It is flagged (`single_true_class`) and reported as not evaluable.
- Each test case has exactly two OT runs, A and B (`inputs/run_manifest.tsv`).

## Metrics (per test case, method, family / clone, level L1 and L2)

**Weights.** For a target cell j and a source subclone g of its family / clone:
- q_ij = P_ij / Σ_k P_kj;
- S_gj = Σ_{i∈g} q_ij;
- within-family S̃_gj = S_gj / Σ_{g∈F} S_gj, where F also contains `family unlabelled`.
- π_g^F = Σ_{i∈g} m_i / Σ_{i∈F} m_i, with m_i = Σ_j P_ij each method's **actual** row mass.
  - For A (BOT) m_i = 1/n, so π is the cell share.
  - For B (UOT) it is the mass the plan actually sends. This is the share the independent coupling with the same marginals would give.
  - The cell-count share is reported beside it.
- Enrichment E_gj = S̃_gj / π_g^F, left undefined when π_g^F < 0.01 or g has < 5 source cells (the candidates are the subclones with E defined).

**Scored cells.** Target cells of the family with a level-1 (or level-2) label whose subclone has ≥ 1 source cell. Cells without a label, or whose subclone has no source cell (`unmatched_observed`), are counted, not scored.

**Reported, A vs B:**
1. **CRISPR-defined subclone identification.**
   - The primary rule is the enrichment argmax over the candidates, because raw mass follows the source marginals; the raw-mass argmax is reported for comparison.
   - Accuracy, balanced accuracy (mean per-true-subclone recall), confusion matrix, per site (`per_site.tsv`) and per subclone (`per_subclone.tsv`).
2. **Rare primary subclone recovery.** Recall of the enrichment argmax for true subclones with π_t^F < 0.10.
3. **Matching enrichment relative to each method's actual marginals.** Per true subclone t: mean S̃_tj, π_t^F and E_t = mean S̃_tj / π_t^F, and the cell-level mean of log2(S̃_{t(j)} j / π_{t(j)}).
4. **Cell-level lineage correspondence (Level 3).**
   - Computed later from the same saved couplings (`<method>/coupling.npz`, row and column order in `cells.tsv.gz`). A primary → site block is a column slice.
   - Not part of this step; no new OT is run for it.
- **References:** the independent coupling (E = 1 for every g; balanced accuracy of the enrichment argmax at chance, 1 / candidates) and the size rule (always the largest subclone).

No test statistic; cells are not pooled over test cases. Extending from the pilot to every eligible family is decided on whether both methods run cleanly (convergence, marginals, no gated cell) and on the size of the A–B differences, reported with the pilot.

## 2026-10-22, before any real coupling: what one plan can and cannot show

Found while testing the pipeline on synthetic counts (`test_pipeline.py`); no real pair had been run.

**Identity.** For any plan P with column masses c_j, the column-mass-weighted mean over target cells of S_gj equals the plan's own source share π_g:

Σ_j c_j S_gj = Σ_j Σ_{i∈g} P_ij = row mass of g.

- **Under BOT** c_j = 1/m, so the plain mean over target cells of S_g is exactly the source cell share.
  - If every scored target cell belongs to one subclone, the mean own-subclone enrichment is exactly 1 for any representation.
  - On a synthetic one-subclone target the balanced accuracy was 0.34 (chance 0.5).
- **Under our UOT** the same identity holds with the realised row masses: against them the mean own enrichment was ≈ 1 (1.17); against the cell-count share it was 1.65.
- **On a mixed target** both solvers separated the subclones (balanced accuracy 0.86 BOT, 0.89 UOT).

The first pilot (one primary → one metastasis) had nearly one-subclone targets in five of six KP pairs. For example, 3515 L2 was 241 / 241 L1.1.2 and 3724 S1 was 3,779 / 3,789 L1.1.

## 2026-10-22, decision (user, before any real coupling): one plan from the primary to all its family's metastases

Chosen by the user after the identity above ("Pool the family's metastases"). The sites compete for each source cell's mass, so a site can differ from the primary's composition even under BOT. The identity still fixes the mean over *all* target cells, but no longer that of one site. The pooled pilot above replaces the six single-metastasis pairs.

## 2026-10-22, simplification (user, before any real coupling): two methods only

- The user removed the 2×2 preprocessing × solver design and its two cross runs (logcpm + UOT, ranknm256_noscale_ds_cos + BOT).
- Every test case runs exactly Method A and Method B, and the comparison is Standard OT workflow vs Our OT workflow.
- Method B's no-rejection setting is written as `rejection_budget = 0`, as the user specified. It is the same plan as the bounds (0, 0) that the earlier draft used (`verify_zero_rejection.tsv`, rows 2 and 3).
