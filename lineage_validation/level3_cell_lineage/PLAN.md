# Level 3: cell-level CRISPR lineage correspondence of the Level 2 couplings

Status: **fixed 2026-10-22, before any Level 3 quantity was computed from a coupling.** A later change is a dated section at the end, marked post hoc.
- An exploratory analysis of the 54 existing Level 2 couplings (`../level2_pairwise_ot/`, `REPORT.md` there): 9 cases × RNA-only / CNV-only / RNA+CNV × A (Standard OT, Balanced OT) / B (Our OT, UOT with rejection budget 0).
- No OT is re-run. Preprocessing, solver, ε, trees and subclone definitions are unchanged. The frozen protocol (`../VALIDATION_PROTOCOL.md`) and Level 1 are untouched.
- Every case and every eligible family is analysed; nothing is selected by result.

## Lineage ground truth (method-independent; `ground_truth.py`)

**Trees and cells.**
- Each family / clone uses the authors' tree its subclones were defined on (`../within_family_subclones/output/feasibility.tsv`, column `tree`):
  - KP-Tracer: the `_All` or `_Fam` Cassiopeia tree, restricted to the family's leaves;
  - Quinn: the clone's own tree.
- The leaf ↔ cell mapping and the reliability flag (character missing share ≤ 0.5) are those of `../within_family_subclones/output/cells/` (SUBCLONE_RULES.md, a545f1c).
- A primary and a metastatic cell are compared only if both are reliable leaves of the **same** family / clone tree. Distances are never compared across independently reconstructed trees.

**Distance.**
- d(i, j) = depth(i) + depth(j) − 2 · depth(LCA(i, j)), with the contracted depth of the project's `Tree` class (`kp_lineage/validation_v2/04_lineage_concordance.py`, the depth the frozen protocol's Level 3 used).
- A unary node is merged with its child, so the distance counts branching steps only. The trees carry no branch lengths.
- A small d means closer CRISPR relatives in the reconstructed tree. It does not mean ancestry: the tree-nearest sampled primary cell is not taken to be the seeding ancestor.

**Comparable sources.** For metastatic cell j of family F, S_j is the case's primary (source) cells, in the plan's row order (`cells.tsv.gz` of the Level 2 run), that are reliable leaves of F's tree.

**Status of a metastatic cell.** The status depends on the tree only, so it is identical for every method and strategy.

| Status | Rule |
|---|---|
| `not_evaluable` | j is not a reliable leaf of a family / clone of the case (no leaf, missing share > 0.5, or another family); or \|S_j\| < 5 |
| `not_informative` | every source in S_j is at the same distance from j: the tree cannot separate them |
| `ambiguous` | the tree-nearest set R_j (below) holds ≥ 50% of S_j: the tree singles out no close relatives |
| `evaluable` | otherwise |

- Primary analysis: evaluable cells. Ambiguous cells are reported separately. Not-informative and not-evaluable cells are counted with their reasons, not scored.
- **Close relatives.** R_j = { i ∈ S_j : d(i, j) = d_min(j) }, the tree-nearest comparable primary cells. This is fixed here, before any A / B result.

## Matching quantities (per metastatic cell j, strategy, method; `score_level3.py`)

**From the saved coupling P.** P is float32; rows follow the source order of `cells.tsv.gz`, columns the target order.
- q_ij = P_ij / Σ_k P_kj over **all** source cells.
- m_i = Σ_j P_ij is the method's actual row mass: uniform for A, the realised UOT marginal for B.

**Total-mass-aware coverage.**
- c_j^OT = Σ_{i∈S_j} q_ij is the share of j's matched mass that lands on comparable sources.
- c_j^ind = Σ_{i∈S_j} m_i / Σ_k m_k is the same share under the independent coupling with the method's own marginals.
- Both are reported. Nothing is renormalised silently.

**Tree-covered conditional analysis.**
- q̃_ij = q_ij / c_j^OT and π̃_i = m_i / Σ_{k∈S_j} m_k, for i ∈ S_j.
- Weighted lineage distance: D_j^OT = Σ q̃_ij d(i, j).
- Independent baseline: D_j^ind = Σ π̃_i d(i, j), with the method's actual marginals, never assumed uniform for UOT.
- ΔD_j = D_j^ind − D_j^OT; positive means OT favours closer relatives than the independent coupling does.
- Effect size comparable across trees: ΔD_j^norm = ΔD_j / (D_j^ind − d_min(j)). It is 1 when all mass is on the nearest relatives and 0 at the independent coupling; it can go negative.

**Top-k (k = 1, 5, 10).**
- Ranking: sources ranked by q_ij among all sources.
- Total-mass-aware: the share of the top k that are comparable.
- Conditional: the mean d of the k comparable sources with the largest q̃, and whether the top-1 is in R_j.
- References:
  - achievable: d_min(j);
  - expected under the independent coupling: D_j^ind, the mean distance of draws proportional to π̃;
  - expected top-1-in-R_j rate under it: Σ_{i∈R_j} π̃_i.

**Close-relative enrichment.**
- M_j^OT = Σ_{i∈R_j} q̃_ij and M_j^ind = Σ_{i∈R_j} π̃_i.
- Reported: log2(M_j^OT / M_j^ind), and the total-mass-aware Σ_{i∈R_j} q_ij against Σ_{i∈R_j} m_i / Σ_k m_k.

**Primary comparisons.**
- A and B are compared on the same evaluable cells and the same comparable sources. They differ only in P and in the marginal that defines π̃.

## Aggregation

- **Cell → family / clone:** mean over its evaluable metastatic cells, also per metastatic site. 95% bootstrap interval over cells (1,000 draws, fixed seed).
- **Family → mouse:** mean over the mouse's families / clones (KP 3457 has two; Quinn mice have 2–12 clones). A large family therefore cannot dominate the summary.
- **Across mice:** median, and the number of mice with mean ΔD > 0, per strategy and method; A − B paired by family.
- **Coverage is reported beside every effect:**
  - evaluable cells / all metastatic cells;
  - comparable sources / all sources;
  - c^OT and c^ind.
- No test statistic is the result.

## Flags (reported separately, not removed)

- **High missingness:** the family's evaluable metastatic cells have a mean character missing share > 0.3, or > 20% of the family's tree leaves are unreliable.
- **Low tree resolution:** more than half of the family's metastatic cells are ambiguous or not informative.

## Figures (`plot_level3.py`; rules fixed here)

- **A — cell-level matching on a real branch.**
  - For every family with evaluable cells, take the frozen level-1 subclone holding the most evaluable metastatic cells. Draw its clade from the authors' tree.
  - Keep at most 40 primary and 40 metastatic leaves, thinned deterministically every n-th in leaf order.
  - For each drawn metastatic leaf, show its top-1 OT primary under A and under B (RNA-only; the other strategies in separate files).
  - One file per family: no example is chosen by result.
- **B — OT vs independent.** Per family, the mean ΔD^norm with its 95% interval, A and B side by side, in three panels (RNA, CNV, RNA+CNV). Per-mouse means are marked.
- **C — success and failure examples.**
  - Per strategy and method, among all evaluable cells pooled, take the cell at the 90th and at the 10th percentile of ΔD^norm; ties go to the smaller cell id.
  - Each panel shows the mass on sources at each tree distance (OT q̃ against independent π̃) and the number of sources available at each distance. It is annotated with d_min, the top-1 distance and weight, and c^OT.
