# Dataset A: Splatter results (local N = 200 per side, replicates 1-8)

Run 2026-11-17, code as committed with this file; definitions in PLAN.md (post-hoc notes: UOT solver). Tables:
`results/ot_depth_benchmark_20261117/splatter/tables/`, figures: `.../splatter/figures/` (one PDF + PNG per panel).
Means over 8 replicates; 256 native pairs (S1) and 96 conflict / aligned / global fits (S2) x 2 costs x 6 solvers,
0 failures, balanced-solver marginal error < 2e-7. This is the local test scale; the same scripts run on the TACC
N = 5,000 root (`tacc/submit_otdb.sh splatter_s1|splatter_s2`).

## 1 Native scenarios (S1): depth is independent of population in Splatter

Depth carries almost no population information here (η² of log nCount by group 0.013 source / 0.027 target), so any
coupling-depth association is technical.

All-shared pairs, mean over the 8 levels:

| Cost | Solver | Correct-group mass | Corr. log nCount | Corr. log nFeature | Within-group corr. |
|---|---|---|---|---|---|
| C1 | Independent | 0.200 | 0.000 | 0.000 | 0.000 |
| C1 | Exact balanced | 0.838 | 0.099 | 0.096 | 0.073 |
| C1 | Entropic balanced | 0.692 | 0.050 | 0.050 | 0.043 |
| C1 | Entropic UOT | 0.712 | 0.051 | 0.051 | 0.043 |
| C1 | Partial (m 0.8) | 0.937 | 0.116 | 0.116 | 0.101 |
| C1 | ConfidenceOT | 0.706 | 0.052 | 0.052 | 0.044 |
| C2 | Exact balanced | 0.833 | 0.216 | 0.212 | 0.194 |
| C2 | Entropic balanced | 0.574 | 0.095 | 0.096 | 0.080 |
| C2 | Entropic UOT | 0.579 | 0.075 | 0.077 | 0.067 |
| C2 | Partial (m 0.8) | 0.910 | 0.178 | 0.177 | 0.176 |
| C2 | ConfidenceOT | 0.578 | 0.082 | 0.085 | 0.069 |

- Biology dominates: every solver puts 3-4.7x the independent-coupling mass on the correct population.
- Depth association is small but real, and it is mostly **within** the correct population (within-group correlation
  close to the total). Among equivalent partners, OT prefers depth-similar ones; this does not move mass to a wrong
  population.
- The cost matters more than the solver: C2 (log1p-PCA) roughly doubles the depth correlation of every solver.
  Sparse solvers (exact, partial) show the most depth association; entropic ones the least.
- Batch levels (L3-L5) lower the depth correlation on C1 (about 0.02-0.06) and keep correct mass at 0.62-0.73
  (entropic) / 0.87-0.96 (sparse).

**Unmatched populations.** Entropic UOT's mass loss is not specific to the unmatched population at ε 0.1, λ 1:
transported share of the lost source population 0.606 vs 0.682 for shared cells (C1). Partial OT is far more specific
(0.261 vs 0.935, C1). ConfidenceOT's coupling equals its UOT backbone; its specificity is in the gate: C1 rejects 0.55
of lost-source cells and 0.30 of emerged-target cells while rejecting 0.01-0.08 of shared cells. On C2 the gate
rejects almost nothing (unmatched 0.001-0.009). At N = 200 the gate is weaker than at N = 5,000 (L1 lost-source 0.27).
Gate rejections among shared cells lean to shallow cells (Spearman rejection vs log nCount -0.12 to -0.19), but they
are 1-2% of shared cells.

## 2 Depth-identity conflict (S2): the key figure

Constructed from L0_matched all_shared by binomial thinning (PLAN.md S2). Read for Group1/2 source cells.

**Conflict arm** (correct target groups thinned, wrong groups keep source-like depth), share of the transported mass:

| f | Cost | Exact | Entropic bal. | UOT | Partial | ConfidenceOT | Independent |
|---|---|---|---|---|---|---|---|
| 1 | C1 | 0.904 | 0.777 | 0.789 | 0.967 | 0.788 | 0.204 |
| 0.1 | C1 | 0.892 | 0.743 | 0.755 | 0.955 | 0.744 | 0.204 |
| 1 | C2 | 0.924 | 0.796 | 0.805 | 0.961 | 0.798 | 0.204 |
| 0.5 | C2 | 0.905 | 0.796 | 0.663 | 0.842 | 0.662 | 0.204 |
| 0.25 | C2 | 0.912 | 0.866 | 0.350 | 0.854 | 0.350 | 0.204 |
| 0.1 | C2 | 0.886 | 0.860 | **0.081** | 0.873 | **0.081** | 0.204 |

At C2 f = 0.1, UOT / ConfidenceOT send 0.918 of these cells' transported mass to the wrong, depth-similar groups
(Group3-5) while still transporting 0.485 of their mass. This is the one setting where technical similarity overrides
biology: an unbalanced coupling on a depth-sensitive cost. Two things prevent it:
- **a depth-robust cost**: on C1 (rank, cosine) the correct share stays 0.74-0.79 at every f; the small loss goes to the
  other thinned group (0.148 at f 0.1), i.e. thinning blurs the two thinned populations, not a depth lure;
- **balanced marginals**: exact and entropic balanced OT stay at 0.86-0.91 on C2, because the thinned target cells must
  receive mass.

The ConfidenceOT gate does not flag this failure on C2 (rejected shared target 0.002 at f 0.1); the coupling is wrong
while the gate stays open.

**Aligned arm** (Group1/2 thinned on both sides, so depth tracks biology, η² 0.97 at f 0.1): coupling-depth
correlation rises to 0.73-0.90 while the correct-group share stays 0.61-0.88. This is depth-associated matching that is
correct, and it is why coupling-depth correlation alone is not an error measure (`S2_association_vs_harm`).

**Global arm** (whole target thinned, same populations): C1 keeps correct mass 0.73-0.75 down to f 0.1; C2 UOT /
ConfidenceOT fall to 0.48. On C1 the gate starts rejecting shared target cells at f 0.1 (0.122 overall, 0.35 of the
shallowest quartile): a depth-driven false rejection once depth differs 10-fold.

## Answer for Dataset A

- In Splatter, OT couplings follow population identity; depth explains a small, mostly within-population share of
  the matching (correlation 0.05-0.22 depending on cost and solver).
- Depth dominates only when an unbalanced solver runs on a depth-sensitive cost and depth conflicts with identity
  (C2, f ≤ 0.25). Rank-cosine C1 and balanced marginals each remove the effect; the gate does not detect it.
- Depth association without harm (aligned arm) is common and must not be called an error.

Figures: `S2_conflict_C2_correct_share`, `S2_conflict_C1_correct_share`, `S2_conflict_C2_wrong_similar_depth_share`,
`S2_conflict_C2_transported_fraction`, `S2_aligned_*`, `S2_global_*`, `S2_association_vs_harm`,
`S1_allshared_{C1,C2}_{depth_corr_count,depth_corr_count_within,correct_mass}`,
`S1_{population_lost,population_emerged}_*_transported_share_unmatched`, `S1_gate_*_rejected_fraction`, `legend_solvers`.
