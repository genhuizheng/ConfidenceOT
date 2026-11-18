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

## 3 S2b: the conflict on L1-L5, A4, A5 (post-hoc extension 2026-11-18, PLAN.md)

672 fits (7 levels x 3 arms x 4 f x 8 replicates; x 2 costs x 6 solvers), 0 failures, balanced marginal error < 3e-6.
Read for Group1/2 source cells. The fair balanced-vs-unbalanced contrast is entropic balanced OT (Sinkhorn) vs entropic
UOT at the same ε = 0.1. Table: `splatter_s2b/tables/S2_S2b_group12_mass_shares.tsv`; figures `splatter_s2b/figures/`.

Correct-group share of the transported Group1/2 mass at f = 0.1 (f = 1 in brackets: the two are within 0.01 at every
level and cost):

| Level | C1 balanced | C1 UOT | C2 balanced | C2 UOT | C2 UOT share on wrong, depth-similar groups |
|---|---|---|---|---|---|
| L0 | 0.74 (0.78) | 0.75 (0.79) | 0.86 (0.80) | 0.08 (0.81) | 0.92 |
| L1 | 0.58 (0.69) | 0.52 (0.70) | 0.53 (0.67) | 0.01 (0.67) | 0.99 |
| L2 | 0.61 (0.73) | 0.54 (0.74) | 0.52 (0.56) | 0.04 (0.56) | 0.96 |
| L3 | 0.59 (0.72) | 0.61 (0.73) | 0.52 (0.53) | 0.05 (0.53) | 0.94 |
| L4 | 0.60 (0.70) | 0.62 (0.71) | 0.51 (0.52) | 0.05 (0.52) | 0.94 |
| L5 | 0.66 (0.84) | 0.65 (0.84) | 0.48 (0.68) | 0.28 (0.67) | 0.59 |
| A4 | 0.61 (0.72) | 0.54 (0.72) | 0.50 (0.55) | 0.04 (0.55) | 0.96 |
| A5 | 0.74 (0.87) | 0.66 (0.87) | 0.50 (0.73) | 0.29 (0.72) | 0.59 |

- Without a within-side depth conflict (f = 1) balanced and unbalanced entropic OT give the same matching at every level.
- On C1 neither collapses under the conflict: UOT is 0.06-0.08 below balanced at L1, L2, A4, A5 and 0.02 above at L3, L4.
- On C2 UOT collapses at every level (0.01-0.29) and sends 0.59-0.99 of the mass to wrong, depth-similar groups;
  balanced OT on C2 also degrades from L1 on (0.48-0.53, 0.30-0.42 on depth-similar groups) but does not collapse.
- **Aligned arm** (depth tracks biology on both sides): C2 UOT still falls to 0.18-0.22 at L1-L4 (balanced 0.63-0.66),
  so C2 + UOT fails whenever cells within a side differ strongly in depth, even when depth is biologically
  informative. C1 UOT stays at 0.68-0.85.
- **Global arm** (whole target thinned to about 1/20 of the source depth at L1-L5): balanced and UOT degrade equally
  (C1 0.34-0.62, C2 0.29-0.49); exact OT and partial OT keep 0.68-0.94. ConfidenceOT on C1 rejects 0.97-1.00 of the
  source (asserted mass ≈ 0): with every population shared, a 20-fold depth gap between sides closes the gate, a
  depth-driven reject-all. On C2 it rejects ≤ 0.02 and asserts mass with a false share of 0.69-0.71.

**ConfidenceOT readout.** Its full coupling contains uniform mass on rejected rows, so its correct-group share must be
read on the retained x retained block (false_asserted_share) together with the rejected fractions, as in the table
above for the global arm; the per-solver tables keep the full-coupling value for comparability and flag this here.
