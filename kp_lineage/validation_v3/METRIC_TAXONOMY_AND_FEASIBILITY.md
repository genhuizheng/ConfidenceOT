# KP-Tracer validation_v3: metric taxonomy and feasibility

Companion to `LEVEL3_VALIDATION_DESIGN.md`; frozen with it. Every metric has one role. A metric not listed here is post hoc if it is ever computed.

**Scope.**
- Both labels (`ranknm256_noscale_ds_cos` primary, `raw` sensitivity).
- The 28 same-family pairs in 7 mice (primary set).
- Different-family shared-tree pairs meeting the design's rules are reported separately, never pooled and never tested.

**Aggregation.** Pair, then metastasis, then mouse. The mouse is the unit.

**Multiplicity.** The exploratory families below are each corrected by Benjamini-Hochberg within the family and label, and never used to restate the primary result.

## 1. Roles fixed in the design

| Role | Metric | Design section |
|---|---|---|
| Primary | ΔS_norm of M4-E, mouse-level one-sided Wilcoxon | 3, 8 |
| Key secondary | ΔR of M4-E, same test | 3, 8 |
| Baseline contrasts (Holm, 2) | level 1 ΔS_norm(M4-E) − ΔS_norm(EOT_C(a,b)); level 2 ΔS_norm(M4-E \| Rs) − ΔS_norm(balanced \| Rs) | 4, 8 |
| Descriptive baselines | kNN at levels 2 and 3; every full-cell comparison; bridge decomposition | 4, 9 |
| Sensitivity | raw arm; M4-R; NJ trees; matched-retained set; without low-information pairs; leave one mouse out (3724 named); ΔS_depth; expression-stratified null (within-cluster part); raw ΔS; ΔS / SD(L); P2 on raw edges | 5, 8 |
| Diagnostics (no test) | the lineage-resolution table (distinct L, modal share, tie probability, root share, polytomy shares, ΔS_max, R_rowmax, recorder coverage, Cassiopeia vs NJ Spearman, normalised Robinson-Foulds, NJ sign agreement); m; |Rs|/n; exp(H(a))/n; per-pair null p values | 5, 6 |

## 2. Exploratory, Level 2: the gate label against the tree (no coupling)

Unit of analysis: source cells of the primary set, label g = retained or rejected by M4-E, and the continuous score r = −signed_rejection_margin. These ask whether rejection is lineage-structured. That is the same question family as E1, so none of them is reported as confirming E1 (design section 10).

| Family | Metric | Null | Feasibility |
|---|---|---|---|
| L2-a nearest relative | m(i) = min over target cells of the tree path distance (contracted edges), per source cell; AUROC of rejection against m(i) | permutation of g among the pair's source cells, 1,000, seed 20261016 | feasible on the released Cassiopeia and NJ trees |
| L2-a, tree-free twin | m(i) under Cassiopeia's prior-weighted Hamming distance on the character matrix (−log indel priors; missing characters skipped) | as above | feasible: character matrices and priors are released; a custom implementation of the published rule |
| L2-b clade structure | Fitch-Hartigan parsimony score of g on the pair's source leaves, normalised by edges | as above | feasible with a custom implementation (Hartigan's multistate rule handles polytomies) |
| L2-b, continuous | phylogenetic Moran's I of r, weights 1/d(i,j) on source leaves | as above | feasible (custom; d = 0 pairs excluded) |
| L2-c group coupling | Evolutionary Coupling: mean tree distance between retained (and rejected) sources and targets, z against 2,000 shuffles of g | shuffles of g | feasible |

## 3. Exploratory, Level 3: the coupling against the tree

| Family | Metric | Rule fixed here | Feasibility |
|---|---|---|---|
| L3-a retrieval | per target cell j, rank sources by πᵢⱼ; relevant = sources at the largest L with j; MRR and AUPRC, each against the target-permutation baseline | computed only for target cells whose relevant set holds fewer than half of the pair's source cells, otherwise "no defensible split" and counted | feasible |
| L3-b tree-free enrichment | the P1 construction with relatedness = negative prior-weighted Hamming distance on alleles: Δ_allele = E_π[−h] − E_{a⊗b}[−h] | same nulls as P1 | feasible (character matrices); dense n × k, largest 14,775 × 90 |
| L3-c clade-level fate bias | for each top-level clade (children of the family tree's root holding ≥ 5% of the pair's source cells), π's share of mass sent from the clade vs the share of target cells in the clade; Pearson correlation and Jensen-Shannon divergence | clade cut fixed by the 5% rule | feasible; few clades per family, so mostly descriptive |
| L3-d descriptors | conditional entropy H(target given source) of π; Evolutionary Coupling of π-weighted groups | none | feasible; descriptors only, not lineage validation |

## 4. Not done this round

| Metric | Reason |
|---|---|
| Yang's mutation-count distance (Camin-Sokal ancestral states) | needs ancestral reconstruction; Cassiopeia availability in the analysis environment is to be checked. Not computed unless available, and then exploratory |
| intBC-block bootstrap trees | allele tables are released, but every replicate needs a Cassiopeia-Hybrid reconstruction |
| Fritz-Purvis D, Pagel's λ, Blomberg's K | need branch lengths; the trees have none |
| LineageOT descendant/ancestor errors, moslin top-ancestor accuracy | need ancestor-descendant ground truth; same-time sampling has none |
| Bayesian tree posteriors (TiDeTree) | computational cost at thousands of cells |
| Single-cell plasticity scores (scEffectivePlasticity) | strongly autocorrelated across cells; no independent test |
| Unweighted UniFrac | almost always maximal when every cell is its own leaf |
