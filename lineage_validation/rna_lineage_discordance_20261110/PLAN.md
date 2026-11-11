# RNA–lineage discordance analysis (KP-Tracer, Quinn 2021): exploratory plan

Status: **FIXED 2026-11-10, before any RNA similarity was computed.** Exploratory.
- Every definition below is set before looking at results.
- A later change is a dated section at the end, marked post hoc.
- Negative results and QC are reported as they are.

**Question (user, 2026-11-10).** In which cell populations do RNA transcriptional similarity and CRISPR lineage
ancestry agree or disagree, and is the disagreement biologically meaningful?

**Ruled out at this stage:**
- no new OT;
- no RNA + tree fused OT;
- no new distance or solver.

**Reused, unchanged:**
- the v3 RNA-only ConfidenceOT gates and M4-E couplings of two configurations: Splatter L5 best
  `ranknm256-midrank|none|scale|cos|pca50` and logCPM + PCA `logcpm|none|noscale|sqeuc|pca30`;
- the CRISPR unit bundles of `crispr_matrix_benchmark_20261108` (allele-table events, edit agreement s_ij,
  MATRIX_METRICS.md);
- the Cassiopeia trees;
- the authors' cell metadata.

## 0. Wording

The report keeps three levels apart:
- what was observed;
- possible biological interpretations;
- hypotheses not yet tested.

Two further rules:
- "Lineage-close" means close CRISPR relatives in a same-time sample, not ancestry.
- No direction (seeding) is claimed.

## 1. Lineages and their resolution (decided before any RNA quantity)

**Scope.** One lineage = one KP MetFamily or one Quinn LineageGroup, taken within one mouse (Quinn experiment). Its
cells are the union of the OT cells of every unit of that mouse in the bundles, on both sides.

**Resolution metrics** (per lineage):
- cells;
- informative events (non-founder);
- median informative events per cell;
- distinct event sets / cells;
- share of cell pairs with 0 < s < 1;
- interquartile range of s;
- for tree lineages, the share of leaves in polytomies with > 50 children.

**Resolved lineage.** All of the following, a fixed rule:
- ≥ 50 cells;
- median informative events per cell ≥ 3;
- distinct event sets / cells ≥ 0.1;
- IQR of s ≥ 0.05.

**Every lineage is reported**, with the reason it is or is not resolved. The analyses of sections 3-6 run on resolved
lineages only.

## 2. Similarities

**RNA (primary).** Computed per mouse / experiment on the cells of section 1:
- the depth-equalised counts (27_, the frozen ds files), so sequencing depth cannot drive it;
- `normalize_total` 1e4, `log1p`, 2,000 HVGs (scanpy `seurat`);
- PCA 30, centred, no scaling;
- r_ij = cosine similarity in PCA space.

This is scanpy's default workflow; nothing is tuned.

**Lineage (primary, observed edits).** s_ij from MATRIX_METRICS.md (prior-weighted event Jaccard; missing and shared
uncut excluded). Pairs that are not evaluable are left out, never set to 0.

**Lineage (secondary, tree).** d_ij = tree distance (level3_cell_lineage), where both cells are reliable leaves of one
family tree. It is labelled "tree-dependent".

## 3. Global agreement

**Mantel test** per resolved lineage:
- Spearman correlation of r_ij and s_ij over evaluable pairs;
- permutations: 999, of cell labels within site (stratified, so site composition is held fixed);
- seed 20261110.

The Spearman correlation is reported for three pair sets: all pairs, within-site pairs and cross-site pairs. The same
is done with the tree distance d_ij.

## 4. Cell-level discordance (k-nearest neighbours, k = 15 fixed)

Within each resolved lineage, for each cell j:
- LN(j): the 15 lineage nearest neighbours (largest s_ij among evaluable pairs, ties broken by a fixed-seed
  permutation);
- RN(j): the 15 RNA nearest neighbours (largest r_ij).

Scores (each relative to the same lineage's background):
- **plasticity score:** P(j) = mean r over LN(j) − mean r over all other cells of the lineage. Low P = lineage-close /
  RNA-far.
- **convergence score:** C(j) = mean s over RN(j) − mean s over all evaluable other cells. Low C = RNA-close /
  lineage-far.
- **neighbourhood agreement:** |LN(j) ∩ RN(j)| / 15, beside the two scores.

Each score is computed three times: all neighbours, within-site neighbours only, and cross-site neighbours only.

**Flags (exploratory):**
- "plastic": P in the bottom 10% of its lineage;
- "convergent": C in the bottom 10%;
- "concordant": both in the middle 40-60%.

The flags define the groups of section 5. They are never tested as discoveries.

## 5. Covariates and controls

- **Covariates per cell:** pre-equalisation total counts (depth), informative events and TS_UMI / NUM_INTBC /
  PercentUncut (KP), site, lineage size.
- **Score vs covariates:** OLS of each score on the covariates, within lineage. Partial correlations are reported, and
  the flags are recomputed on the residuals as a sensitivity.
- **Null for the scores:** within-site permutation of the RNA coordinates (99 draws). Discordant-cell rates beyond the
  null are reported per lineage.
- **3724:** every summary is given with and without mouse 3724, and per mouse. No pooled number is reported without
  its per-mouse breakdown.

## 6. Biology (exploratory)

**Cell states:**
- KP: the authors' `Cluster-Name` (Yang et al. 2022);
- Quinn: Leiden clusters (resolution 1.0, 15 neighbours) of section 2's PCA.

**Lineage groups:** intBC-allele / UPGMA groups from the bundles (Quinn clones and their groups).

**Lineage × state:** one heatmap per resolved lineage, showing the fraction of the group's cells in each state (one
quantity per heatmap).

**DEG / GSEA:** plastic vs concordant cells, and convergent vs concordant cells.
- Comparisons are made within the same lineage and site; a group needs ≥ 20 cells.
- Expression: equalised counts.
- Test: Wilcoxon; one tumour per site, so effects are read, not p-values.
- GSEA: hallmark prerank, all 50 sets, both directions.
- No program (EMT, proliferation, hypoxia, stress) is assumed.

## 7. OT relation (two configurations, M4-E)

**Gate:** retained vs rejected target cells, compared on P, C and the agreement score (AUROC). Per unit and per mouse.

**Coupling:** per retained target, the coupling mass on four kinds of source pairs:
- lineage-close / RNA-far: s in the top 25% and r in the bottom 25% of the unit's pairs;
- lineage-far / RNA-close;
- concordant-close;
- concordant-far.

Each share is compared with the independent coupling (the same marginals).

## 8. Figures (separate panels)

1. r vs s per resolved lineage (hexbin), with the Spearman correlation and the Mantel p;
2. distributions of P and C, observed against the permutation null;
3. lineage × state heatmaps;
4. DEG / GSEA dot panels for the lineages with the most discordant cells;
5. gate AUROC and coupling-mass ratios against discordance.

## 9. Code

`lineage_validation/rna_lineage_discordance_20261110/`. Outputs go to `results/rna_lineage_discordance_20261110/`.

## Amendment 2026-11-10 (user, before any RNA quantity was computed): RNA similarity as the OT sees it, from raw counts

**The change.** The primary RNA similarity becomes the one ConfidenceOT itself uses.
- Raw counts.
- Per unit, the exact joint coordinates of a v3 configuration, regenerated with `realdata_benchmark/v3_grid/
  v3_common.representations`: the same encoding, PCA, seed and `mnncovlocal` correction as the fit.
- r_ij = cosine similarity for `cos` configurations and −squared Euclidean distance for `sqeuc`, on those
  coordinates.
- Configurations:
  - Splatter L5 best, `ranknm256-midrank|none|scale|cos|pca50`;
  - logCPM + PCA, `logcpm|none|noscale|sqeuc|pca30`.

**Consequences:**
- The analysis unit becomes the OT unit (a primary → metastasis pair, or a pooled L2/L3 case), per lineage within the
  unit, because coordinates of different units are not comparable.
- Mantel tests and the kNN scores of sections 3-4 are computed per unit × lineage.
- Depth enters as a covariate (section 5).
- The section 2 depth-equalised similarity is kept as a sensitivity only.
- The resolution rule of section 1 is applied per lineage as before.

**Scope (user):** every lineage is analysed, resolved or not, with its resolution flag. The filter is decided after
the results are seen and is reported as a post hoc choice.
