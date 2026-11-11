# RNA vs CRISPR lineage relationships: three views (exploratory plan v2)

Status: **FIXED 2026-11-10, before any CRISPR-OT fit, RNA similarity or comparison was computed.** It replaces
PLAN.md (v1 and its amendment), which stays as the record. A later change is a dated section at the end, marked post
hoc. All QC and negative results are kept.

**Goal (user, 2026-11-10).** Compare RNA transcriptional relationships with CRISPR genetic-lineage relationships,
where they agree and where they do not, and whether the differences reflect phenotypic plasticity, transcriptional
convergence or metastatic adaptation.
- The aim is not to decide which OT is correct.
- No fused RNA + CRISPR OT.
- No re-optimisation of the RNA preprocessing.

**Wording.** Observed facts, model inferences and biological interpretations are kept apart in every report.
"Lineage-close" means close CRISPR relatives in a same-time sample, not ancestry or seeding direction.

## 1. The three views (kept separate)

**A. RNA-OT** (reused, not refitted). The v3 ConfidenceOT results: raw counts, frozen calibration v3, M4-E and M4-R.
- Couplings and gates of two configurations:
  - Splatter L5 best, `ranknm256-midrank|none|scale|cos|pca50`;
  - logCPM + PCA, `logcpm|none|noscale|sqeuc|pca30`.
- RNA similarity between cells: the same configuration's joint coordinates, regenerated from raw counts with
  `realdata_benchmark/v3_grid/v3_common.representations` (encoding, PCA, seed and `mnncovlocal` exactly as in the
  fit). Similarity is cosine for `cos` and −squared Euclidean for `sqeuc`.

**B. CRISPR-OT** (new fits, same solver). The CRISPR distance becomes the OT cost, so the solver is unchanged.
- **Distance:** D_ij = 1 − s_ij, where s_ij is the prior-weighted event Jaccard of MATRIX_METRICS.md:
  - missing excluded;
  - shared uncut excluded;
  - indel priors (homoplasy);
  - one event per intBC and string (cut-site dependence);
  - indel weights −log q differ by intBC and string, which carries the information difference between intBCs.
- **Pairs without a comparable intBC:**
  - Quinn cells of **different LineageGroups**: D = 1. The groups are defined by distinct lentiviral integrations,
    which is an observed genetic difference, not a missing value.
  - Any other pair without a comparable intBC: the unit's median evaluable D. The share of such pairs is reported per
    unit.
- **Cost:** C = D / median(D) over random cross-side pairs (distance 0 excluded), as `median_pair_scale` does for RNA.
- **Calibration:**
  - within-side split null built from the same D: two halves of one side, up to 2,000 cells per side;
  - 5 calibration + 5 validation replicates, grid 5, acceptance minimum 0.99;
  - **no MNN / `mnncovlocal`**: that correction is RNA-specific and is not validated for this distance.
- **Fit:**
  - `ConfidenceOT(backbone="uot")`, M4-E (exact) and M4-R (reversible);
  - ε 0.1, λ_a = λ_b = 1, tolerance 1e-4, rejection budgets 0.95 (not enforced), CPU;
  - the arguments of the v3 / `02_run_pair.py` fits.
- **Saved:** gates (retained, margin), couplings (float32), calibration payload and validity flags, cost statistics.
- **Units:** the 49 bundle units (9 L2/L3 cases; 40 KP same-family L1 pairs), the same cells as A.

**C. Raw CRISPR matrix** (no OT). The allele-table states, shown on 2026-11-10 to equal the authors' integer character
matrices cell by cell (`02_matrix_concordance.py`: 3,849 columns, 100% matched).
- **Cell-pair similarity:** s_ij.
- **Edit-defined groups:** intBC-allele classes and UPGMA on 1 − s (bundles; MATRIX_METRICS.md). Never k-means or
  Euclidean on integer codes.
- **Per-cell coverage:** observed intBCs, informative events, events per cell.
- **Lineage resolution:** step 1 (`01_lineage_resolution.py`); every lineage is flagged and none is dropped.
- **Cassiopeia trees:** auxiliary only (tree distance where available), never ground truth.

## 2. Audit of B before its results are read

Per unit:
- the share of non-evaluable pairs and how each was filled (median vs different clone);
- the distribution of D (share of zeros, ties);
- calibration: rejection cost, `m4e_inference_valid`, `feasible_cost_found`, M4-R clean;
- convergence;
- retained fractions on each side.

A unit whose calibration is invalid is kept and flagged.

## 3. Comparisons (same cells, mouse and sites)

All comparisons are per unit, then per mouse; KP and Quinn are separate.

1. **RNA-OT vs CRISPR-OT:**
   - gate agreement (retained / rejected cross-table, Cohen's κ, per side);
   - coupling agreement: per retained target, the total-variation distance between the two couplings' source
     distributions, and the overlap of the top-15 sources;
   - mass on the same source groups (edit groups, sites).
2. **Explained by the raw matrix?** For targets where the two couplings disagree (top quartile of total-variation
   distance), the s of their RNA-OT top sources against their CRISPR-OT top sources, and the edit coverage of those
   targets.
3. **Lineage-close / RNA-far** (potential divergence): pairs in the unit's top 25% of s and bottom 25% of RNA
   similarity. Per cell, the share of its lineage-nearest 15 cells that are RNA-far.
4. **RNA-close / lineage-far** (potential convergence): pairs in the top 25% of RNA similarity and bottom 25% of s.
   Per cell, the share of its RNA-nearest 15 cells that are lineage-far.
5. **Concentration:** discordance and OT disagreement by clone / edit group, site and RNA state:
   - KP: the authors' `Cluster-Name`;
   - Quinn: Leiden clusters (resolution 1.0) on the RNA coordinates.

## 4. Controls

- **Covariates per cell:** depth (pre-equalisation total counts), informative events / observed intBCs / TS_UMI,
  clone size, site, lineage resolution flag.
- **Partial correlations,** and discordance flags recomputed on covariate residuals as a sensitivity.
- **Null:** within-site permutation of RNA coordinates, 99 draws, seed 20261110.
- **KP 3724:** analysed on its own, and every KP summary is given without 3724.

## 5. Biology (only where the evidence supports it)

For discordant populations with ≥ 20 cells in one lineage and site, against concordant cells of the same lineage and
site:
- Wilcoxon on raw counts (normalize_total, log1p) and on depth-equalised counts as a check;
- hallmark prerank GSEA, all 50 sets, both directions;
- read effects, not p-values (one tumour per site).

No program (EMT, proliferation, hypoxia, stress) is assumed in advance; whatever is enriched is reported.

## 6. Figures (separate panels)

1. RNA-OT vs CRISPR-OT correspondence: gate cross-tables, coupling total-variation distributions, top-k overlap.
2. Raw editing similarity: s distributions, edit-group structure, coverage.
3. Discordance maps: RNA similarity vs s per lineage (hexbin), with the two discordant quadrants marked.
4. Clone / state / site association heatmaps (one quantity per heatmap).
5. Representative DEG / GSEA dot panels.

## 7. Code and outputs

- Code: `lineage_validation/rna_lineage_discordance_20261110/`.
- Outputs: `results/rna_lineage_discordance_20261110/`.
- CRISPR-OT runs locally (49 units, one fit per unit per method).
