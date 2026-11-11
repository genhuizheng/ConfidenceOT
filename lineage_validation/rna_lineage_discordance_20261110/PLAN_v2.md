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

## Amendment 2026-11-11 (user), fixed before any of the analyses below was computed

### A1. Lineage filtering
- **Primary analysis:** resolved lineages only, by the step-1 rule fixed in PLAN.md section 1:
  - ≥ 50 cells;
  - median informative events per cell ≥ 3;
  - distinct event sets / cells ≥ 0.1;
  - IQR of s ≥ 0.05.
- **Sensitivity:** every lineage, each flagged with its informative events, missingness (share of missing intBC
  states), clone size, tree resolution (share of leaves in polytomies with > 50 children) and median depth.
- The thresholds are not changed after results are seen.

### A2. KP state–lineage association (focus)

**Cells.** Per mouse, every OT cell of the mouse's units (both sides), with:
- the authors' state (`cell_type`);
- the site (tumour);
- depth;
- CRISPR events.

**Lineage levels.** Both are reported. Clades nested in one family are never treated as independent replicates.
- **Family:** the authors' MetFamily, a separate tumour-founding event, so families are independent.
- **Clade:** within a family, average-linkage (UPGMA) clusters on 1 − s (raw edits; non-evaluable pairs get the
  family median for the clustering only), cut at distance 0.5. Clusters of < 20 cells are pooled as "minor" and are
  not counted as clades.

**States analysed.** States with ≥ 50 cells in the mouse. Every state is listed.

**Metrics** (mature, no new score):
- per state:
  - the number of families and of clades with ≥ 5 cells of the state;
  - Shannon entropy and the Hill-number diversity (exp entropy) of its clade composition;
  - the largest clade share (concentration);
- per clade: Shannon entropy and Hill diversity of its state composition;
- per mouse: Cramér's V of clade × state, with standardised residuals per cell of the table for the heatmap.

**Null.** State labels permuted within site, within mouse, 999 draws, seed 20261111. This keeps clade size, state
abundance per site and site composition. Each metric is reported as observed, null mean, z and empirical
two-sided p.

**Strata.** Per mouse; KP 3724 on its own, and every pooled summary also without 3724. Pooled summaries are the mean
over mice.

### A3. Primary vs metastasis
- **Unit:** within one mouse.
- **Cells:** primary cells and metastatic cells of states present in both (≥ 20 cells each).
- **Metric:** Cramér's V of clade × state computed separately for primary and for metastatic cells, after
  subsampling both to the same number of cells (the smaller; 200 draws, seed 20261111), against the within-site
  permutation null of A2.
- **Beside it:** the view-C divergence / convergence excess (observed − null) per side and state, from the
  comparison tables.
- **Controls:** the comparison is repeated within strata of informative events per cell (tertiles) and depth
  (tertiles).
- **Reporting:** every mouse separately, then the mean over mice, with and without 3724.

### A4. RNA-OT vs CRISPR-OT on a common support
- Gate disagreement is reported apart, as before.
- **Conditional coupling disagreement:**
  - both couplings restricted to sources retained by both and targets retained by both;
  - every column renormalised to 1 on that common block;
  - pairs restricted to evaluable cell pairs (s defined);
  - then the per-target total-variation distance and top-15 overlap.

### A5. GSEA
- The existing DEG / GSEA (deg_gsea_v1, deg_gsea_v1_ds) is kept unchanged.
- Programs are read only where they are consistent across mice in both the raw and the depth-thinned versions.

### A6. Figures
1. Lineage × state association heatmaps (standardised residuals, one per mouse).
2. State-specific lineage concentration / diversity against the null.
3. Primary vs metastasis Cramér's V per mouse.
4. RNA-OT / CRISPR-OT gate and conditional disagreement.
5. A representative lineage: RNA embedding coloured by state, clade and site, beside its informative-edit matrix.

## Post hoc 2026-11-11 (after the A2-A4 results were seen; no metric, threshold or cell set changed)
- **Diagnostic added to A2 / A3:** `permutable_share`, the share of cells in sites holding at least 2 states and at
  least 2 clades. Only there can a within-site permutation change the clade x state table. A null equal to the
  observed value with share 0 (3457 metastases, where each metastatic site holds one clade) means the null cannot
  move. It is not evidence of no coupling. Reported beside every V.
- **A1 tree resolution:** computed for Quinn (authors' hybrid trees, 116 of 148 lineages). NA for KP, because only
  the tree priors, not the trees, are in the local inputs.
- **70_ `--reuse-clades`:** re-reads the clades instead of recomputing them. The clade step is deterministic and
  draws no random numbers, so the outputs are unchanged; checked against the first run.

## Extension 2026-11-11 (user), fixed before any of it was computed

New, dated sensitivity / extension analyses. The frozen results (`state_lineage/`, A1-A4 above) are kept unchanged.
New outputs go to `state_lineage_ext/<cellset>__<annotation>/` and `kp_all_audit/`.

### E1. Harmonized state annotation
- **Candidates:** an authors' label `X (Met)` whose parent `X` exists verbatim as another authors' label. No other
  label is touched.
- **Data:** every eligible KP cell (E2), logCPM (each cell scaled to 1e4, log1p).
- **Checks, reported for every candidate:**
  1. **Markers:** scanpy Wilcoxon, each label vs all other cells, top 50 genes per label. Report the overlap of the
     candidate's top 50 with its parent's, against its overlap with every other label.
  2. **Marker expression:** Spearman correlation of the label-mean logCPM over the union of all labels' top-50
     genes. Report the candidate vs parent, against the candidate vs every other label.
  3. **RNA distance:** in PCA-30 of the 2,000 highest-variance genes (`rb_common.pca_coordinates`, seed 20261111),
     the cosine distance between label centroids. Report the candidate vs parent, against the candidate vs every
     other label. Also per mouse wherever both labels have ≥ 20 cells.
  4. **State composition:** mice, sites and the primary / metastasis split of each label.
- **Merge rule (fixed now):** the candidate is merged into its parent only if the parent is its closest label on
  check 2 **and** on check 3, pooled over mice. Otherwise it stays separate and the result is reported.
- Original and harmonized annotations are both analysed.

### E2. All eligible KP cells
- **Eligible:** a cell of the authors' processed object (`adata_processed.combined.h5ad`) that has all of:
  - `is_normal` False;
  - a non-missing `Cluster-Name`;
  - a tumour (`Tumor` set, not "Normal");
  - a `MetFamily`;
  - a record in the allele table, by the frozen rule (FINAL, plus unfiltered for cells FINAL lacks; `allele_source`
    flagged per cell).
- **Audit before any A2 / A3** (`kp_all_audit/`), per mouse:
  - expression cells and allele-table cells, and the ID overlap (exact, else `map_ids`);
  - normal vs malignant;
  - annotated vs missing state;
  - family coverage, informative events, missing intBC share;
  - depth (total counts, detected genes);
  - the share coming from the unfiltered table.
  The authors' QC is taken as given; no new QC threshold is applied.
- **Overlap with the OT units:** reported as the share of OT-unit cells inside the eligible set, and the share of
  eligible cells that were in an OT unit.

### E3. A2 / A3 rerun
**Methods unchanged:**
- clades: UPGMA on 1 − s within family, cut 0.5, minor < 20;
- states ≥ 50 cells per mouse;
- metrics and the within-site permutation null: 999 draws, seed 20261111;
- primary vs metastasis at equal n: 200 draws;
- tertile controls;
- `permutable_share`.

**Resolution:** the frozen rule (≥ 50 cells, median informative events ≥ 3, distinct event sets / cells ≥ 0.1,
IQR s ≥ 0.05), recomputed per family on the cell set analysed. Primary scope = resolved families; sensitivity = all.

**Runs:** cell set {OT units, all eligible} × annotation {original, harmonized}. The OT-units × original run must
reproduce the frozen `state_lineage/` exactly (regression check).

**Added outputs** (defined now):
- **Site identifiability:** per site and mouse, the non-minor clades with ≥ 5 cells in the site and the analysed
  states with ≥ 5 cells in the site. A site with fewer than 2 of either is "not identifiable". Primary vs metastasis
  reports the share of each side's cells in identifiable sites. Not identifiable is never read as no association.
- **Per state, primary vs metastasis:** for each state with ≥ 20 cells on both sides of a mouse, both sides are
  subsampled to the same n (200 draws). Each draw computes on each side:
  - the Hill diversity of its clade composition;
  - its largest clade share;
  - the same two after one within-site permutation of the states among that side's cells.
  Reported: observed, null and excess per side, plus the share of the state's cells in identifiable sites.

**Reporting:** per mouse and per state; pooled means over mice with and without 3724; the four runs compared side by
side.

### E4. Unchanged
- A4 stays on the existing RNA-OT / CRISPR-OT cells (49 units, no new fits).
- DEG / GSEA is kept and not extended.
- No new solver, no RNA grid rerun, no parameter change after results.
