# LTL331 state-level readout of ConfidenceOT (Non-Harmony branch; author-Harmony branch audited, fits pending)

Written 2026-11-01, coupling section 2026-11-02. Tables: `results/ltl331_nepc/state_biology_20261101/nonharmony_v4/`
(10_state_biology.py; v3 for the same retention and signature tables, couplings of two labels only) and
`signatures_v1/` (11_signature_scores.py), from the existing gates of all 64 fitted configurations
(`PDX_CELL_GATES.tsv.gz`, 7 adjacent pairs, M4-E and M4-R). Nothing was refitted, and no configuration was excluded for
calibration, convergence or depth; those are columns. A gate says which cells have, or lack, a counterpart in the
expression space of the other time point. A coupling links expression states. Neither traces a cell to its ancestor or
its descendant.

## 1. Annotations checked against the paper

Sar et al., Genome Med 2026 (GSE297328), Additional files 1 and 2.
- **Cells and clusters:** our `seurat_clusters` is the authors' 18-cluster clustering of the same 51,726 cells. The
  authors kept doublets: their DoubletFinder removal (to 49,568 cells) was a sensitivity analysis only.
- **Composition (PAPER_CLAIMS_CHECK.tsv):** every statement in the paper reproduces.
  - preCx: cluster 11 = 62.9% (paper 63%), cluster 15 = 15.4% (paper 15%).
  - The dominant cluster per time point is 11 / 2 / 2 / 0 / 0 / 4 / 10 / 1. At wk2 / wk4 / wk22R it holds 48–49%,
    against the paper's "50% or more".
  - preCx has no NEPC-cluster cells, and wk2 has no AR-high cells.
  - Cluster 17 first appears at 8 weeks and peaks at 16 weeks (1.4%).
- **State map used:**

  | State | Clusters |
  |---|---|
  | AR-high PRAD | 11, 15 |
  | AR-low PRAD | 0, 2, 4, 5, 8, 12, 16 |
  | pre-EMT | 6 |
  | intermediate EMT-like | 17 |
  | NEPC ASCL1+ | 10 |
  | early NEPC ASCL1− | 13 |
  | terminal NEPC ASCL1− | 1, 7, 9 (the published CellRank terminal macrostates) |
  | NEPC other | 3, 14 |

- **The authors' UCell programs (Table S2) reproduce the annotation (CLUSTER_SIGNATURES.tsv):**
  - Cluster 17 has PRAD/AR 0.002 and NE 0.004, both at baseline as the paper reports. Its EMT is the highest
    (0.30; Fig. 2D panel 0.47), and VIM is detected in 95% of its cells.
  - AR-high: cluster 11 has PRAD/AR 0.47, cluster 15 0.29.
  - ASCL1 is detected in 66% of cluster 10 cells vs 4–14% in clusters 1, 7, 9 and 13.
  - Cluster 6 has the highest ciliary program (0.46), and clusters 3 and 8 the highest proliferation.
  - The proliferation program lacks FAM64A / HN1 (PIMREG / JPT1 in these matrices), so 53 of its 55 genes are used.
  - The authors define no REST program, and none was added. SRRM4 is detected in 33–72% of NEPC-cluster cells vs < 1%
    of PRAD cells, while REST mRNA is detected at a low rate everywhere.
- **`coarse_celltype` is not used as a state.** It comes from the authors' three-cohort Harmony object, and its Basal
  / Other are not the paper's intermediate: the paper reports KRT5 / KRT14 absent.

## 2. State retention (M4-E, all 64 configurations; ROBUSTNESS.tsv)

Each finding is the median over configurations; "configurations" is the share in which the state is retained more
(or less) than the rest of its side.

**8→12 and 12→16 weeks (the paper's transition window)**
- At 12 weeks (source of 12→16), retention against the rest of the side:
  - intermediate EMT-like (66 cells): +0.19, 97% of configurations;
  - NEPC ASCL1+ (96 cells): +0.15, 95%;
  - pre-EMT: +0.12, 97%;
  - AR-low PRAD (11,186 cells): −0.13 (retained 0.28).
- At 16 weeks (target), intermediate cells are retained at 0.96 (100% of configurations) and ASCL1+ at 0.92 (97%).

**16→20 weeks**
- At 16 weeks (source): ASCL1+ cells are retained at 0.85, +0.84 above the rest in 100% of configurations; AR-low PRAD
  at 0.01 and pre-EMT at 0.00.
- 20 weeks (R) is 94% NEPC, so this follows the target composition.
- At 20 weeks (target): ASCL1+ is retained at 0.74 (above the rest in 98%); terminal ASCL1− at 0.54 (below the rest in
  100%).

**20→22 weeks**
- At 20 weeks (source):
  - terminal ASCL1−: 0.88, +0.47 above the rest in 100% of configurations;
  - NEPC other: 0.77, above the rest in 100%;
  - ASCL1+ (817 cells): 0.35, below the rest in 100%.
- This agrees with the paper's description of a transition from the ASCL1+ dominant state (cluster 10, 20 wk) to the
  ASCL1− dominant state (cluster 1, 22 wk), read as expression correspondence.

**Robustness checks**
- Standardising to pre-equalisation depth quintiles or nFeature quintiles hardly changes any of these fractions: median
  absolute change 0.001, at most 0.28 (STATE_RETENTION.tsv).
- M4-R (cycling in every fit; diagnostic): at 20→22 it retains nearly all source cells (0.80–0.96 in every state), so
  its state contrast is weak. It reverses some of M4-E's directions at 16→20, e.g. at 20 weeks it retains AR-low PRAD
  at 0.93 and the NEPC states at 0.24–0.37. M4-R is not used as the main readout.

## 3. Programs of retained vs rejected cells (SIGNATURE_ROBUSTNESS.tsv)

Within each state, retained minus rejected mean UCell score, original counts. Each finding gives the median over
configurations, then the same difference within depth quintiles and within nFeature quintiles, and the share of
configurations with that sign.

| Pair, side | Finding | Raw | Depth strata | nFeature strata | Configurations |
|---|---|---|---|---|---|
| 0→2, preCx | rejected cells carry more PRAD/AR program (all cells) | −0.150 | −0.103 | −0.099 | rejected higher in 97% |
| 0→2, preCx | same, within AR-high PRAD | −0.151 | −0.081 | | |
| 12→16, 12 wk | intermediate: retained cells carry more EMT | +0.119 | +0.080 | +0.103 | 100% |
| 12→16, 12 wk | ASCL1+: retained cells carry more NE | +0.065 | +0.044 | +0.050 | 100% |
| 12→16, 12 wk | PRAD/AR program is lower in retained cells, all states | | | | 98–100% |
| 16→20, 16 wk | all cells: NE higher in retained | +0.089 | +0.107 | | 100% |
| 16→20, 16 wk | all cells: PRAD/AR lower in retained | −0.072 | | | 100% |
| 16→20, 20 wk | within ASCL1+: NE higher in retained | +0.050 | +0.055 | | 100% |
| 16→20, 20 wk | within early ASCL1−: EMT lower in retained | −0.044 | | | rejected higher in 93% |
| 20→22, 20 wk | within ASCL1+: NE lower in retained | −0.056 | −0.054 | | rejected higher in 100% |

- **0→2 weeks:** at preCx, half of the PRAD/AR difference goes with depth (the stratified effect is about −0.08 to
  −0.10).
- **20→22 weeks:** the canonical NE program contains ASCL1, so this follows the ASCL1+ → ASCL1− shift of the dominant
  NEPC state.

These are associations between a gate and an expression program within one tumour. Cells are not replicates; the
configuration share is a robustness count, not a p-value.

## 4. State-to-state coupling (all 64 configurations; nonharmony_v4/STATE_COUPLING.tsv)

- **Source of the couplings:** `PDX_STATE_COUPLING.tsv.gz`, TACC job 1117157. All 448 fits (64 configurations × 7
  pairs) were read, and none was missing.
- **Enrichment** = mass × total / (row mass × column mass), so 1 is what the two state marginals alone give. Each
  finding is for M4-E, in the retained block: median enrichment over configurations (interquartile range), the share
  of configurations with enrichment > 1, and P(target state | source state).
- **Cell counts:** a state that is absent from the retained block in a configuration has no ratio there, so some rows
  cover fewer than 64 configurations. States with few cells give wide ranges; read them with their counts.
- **Upper bound:** when one target state dominates, as terminal ASCL1− does at 22 weeks, enrichment cannot exceed
  1 / its share. A small ratio there is not a weak correspondence.

| Pair | Source → target | Median (IQR) | > 1 | P(target given source) | Configurations |
|---|---|---|---|---|---|
| 8→12 | intermediate → intermediate | 23.6 (16.7–38.1) | 100% | 0.12 | 52 |
| 8→12 | ASCL1+ → intermediate | 7.6 (5.1–10.7) | 94% | 0.04 | 62 |
| 12→16 | ASCL1+ → ASCL1+ | 13.2 (10.7–23.8) | 100% | 0.60 | 64 |
| 12→16 | intermediate → intermediate | 10.4 (7.9–15.4) | 100% | 0.24 | 64 |
| 12→16 | ASCL1+ → intermediate | 1.50 (1.24–1.77) | 83% | 0.04 | 64 |
| 12→16 | intermediate → ASCL1+ | 1.05 (0.91–1.38) | 56% | 0.05 | 64 |
| 16→20 | intermediate → early ASCL1− | 3.26 (2.45–4.22) | 100% | 0.17 | 63 |
| 16→20 | intermediate → NEPC other | 2.29 (1.36–3.13) | 95% | 0.13 | 63 |
| 16→20 | intermediate → terminal ASCL1− | 1.15 (0.97–1.38) | 74% | 0.14 | 62 |
| 16→20 | ASCL1+ → ASCL1+ | 1.13 (1.02–1.32) | 98% | 0.81 | 64 |
| 16→20 | ASCL1+ → terminal ASCL1− | 0.89 (0.75–0.99) | 13% | 0.13 | 63 |
| 20→22 | early ASCL1− → early ASCL1− | 9.10 (6.78–10.45) | 100% | 0.32 | 64 |
| 20→22 | intermediate → early ASCL1− | 5.14 (3.76–7.18) | 100% | 0.20 | 54 |
| 20→22 | terminal ASCL1− → terminal ASCL1− | 1.20 (1.19–1.22) | 100% | 0.86 | 64 |
| 20→22 | ASCL1+ → terminal ASCL1− | 1.06 (1.05–1.07) | 100% | 0.74 | 64 |
| 20→22 | ASCL1+ → early ASCL1− | 0.75 (0.71–0.84) | 0% | 0.03 | 64 |

- **Same state on both sides:** every state present on both sides is enriched in 89–100% of configurations at every
  pair. The pre-EMT diagonal ranges from 2.8× (12→16) to 15.8× (2→4).
- **Leaving a state:**
  - 16→20: the intermediate EMT-like cells of 16 weeks correspond above independence to early ASCL1− at 20 weeks
    (100% of configurations). Their correspondence to terminal ASCL1− is weaker (74%).
  - 20→22: the ASCL1+ cells of 20 weeks sit mostly in terminal ASCL1− at 22 weeks (P 0.74, enrichment 1.06 in every
    configuration, near the bound set by terminal ASCL1−'s share). Their correspondence to early ASCL1− is below
    independence in every configuration.
- **Frozen labels:** the two frozen labels of v3 sit inside the full range over configurations, though not always
  inside the interquartile range. ranknm256_noscale_ds_cos is below the median and logcpm above it. For example, at
  12→16 intermediate → intermediate is 6.1× in ranknm256_noscale_ds_cos and 31.2× in logcpm, against a range of
  3.0–218 over configurations. The size of an enrichment therefore depends on the preprocessing; its direction (above
  or below 1) is the robust part.
- **What this is not:** these are correspondences between expression states at two time points, not transitions of
  cells. No transition is required to carry mass.

## 5. Author-provided Harmony branch (pending)

- **What the authors' Harmony is** (Additional files 1 §6.2 and 2 Table S9): a joint integration of LTL331 with the Gao
  and Li tumour-epithelium cohorts, by cohort (theta 2, max_iter 10, human one-to-one orthologs, Leiden res 0.55, 27
  clusters). LTL331's time points are not corrected against each other. LTL331's dispersion is almost unchanged by it
  (22.8 → 22.1; Table S8).
- **How it enters OT:** the coordinates as published, with no HVG selection, PCA or scaling; cosine and sqeuc cost
  scaled by the median pair cost; calibration recomputed; the same cells, solver and metrics as the 64 configurations.
- Its genes (ortholog set, three-cohort HVGs), dimensions and joint PCA differ from every Non-Harmony configuration.
  A difference between the branches is therefore not the Harmony correction alone.
- **Audit of the authors' object** (`GSE297328_ltl331_harmony_integrated.h5ad`, 09_author_harmony_audit.py, TACC job
  1115473, 2026-11-01):
  - **Object:** 73,934 cells (LTL331 51,726; Gao 12,203; Li 10,005). X has the 3,000 vst HVGs of the joint object;
    raw has 29,622 genes.
  - **Embedding:** `obsm.X_harmony_dataset`, 30 dimensions, with loadings `varm.HARMONY_DATASET` (3,000 × 30). The
    integration variable is `dataset` (cohort), as the paper states, not sample or time point. The object also holds
    the uncorrected joint `X_pca` (50 dimensions) and two UMAPs.
  - **Alignment:** all 51,726 LTL331 rows match our cells one to one by `cellid`, with no duplicates and none
    missing either way. Their `ltl331_cluster_original` equals our `seurat_clusters` in every cell, and
    `coarse_celltype`, nCount_RNA, nFeature_RNA and percent.mt are equal in every cell. Obs names carry a prefix, so
    matching is by `cellid`.
  - **Cluster count:** the object's own `seurat_clusters` (29 levels) is the joint clustering at resolution 0.8; its
    resolution 0.55 column has 25 levels, against the 27 clusters of the paper. Neither is used here: states come
    from `ltl331_cluster_original`.
  - **Difference from Non-Harmony:** 30 Harmony dimensions from a 3,000-HVG, three-cohort PCA, against every
    Non-Harmony configuration's own genes and dimensions; the same 51,726 cells and the same frozen equalised
    manifest.
