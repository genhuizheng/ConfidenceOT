# Local lineage visualization: real CRISPR lineage beside the existing ConfidenceOT outputs

Status: **scripts written; KP-Tracer lineage figures checked on the authors' files; the OT, UMAP and Quinn parts wait for the TACC export.** Nothing here re-runs OT, re-infers a tree, classifies clones or computes a new validation metric. The frozen protocol (`../VALIDATION_PROTOCOL.md`, 508609f) is not touched.

## Pipeline

| Step | Where | Script | Output |
|---|---|---|---|
| 1. Export (read only) | TACC, one gg job | `submit_export.sh` → `tacc_export.py` | `<lineage_validation>/local_lineage_visualization_export/`, packed as `local_lineage_visualization_core.tar.gz` and, separately, `local_lineage_visualization_couplings.tar` |
| 2. Download + md5 check | local (Git Bash) | `download_from_tacc.sh [--couplings]` | `data/local_lineage_visualization_export/` |
| 3. Figures | local | `plot_lineage.py --data data/local_lineage_visualization_export` | `figures/`, `annotations/`, `figures_index.tsv` |

`data/`, `figures/` and `annotations/` are not committed (`.gitignore`); they are reproducible from steps 1-3.

## What is exported (all real data, unchanged)

- **KP-Tracer 3508, 3515:**
  - the authors' Cassiopeia `_All` and `_Fam` trees and their character matrices, byte-identical copies from `KPTracer-Data/trees/` (Yang et al. 2022, Zenodo 5847462);
  - the `KPTracer_meta.csv` rows of both mice.
- **Quinn 2021 (all four mice):**
  - the deposited `character_matrices/` and `trees/` folders, as converted from GEO GSE161363 (files over 50 MB are listed in `audit/skipped_files.tsv`, not copied);
  - the converted obs table (`sampleID`, `LineageGroup` and the authors' other columns).
  - Allele tables are not copied; their size is listed.
- **Frozen pipeline labels:** lineage per cell, the tree used per lineage and the exact leaf → cell mapping (`labels/`). In Quinn 10k a cell is `<sampleID>.<leaf>`.
- **Existing OT outputs of these mice, all five strategies, M4-E and M4-R:**
  - `cell_confidence.csv`, `pair_metrics.csv` and `run.json` as written;
  - per-cell retain / reject / exclude;
  - the existing scores (`03_score_runs.py`) and summary tables (`05_aggregate.py`);
  - the fixed joint UMAP (`48_reference_umap.py`);
  - the existing `06_figures.py` images.
- **One derived display table per run (`ot/matching/`).** It is the protocol's own q_ij (§6, rejected cells weigh 0) summed by source group: the KP-Tracer primary tumour or the Quinn LL clone. For each target cell it gives the group with the largest weight, that weight, ties, the runner-up, and the single source cell with the largest q_ij. This is a display of the saved coupling, not a new metric.
- **Couplings** (`coupling_m4e.npz`, `coupling_m4r.npz`, byte-identical) for 3508, 3515 and Quinn 10k only, in the separate optional archive. Quinn 5k is about 0.6 GB per coupling.
- **DEG/GSEA:** the export lists every DEG/GSEA file it finds for these mice (`audit/deg_gsea_inventory.tsv`). The lineage-validation runs have no DEG/GSEA so far; the existing downstream covers only the three human cancers. **Figure E is therefore not produced; nothing is re-analysed to make one.**

## Cell-ID mapping (checked)

`audit/cell_id_mapping.tsv` gives, for every tree:
- leaves;
- character-matrix rows, and leaves ∩ rows;
- leaves with metadata;
- leaves mapped to cells;
- mapped cells that are in the OT runs.

Already checked on the local copies of the authors' files:

| Tree | Leaves | Matrix rows | Leaves in matrix | Leaves with KP metadata | Characters | Missing entries |
|---|---|---|---|---|---|---|
| 3508_Apc_All | 4,798 | 4,798 | 4,798 | 4,798 | 29 | 15.0% |
| 3515_Lkb1_All | 2,209 | 2,209 | 2,209 | 2,209 | 30 | 11.8% |

KP-Tracer leaf names equal cell ids exactly.

- **3508:** families T1 and T2. T1 is 3,731 cells, all on the primary. The T2 family is 312 T2 cells plus every metastasis (N1-N6, PF, Rib; 755 cells).
- **3515:** all metastases (K1-K3, L1, L2, N1; 1,168 cells) are in the T1 family with 710 T1 cells; T2 (213) and T3 (118) are their own families.

## Figures (per tree)

| File | Content |
|---|---|
| `<tree>_A_matrix` | character matrix, rows in tree leaf order. Colour = edit state id (the same colour means the same state within one column; state ids are not comparable across columns). Light grey = unedited (0), white = missing. Strips: primary / metastasis, site, family or clone |
| `<tree>_A_tree` | the authors' tree (x = branching depth, unary nodes merged) and its site strip, with two zoomed branches |
| `<tree>_A_tree_matrix` | tree and matrix with one shared row order: the whole tree and the two zooms |
| `<tree>_B_tree_ot_rejection` | tree, site, then the M4-E retain / reject / exclude of each strategy in the run that holds every cell of the mouse (KP-Tracer D_C3: pooled primaries → pooled metastases; Quinn D_B: LL → pooled tissues). Not in run = not analysed there (e.g. not malignant, or not an eligible site) |
| `<tree>_C_tree_ot_coupling` | tree, site, the CRISPR / author lineage label, and per strategy the OT-predicted source group of each target cell with its weight. Predictions are expression-based associations, drawn beside the tree, never as tree edges |
| `<mouse>_D_umap` | the fixed joint UMAP: site, lineage, the five gates, the OT-predicted source group (RNA only) |

**Zoom rule** (fixed in the code): clades of 80–400 leaves holding at least two sites, ranked by their number of metastatic (KP-Tracer) or non-LL (Quinn) leaves; the top two that do not overlap.

Large trees are drawn without leaf labels; every leaf is a row and the strips carry its annotations.

## Annotation tables

`annotations/<dataset>_<mouse>_<tree>_cells.csv` has one row per leaf, in leaf order, with:
- the cell id, site, primary / metastasis, lineage label and author metadata;
- whether the leaf has a matrix row, and its missing share;
- the UMAP coordinates;
- per strategy (M4-E): the state, the predicted source group, its weight and the top source cell.

The `leaf_order` column links every figure row back to its cell.
