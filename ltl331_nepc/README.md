# LTL331 neuroendocrine transdifferentiation time course (GSE297328)

**Data.**
- Sar et al., *Genome Medicine* 2026, doi 10.1186/s13073-026-01806-8; GEO GSE297328.
- The LTL331 patient-derived xenograft under the kidney capsule of NOD-SCID mice, at 8 time points: before castration, then 2, 4, 8, 12 and 16 weeks after host castration, and 20 and 22 weeks for the relapsed neuroendocrine LTL331R.
- One sample per time point; human-specific reads only; 51,726 cells after the authors' QC.
- The paper's findings:
  - a single clone reprogrammes from prostate adenocarcinoma (PRAD) to neuroendocrine prostate cancer (NEPC);
  - it passes through a transient mesenchymal gateway at 12–16 weeks (~0.25% of cells);
  - it then commits mostly to an ASCL1− fate.

**Pipeline** (`submit_ltl331.sh`; the project's chain for a new dataset):
1. **Download** every GEO file (`cancer_application/GSE297328_longitudinal/scripts/download_gse297328.sh`, sizes checked).
2. **`01_convert_geo.py`** (the files as checked on TACC on 2026-10-30, `00_inspect_data.py` and `00b_match_cells.py`):
   - the annotated h5ad does not match its README: `.raw` is log1p(1e4 × counts / nCount_RNA) on the authors' 26,175 genes, not counts, there is no counts layer, and obs is a Seurat export without `cell_type` or `phase`;
   - the counts therefore come from the RAW CellRanger matrices: every one of the 51,726 annotated cells is found by barcode in the archive of its week, and its counts reproduce `.raw` to 1e-15;
   - one H5AD per time point holds those integer counts (all 29,622 features) for the authors' QC cells;
   - labels: the authors' annotated columns (`seurat_clusters`, `nCount_RNA`, `percent.mt`, …), `coarse_celltype` from the harmony object (linked by `cellid`, all cells), and `phase`, `S_score`, `G2M_score`, `latent_time` from the velocity object (51,499 cells; missing for the other 227);
   - plus `time_weeks`, `relapse` (`is_regrowth`) and `malignant` (every cell: human-specific reads of a PDX).
3. **`02_build_manifest.py`:** the 7 adjacent time-point pairs.
4. **`cancer_metastasis/27_downsample_counts.py`:** the project's QC and depth equalisation.
5. **`03_raw_on_ds_cells.py`:** the original counts of exactly the equalised cells.
6. **ConfidenceOT** (`cancer_metastasis/tacc_preprocessing_factorial.slurm`):
   - two representations on the same cells: **`ranknm256_noscale_ds_cos`** (the project's preprocessing, equalised counts) and **`logcpm`** (library size → log1p → HVG → PCA with gene scaling, original counts; as MOSTA's logcpm arm);
   - M4-E / M4-R, within-side split null, acceptance 0.99, no cell cap, couplings saved.

**Not used.** The authors' log-normalised `.X` / `.raw`, PCA, UMAP, Harmony embedding, velocity layers and the RDS (only obs labels are read from the harmony and velocity objects). The biological question and any further pairing (e.g. non-adjacent time points) come later.

**Test:** `tests/test_prepare.py` runs on fake data shaped like the real files (14 / 14).
