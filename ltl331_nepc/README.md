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
1. **Download** on the login node: `GSE297328_RAW.tar` (per-sample CellRanger matrices, 662 MB) and `GSE297328_ltl331_annotated.h5ad` (2.6 GB). Commands are at the end of `submit_ltl331.sh`.
2. **`01_convert_geo.py`:**
   - one H5AD per time point holding the authors' raw integer UMIs (`.raw`) for their QC cells;
   - their labels (`cell_type`, `leiden`, `phase`, `model`), plus `time_weeks`, `relapse` and `malignant`;
   - every cell is a human tumour cell in a PDX with human-specific reads, so `malignant` is set for all;
   - cross-checked against the RAW matrices.
3. **`02_build_manifest.py`:** the 7 adjacent time-point pairs.
4. **`cancer_metastasis/27_downsample_counts.py`:** the project's QC and depth equalisation.
5. **`03_raw_on_ds_cells.py`:** the original counts of exactly the equalised cells.
6. **ConfidenceOT** (`cancer_metastasis/tacc_preprocessing_factorial.slurm`):
   - two representations on the same cells: **`ranknm256_noscale_ds_cos`** (the project's preprocessing, equalised counts) and **`logcpm`** (library size → log1p → HVG → PCA with gene scaling, original counts; as MOSTA's logcpm arm);
   - M4-E / M4-R, within-side split null, acceptance 0.99, no cell cap, couplings saved.

**Not used.** The authors' log-normalised `.X`, PCA, UMAP, Harmony integration and velocity objects. The biological question and any further pairing (e.g. non-adjacent time points) come later.

**Test:** `tests/test_prepare.py` runs on fake data (11 / 11).
