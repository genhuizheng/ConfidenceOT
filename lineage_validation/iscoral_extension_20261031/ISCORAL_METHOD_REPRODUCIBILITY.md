# isCORAL method reproducibility

**Status (2026-10-31).** What I accessed:
- The GitHub user listing https://api.github.com/users/YongjieLin1998/repos, which shows 4 repositories: CORAL, isCORAL_beta, isCORAL_reproduction_notebooks and a profile repo.
- `isCORAL_beta` @ `f9fa160dbab2309e0716fe58943f340951d4670b`, `isCORAL_reproduction_notebooks` @ `b02d5673902293f011e859f60458c615a1bd6658` and `CORAL` @ `68b30dc386b6bee242eaa9aadf4f64a7733f4ffa`. I cloned all three to the session scratchpad and **read them only**: nothing was installed or executed, so runnability below is judged from the code, not tested.
- The preprint bioRxiv 10.64898/2026.10.19.760045 v1, read as a WebFetch digest. It names both isCORAL repositories as the code location.

What I could not access:
- the input data objects (`./1_data/...RData`);
- Supplementary Table 2;
- the "full traceability files", which R's README says are available on request.

All three repositories have a single branch, `main`.

## 1. Repositories

| Repo | Role | Last commit | Commits | Licence |
|---|---|---|---|---|
| https://github.com/YongjieLin1998/isCORAL_beta | **The isCORAL package** (R package `isCORAL`, Version 1.0.0, "beta"). Created 2026-10-11. | `f9fa160` 2026-10-21 21:23 +0800, "Add citation section to README" | 19 | DESCRIPTION says `License: MIT`. There is **no LICENSE file**, and a README licence section was removed in commit `9a76c9b`. |
| https://github.com/YongjieLin1998/isCORAL_reproduction_notebooks | Analysis scripts for the paper, plus archived package versions 6.1.0, 6.1.1, 6.1.3 and 6.1.4 under `packages/`. Created 2026-10-11. | `b02d567` 2026-10-22 00:24 +0800 | 3 | No licence file or field. |
| https://github.com/YongjieLin1998/CORAL | **CORAL-base** (R package `CORAL` v5.0.0). Supervised; needs barcodes. **Not isCORAL.** | `68b30dc` 2026-02-01; on that date the README notes a fix to the "fluctuation mode" numerics | 40 | DESCRIPTION says MIT, but the LICENSE file is **GPL-3** (inconsistent). |

The CORAL README also points to the original CORAL-paper code and data in a Google Drive folder (not accessed).

## 2. Installation (as documented; not tested)
- **isCORAL.** Run `remotes::install_github("YongjieLin1998/isCORAL_beta")`, then `library(isCORAL)` (B README).
  - Imports include Seurat/SeuratObject, RcppML, RSpectra, kernlab, irlba, glmnet, callr, doMC, doRNG, doParallel, Matrix and matrixStats (B `DESCRIPTION`).
  - Leiden needs `leidenbase`; otherwise it falls back to Louvain.
- **CORAL.** Run `devtools::install_github("YongjieLin1998/CORAL")`.
- **Operating system.** The core scan uses `callr` plus fork-based `doMC`. The README and DESCRIPTION `SystemRequirements` state that **native Windows is not supported**, so it needs Linux or macOS (TACC is fine).
- **GPU.** None. Everything runs in CPU R.

## 3. Public API and core files (`isCORAL_beta`)

| Function | File | Needs barcodes? |
|---|---|---|
| `isCORAL()`: core scan, memory scores, `S_stab` | `R/isCORAL_core_functions.R:415` | no |
| `CORALembedding()`: pCORAL cell and clone embeddings | `R/isCORAL_embedding.R:35` | no |
| `SmoothEmbedding()` | `R/isCORAL_embedding.R:609` | no |
| `Assign_deterministic_pseudoclone()`: hclust of 1 − S_stab | `R/clonal_expression.R` | no |
| `FindMetaclone()`: Leiden or Louvain on the clone embedding | `R/isCORAL_metaclone.R:131` | no |
| `PlotMemoryGenes()`, `PlotPseudoclone()`, `PlotCORALEmbedding()`, `PlotMetacloneLandscape()`, `PlotMetaclone()` | `R/*visualization*.R` | optional (ground-truth overlay) |
| `calculate_heritable_genes()`: ground-truth ω² plus permutation threshold | `R/helper.R:26` | **yes** (validation only) |
| `CompareMemoryScores()`: AUROC/AUPRC/max-F1/correlations against ground truth | `R/memory_gene_visualization.R` | **yes** (validation only) |
| `PlotValidationSuite()`, `PlotMetacloneGroundTruthHeatmap()` | `R/isCORAL_metaclone.R:402`, `:940` | **yes** (validation only) |

Unit tests are in `tests/testthat/` (9 files; not run).

## 4. Expected input and output
- **Input.** A Seurat v4 or v5 object with unique cell names.
  - In `input_mode="raw_counts"` (the default), integer counts sit in `RNA/counts`. A warning fires if the values look non-integer (`R/isCORAL_core_functions.R:322-349`).
  - In `input_mode="pre_normalized_data"`, a non-negative log-normalised layer is used instead.
  - `species` = "human" or "mouse". It controls only the `MT-`/`mt-` and `cc.genes` matching.
- **Output of the scan.** It is stored in `seurat_obj@misc$CORAL_scan_results[["binsize_<b>"]]`:
  - `ss_filter_average`, a gene table with mean, sst, fano and `omega_square_infer`;
  - `S_stab`, an N×N sparse consensus matrix;
  - repeat counts.

  Logs go to `results_dir/logs`.
- **Output of the embedding.** It adds the reductions `CORAL_cell_embed` and `CORAL_clone_embed`, plus a cache in `@misc$isCORAL_embedding` holding P_norm, the MDS coordinates and the glmnet model. Note that `CORALembedding()` reads `RNA/data` hard-coded (`R/isCORAL_embedding.R:140,146`).
- **Shipped data.** There are **no checkpoints or pretrained models**; none are needed, because nothing is trained across datasets. There are **no shipped memory-gene lists**, since memory genes are computed per dataset. `isCORAL_beta` has no `inst/extdata`. `CORAL` ships `inst/extdata/pancancer_nmf_programs.csv`, which no R file references.

## 5. Compute requirements (from the code; no benchmarks run)
- **Spectral clustering.** Each call builds a **dense N×N kernel matrix** and its normalised Laplacian (`R/isCORAL_core_functions.R:142-148`). It then takes `eigs_sym` with k ≈ max(30, 2.5·B) and runs k-means with nstart = 100.
  - There are 51 such calls per repeat (50 components plus the final one) × 100 repeats = **about 5,100 dense N×N eigenproblems** per dataset.
  - They run in `parallel_cores` forked workers, each holding its own N×N matrices.
- **Embedding.** `CORALembedding()` allocates several dense N×N double matrices (`E_cross`, `Dist_E_MMD2`, `B_mat`; `R/isCORAL_embedding.R:257-276`).
  - At N = 10⁴ that is about 0.8 GB per matrix.
  - At N = 5×10⁴ it is about 20 GB per matrix, so very large samples need `max_cells` subsampling.
- **Paper runs.** The configs used 10 to 40 cores (R `config/*.yml`).
- **Unknowns.** The paper's runtime and memory figures were **not verified**.

## 6. Code that produced the paper figures
`isCORAL_reproduction_notebooks` contains:
- **`config/*.yml`.** One file per dataset: `01_full_data_HSC_D7`, `02_full_data_A549_invitro_MALAT1`, `06_full_data_mHSC`, `07_full_data_SUM159_1_MALAT1`, `11_full_data_Fatemap1_MALAT1`, `15_full_data__embryo`, `16_full_data__celltag_correct`, `18_full_data_cancerspatial`. Each records the input file, package version and bin size:

  | Config | Package version | bin size | Input object |
  |---|---|---:|---|
  | HSC D7 | 6.1.0 | 200 | `1_singh_HSC_D7.RData` |
  | A549 | 6.1.1 | 200 | `6_A549_invitro.RData` |
  | mHSC | 6.1.0 | 10 | `4_methyltree_HSC.RData` |
  | SUM159 | 6.1.1 | 200 | |
  | FateMap | 6.1.1 | 100 | |
  | embryo | 6.1.1 | 30, `species: "human"` | |
  | CellTag | 6.1.3 | 100 | |
  | cancer spatial | 6.1.1 | 30 | `slidetag_1_tumor_subset_spatial.RData` |

- **`2_analysis_scripts/01_run_full_analysis.R`** plus a launch shell script. It calls `isCORAL_v6_1_0()` from the package version named in the config.
- **`2_analysis_scripts/03_run_downsampling_analysis.R`.** Cell and UMI (binomial) downsampling.
- **`4_benchmark/1_baseline_generation.R`.** Builds the baselines: HVG 2,000, 30 PCs, SNN/KNN at resolution 1.5, cell-cycle-regressed variants and GEMLI (cutoff 50).
- **`4_benchmark/2_isCORAL_comparison.R`.** Metrics and controls.
- **`4_benchmark/3_*`.** Figure plotting.
- **`4_benchmark/4_downsample_memory_benchmark.R`.**
- **`5_embeddings/5_1.R`, `5_2.R`, `5_4.R`.** HSC embedding figures. The barcode column is `larry_sapphire_merged`.
- **`6_dynamic_inference/`.** The HSC and embryo metaclone DAG, using CytoTRACE2.

**Missing or broken pieces (judged from reading the code):**
1. **No input data.** No raw-download-to-`RData` preprocessing is included for any dataset, and no `1_data/` objects are shipped. The README defers to an on-request policy until publication.
2. **No tumour tree code.** There is no script for the mouse-tumour Cas9-tree comparison (Fig. 6A–C, including the F1-maximising branch matching and energy-distance-to-tree). A grep for phylo, newick, ape, subclone, cas9 and `tumor ==` found nothing.
3. **Broken default package for embeddings and dynamics.** `5_embeddings/5_1.R:12` and `6_dynamic_inference/{HSC,embryo}.R:13` load `packages/isCORAL_v6.1.4` by default. That folder ships only `R/helper.R`: there is no `CORALembedding` and no `FindMetaclone` source. These scripts will not run as shipped unless `ISCORAL_PACKAGE_DIR` points at another version, such as `isCORAL_beta`.
4. **Inconsistent clone column.** `4_benchmark/1_baseline_generation.R:57` computes ground truth with clone column `barcode_day0`. `2_isCORAL_comparison.R:22` scores with `barcode`. Mismatched columns only trigger a warning (`2_…R:98-110`).
5. **Bin size not passed through.** `2_isCORAL_comparison.R:54` hard-codes `ISCORAL_BIN_SIZE <- 100`. The dataset configs use 10–200, so each run must be edited by hand, and the value used per figure is not recorded in the repo.
6. **Embryo bin size disagrees.** The paper Methods (WebFetch digest) give 50; `config/15_full_data__embryo.yml:48` gives 30.
7. **Released package differs from the paper's.** The released `isCORAL_beta` 1.0.0 is not the version used for the paper (6.1.0/6.1.1/6.1.3). Its defaults differ: `bin_size` 200 vs 50, `downsample_gemli_ratio` 1 vs 0.75, `nmf_eval_downsample_cells` NULL vs 2000, and `species` human vs mouse (B `inst/doc/ISCORAL_VERSION_DIFFERENCES_AND_UPDATES_BILINGUAL.md` §3).
   - The paper configs set these explicitly.
   - The helper functions are identical apart from whitespace (I diffed lines 1-240 of `packages/isCORAL_v6.1.1/R/isCORAL_core_functions.R` against `isCORAL_beta/R/isCORAL_core_functions.R`).
   - 6.1.0 hard-codes the exclusion of `MALAT1` from the candidate set; 6.1.1 drops that rule, which is what the "_MALAT1" config names refer to.

**Verdict.**
- **isCORAL code:** published and readable. The core scan and embedding look complete and self-contained.
- **Paper reproduction:** **not reproducible end to end from public materials**. The data-preparation steps, the input objects and the tumour-tree evaluation are missing, and some scripts point at an incomplete package folder.
- **CORAL:** a separate, supervised package. Using it would need barcodes and is not a substitute for isCORAL.
