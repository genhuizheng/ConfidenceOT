# R CNV tools: source-verified usage (inferCNV, CopyKAT, SCEVAN, CONICSmat)

**Status, 2026-11-23.** I read the sources only. Nothing was installed or run. The clones are in the session scratchpad, under `cnvtools/`.

| package | source read | commit (date) | version |
|---|---|---|---|
| inferCNV | github.com/broadinstitute/infercnv | `1def6c930335d00ddf251474abb0a31a1c74b0da` (2026-10-25) | DESCRIPTION says 1.23.0 (stale) |
| inferCNV | git.bioconductor.org/packages/infercnv, branch RELEASE_3_23 | `b421d9405c97a309b081ef86d455e976df93eae4` (2026-04-28) | 1.28.0; `R/` is byte-identical to GitHub master (`diff -rq R` is empty) |
| inferCNV wiki | github.com/broadinstitute/inferCNV.wiki | `88b416c7bac5b7dae2750c6e7aada6e8dcfcc8f7` (2026-10-25) | |
| CopyKAT | github.com/navinlabcode/copykat | `01924c0999bb660064da8d3e1561615ac6efb76f` (2026-06-16) | 1.2.5 |
| SCEVAN | github.com/AntonioDeFalco/SCEVAN | `5a49b88ac9445eeffcebb95404e3190992faac04` (2026-03-19) | 1.0.3 |
| yaGST (SCEVAN dependency) | github.com/miccec/yaGST | `56227df3ae183070c9d156af11c306ee799435e6` (2017-11-02) | 2017.08.25 |
| CONICS / CONICSmat | github.com/diazlab/CONICS, subfolder CONICSmat | `053cf267cc819694096d923f17081937ed1a53ea` (2021-08-24) | 0.0.0.1 |
| CONICS wiki | github.com/diazlab/CONICS.wiki | `cf59bba607f1a68e153c4ce8f68fcbeda02c19f3` (2019-06-13) | |

Conda availability comes from `api.anaconda.org/package/<channel>/<name>`, queried 2026-11-23. Line references are `file:line` at the commits above. To link one, use `https://github.com/<repo>/blob/<commit>/<file>#L<line>`.

Points marked **not verified** need someone to run R, or to check something I could not read.

Input files: `01_export_r_inputs.py` (this folder) writes `counts.mtx` (raw integer counts, genes x cells), `genes.tsv` (gene, chr, start, stop) and `cells.tsv` (group, is_reference = group starts with `ref_`). Every tool below takes raw counts with genes in rows and cells in columns, so those files can be used directly. The exceptions are tool-specific annotation formats and naming, noted per tool.

---

## 1. inferCNV (Broad)

### Installation and aarch64
- **Bioconductor (the preferred route in the wiki):** `BiocManager::install("infercnv")` (wiki `Installing-infercnv.md:70-76`). The Bioconductor 3.23 release has 1.28.0 for R 4.6. That release offers binaries for Windows and macOS only, so Linux installs from source.
- **bioconda:** `bioconductor-infercnv` 1.26.0, **noarch** (r-base >=4.5,<4.6). Its run dependencies include `jags` and `r-rjags`. Both have conda-forge **linux-aarch64** builds (jags 5.0.2; r-rjags up to 4_16). The package itself is pure R (`DESCRIPTION`: `NeedsCompilation: no`), so the noarch build is fine on ARM.
- **JAGS is needed even without the HMM.** The `NAMESPACE` has `importFrom(rjags,coda.samples)` and `importFrom(rjags,jags.model)` (`NAMESPACE:93-94`), so loading infercnv loads rjags, and rjags needs the JAGS shared library. `DESCRIPTION`: `SystemRequirements: JAGS 4.x.y`. The wiki (`Installing-infercnv.md:50-55`) also says JAGS must be installed. JAGS is used at runtime only by the BayesNet step (`R/inferCNV_BayesNet.R:1095,1105`), which runs only when `HMM=TRUE`.
- **Dependency tree:** Seurat, igraph, edgeR, SummarizedExperiment, SingleCellExperiment, phyclust, parallelDist, fastcluster, RANN, HiddenMarkov, coin, caTools, among others (`DESCRIPTION` Imports). On conda-forge/bioconda, linux-aarch64 builds exist for r-seurat 5.5.1, r-igraph, bioconductor-edger, r-phyclust, r-paralleldist, r-fastcluster, r-rann, r-hiddenmarkov, r-coin and r-catools.
- **Containers:** the wiki points to `trinityctat/infercnv` on Docker/Singularity (`Installing-infercnv.md:7-38`). Whether an arm64 image exists is **not verified**.
- **Risk on aarch64:** low to moderate. Use the bioconda/conda-forge route, so that jags and rjags come prebuilt.

### Entry functions
`R/inferCNV.R:133-140`
```r
CreateInfercnvObject(raw_counts_matrix, gene_order_file, annotations_file, ref_group_names,
                     delim="\t", max_cells_per_group=NULL,
                     min_max_counts_per_cell=c(100, +Inf), chr_exclude=c('chrX','chrY','chrM'))
```
`R/inferCNV_ops.R:242-348`. These are the main defaults, with their lines.
```r
run(infercnv_obj, cutoff=1, min_cells_per_gene=3, out_dir=NULL, window_length=101,
    smooth_method=c('pyramidinal','runmeans','coordinates'), num_ref_groups=NULL,
    ref_subtract_use_mean_bounds=TRUE, cluster_by_groups=TRUE, cluster_references=TRUE,
    k_obs_groups=1, hclust_method='ward.D2', max_centered_threshold=3, scale_data=FALSE,
    HMM=FALSE, HMM_transition_prob=1e-6, HMM_report_by=c("subcluster","consensus","cell"),
    HMM_type=c('i6','i3'), HMM_i3_pval=0.05, HMM_i3_use_KS=FALSE, BayesMaxPNormal=0.5,
    sim_method='meanvar', sim_foreground=FALSE, reassignCNVs=TRUE,
    analysis_mode=c('subclusters','samples','cells'),
    tumor_subcluster_partition_method=c('leiden','random_trees','qnorm','pheight','qgamma','shc'),
    tumor_subcluster_pval=0.1, k_nn=20, leiden_method=c("PCA","simple"),
    leiden_function=c("CPM","modularity"), leiden_resolution="auto", ..., z_score_filter=0.8,
    denoise=FALSE, noise_filter=NA, sd_amplifier=1.5, noise_logistic=FALSE,
    outlier_method_bound="average_bound", ..., num_threads=4, plot_steps=FALSE,
    inspect_subclusters=TRUE, resume_mode=TRUE, png_res=300, plot_probabilities=TRUE,
    save_rds=TRUE, save_final_rds=TRUE, diagnostics=FALSE, remove_genes_at_chr_ends=FALSE,
    prune_outliers=FALSE, mask_nonDE_genes=FALSE, ..., no_plot=FALSE, no_prelim_plot=FALSE,
    write_expr_matrix=FALSE, write_phylo=FALSE, output_format="png", ..., useRaster=TRUE,
    up_to_step=100)
```
Defaults given as `c(...)` resolve to their first element through `match.arg`. That makes the default `analysis_mode` `"subclusters"`, the default partition method `"leiden"` and the default HMM type `"i6"`.

### Input
- **Counts:** raw counts, **genes x cells**. The argument can be a `matrix`, a `dgCMatrix`, a data.frame or a file name (`inferCNV.R:143-165`). The doc string reads "genes (rows) vs. cells (columns) containing the raw counts" (`inferCNV.R:111-113`).
- **Gene identifiers:** any identifier works, as long as it matches the row names of the gene order file. Only genes present in both files are kept (`.order_reduce`, `inferCNV.R:377`; wiki `File-Definitions.md:57-59`).
- **Gene order file:** tab-separated with no header: gene, chr, start, stop. Pass it as a file name or as a data.frame with gene row names and three columns (`inferCNV.R:168-178`). `chr_exclude` drops rows whose chr is in `c('chrX','chrY','chrM')` (`inferCNV.R:179-181`). That match needs the `chr` prefix, so with bare names ("X") pass `chr_exclude=c("X","Y","MT")`. Genes are ordered by `order(chr, start, stop)`, with chr as a factor in the order the chromosomes first appear in the file (`inferCNV.R:401-407`).
- **Ready-made gene order files:** TrinityCTAT (`https://data.broadinstitute.org/Trinity/CTAT/cnv/`, listed 2026-11-23) has human files (`hg38_gencode_v27.txt`, gencode v19/v21) and mouse files: `mouse_gencode.GRCm38.p6.vM25.basic.annotation.by_gene_name.infercnv_positions` and the GRCm39 vM32 equivalent, each by gene_name or by gene_id. The repo also has a GTF converter, `scripts/gtf_to_position_file.py` (wiki `instructions-create-genome-position-file.md`). Our own `genes.tsv` columns would work as well.
- **Annotations:** two columns, cell and group, tab-separated with no header. A data.frame needs cell row names and one column (`inferCNV.R:184-198`). Every annotated cell must exist in the matrix (`:201-210`), and cells that are not annotated are dropped (`:285`).
- **Reference cells:** `ref_group_names` = the vector of group labels for normal cells (`:291-300`). Each reference group must exist (`:296-297`). Every other group becomes an observation group (`:303-312`).
- **No reference cells:** the wiki says to set `ref_group_names=NULL`, "in which case the average signal across all cells will be used to define the baseline" (`Running-InferCNV.md:30`). In code, `has_reference_cells()` is `length(reference_grouped_cell_indices) != 0` (`inferCNV.R:526-528`). With no reference, `subtract_ref_expr_from_obs` uses `proxyNormal` = all observation cells (`inferCNV_ops.R:1683-1689`). Denoising then uses the mean and sd of all cells (`:2304-2311`). `num_ref_groups` stops with no reference (`:692-693`), as does `mask_nonDE_genes` (`:1512`).
- **Cell filter:** cells with total counts `< min_max_counts_per_cell[1]` (100) are removed in `CreateInfercnvObject` (`inferCNV.R:236-256`). That is the only cell filter. The wiki says inferCNV expects low-quality cells to be removed beforehand (`Running-InferCNV.md:32`).
- **Gene filters in `run`:** the mean count must be at least `cutoff` (`inferCNV_ops.R:560`, `:2128`). The gene must also be expressed in at least `min_cells_per_gene` cells. That second filter is computed over **all** cells, not only references (`:2184`), although the doc string says reference cells.

### Species
inferCNV has no built-in genome and is species-agnostic: species is set entirely by the gene order file. For mouse, use a mouse gene order file such as TrinityCTAT's GRCm38 vM25 by_gene_name. The wiki mentions the mouse MHC cluster on chr17, which shows a strip of signal even in references (`There-is-duplication-and-loss-signal-in-a-strip-of-genes-even-in-my-references.md:1`).

### Output (per-cell profile)
- **Returned object:** `run()` returns the infercnv object (`inferCNV_ops.R:1650`). With `save_final_rds=TRUE` the same object is written to `out_dir/run.final.infercnv_obj` (`:1617-1618`). The object written to `out_dir/preliminary.infercnv_obj` (`:1156`) is the profile before denoising and HMM.
- **The matrix:** `infercnv_obj@expr.data` is **genes x cells**, covering every retained cell, references included. Use `@reference_grouped_cell_indices` and `@observation_grouped_cell_indices` (column indices) to separate them (wiki `Output-Files.md:34-42`). Rows are the genes kept after filtering, in genomic order; `@gene_order` holds chr/start/stop.
- **Scale:** counts are depth-normalised to the median library size (`:3082-3097`), then log2(x+1) (`:2760`). The reference mean is subtracted, the result is capped at ±`max_centered_threshold` (3), smoothed (101-gene pyramidal window), centred per cell, and the reference subtracted again. Finally `2^x` is applied (`invert_log2`, `:2814-2818`). **The final values are fold changes centred at 1** (wiki `Running-InferCNV.md:72-90`). With `denoise=TRUE`, values within `mean ± sd_amplifier*sd` of the reference are set to the reference mean (`:2312-2335`).
- **HMM:** `HMM=TRUE` does not overwrite `expr.data`. The HMM states go to a separate `hmm.infercnv_obj` (`:1255-1262`, `:1469-1475`).
- **Text matrices:** `infercnv.observations.txt` and `infercnv.references.txt` are written only when plotting is on and `write_expr_matrix=TRUE` (`inferCNV_heatmap.R:143-147`, `:544`, `:898-906`). The heatmap's `x.range` clips those text files (wiki `There-is-a-discrepancy...md:1`), so read `@expr.data` from the RDS instead.

### Recommended settings (docs)
- **10x:** `cutoff=0.1`. Sources: vignette `inferCNV.Rmd:124` ("cutoff=1 works well for Smart-seq2, and cutoff=0.1 works well for 10x Genomics"), wiki `infercnv-10x.md:51` and wiki `Running-InferCNV.md:40,48`.
- **Wiki example:** `cluster_by_groups=T, denoise=T, HMM=T` (`Running-InferCNV.md:39-45`). The package's own example uses `analysis_mode="samples", denoise=TRUE, HMM=FALSE, no_plot=TRUE` (`inferCNV_ops.R:232-240`).
- **Threads:** `num_threads` (default 4) sets `infercnv.env$GLOBAL_NUM_THREADS` (`:388`), which is used by `parallelDist` hclust and by BayesNet `CORES` (`:1377`).
- **Clean runs:** `resume_mode=TRUE` reloads step RDS files that already exist in `out_dir`, so use a fresh `out_dir`.

### Runtime / memory
- **Docs:** the wiki gives no figures for 20-30k cells. It says Leiden subclustering was added because hclust "scales poorly with the number of cells" (`infercnv-tumor-subclusters.md:17`).
- **Project experience** (memory note, not docs): with R inferCNV 1.26 on units of up to about 75k cells, BayesNet forks ran out of memory at 16-32 threads. `inspect_subclusters=TRUE` (the default) draws a heatmap after STEP 15 even with `no_plot=TRUE`. That raster overflowed past about 5.4e8 genes x cells. Units of 20-30k cells stayed below that size. With `HMM=FALSE`, `analysis_mode="samples"` and `inspect_subclusters=FALSE`, neither failure path is reached.

### Calls I would use
```r
library(infercnv); options(scipen = 100)   # the scipen advice is printed by inferCNV.R:314-318
# (a) KP mouse, author normal cells as reference
obj <- CreateInfercnvObject(raw_counts_matrix = counts,          # dgCMatrix genes x cells
                            gene_order_file   = gene_order,      # data.frame: rownames gene; chr, start, stop
                            annotations_file  = annot,           # data.frame: rownames cell; one column = group
                            ref_group_names   = ref_groups,      # unique groups starting with "ref_"
                            chr_exclude       = c("chrX","chrY","chrM"))   # match genes.tsv chr naming
obj <- run(obj, cutoff = 0.1, out_dir = od, cluster_by_groups = TRUE, denoise = FALSE, HMM = FALSE,
           analysis_mode = "samples", num_threads = 4, no_plot = TRUE, inspect_subclusters = FALSE,
           resume_mode = FALSE, save_rds = FALSE, save_final_rds = TRUE)
prof <- obj@expr.data   # genes x cells, FC centred at 1
# (b) Quinn human A549, no normal cells: identical, but ref_group_names = NULL (baseline = mean of all cells)
```
`denoise` and `HMM` are a protocol choice. Both default to FALSE in `run()`.

---

## 2. CopyKAT

### Installation and aarch64
- **Route:** GitHub only, `devtools::install_github("navinlabcode/copykat")` (`README.md:25-30`). `r-copykat` does not exist on bioconda or conda-forge (API lookups 2026-11-23).
- **Pure R:** the repo has no `src/` directory, and `DESCRIPTION` declares no compiled code.
- **Dependencies:** all on CRAN (`DESCRIPTION` Imports: parallel, parallelDist, dlm, gplots, RColorBrewer, mixtools, cluster, MCMCpack, transport, stats, utils). The compiled ones have conda-forge linux-aarch64 builds: r-paralleldist 0.2.7, r-dlm 1.1_6.1, r-mixtools 2.0.0.1, r-mcmcpack 1.7_1 and r-transport 0.15_4.
- **Risk on aarch64:** low. Install the dependencies from conda-forge, then `remotes::install_github` with `dependencies = FALSE`.

### Entry function
`R/copykat.R:31`
```r
copykat(rawmat=rawdata, id.type="S", cell.group="", cell.line="no", ngene.chr=5, min.gene.per.cell=200,
        LOW.DR=0.05, UP.DR=0.1, win.size=25, norm.cell.names="", KS.cut=0.1, sam.name="",
        distance="euclidean", test.emd="FALSE", output.seg="FALSE", plot.genes="TRUE",
        genome="hg20", n.cores=1)
```
The README text says the default is "UP.DR=0.2" (`README.md:66`), but the code default is `UP.DR=0.1`.

### Input
- **Counts:** raw UMI counts, genes x cells. Symbols (`id.type="S"`) or Ensembl IDs (`"E"`) both work (`copykat.R:3-4`; `README.md:47`).
- **Dense matrix recommended:** the README builds `as.matrix(...)` (`README.md:54`). `annotateGenes.*` runs `cbind(anno, mat)` on a data.frame (`annotateGenes.hg20.R:30`, `annotateGenes.mm10.R:35`). Whether a `dgCMatrix` survives that is **not verified**.
- **Gene positions:** built in. `annotateGenes.hg20` uses `full.anno` (hg38, 56,051 genes), and `annotateGenes.mm10` uses `full.anno.mm10` (51,450 genes) (`README.md:173-176`, data in `R/sysdata.rda`). Genes are matched on `hgnc_symbol` or `mgi_symbol` (`annotateGenes.mm10.R:27-34`), so user-supplied positions are not used.
- **Reference cells:** `norm.cell.names` = a vector of cell names. The known-normal branch requires `length(norm.cell.names) > 1` (`copykat.R:136`), so the default `""` means none. The baseline is the per-gene median of those cells on the smoothed matrix (`:145`). The code stops if none of the names is in the data (`:142`).
- **No reference, tissue mode** (`cell.line="no"`): `baseline.norm.cl` looks for a diploid cluster, falling back to `baseline.GMM` (`:166-182`). The README says the method assumes aneuploid tumour cells mixed with diploid stroma (`README.md:3`). On pure A549 this would treat one A549 cluster as "diploid" (inference from the code).
- **No reference, cell line mode** (`cell.line="yes"`): uses `baseline.synthetic` (`:128-134`). The README says this mode "uses synthetic basline from the data variations, which does not represent the published algorithm" and "does not guarantee the success nor the accuracy" (`README.md:78`). It is implemented only inside `if(genome=="hg20")` (`:242-346`).
- **Filters** (`:42-61`, `:86-103`, `:187-213`):
  - Cells with fewer than `min.gene.per.cell` (200) detected genes are removed, but only if more than one cell falls below (`:45`).
  - Genes must be detected in more than `LOW.DR` (5%) of cells for smoothing, and in at least `UP.DR` (10%) for segmentation and the output.
  - Fewer than 7000 genes left sets `UP.DR <- LOW.DR` (`:57-61`).
  - For human only, cell-cycle genes and `HLA-*` are removed (`:75-81`).
  - Cells are removed if any chromosome present in the annotated matrix has fewer than `ngene.chr` (5) expressed genes in that cell. This runs twice, on all genes (`:93`) and on the UP.DR subset (`:201`).
  - **Inference from code (not verified):** if any chrY gene survives LOW.DR, cells that express no chrY genes fail the filter, which would drop female cells in mixed-sex data. Dropping chrY genes from the input avoids that.

### Species
- **Mouse:** `genome="mm10"`. The README says mouse uses "the same method as human data", and the result is "in gene space instead of genomic space". The aneuploid/diploid call is "forced to execute, but users need to eyeball to check its accuracy" (`README.md:84`).
- **What the mm10 branch skips** (`copykat.R:679-963`): the 220 kb bin conversion, the cell-cycle/HLA removal (`:75`), and the cell line mode, which is handled only under hg20.

### Output (per-cell profile)
All files are written to the **working directory** with prefix `<sam.name>_copykat_`, because the names are built with `paste(sample.name, ...)` and no directory. Each run therefore needs its own `setwd()`.

- **Both genomes:** `<s>_copykat_raw_results_gene_by_cell.txt` (`:239`). This is genes x (7 annotation columns + cells), referenced to the chosen baseline (the known normals when given), DLM-smoothed, MCMC-segmented and centred per cell. The README calls it the result "in which cells are referred to normal cells" (`README.md:6-8`). It has the UP.DR genes and the cells that passed the filters. The 7 columns are `abspos chromosome_name start_position end_position ensembl_gene_id hgnc_symbol band` (header of `vignettes/test_copykat_CNA_raw_results_gene_by_cell.txt`; mouse has `mgi_symbol`, `copykat.R:836`).
- **hg20:**
  - `<s>_copykat_raw_results_bin_by_cell.txt` (`:245`).
  - The returned `$CNAmat` = `cbind(chrom, chrompos, abspos, mat.adj)`: 220 kb bins x cells (`:672-674`; `README.md:112`). That is **12,167 bins** including chrX (chrom 23) and excluding Y (`convert.all.bins.hg20.R:17`; 12,168 lines in the shipped `test_output/test_copykat_CNA_results.txt`).
  - `mat.adj` is re-baselined to the cluster that holds most of the known or predicted normals, not to the normals themselves. Values within 0.25 sd of the diploid mean are flattened, and every cell is then centred (`:355-408`).
  - The file is `<s>_copykat_final_results_bin_by_cell.txt` (`:504`). In the current code, every `write.table` in the hg20 tissue branch is at `:239`, `:245`, `:501`, `:504` and `:666`, and none of them writes `CNA_results.txt`. The `test_output/test_copykat_CNA_results.txt` file in the repo dates from an earlier naming.
- **mm10:** `$CNAmat` = `cbind(anno.mat2[,1:7], mat.adj)`, genes x cells (`:961-963`), also written to `<s>_copykat_CNA_results.txt` (`:830`).
- **Cell line mode (hg20):** returns `list(CNAmat, hclustering)` with no `$prediction` (`:344-346`). `CNAmat` = bins x cells, and `raw_results_bin_by_cell.txt` is rewritten with it (`:254`).
- **Scale:** log Freeman-Tukey, `log(sqrt(x)+sqrt(x+1))`, centred per cell (`:108-109`). The baseline is subtracted and the result segmented, so values are **log-ratio-like, centred at 0 per cell**, without a defined unit. The README heatmap uses breaks over [-1, 1] (`README.md:134`).
- **Dropped cells:** cells that fail a filter are missing from `CNAmat`. They are added back to `$prediction` / `<s>_copykat_prediction.txt` as `"not.defined"` (`:487-501`; `README.md:106`).

### Recommended settings (docs)
- **Vignette call (10x breast tumour):** `copykat(rawmat=exp.rawdata, id.type="S", ngene.chr=5, win.size=25, KS.cut=0.1, sam.name="test", distance="euclidean", norm.cell.names="", output.seg="FLASE", plot.genes="TRUE", genome="hg20", n.cores=1)` (`README.md:90`).
- **Ranges:** `win.size` 15-150, and `KS.cut` usually 0.05-0.15 (`README.md:68`).
- **Segment check:** if fewer than 25 breakpoints are found, KS.cut is halved twice and then the run stops (`:223-233`).
- **Threads:** `n.cores` drives `parallel::mclapply` (forks) and `parallelDist` threads (`:123`, `:147`, `CNA.MCMC.R:73`).

### Runtime / memory
- **Docs:** only "It might take a while to run a dataset with more than 10,000 single cells. It is suggested to run one sample at a time" (`README.md:93`).
- **My estimate:** a dense 25k-gene x 30k-cell double matrix is about 6 GB, and the code copies it several times. Expect tens of GB.
- **Heatmaps:** they are drawn regardless of size. hclust on 30k cells is O(n^2) memory, about 3.6 GB for the distance object. `plot.genes="FALSE"` skips the gene-level heatmap.

### Calls I would use
```r
library(copykat); setwd(unit_dir)
# (a) KP mouse with author normal cells
ck <- copykat(rawmat = as.matrix(counts), id.type = "S", genome = "mm10", norm.cell.names = ref_cells,
              ngene.chr = 5, win.size = 25, KS.cut = 0.1, sam.name = unit, distance = "euclidean",
              output.seg = "FALSE", plot.genes = "FALSE", n.cores = 8)
prof <- ck$CNAmat[, -(1:7)]          # genes x cells; or read <unit>_copykat_raw_results_gene_by_cell.txt
# (b) Quinn human A549, no normal cells: cell line mode (README: not the published algorithm)
ck <- copykat(rawmat = as.matrix(counts), id.type = "S", genome = "hg20", cell.line = "yes",
              sam.name = unit, plot.genes = "FALSE", n.cores = 8)
prof <- ck$CNAmat[, -(1:3)]          # 12,167 bins x cells
```

---

## 3. SCEVAN

### Installation and aarch64
- **GitHub route** (`README.md:15-22`): `devtools::install_github("miccec/yaGST"); devtools::install_github("AntonioDeFalco/SCEVAN")`.
- **yaGST** is pure R (its DESCRIPTION has `Imports: ggplot2, doParallel`, and the repo has no `src`). It is not on conda.
- **SCEVAN has compiled C** (`DESCRIPTION`: `NeedsCompilation: yes`; `src/vegaMC.c`, `src/run_vegaMC.c`; `NAMESPACE`: `useDynLib(SCEVAN, .registration=TRUE)`).
  - **The repo ships prebuilt x86-64 objects:** `file src/*.o` reports `ELF 64-bit LSB relocatable, x86-64` for `run_vegaMC.o`, `vegaMC.o` and `sort_data.o`.
  - make may reuse them rather than recompile, which would fail to link on aarch64. Delete `src/*.o` in a local clone and run `R CMD INSTALL` on that clone (or use `R CMD INSTALL --preclean`). That this failure actually happens is **not verified** (I did not build anything).
- **bioconda `r-scevan` 1.0.3 exists but is `noarch: generic`.** Its `build.sh` (bioconda-recipes `recipes/r-scevan/build.sh`) runs `devtools::install_github('miccec/yaGST')` and `R CMD INSTALL --build .` on the bioconda builder. The compiled `SCEVAN.so` inside the noarch package is therefore built for the builder's architecture. Inferred from the recipe, **not tested**: it would not load on linux-aarch64. The recipe's run dependencies also omit yaGST.
- **Other dependencies:** parallelDist, pheatmap, forcats, dplyr, fgsea, cluster, ggplot2, Rtsne, scran, ape, ggtree, tidytree and ggrepel (`DESCRIPTION`). linux-aarch64 builds exist for bioconductor-fgsea, bioconductor-scran, r-paralleldist, r-rtsne and r-ggrepel. ggtree, tidytree and pheatmap are noarch.
- **Undeclared dependency:** `library(igraph)` in `R/testFunc.R:70`.
- **Gene annotation:** shipped in `R/sysdata.rda` (EnsDb Hsapiens v86 / Mmusculus v79 tables, `preProcessing.R:13-17`). No EnsDb package or internet is needed.
- **Risk on aarch64:** moderate, because of the source build with the stale `.o` files and the unusable bioconda build.

### Entry function
`R/pipelineCNA.R:37`
```r
pipelineCNA(count_mtx, sample="", par_cores=20, norm_cell=NULL, SUBCLONES=TRUE, beta_vega=0.5,
            ClonalCN=TRUE, plotTree=TRUE, AdditionalGeneSets=NULL, SCEVANsignatures=TRUE,
            organism="human", ngenes_chr=5, perc_genes=10, FIXED_NORMAL_CELLS=FALSE,
            output_dir="./output")
```
The README differs from the code in three places. It calls the argument `norm_cells` (`README.md:32`), but the argument is **`norm_cell`**. It gives `plotTree` a default of FALSE and lists `ClonalCN` as optional (`README.md:36-37`), but the code defaults are `plotTree=TRUE` and `ClonalCN=TRUE`. The Rd usage (`man/pipelineCNA.Rd:7-22`) does not list `output_dir`.

### Input
- **Counts:** a raw count matrix, genes x cells, with symbols or Ensembl IDs (`pipelineCNA.R:17`). The ID type is detected: symbols are used if any row name matches `gene_name` (`preProcessing.R:23-27`). A `dgCMatrix` is accepted and densified (`preProcessing.R:44-45`).
- **Gene positions:** built in. Only chromosomes 1:22 are used (`preProcessing.R:19-20`), so **chrX and chrY are dropped**. For mouse the filter `1:22` keeps 1-19.
- **Filters** (`preprocessingMtx`, `preProcessing.R:74-185`):
  - Cells with fewer than 200 detected genes are removed, but only when more than one cell has fewer than 100 (`:84-89`, exact code).
  - Genes must be detected in more than `perc_genes` of cells. `pipelineCNA` passes `perc_genes/100`, so 10% by default (`pipelineCNA.R:45`). If 7000 genes or fewer pass, the threshold drops by 0.05 (`:91-101`).
  - Cell-cycle genes (reactome, human or mouse) and `HLA-*` genes are removed (`:127-141`).
  - Cells are removed if any of the 22 (human) or 19 (mouse) chromosomes has fewer than `ngenes_chr` (5) expressed genes (`:145-161`).
  - The code stops if fewer than 15 cells remain (`:163`).
- **Transform:** log Freeman-Tukey, centred per cell (`:165-170`).
- **Reference cells:** `norm_cell` = a vector of cell names. When it is given, the search for confident normal cells is skipped (`pipelineCNA.R:43-47`). The baseline is the per-gene median of those cells (`classifyTumor.R:181-188`). With `FIXED_NORMAL_CELLS=TRUE` the given cells are the non-malignant set; otherwise they only seed the classification of 2 clusters (`classifyTumor.R:292-299`, `:324`). README: "TRUE if norm_cell vector to be used as fixed reference, if you are only interested in clonal structure and not normal/tumor classification" (`README.md:33`).
- **No reference** (`norm_cell=NULL`):
  - `pipelineCNA` always runs `getConfidentNormalCells`. That is an ssMWW gene-set test of normal-cell signatures (`confidentNormal.R:61-83`; thresholds `pval_cutoff=1e-10, nes_cutoff=1.0` at `:78`).
  - Cells it finds become the baseline.
  - If it finds none, `classifyTumorCells` uses `removeSyntheticBaseline`: hclust plus Calinski selection of k, then one random-normal "synthetic normal" per cluster (`classifyTumor.R:11-30`, `:174-176`; `set.seed(1)` at `:172`). In that case **segmentation is skipped**, because it requires normals (`:238`), every cell is called tumour (`:336-338`), and the saved matrix is the smoothed, per-cell-centred profile.
  - **Forcing the synthetic baseline on pure A549:** this means bypassing the confident-normal search. `pipelineCNA` has no argument for it. Instead call `SCEVAN:::preprocessingMtx(..., findConfident=FALSE)` (not exported) and then `SCEVAN::classifyTumorCells(..., norm_cell_names=NULL)` (exported). This mirrors `pipelineCNA.R:45-49`.

### Species
`organism="mouse"` (`pipelineCNA.R:27`) switches the annotation to `EnsDb_Mmusculus_v79` (GRCm38/mm10), the signatures to `geneSetMouse` (`confidentNormal.R:64-68`), the cell-cycle list to the mouse list and the chromosome count to 19 (`preProcessing.R:129-135`). Plotting uses `sizeGRCm39` (`plotSegmentation.R:33,60`), which is cosmetic only.

### Output (per-cell profile)
- **Return value:** `pipelineCNA` returns only `classDf`, a data.frame with one row per input cell. Its `class` column is `"tumor"`, `"normal"` or `"filtered"`, plus `confidentNormal` and `subclone` (`pipelineCNA.R:52-55`, `:87`). Filtered cells are labelled there, not dropped.
- **The matrix:** `file.path(output_dir, paste0(sample, "_CNAmtx.RData"))`, **genes x cells**, covering every cell that passed preprocessing, tumour and normal.
  - **With normal cells:** the object is `CNA_mtx_relat` (`classifyTumor.R:333`). This is the VegaMC-segmented profile (segment means; segments that are not altered are set to 0, `:249-252`), re-baselined on the predicted non-malignant cells, centred per cell and divided by `0.5*(max-min)`, so the global range is about [-1, 1] (`:302-304`).
  - **Without normal cells:** the object is `CNA_mtx`, the smoothed, per-cell-centred relative expression with no segmentation (`:279`, `:258-262`). README example: `load("output/MGH106_CNAmtx.RData")` and use `CNA_mtx_relat` (`README.md:76-80`).
- **Row annotation:** `file.path(output_dir, paste0(sample, "_count_mtx_annot.RData"))`, object `count_mtx_annot`, in the same row order (`pipelineCNA.R:79-81`; README example `:79`).
- **Other files:** segmentation files in the same folder (`*vega_output`, `*_Clonal_CN.seg` when `ClonalCN=TRUE`).

### Recommended settings (docs)
- **No 10x-specific settings.** The vignette call is `pipelineCNA(count_mtx, sample="MGH106", par_cores=20, SUBCLONES=TRUE, plotTree=TRUE)` (`vignettes/IntratumoralHeterogeneityInGlioblastoma.Rmd:27`).
- **Parameters:** `beta_vega` controls segmentation coarseness (`pipelineCNA.R:22`).
- **Threads:** `par_cores` (default 20) drives `mclapply` forks on Linux (`classifyTumor.R:223`, `:99`) and `parallelDist` threads (`:269`).
- **Profile-only runs:** `SUBCLONES=FALSE, ClonalCN=FALSE, plotTree=FALSE` skips the subclone, DE and pathway analyses (`pipelineCNA.R:60-72`).

### Runtime / memory
- **Docs:** no figures. The README says only that it is "particularly fast even for large datasets" (`README.md:13`).
- **Memory:** the matrix is densified (`preProcessing.R:45`) and hclust runs over all cells (`classifyTumor.R:269/284/316`), so expect memory similar to CopyKAT.

### Calls I would use
```r
library(SCEVAN)
# (a) KP mouse, author normal cells as fixed reference
cls <- pipelineCNA(counts, sample = unit, par_cores = 8, norm_cell = ref_cells, FIXED_NORMAL_CELLS = TRUE,
                   SUBCLONES = FALSE, ClonalCN = FALSE, plotTree = FALSE, organism = "mouse", output_dir = od)
load(file.path(od, paste0(unit, "_CNAmtx.RData")))         # CNA_mtx_relat, genes x cells
load(file.path(od, paste0(unit, "_count_mtx_annot.RData"))) # count_mtx_annot, row annotation
# (b) Quinn human A549, no normal cells
cls <- pipelineCNA(counts, sample = unit, par_cores = 8, SUBCLONES = FALSE, ClonalCN = FALSE,
                   plotTree = FALSE, organism = "human", output_dir = od)
load(file.path(od, paste0(unit, "_CNAmtx.RData")))         # CNA_mtx if no confident normals were found,
                                                            # CNA_mtx_relat if the signature search found some
```

---

## 4. CONICSmat

### Installation and aarch64
- **Route:** GitHub subfolder only, `devtools::install_github("diazlab/CONICS/CONICSmat", dep = FALSE)` (wiki `Tutorial---CONICSmat;---Dataset:-SmartSeq2-scRNA-seq-of-Oligodendroglioma.md:19-23`). It is not on bioconda or conda-forge.
- **Pure R:** the package folder has no `src`.
- **Dependencies:** `DESCRIPTION` Imports are beanplot, mixtools, pheatmap, biomaRt, Rtsne, squash and scran. The code also calls `zoo::rollapply` (`InferBreakpoints.R:82,218`), but zoo is **not declared**. The tutorial installs it by hand (`:14`). linux-aarch64 or noarch builds exist on conda-forge/bioconda for every one of these (r-zoo, r-mixtools, r-rtsne and bioconductor-scran have aarch64 builds; beanplot, pheatmap, squash and biomaRt are noarch).
- **Risk on aarch64:** low.
- **`getGenePositions()` is not usable as shipped.** It queries biomaRt live at the default host `dec2016.archive.ensembl.org` (`GetPositions.R:12-20`). That host now answers HTTP 301 with a redirect to `ensembl.org/help/articles/archives` (checked 2026-11-23), and compute nodes may have no internet anyway. Build `gene_pos` from `genes.tsv` instead, as described below.

### Main functions (signatures)
- `getGenePositions(gene_names, ensembl_version="dec2016.archive.ensembl.org", species="human", ignoreAlt=F)` (`GetPositions.R:12`)
- `normMat(expmat)`: raw counts to log2(CPM/10+1) (`Preprocess.R:64-69`)
- `filterMatrix(mat1, genelist, minExp=10, minCells=5)` (`Preprocess.R:33`)
- `calcNormFactors(expmat)`: the per-cell column mean (`Preprocess.R:50-53`)
- `plotAll(mat, normFactor, regions, gene_pos, fname, normal=NULL, tumor=NULL, postProb=0.8, repetitions=4)` (`VisualizePosterior.R:178`), which fits the GMM through `plotChrEnichment(expmat, chr, normFactor, gene_positions, n=1, groups1=NULL, groups2=NULL, start=NULL, end=NULL, k=2, vis=T, postProb=0.95, repetitions=5)` (`:93`)
- `plotHistogram(pmat, expmat, clusters, zscoreThreshold=4, patients=NULL, celltypes=NULL)`: returns cluster IDs (`VisualizePosterior.R:233`)
- `binarizeMatrix(mixmat, normal, tumor, threshold=0.8, withna=T)` (`BinarizeMatrix.R:49`) and `generatePvalMat(mat, regions, normfactor, normal, tumor, gene_pos, threshold=0.8)` (`BinarizeMatrix.R:137`)
  - **Bug in `generatePvalMat`:** the argument is `normfactor`, but the body uses `normFactor` (`:140`), so a global `normFactor` must exist.
- `plotChromosomeHeatmap(mat, normal, plotcells, gene_pos, windowsize=121, chr=FALSE, expThresh=0.4, thresh=1, colo=NULL, colo2=NULL, retMat=FALSE, plotdendrogram=FALSE)` (`InferBreakpoints.R:184`)

### Input
- **Expression:** genes x cells, on the **log2(CPM/10+1)** scale (wiki tutorial `:36`, `plotAll` doc `VisualizePosterior.R:166`). Get it from raw counts with `normMat()`. Gene names are HUGO symbols for human and MGI symbols for mouse, and must match column 2 of `gene_pos`.
- **`gene_pos`** (the format of `getGenePositions`):
  - Columns: `ensembl_gene_id`, `hgnc_symbol` (or `mgi_symbol`), `chromosome_name`, `start_position`, `end_position`.
  - Column 2 must be the symbol (`VisualizePosterior.R:102,105`).
  - Column 3 must hold numeric chromosome codes, with X coded 23, Y 24 and MT 0 (`GetPositions.R:25-28`).
  - The names `chromosome_name` and `start_position` must exist, because `plotChromosomeHeatmap` uses them (`InferBreakpoints.R:206-207`).
- **Regions:** read with `read.table(file, sep="\t", row.names=1, header=T)`, giving columns Chrom, Start, End, Length (wiki `:89-99`). `plotAll` uses `regions[i,1..3]` as chr, start and end (`VisualizePosterior.R:185`).
  - A region is fitted only if **more than 100 genes** of `gene_pos` lie strictly inside it (`:102`, `:107`). The tutorial describes these as expressed genes (`:116`).
  - Regions with too few genes, or where EM fails, are silently left out of the result.
- **Repo region files** (repo root): `chromosome_arm_positions_grch38.txt` (44 autosomal arms), `chromosome_full_positions_grch38.txt` (22 whole autosomes) and `chromosome_full_positions_mm10.txt` (19 whole mouse autosomes). There is **no mouse arm file**. The positions must come from the same build as `gene_pos`.
- **Reference cells: optional by design.** The wiki says it "can be run without a definitive normal control" (`Home.md`). In `plotAll`, `normal`/`tumor` (column indices) only colour the plots (`VisualizePosterior.R:95-100`, `:137-150`), so the posterior matrix does not depend on them. They matter in three places:
  - `binarizeMatrix` and `calcPvalue`, which decide whether the normals sit in the high or the low component, and so whether a call is a gain or a loss (`BinarizeMatrix.R:101-117`);
  - `generatePvalMat`;
  - `plotChromosomeHeatmap`, whose reference is the `rowMeans` of the normals (`InferBreakpoints.R:187`). It cannot run without normal cells.

### Species
- **Mouse:** `getGenePositions(species="mouse")` switches to `mmusculus_gene_ensembl` / `mgi_symbol` (`GetPositions.R:17-19`). That query fails on the retired host, so supply `gene_pos` yourself.
- **Mouse regions:** mouse chromosomes are acrocentric (telocentric), so there are no p/q arms. Use whole chromosomes from `chromosome_full_positions_mm10.txt`, or custom sub-chromosomal regions in the same format.
- **`plotChromosomeHeatmap` is hard-coded to human:** it keeps chromosomes `1:22` (`:206`) and loops `for (i in 1:21)` to place chromosome boundaries (`:290-296`). With `retMat=TRUE` it builds the chromosome label row from those boundaries (`:300-302`). For mouse, chromosomes 20-22 are missing, so the boundaries contain NA and `rep(..., NA)` would error. That conclusion comes from reading the code, **not from running it**.

### Output (per-cell profile)
- **Main output:** `plotAll()` returns `l`, a **cells x regions** matrix (`rownames(l)=colnames(mat)`, `VisualizePosterior.R:213-215`). Each column is the posterior probability, for every cell, of the GMM component with the **larger mean**, that is higher expression of the region (`:193-199`).
  - **Scale:** probabilities in [0,1]. This is not a copy-number ratio.
  - Every cell is kept; only regions are dropped.
  - It also writes `<fname>_CNVs.pdf` (`:179`) and `<fname>_BIC_LR.txt`, which holds the BIC for 1 and 2 components, the BIC difference and the Bonferroni-scaled LRT p-value (`:206-214`; `likelihoodRatioTest`, `:70-75`).
  - The tutorial keeps regions with `BIC difference > 200 & LRT adj. p-val < 0.01` (wiki `:165-167`).
- **Gene-window profile:** `plotChromosomeHeatmap(..., retMat=TRUE)` returns `rbind(chrpos, t(d))` (`InferBreakpoints.R:299-303`).
  - Row 1 holds chromosome labels; the remaining rows are cells, ordered normal then tumour among `plotcells`.
  - `d` is the log-ratio to the normal mean, smoothed with a `windowsize`-gene rolling mean, centred per cell and clipped to ±`thresh` (`:214-225`).
  - It needs normal cells, is human-only as written, and always plots.
- **Per-cell p-values:** `generatePvalMat` returns BH-adjusted p-values, tumour cells x regions (`BinarizeMatrix.R:137-148`).

### Recommended settings (docs)
- **No 10x-specific parameters.** The README says only that it works with "5'/3' tagged (e.g. 10X Genomics) data" (`README.md:21`).
- **Tutorial values:** `filterMatrix(..., minCells=5)`, then `plotAll(..., repetitions=10, postProb=0.75)` in the GBM tutorial (`Tutorial---CONICSmat:-Inference-of-clones-in-GBM...md:43-45`), and `windowsize=121` or `151` with `expThresh=0.2` for the heatmap.
- **Threading:** none. Everything is single-threaded `apply`. The package masks `colMeans`/`rowMeans`/`rowSums`/`colSums` with `apply` versions (`Preprocess.R:1-19`), which forces dense matrices and runs slowly.

### Runtime / memory
- **Docs:** none.
- **Expected cost:** dense log2(CPM/10+1) of 20-30k cells, plus `normalmixEM` with `repetitions` restarts per region on one vector of n cells. The GMM itself is cheap. `plotChromosomeHeatmap(chr=TRUE)` computes a full cells x cells distance matrix (`InferBreakpoints.R:233`).

### Calls I would use
```r
library(CONICSmat)
expr <- normMat(as.matrix(counts))                                  # log2(CPM/10+1), genes x cells
gene_pos <- data.frame(ensembl_gene_id = g$gene, hgnc_symbol = g$gene,   # mouse: column 2 = MGI symbols
                       chromosome_name = chr_numeric, start_position = g$start, end_position = g$stop)
expr <- filterMatrix(expr, gene_pos[, 2], minExp = 10, minCells = 5)
normFactor <- calcNormFactors(expr)
# (a) KP mouse: whole-chromosome regions; reference indices only affect the plots in plotAll
regions <- read.table("chromosome_full_positions_mm10.txt", sep = "\t", row.names = 1, header = TRUE)
l <- plotAll(expr, normFactor, regions, gene_pos, unit, normal = ref_idx, tumor = obs_idx)  # cells x regions
# (b) Quinn human A549, no reference: chromosome arms
regions <- read.table("chromosome_arm_positions_grch38.txt", sep = "\t", row.names = 1, header = TRUE)
l <- plotAll(expr, normFactor, regions, gene_pos, unit)                                       # cells x regions
```

---

## Cross-tool summary

| | inferCNV | CopyKAT | SCEVAN | CONICSmat |
|---|---|---|---|---|
| install | Bioconductor / bioconda noarch + conda-forge jags, rjags | GitHub only, pure R | GitHub (yaGST + SCEVAN, C code); bioconda noarch build unusable on ARM | GitHub subfolder, pure R (+ zoo) |
| aarch64 risk | low to moderate (JAGS) | low | moderate (stale x86 `.o`, compile from source) | low |
| gene positions | user file (TrinityCTAT mouse/human, or `genes.tsv`) | built in hg38 / mm10 | built in EnsDb v86 / v79 | user `gene_pos` + region file |
| mouse | via gene order file | `genome="mm10"` (gene space, no bins) | `organism="mouse"` | whole-chromosome regions (`chromosome_full_positions_mm10.txt`) |
| reference | `ref_group_names` | `norm.cell.names` (>1 name) | `norm_cell` (+ `FIXED_NORMAL_CELLS`) | optional; plots and gain/loss only |
| no reference | `ref_group_names=NULL`: mean of all cells | `cell.line="yes"` (hg20 only; not the published method) | synthetic baseline only if the normal-signature search finds none | native |
| per-cell matrix | `@expr.data`, genes x cells, FC around 1 | `$CNAmat`: bins x cells (hg20, 12,167 bins) or genes x cells (mm10), per-cell-centred log scale | `<sample>_CNAmtx.RData`, genes x cells, about [-1,1] (with normals) or smoothed (without) | `plotAll` result: cells x regions posterior in [0,1] |
| cells dropped | total counts < 100 | <200 genes, <5 genes on any chromosome; "not.defined" in prediction | <200 genes, <5 genes on any autosome; "filtered" in classDf | none |
| chrX | excluded by default (`chr_exclude`) | included (chrom 23) | excluded (1:22) | not in region files |
| threads | `num_threads` | `n.cores` (forks) | `par_cores` (forks, default 20) | none |
