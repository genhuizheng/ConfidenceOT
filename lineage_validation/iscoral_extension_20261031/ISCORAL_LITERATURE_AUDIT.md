# isCORAL literature audit

**Status (2026-10-31).** What I accessed:
- The isCORAL preprint: Lin Y. & Lin Y., "Virtual lineages: in silico lineage tracing from single-cell and spatial transcriptomes", bioRxiv 2026.10.19.760045, DOI 10.64898/2026.10.19.760045, **v1**, posted 2026-10-20. Full-text HTML: https://www.biorxiv.org/content/10.64898/2026.10.19.760045v1.full. No later version was visible.
- The isCORAL package `YongjieLin1998/isCORAL_beta` at commit `f9fa160dbab2309e0716fe58943f340951d4670b` (2026-10-21).
- The reproduction repo `YongjieLin1998/isCORAL_reproduction_notebooks` at commit `b02d5673902293f011e859f60458c615a1bd6658` (2026-10-22).
- The CORAL repo `YongjieLin1998/CORAL` at commit `68b30dc386b6bee242eaa9aadf4f64a7733f4ffa` (2026-02-01).
- The earlier CORAL preprint, bioRxiv 10.1101/2025.08.21.671653 v1.

What I could **not** access, or accessed only partly:
- **How the paper was read.** WebFetch returns a model-written digest of the HTML, not the verbatim text. Every paper statement below is at that level of fidelity, and the key ones were cross-checked against the code.
- **Formulas.** Equations that the page renders as images were unreadable, including the candidate-gene thresholds and the bin-size rule.
- **Figures.** I did not read figure panels or their numeric labels: Fig. 2C, 2E, 3E–H, 6B and the supplementary figures.
- **Supplementary material.** The bioRxiv supplementary page listed no files. Supplementary Table 2 (dataset overview and bin sizes) was therefore **not verified**.
- **Journal version.** No journal version of isCORAL was found.
- **CORAL's journal version.** The isCORAL reference list cites CORAL as Cell Syst. 17, 101621 (2026). I did not open that journal version.

The code statements were read directly from the cloned repositories, with file and line given. I installed and ran nothing.

Sources: P = isCORAL preprint full text (URL above); C = CORAL preprint, https://www.biorxiv.org/content/10.1101/2025.08.21.671653v1.full; B = `isCORAL_beta` @ f9fa160; R = `isCORAL_reproduction_notebooks` @ b02d567; K = `CORAL` @ 68b30dc; Z = https://zenodo.org/records/19771805.

---

## 0. CORAL and isCORAL are different methods

| | CORAL (CORAL-base) | isCORAL |
|---|---|---|
| Paper | Lin, Chen, Wu, Zhou, Lin, "Widespread transcriptional memory shapes heritable states and functional heterogeneity in cancer and stem cells", bioRxiv 10.1101/2025.08.21.671653 (posted 2025-08-25) (C). isCORAL ref. 32 cites it as Cell Syst. 17, 101621 (2026); that version not opened. | Lin & Lin, bioRxiv 10.64898/2026.10.19.760045 v1 (P) |
| Code | https://github.com/YongjieLin1998/CORAL (R package "CORAL", v5.0.0) | https://github.com/YongjieLin1998/isCORAL_beta (R package "isCORAL", beta 1.0.0). The paper itself was run with archived versions 6.1.0, 6.1.1 and 6.1.3 (see the reproducibility file). |
| Needs lineage barcodes? | **Yes.** It works on experimental clones (K: README.md, which requires a `barcode` metadata column; `R/coral_base_analysis.R:214`, `run_coral_ground_truth_analysis(true_barcode_col=...)`). | **No.** The core scan and embedding take no clone labels (B: `R/isCORAL_core_functions.R:415-443`; `R/isCORAL_embedding.R:35-45`). Barcodes are used only by the validation and plotting helpers. |
| What it does | Energy distance between experimental clones, then MDS and hierarchical clustering into "heritable states". A per-gene ω² with clone as the factor. A permutation null with threshold Q3 + 1.5·IQR of ω² under shuffled clone labels (K: `R/coral_base_analysis.R:407-445`; C, Methods). A "fluctuation mode" curve of ω² across tree cuts (K: `R/coral_base_analysis.R:489-519`). | Replaces experimental clones with transcriptome-derived temporary bins, then builds a consensus cell–cell co-association matrix ("pseudoclones"), memory scores, pCORAL embeddings, metaclones and a dynamic map (P; B). |

The earlier CORAL paper (C) is a lineage-*supervised* analysis. It used the authors' own A549 lentiviral-barcode data in vitro (8,538 barcoded cells in 549 clones in the text, 8,972 cells in the Fig. 1C legend; C says both) and an A549 xenograft. It also used public data:
- **Quinn et al. 2021, GEO GSE161363**: the "m5k" mouse, left-lung cells, with the static barcode "iniBC" used as the clone label (C, Methods);
- WM989 melanoma (Harmange et al. 2023);
- mouse HSC from Singh et al. 2025 (figshare 10.6084/m9.figshare.25822948);
- mESC data (GSE226169).

So **Quinn data were used by CORAL, not by isCORAL**. In the isCORAL reference list, Quinn et al. 2021 is ref. 68. The text cites it only once, in the Discussion, as an example of cancer-evolution systems (P).

---

## A. Memory-gene identification in isCORAL

### What "memory gene" means
- **Ground truth (only when barcodes exist).** A gene's "transcriptional memory" is the one-way ANOVA ω² with experimental clone identity as the factor (P, Results/Methods; B: `R/helper.R:26`, `calculate_heritable_genes()`).
- **Inferred score.** isCORAL's inferred memory score is the same ω² statistic, but computed against **transcriptome-derived bins instead of clones**, and averaged over repeats.
- **No threshold for calling a gene "memory".** isCORAL produces a continuous ranking (`omega_square_infer`). `PlotMemoryGenes()` keeps genes with mean > 0.1 and ranks them (B: `R/memory_gene_visualization.R:113`).
- **Benchmark positives.** These are the top 2,000 genes by experimental ω² (B: README, `CompareMemoryScores()`; R: `4_benchmark/2_isCORAL_comparison.R:202-205`).
- **CORAL's threshold.** CORAL's permutation threshold (Q3 + 1.5·IQR, K: `R/coral_base_analysis.R:445`; B: `R/helper.R:188`) applies only to the barcode-based ground truth.

### Exact criterion as implemented (B, `R/isCORAL_core_functions.R`)

1. **Normalisation.** `Seurat::NormalizeData` (log1p of counts per 10⁴) is applied to raw counts (l.490-509). In `pre_normalized_data` mode, the supplied layer is used as is (l.510-527).

2. **Gene statistics (l.538-548).** Mean μ_g; SST_g = Σx² − N·μ_g²; Fano F_g = (SST_g/N)/μ_g.

3. **Candidate ("GEMLI") genes (l.563-571).** A gene is a candidate if any one of three conditions holds, where q_p(·) is the p-quantile across all genes:

   μ_g ≥ q₀.₉₈(μ)  **or**  (μ_g ≥ q₀.₉₀(μ) and F_g ≥ q₀.₄₀(F))  **or**  (μ_g ≥ q₀.₆₀(μ) and F_g ≥ q₀.₉₀(F))

   Then mitochondrial genes (`^MT-` / `^mt-`), Seurat `cc.genes` S and G2M genes, and any user `genes_to_remove` are removed (l.550-571).

   The selection scheme follows GEMLI (Eisele et al. 2024, isCORAL ref. 33). The quantile values match the defaults listed in B `inst/doc/ISCORAL_VERSION_DIFFERENCES_AND_UPDATES_BILINGUAL.md` §3.

4. **Per repeat q = 1…Q (Q = `n_repeats` = 100), l.671-739:**
   - a. **Gene subsample.** Draw a fraction `downsample_gemli_ratio` of the candidates (1 in beta; 0.75 in the 6.1.0/6.1.1 defaults, but the paper configs set it to 1, R: `config/*.yml`).
   - b. **NMF (l.174-230).** Clip negatives to 0. Apply log1p if max > 50. L2-normalise each cell. Run `RcppML::nmf` with k = 50 and no L1/L2 penalty. Take the top 5 genes by loading for each component.
   - c. **Temporary partition per component j (l.698-713).** Spectral clustering of all cells on those 5 genes:
     - Laplace kernel, with σ from `kernlab::sigest` (median);
     - L_sym = D^-½ W D^-½;
     - top eigenvectors from `RSpectra::eigs_sym`, row-normalised;
     - k-means with k = B = max(2, ⌊N / bin_size⌋) and `nstart` = 100 (l.102-169, l.669).

     Then compute ω² of **every gene** against that partition.
   - d. **Component scoring (l.715-730).** Rank genes by ω² among genes with mean > 0.05. A component's score is the mean rank of the candidate genes. Keep the 20% of components with the **lowest** mean rank (10 of 50).
   - e. **Final partition.** Spectral clustering on the union of the kept components' top genes, with B bins. Then compute ω²_g,q of every gene against this final partition (l.733-737).

5. **ω² formula (l.64-84).** For a partition with k groups, with SSB, SST and SSW = SST − SSB, and MSW = SSW/(N − k):

   **ω²_g = (SSB_g − (k−1)·MSW_g) / (SST_g + MSW_g)**

   Code: numerator l.81, denominator l.82, ratio l.84.

6. **Memory score (l.793-807).** score_g = max(0, (1/Q)·Σ_q ω²_g,q), where NA is set to 0 before averaging and negatives are set to 0 afterwards.

### Questions

- **Fully unsupervised?** Yes, as implemented. No clone label enters `isCORAL()` or `CORALembedding()`. The paper says the measured lineages supply ground truth only, and that LARRY barcodes in the HSC data and the Cas9 tree in the tumour data were used only for validation (P, Results).

- **Barcodes during training?** There is no training step: no model is fit and transferred, and every run is transductive on one dataset.
  - Hyperparameters are fixed defaults (the quantiles, k = 50, 5 genes, 20%, Q = 100, Laplace kernel) plus a per-dataset bin size.
  - **I could not verify how the defaults were originally chosen, or whether lineage-traced data informed them.** The paper text visible to me says only that standard settings were used without additional tuning (P).
  - The bin size per dataset is in Supplementary Table 2, which I could not access. The README rule is bin_size ≈ N/50 (B: README §1).

- **Pretrained information from other lineage-traced datasets?** **None in the code.**
  - There are no checkpoints, no shipped gene lists and no reference objects (B has no `inst/extdata`).
  - The CORAL repo ships `inst/extdata/pancancer_nmf_programs.csv`. No R file in either package references it (grep).
  - The only external gene lists used are Seurat's `cc.genes` and the mt- prefix (B l.550-558).

- **Dataset-specific?** Yes. Memory scores are recomputed from scratch for each dataset and depend on that dataset's partition structure and bin size.

- **Sensitivity to cell type, batch, cell cycle and treatment:**
  - **Cell cycle.** The S/G2M gene lists are removed only from the **feature** set used to build partitions. ω² is still computed for all genes (l.711, l.737), so a cell-cycle gene can still score high if cycle phase aligns with the bins.
    - The paper reports HIST1H1E scored low by isCORAL (0.035, rank 1,798; true score 0.013), versus 0.639 by GEMLI (Fig. 2D, P).
    - It also reports that isCORAL slightly overestimates Top2a (Fig. S1H, P; not opened).
    - The S/G2M lists are a fixed reference set, so phase signal carried by other genes is not removed.
  - **Cell type, batch, treatment.** The method has **no covariate, batch or cell-type handling** (no such argument in `isCORAL()`, B l.415-443).
    - The bins are fine transcriptomic clusters of the whole sample (about 50 bins at the README rule).
    - Any between-type, between-batch or between-condition difference is therefore a strong driver of the bins. Genes marking it get high ω² by construction.
    - The paper's Discussion, as rendered by WebFetch, does not discuss batch or treatment. On ancestry it states that the approach "does not reliably distinguish current cell state from ancestry" (P, Discussion).

- **Can it separate heritable expression from persistent cell-state correlation?** **No, not by construction.**
  - The inferred score is the variance of a gene explained by unsupervised transcriptomic bins.
  - A gene that marks a stable non-clonal state, such as a cell type, niche or treatment response, scores just as high as a clonally inherited one.
  - Clone ancestry enters only through the assumption that fine transcriptomic substructure mirrors clones.
  - Any separation is shown only empirically, against barcode ω² ground truth on the datasets tested (Fig. 2 and S1–S2, P).
  - "Heritable" here means clonally shared, measured by barcodes. Neither paper, as read here, shows genetic inheritance. CORAL (C) attributes memory to epigenetic maintenance, not genetics.

---

## B. Lineage reconstruction

- **Probabilistic clone relationships (B, l.235-256).**
  - For each of the Q final partitions, build a cell-by-bin indicator matrix.
  - S_stab = (1/Q)·Σ_q 1[cell i and cell j share a bin in repeat q], with diag = 1. This is a consensus co-association matrix.
  - The "pseudoclone" of cell i is row i of S_stab, row-normalised: P_norm = D⁻¹ S_stab (B: `R/isCORAL_embedding.R:250-252`).
  - The README defines enrichment(s, j) = P_norm(s, j)·(N − 1).

- **Clone-oriented embedding, pCORAL (B: `R/isCORAL_embedding.R`):**
  1. Take the top G = 3,000 genes by memory score (l.237-238) from the log-normalised `RNA/data` layer.
  2. Compute the squared energy distance between the pseudoclone distributions of cells i and j (l.254-271):

     ED²(i,j) = 2·E[‖x−y‖] − E[‖x−x′‖] − E[‖y−y′‖], with x, x′ ~ P_norm(i,·) and y, y′ ~ P_norm(j,·), using Euclidean distance on those genes.
  3. Classical MDS by double centring, then `irlba`, giving 30 dimensions (l.274-285).
  4. A multi-response elastic net (`glmnet`, family "mgaussian", α = 0.05, 5-fold CV, λ.min) from the cell-L2-normalised, z-scored memory-gene expression to the MDS coordinates (l.290-373).
  5. Predictions are rescaled to the MDS per-dimension SD, giving `CORAL_cell_embed` (l.401-413).
  6. `CORAL_clone_embed` = P_norm · CORAL_cell_embed (l.415-423).

- **Outputs:**
  - S_stab / P_norm, a pairwise cell–cell co-association. This is not a probability that two cells are sisters.
  - Memory scores.
  - Deterministic pseudoclones from `hclust(1 − S_stab)` cut at k (`Assign_deterministic_pseudoclone`).
  - The two embeddings.
  - Metaclones: Leiden on an SNN graph of the clone embedding (`FindMetaclone`, B `R/isCORAL_metaclone.R:131`).
  - In R only: a metaclone DAG built from CytoTRACE2 potential (`6_dynamic_inference/`).
  - **No lineage tree** is produced. The hclust dendrogram of 1 − S_stab is not a phylogeny.

- **Uncertainty or confidence.** The only uncertainty measure is the co-association frequency across the Q repeats.
  - It is not calibrated against clonal truth.
  - There is no per-cell statistic for "this cell has no clonal relative in the sample".
  - Every cell receives a distribution, because diag(S_stab) = 1 and each cell always shares a bin with about bin_size others.
  - There is no p-value, posterior or null model for pseudoclone membership.

- **Unseen clones.** The method is transductive: it is re-run on each dataset and has no train/test split. The elastic net could in principle project new cells into `CORAL_cell_embed`, but no API does this (not verified as a supported use).

- **Number of clones needed?** No. However, `bin_size` fixes the number of temporary bins B = ⌊N / bin_size⌋, which sets the resolution of the pseudoclones. Deterministic pseudoclones and metaclones need k or a resolution.

- **Required biological assumptions (from the paper's framing and the construction):**
  - Clonal relatives remain transcriptionally closer on high-mean/high-Fano gene modules than non-relatives.
  - That memory persists over the time between clone founding and sampling. The paper states that memory typically persists over short timescales, so pseudoclones capture recent relationships (P, Results).
  - Clonal structure is not dominated by stronger non-clonal structure. The paper notes that dispersed, plastic clones were hard for all methods (P, Results).
  - One sample in one condition. There is no batch model.

---

## C. Experimental validation (summary; full rows in ISCORAL_DATASET_COMPARISON.tsv)

**Datasets.** The paper benchmarks on six lineage-traced scRNA datasets:
- A549 (own data, ref. 32);
- mouse HSC (Chen et al. 2025, ref. 28);
- LARRY HSC day 7 (Singh et al. 2025, ref. 37);
- SUM159 (Nadalin et al. 2024, ref. 44);
- FateMap melanoma (Goyal et al. 2023, ref. 43);
- CellTag iEP (Biddy et al. 2018, ref. 42).

It adds two application datasets:
- a mouse lung tumour Slide-tags subset with a Cas9 tree (Jones et al. 2026, ref. 38);
- a human CS6 Stereo-seq embryo (Xiao et al. 2026, ref. 39), which has no lineage labels.

**Metrics:**
- Memory genes: AUROC, AUPRC and max-F1 with the top 2,000 experimental-ω² genes as positives; Pearson and Spearman of ω².
- Pseudoclones: cell-pair AUROC/AUPRC with same-clone pairs as positives, AUPRC enrichment over prevalence, energy distance in the top 30 PCs, and pseudobulk L1 (P, Methods; R: `4_benchmark/2_isCORAL_comparison.R:212-346`).
- Controls: a "bootstrap floor" and uniform allocation (R l.389-418).

**Baselines** (R: `4_benchmark/1_baseline_generation.R:61-171`; `2_isCORAL_comparison.R:158-200`):
- Seurat VST, Seurat MVP and scran `modelGeneVar`, which are HVG scores;
- a "PowerSeek" approximation: PCA variance loading on 30 PCs;
- GEMLI with cutoff 50;
- standard SNN and KNN graphs on 2,000 VST HVGs, 30 PCs, resolution 1.5;
- the same graphs after cell-cycle regression.

The SNN/KNN graphs are the "ordinary RNA PCA/HVG" baselines.

**Reported results (P; numbers only where they appear in the text I read):**
- **Memory genes, A549 (8,972 cells, 6,556 genes; Fig. 2B–D).**
  - isCORAL is reported best on all five metrics.
  - Pearson r = 0.84 between inferred and experimental ω².
  - Per-method AUROC/AUPRC values are in Fig. 2C (**not verified**).
  - Gene examples: NNMT has true ω² 0.364 (rank 23), isCORAL 0.170 (rank 102), GEMLI 0.458 (rank 400). HIST1H1E: see above.
  - The text says the largest gains were in A549 and SUM159. The cross-dataset Spearman values in Fig. 2E were **not verified**.
- **Pseudoclones.**
  - On mouse HSC (549 cells, Fig. 3C–F), isCORAL is reported to have the highest AUPRC enrichment and the lowest energy-distance and L1 errors. KNN baselines had the lowest AUCs. **Numbers are in the figures, not verified.**
  - Fig. 3H summary across datasets: isCORAL reported best everywhere shown, with roughly two- to four-fold AUPRC-enrichment gains over the strongest baseline in some cases (P). **Per-dataset values not verified.**
- **Mouse tumour (Fig. 6A–C).**
  - Each pseudoclone was matched to a branch of the recorded Cas9 tree. The Fig. 6B legend says branches were selected to maximise the F1 of pseudoclone recovery, which uses the tree labels at **evaluation**.
  - Median energy distance: isCORAL 1.42 versus its random control 5.14. The cell-cycle-regressed SNN had a median of 4.18, and its random control was computed separately (value not verified).
  - 9 metaclones versus 6 tree subclones (Fig. 6C).
- **Use of lineage for training or model selection.**
  - The method is unsupervised.
  - Lineage labels are used for scoring, including the F1-optimal operating points: the memory-gene threshold in Fig. 2B and branch selection in Fig. 6B.
  - The per-dataset bin size (Supplementary Table 2) and the tumour metaclone resolution (0.5) were chosen by the authors. **Whether lineage labels informed those choices is not stated (not verified).**

### The mouse tumour spatial dataset: KP-Spatial, not KP-Tracer 2022 and not Quinn
This is decided from accession and citation only:
- **The Methods give the data source as Zenodo 19771805, "slidetag_1" subset (P).** The reproduction config reads `./1_data/cancer_spatial/slidetag_1_tumor_subset_spatial.RData` (R: `config/18_full_data_cancerspatial.yml:9`).
- Zenodo 19771805 is "Processed data for study 'Spatiotemporal lineage tracing reveals the dynamic spatial architecture of tumour growth and metastasis'" by Jones, Sun, Min, Colgan et al. It was published 2026-04-25, generated with Slide-seq and Slide-tags, and is one archive, `KPSpatial_Data.tar.gz` (6.1 GB) (Z).
- Ref. 38 in isCORAL is Jones, M. G. et al., Nat. Genet. 58, 2398–2410 (2026), with the same title (P, reference list).
- **=> This is the KP-Spatial dataset (Jones, Sun et al.).**
  - Cells were filtered to `tumor == TRUE`, 500–5,000 genes and < 10% mitochondrial reads.
  - isCORAL was run with bin size 30 and metaclone resolution 0.5 (P, Methods).
  - The cell count was **not reported** in the text I read.
  - That a file or sample named "slidetag_1" exists inside `KPSpatial_Data.tar.gz` is **not verified**: the archive contents are not listed on Zenodo.
- **The text uses "KPTracer" as the name of the Cas9 lineage-recording system used in KP-Spatial.** It does **not** use the Yang et al. 2022 Cell data:
  - there is no Yang 2022 entry in the reference list;
  - Zenodo 5847462 is not cited (P, reference list).
- **Quinn (GSE161363) is not used as data by isCORAL.** It is cited only in the Discussion (ref. 68). GSE161363 does not appear in the isCORAL text (P).

### Was KP-Tracer, KP-Spatial or Quinn used to train, tune or pretrain isCORAL?
- **Pretrain or train:** no, for all three. There is no training or pretrained state (B).
- **Evaluation:** KP-Spatial (slidetag_1 subset) **was used as an evaluation and application dataset**, and its Cas9 tree was used to score results, including F1-optimal branch matching.
  - The tumour hyperparameters (bin 30, resolution 0.5) were set by the authors on this dataset.
  - Whether the tree was in view when they were set is **not stated**.
  - The paper says no additional tuning was done; I cannot verify that.
- **KP-Tracer 2022 (Zenodo 5847462):** not used anywhere in isCORAL, as far as the reference list and Methods show.
- **Quinn 2021:** not used by isCORAL. It was used by the earlier CORAL paper (C), with the "m5k" left lung and static barcodes. That shaped the authors' notion of memory genes, but no parameter, gene list or model carries from it into isCORAL's code.

---

## Implications for a leakage-free use on KP-Tracer / Quinn

**Supervision isCORAL needs:**
- **At training:** none. There is no training phase.
- **At inference:** only the expression matrix (raw counts or a log-normalised layer), a species flag for the cell-cycle and mt gene lists, and a bin size.
- **Risks to avoid:**
  - Fix `bin_size` (for example the README's N/50 rule) **and the downstream k or resolution before looking at barcodes**.
  - Do not pick operating thresholds by F1 against barcodes, as the paper's Fig. 2B and Fig. 6B do for display.
  - Score with label-free threshold-independent metrics (AUROC/AUPRC over all pairs), or with thresholds fixed in advance.
- **KP-Spatial specifically.** isCORAL has already been evaluated, and its tumour hyperparameters set, on a KP-Spatial Slide-tags subset. A result on KP-Spatial is therefore not an independent test of the published configuration.
- **KP-Tracer 2022 and Quinn.** isCORAL has not seen these data in the published work, so a run with frozen defaults is a clean held-out application.
- **Batch handling.** isCORAL has no batch handling. Running it across pooled tumours or mice makes the bins track tumour or mouse identity, which inflates same-clone AUPRC trivially, because clones are nested in tumours. Run it per tumour or per sample, or evaluate pairs only within a tumour.

**Which settings the published evaluation corresponds to:**

| Setting | Published isCORAL evaluation |
|---|---|
| (1) RNA-only inference | **Yes**, for inference on every dataset (B code; P). |
| (2) External pretrained lineage information | **No.** No pretrained model or shipped gene lists. The defaults' provenance is not verified. |
| (3) Lineage labels of training clones | **No.** There is no train/test split. |
| (4) Oracle labels of evaluation clones | **Used in scoring only.** Ground-truth ω² defines the positives. F1-maximising operating points are chosen with the labels (Fig. 2B threshold; Fig. 6B branch selection). Whether labels informed the per-dataset bin size or metaclone resolution is not stated. |
