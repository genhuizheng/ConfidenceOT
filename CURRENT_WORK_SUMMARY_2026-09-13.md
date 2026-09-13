# ConfidenceOT Current Work Summary

**Last updated:** 2026-09-13  
**Local repository:** `D:\Xia_lab\CellOT`  
**TACC repository:** `/scratch/10119/ghzheng/OT_project/code/ConfidenceOT`  
**TACC result root:** `/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results`

## 1. Overall objective

The current project evaluates whether ConfidenceOT can identify cell states that are compatible or incompatible between paired primary and metastatic single-cell RNA-seq samples.

The central biological question is:

> What distinguishes primary malignant cells whose expression states are compatible with a matched metastatic lesion from primary malignant cells whose states are not represented in that lesion?

This wording is intentionally descriptive. A ConfidenceOT gate does not by itself prove metastatic potential, clonal origin, or lineage. Such claims require orthogonal validation, for example shared CNV clones, somatic mutations, lineage tracing, or an independent cohort.

## 2. Interpretation of ConfidenceOT outputs

For one exact `patient x primary site x metastatic site` pair:

- `retained=True` means that the cell was retained by the fitted M4 gate and is relatively compatible with the opposite side under that particular model, representation, cost, and initialization.
- `rejected=True` means that the cell was excluded by the fitted gate and is relatively poorly explained by the opposite side under that fit.
- The rejection budget is an upper bound on the rejected fraction, not a requirement to reject that fraction.
- The rejection score is a continuous cell-level diagnostic associated with the binary gate; a larger score indicates stronger evidence toward rejection under the fitted solution.
- Cross-pair/cap consensus states summarize repeated occurrences of the same cell: consistently rejected, consistently retained, or discordant across eligible pair/cap runs.

The labels must not be renamed `metastasis-potential` and `non-metastasis-potential` until they pass robustness and biological validation.

## 3. Main dataset: GSE180661 HGSOC

### 3.1 Dataset inventory

GSE180661 is a high-grade serous ovarian cancer single-cell dataset containing primary lesions and matched metastatic sites such as infracolic omentum, ascites, bowel, and pelvic peritoneum.

| Scope | Coverage |
|---|---:|
| Original patients | 41 |
| Sample files | 74 |
| Cells | 929,690 |
| Enumerable primary-metastasis combinations | 121 |
| All-cell eligible patients | 33 |
| Author malignant label | `Ovarian.cancer.cell` |
| Malignant-only eligible exact pairs | 94 |
| Malignant-only represented patients | 29 |

The number of patients differs by analysis because the malignant-only workflow requires enough author-annotated malignant cells on both sides of an exact pair. Multi-primary origin ranking is a smaller subset of the malignant analysis.

### 3.2 Figure-oriented workflow

#### Figure 1: cohort and sample inventory

- Sample/site availability across patients.
- Primary and metastatic cell counts.
- Cell-type composition by patient and site.
- Sampling time was intentionally ignored because it was not consistently available.

#### Figure 2A: all-cell OT validation

- Unit: one patient, one primary sample/site, and one metastatic sample/site.
- Use all annotated cell types.
- Summarize source-to-target transported mass as a cell-type transition matrix.
- Main validation question: whether biologically corresponding populations, such as T cells, preferentially map to one another.
- This branch validates transport structure; it is not the main malignant-cell discovery analysis.

#### Figure 2B: malignant-only OT

- Use author-annotated malignant cells only.
- Fit each exact primary-metastasis pair independently.
- Visualize primary and metastatic malignant cells jointly for pairwise mapping, while also providing clearly separated primary and metastatic panels.
- Export cell-level confidence and gate state for each side.

#### Figure 3: primary internal differential expression

- Within every exact pair, split primary malignant cells into source-retained and source-rejected groups.
- Sum raw integer counts across cells to form one pseudobulk per `pair_id x comparison_status`.
- Pseudobulk is a raw-count sum, not an average; library-size differences are handled by PyDESeq2 normalization.
- Combine all eligible pair pseudobulks using the paired design `~ pair_id + comparison_status`.
- Test all genes, including genes used in the OT representation.
- Report gene name, log2 fold change, FDR, mean CPM, patient detection fraction, patient direction consistency, and whether the gene was used by OT.

#### Figure 4: pathway analysis

- Run GSEApy prerank using the PyDESeq2 Wald statistic.
- Show positive and negative normalized enrichment scores in a dot plot.
- Retain pathway direction and FDR in the output table.

#### Figure 5: external clinical validation

- Construct a candidate expression signature from the primary internal DEG/GSEA results.
- Test the signature in TCGA-OV with Kaplan-Meier and Cox models.
- Treat survival analysis as external association rather than proof of metastatic lineage.

## 4. Preprocessing and OT settings

### 4.1 Cancer workflows

The standard representation is:

`raw counts -> library-size normalization to 10,000 -> log1p -> joint HVG selection -> gene scaling -> joint PCA -> pairwise cost -> OT`

Settings varied during sensitivity analyses:

- Main GSE180661 malignant analyses used 2,000 HVGs and 30 PCs.
- Post-QC thresholds were `nCount >= 1000`, `nFeature >= 500`, and mitochondrial percentage `<= 20%`.
- The post-QC workflow retained 176,070 of 202,731 source-cell occurrences (86.85%) and 172,683 of 197,090 target-cell occurrences (87.62%).
- The main post-QC sensitivity analysis used source cap 0.85 and target cap 0.95.
- Null-calibrated rejection costs were used unless a workflow was explicitly labelled as a fixed-cost experiment.

### 4.2 All-cell pan-cancer run

The completed all-cell run covered:

- 13 datasets
- 86 patients
- 252 exact primary-metastasis pairs
- 504 method rows: 252 M4-E and 252 M4-R
- Source cap 0.85 and target cap 0.95 in the completed run
- Up to 10,000 observed cells per side

All 252 pairs completed. M4-E converged for all pairs. M4-R had valid calibration for only 3 of 252 pairs, with frequent outer-loop cycles, so M4-R was retained primarily as a diagnostic sensitivity result rather than a biological inference engine.

## 5. Completed sensitivity analyses and findings

### 5.1 Source-cap sensitivity

The target cap was fixed at 0.95 while the source cap was varied. Earlier pre-QC comparisons gave:

| Source cap | Source weighted rejection | Target weighted rejection | Exact winner match vs 0.95 |
|---:|---:|---:|---:|
| 0.95 | 0.911 | 0.615 | 1.000 |
| 0.90 | 0.860 | 0.606 | 0.882 |
| 0.85 | 0.828 | 0.670 | 0.794 |

Additional source caps, including 0.75, 0.60, and 0.50, were subsequently included in sensitivity work. These runs are used to examine cap binding, cell-group sizes, gate stability, DEG direction, and biological coherence. A cap must not be selected solely because it produces more metastasis-related genes.

### 5.2 Post-QC DEG observation

At source cap 0.85, the source-rejected primary cells showed strong inflammatory, interferon, chemokine, stress, and invasion-associated signals, including examples such as `CXCL10`, `CXCL11`, `CXCL8`, `CCL20`, `OASL`, `SAA1`, `RSAD2`, `IDO1`, `MMP7`, `ADM`, and `IGFBP3`.

This observation is biologically interesting but contradicts the initial assumption that source-retained cells could automatically be interpreted as the more metastasis-competent population. It therefore triggered algorithmic and data-quality audits rather than a reversal of labels based only on desired biology.

### 5.3 Initialization sensitivity

M4-E was rerun from deterministic all-one/all-zero-related feasible starts and five random starts for 94 exact malignant pairs. The stored baseline was reconstructed exactly, but random-start robustness was limited:

| Metric | Source | Target |
|---|---:|---:|
| Median whole-gate agreement | 0.816 | 0.762 |
| Median retained-set Jaccard | 0.276 | 0.389 |
| Consensus-stable cell-occurrence fraction | 0.569 | 0.584 |

All fits converged without detected cycles, but convergence did not imply a unique gate. Thus, a single M4-E retained/rejected partition is not currently initialization-independent.

### 5.4 Other diagnostic analyses

The following independent analyses were implemented or run:

- Fixed rejection cost sensitivity at `c = 0.4, 0.5, 0.6, 0.7`.
- A second OT analysis restricted to cells rejected by the first source-cap-0.85 OT.
- One-pair-per-patient analysis, selecting the exact pair with the largest value of `min(source malignant cells, target malignant cells)` for every patient.
- Pre-QC versus post-QC comparison.
- Pair-level QC and library-complexity checks.
- Source-cap-specific pseudobulk DEG and pathway analyses.

The second-OT result contained many positive DEGs but weak patient-direction consistency for most leading genes. It is therefore exploratory and does not establish that first-round rejected cells form a coherent metastatic subpopulation.

## 6. Replication datasets

Independent author-labelled malignant-cell workflows were created for:

- GSE181919: author label `Malignant.cells`.
- GSE225857: author tumour clusters `Tu01` through `Tu11`.

Each dataset is analyzed independently and is never pooled with GSE180661 for OT or DEG. The workflow applies the same post-QC thresholds, 2,000 HVGs, 30 PCs, source cap 0.85, target cap 0.95, exact-pair pseudobulk, paired PyDESeq2, and GSEApy.

Preliminary inspection suggests strong dataset-specific effects. Normal epithelial or goblet-like annotation can create apparently strong matching unrelated to metastatic lineage, particularly when malignant labels are broad. Replication results must therefore be presented with annotation audits and dataset-specific directions rather than as a single pooled effect.

## 7. TACC execution model

### 7.1 Cluster resources

- Allocation: `MCB26031`.
- `gh`: standard GPU work; avoided for the newest small sensitivity jobs when queue behavior was unfavorable.
- `gh-dev`: short GPU analyses and initialization sensitivity.
- `gg`: CPU analyses, pseudobulk, DEG, GSEA, plotting, dependency installation, and the current Splatter scaling benchmark.

The workflow uses Slurm dependencies so that audit/manifest, OT, pseudobulk, DEG, GSEA, and finalization stages start only after their required upstream stages complete successfully. A `DependencyNeverSatisfied` state means an upstream job failed or was cancelled and the downstream job must be cancelled and resubmitted after fixing the cause.

### 7.2 Main TACC result locations

- GSE180661 malignant OT and biological analysis: under `confidenceot_results`.
- Post-QC source-cap-0.85 OT: `gse180661_sourcecap_0p85_postqc_ot_20260908/source_cap_0p85`.
- Initialization sensitivity: `gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_rerun_20260910`.
- Initialization summary: `gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_summary_rerun_20260910`.
- Pan-cancer all-cell OT: `pancancer_all_cell_sourcecap_0p85_targetcap_0p95_20260902`.
- Splatter scaling benchmark: `/scratch/10119/ghzheng/OT_project/benchmark_scaling`.

## 8. Splatter simulation benchmark

### 8.1 Purpose

The simulation benchmark measures population-level anomaly detection, directional rejection accuracy, robustness to batch and gene perturbation, and single-fit OT runtime as the number of cells increases.

### 8.2 Simulation grid

| Component | Values |
|---|---|
| Cell counts | 1,000; 5,000; 10,000; 20,000 |
| Batch condition | none; mild |
| Target gene perturbation | 0%; 5%; 10%; 20% |
| Perturbation magnitude | two-fold signed up/down perturbation |
| Scenarios | S1 extinction; S2 emergence; S3 source outlier; S4 bifurcation; S5 abundance shift |
| Replicates | 2 |

This gives 80 M4 cases: `4 sizes x 2 batches x 5 scenarios x 2 replicates`. Partial OT is run for the 20 `N=1000` comparator cases.

### 8.3 Simulation preprocessing

Every simulated pair uses:

`raw counts -> library-size normalization to 10,000 -> log1p -> top 500 HVGs -> gene-wise standardization -> joint PCA with 20 PCs -> squared Euclidean cost -> median-positive cost scaling -> OT`

The joint PCA is fitted once on the clean source-plus-target counts and then frozen. Every perturbed dataset is projected through the same HVGs, scaler, and PCA. Thus, perturbation doses and methods are compared in exactly the same representation and cost geometry.

### 8.4 Compared methods

- Traditional balanced OT.
- Vanilla UOT mass-deficit baseline.
- Fixed-cost M4-E and M4-R with balanced and UOT backbones.
- Null-calibrated M4-E and M4-R with balanced and UOT backbones.
- Partial OT comparator.

The current scaling runs use source and target rejection budgets of 0.15, fixed binary rejection cost `c = 0.5`, entropy regularization `epsilon = 0.1`, UOT marginal penalties `lambda_a = lambda_b = 1.0`, two calibration nulls, and two held-out validation nulls.

### 8.5 Benchmark outputs

Planned result figures and tables include:

- Population rejection heatmaps by scenario.
- Directional F1 heatmaps across batch and perturbation settings.
- Single-fit runtime scaling across cell counts.
- Method- and size-level metric tables.
- A final audit JSON and downloadable result bundle.

The reported single-OT fit time excludes normalization, HVG selection, PCA, cost construction, null calibration, file I/O, plotting, and scheduler waiting. Preprocessing and calibration are recorded separately where applicable.

### 8.6 Current simulation status

- The original scaling submission did not run because `splatter` was unavailable.
- The TACC repository-local R library was then installed successfully.
- Installed simulation package: Splatter 1.34.0, together with the required Bioconductor dependencies.
- Installation job `994564` completed successfully on `gg`.
- The benchmark root is `/scratch/10119/ghzheng/OT_project/benchmark_scaling`.
- The first 32-thread submission completed all 20 `N=1000` cases but was stopped during `N=5000` because each GG allocation supplied a full 144-core node while the workflow used only 32 threads.
- The replacement configuration uses 144 threads per case, nine independent case workers, and up to nine concurrent GG nodes. All sizes will be rerun so runtime comparisons use identical resources.
- Single-fit runtime is elapsed wall-clock time measured with `time.perf_counter`; aggregate process CPU time is retained separately.

Submission command:

```bash
cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT

conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot

export CONFIDENCEOT_SCALING_ROOT=/scratch/10119/ghzheng/OT_project/benchmark_scaling
export CONFIDENCEOT_WORKER_COUNT=9
export CONFIDENCEOT_MAX_PARALLEL=9

bash scripts/tacc/submit_splatter_scaling.sh
```

## 9. Current interpretation

1. ConfidenceOT can generate pair-specific compatible and rejected cell states and useful continuous confidence outputs.
2. All-cell transition matrices are suitable as a structural validation of cell-type matching.
3. Malignant-only gates identify biologically distinct primary-cell programs, but the current direction is not equivalent to metastatic potential.
4. Strong inflammatory/invasive signals in source-rejected cells are real observations in the fitted partition, not evidence that the software labels were accidentally reversed.
5. Post-QC filtering does not by itself resolve the biological-direction issue.
6. M4-E gate initialization sensitivity is currently the most important algorithmic limitation for cell-level interpretation.
7. M4-R frequently fails calibration/terminal validity in these datasets and should not be used for the primary biological claim.
8. Dataset annotation and epithelial-state contamination are major possible confounders.
9. The next defensible result should emphasize robustness across initialization, cost/cap settings, patients, exact pairs, and independent datasets.

## 10. Immediate next steps

1. Submit and complete the Splatter scaling benchmark on `gg`.
2. Verify all 80 M4 cases and 20 Partial OT comparator cases before final plotting.
3. Finalize DEG comparisons across feasible M4-E initializations rather than choosing the initialization with the most favorable biology.
4. Quantify patient and repeated-primary-sample influence using patient-aware summaries or one-pair-per-patient sensitivity analysis.
5. Complete dataset-specific replication summaries for GSE181919 and GSE225857 with explicit annotation caveats.
6. Use neutral labels (`source-retained`, `source-rejected`, or `compatible`, `restricted`) until independent evidence supports a metastatic-potential interpretation.
7. Reserve TCGA survival analysis for a stable, prespecified signature that survives the robustness checks above.
