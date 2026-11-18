# ConfidenceOT biological application search: literature and candidate applications

**Date:** 2026-11-17

**Question:** which existing biological applications can be studied with discrete, rejection-aware ConfidenceOT, and
what can the rejection step add?

**Sources:**
- paper full texts on PMC / Europe PMC / arXiv;
- GEO FTP listings, Dryad and the pertpy data registry for the data;
- accessions checked against each paper's data statement.

Anything not checked is marked UNVERIFIED.

**Scope rules:**
- No continuous velocity field, SDE or landscape is learned.
- Dynamic-OT papers are read only for whether their application can be posed as a source -> target matching task.

## 1. Discrete OT papers (10)

| # | Paper | Biological question | Real data | How OT is used | Validation | Populations on one side only | Fit for rejection-aware discrete OT |
|---|---|---|---|---|---|---|---|
| 1 | Waddington-OT, Schiebinger et al., Cell 2019, doi:10.1016/j.cell.2019.01.006 | Ancestors / fates in MEF -> iPSC reprogramming | GSE122662, mouse, ~315k profiles (251k after QC), 39 time points over 18 d; Dox d0–8, then 2i or serum | Day t -> t+1, entropic **unbalanced** (ε 0.05, λ1 1, λ2 50), growth from proliferation / apoptosis signatures; fate probabilities, ancestor trajectories | Held-out middle time point vs batch-to-batch baseline; wet lab (Obox6, Zfp42, GDF9 raise efficiency); no barcodes | Dying stromal arm; emerging iPSC / neural / extra-embryonic states (growth / death marginals) | Yes. Source rejection = no descendant at that gap, target rejection = no credible ancestor; no independent truth; large |
| 2 | moscot, Klein et al., Nature 2025, doi:10.1038/s41586-024-08453-2 | Scalable OT for time, space and spacetime; origin of delta / epsilon cells | Mouse embryo atlas 1.7M cells E3.5–E13.5; MOSTA; liver CITE-seq -> MERSCOPE; **pancreas multiome GSE275562** (~22k nuclei, E14.5–E16.5); GSE132188 | Unbalanced W (time), GW / FGW (space), growth / apoptosis marginals; transition matrices, driver genes | Agreement with TOME; known drivers; held-out genes; **NEUROD2 knockout** in hiPSC islets | Apoptosis, new lineages (unbalanced τ) | Yes; pancreas E14.5 -> E15.5 is modest; validation markers + knockout |
| 3 | CellOT, Bunne et al., Nat Methods 2023, doi:10.1038/s41592-023-01969-x | Per-cell perturbation response, unseen patients / species | sciPlex3; **Kang PBMC IFN-β GSE96583**; glioblastoma panobinostat GSE148842; LPS cross-species E-MTAB-6754; LARRY GSE140802; 4i melanoma (in-house) | Neural **balanced** Monge map control -> treated (not a discrete coupling) | Held-out cells; leave-one-patient / species-out; MMD, r², markers | Dasatinib kills some states; death / proliferation **not modelled** | Yes as discrete control -> treated; Kang = natural negative control (all types on both sides) |
| 4 | CINEMA-OT, Dong et al., Nat Methods 2023, doi:10.1038/s41592-023-02040-5 | Causal per-cell treatment effect after removing confounders | Dryad doi:10.5061/dryad.4xgxd25g1: airway organoids mock / RV / CSE / RV+CSE (26,420 cells, raw counts); PBMC cytokines d2 / d7; sciPlex4; AD cortex GSE138852 | ICA confounder space, entropic **balanced** OT control -> treated; CINEMA-OT-W reweights abundance | Mostly qualitative (ISG induction, CSE biology) | Partly (reweighting) | Yes; rejection = treatment-specific states (e.g. infected cells); scripted Dryad download blocked |
| 5 | LineageOT, Forrow & Schiebinger, Nat Commun 2021, doi:10.1038/s41467-021-25133-1 | Use lineage trees to improve couplings | **C. elegans GSE126954** (Packer), 86,024 cells, 130–450 min; simulations | Lineage-shifted later cells, entropic OT; growth via marginals | **Invariant C. elegans lineage = true coupling** | Cells sampled before birth removed; death not modelled | Yes; only real data with an exact true coupling; rejection = lineages that end or are born in the interval (inference) |
| 6 | moslin, Lange et al., Genome Biol 2024, doi:10.1186/s13059-024-03422-4 | Couple lineage-traced cells across time | C. elegans ABpxp (6,476 cells, 7 points); **zebrafish heart regeneration** (Hu et al. 2022, LINNAEUS scars, ~44k cells, ctrl / 3 / 7 dpi; GSE159032 per ZFIN, UNVERIFIED in moslin text) | Fused GW (expression + lineage), unbalanced τa | C. elegans reference; zebrafish cell-type persistence, known fibroblast origins | Injury fibroblasts absent in ctrl (unbalanced τ) | Yes, strong: injury-emergent target-only states with scar support |
| 7 | uniPort, Cao et al., Nat Commun 2022, doi:10.1038/s41467-022-35094-8 | Integrate heterogeneous single-cell / spatial data | PBMC multiome; spleen RNA / ATAC; MERFISH; PDAC GSE111672 | Minibatch **unbalanced** OT in a VAE latent | ARI / F1, FOSCTTM | **Explicit held-out cell types removed from one modality** | Benchmark design for rejection (constructed one-sided types); cross-modality, not a biological contrast |
| 8 | Pamona, Cao, Hong & Wan, Bioinformatics 2021, doi:10.1093/bioinformatics/btab594 | Partial alignment with dataset-specific cells | sc-GEM, scNMT, SNARE-seq, PBMC RNA / ATAC (accessions UNVERIFIED) | **Partial GW**, shared mass s set by user or estimated | Label transfer, FOSCTTM | BJ fibroblasts removed from one side | Conceptually closest (partial transport), but needs s; modalities not conditions |
| 9 | PASTE, Zeira et al., Nat Methods 2022, doi:10.1038/s41592-022-01459-6 | Align adjacent spatial slices | DLPFC Visium, cSCC, HER2+ BC, spinal cord | **Balanced** FGW | Layer / cluster agreement | Not handled | Weak (spots outside overlap) |
| 10 | DeST-OT, Halmos et al., Cell Systems 2025 (bioRxiv 10.1101/2024.03.05.583575) | Spatiotemporal alignment; growth / death without gene sets | Axolotl brain, mouse kidney development (accessions UNVERIFIED) | **Semi-relaxed** FGW; source mass excess = growth / death | Growth distortion vs cell-type growth | Source-side growth / death is the output | Yes conceptually; no independent truth |

## 2. Fluid / dynamic / stochastic OT papers (9) and convertibility

| # | Paper | Real data | What is interpreted | Validation | What a discrete coupling can answer / what needs continuity |
|---|---|---|---|---|---|
| 11 | DiffusionOT, Liu, Zhou, Nie, Li, Sci Adv 2026, doi:10.1126/sciadv.aeb4205 | A549 TGF-β EMT; LARRY GSE140802; zebrafish hair-cell regeneration GSE196211; MOSTA | Fate fractions, growth, EMT genes, in-silico KO / OE | Simulation, markers; **no held-out time point; LARRY barcodes not used** | Fate fractions and growth = d_t -> d_t+1 coupling; landscape, GRN, KO / OE need the dynamic model |
| 12 | TrajectoryNet, Tong et al., ICML 2020, arXiv 2002.04461 | Embryoid body (Mendeley 10.17632/v6n743h5ng.1, ~16k cells, 5 windows); mouse cortex (not public) | Branch genes, growth | Leave-one-time-point-out EMD: dynamic 0.784 vs static OT 0.821 | Growth is fitted to a discrete unbalanced OT solution; interpolation almost matched by static OT |
| 13 | TIGON, Sha et al., Nat Mach Intell 2024, doi:10.1038/s42256-023-00763-w | LARRY d2 / 4 / 6; **A549 TGFB1 EMT (3,133 cells, 0 h–7 d)**; iPSC -> cardiomyocyte qPCR | Growth genes, EMT GRN, cell communication at unmeasured times | **Clonal growth vs inferred growth** (Spearman 0.44); clonal fate probabilities | Clonal growth and fate tests are discrete-coupling questions; GRN and unmeasured-time states need continuity |
| 14 | PRESCIENT, Yeo et al., Nat Commun 2021, doi:10.1038/s41467-021-23518-w | LARRY; Veres β-cell GSE114412 | Fate bias; TF perturbation screens | **Clonal fate bias** r 0.40, AUROC 0.725 (clonal ceiling 0.487 / 0.771) | d2 -> d6 fate bias = a coupling; perturbation screens need the generative model |
| 15 | MIOFlow, Huguet et al., NeurIPS 2022, arXiv 2206.14928 | EB; **AML chemotherapy, SPLINTR GSE161676** (scRNA GSE186305) | Surviving-cell programme (SLPI, MYB, B2M) | EB held-out time points; AML qualitative, barcodes unused | "Which cells survive chemotherapy" = source-side rejection problem; SPLINTR barcodes could validate |
| 16 | DeepRUOT, Zhang, Li, Zhou, ICLR 2025, arXiv 2410.00844 | LARRY, A549 EMT (sizes UNVERIFIED) | Growth, bifurcation, landscape | W1 / W2 at later times | Growth and false-transition removal = what discrete unbalanced OT does |
| 17 | scNODE, Zhang et al., Bioinformatics 2024, doi:10.1093/bioinformatics/btae393 | Zebrafish SCP162; Drosophila GSE190149; reprogramming GSE122662 | Least-action path genes; OE | Interpolation **and extrapolation** of held-out points | Interpolation only = coupling; **extrapolation needs a dynamic model** |
| 18 | PI-SDE, Jiang & Wan, Bioinformatics 2024, doi:10.1093/bioinformatics/btae400 | Veres GSE114412; LARRY | Potential, Mo / Neu split | Leave-out W distance; clonal fate not used | Mo / Neu split = d2 -> d4/6 coupling; potential needs continuity |
| 19 | JKOnet, Bunne et al., arXiv 2106.06345 | EB, 15,150 cells | Energy of a JKO flow | One-step / all-step W | One step ≈ coupling; energy learning is continuous by construction |

**Pattern:**
- **What a coupling between adjacent snapshots answers:**
  - growth;
  - fate probabilities;
  - clonal fate bias;
  - held-out-midpoint interpolation.
- **What needs a dynamic model:**
  - extrapolation;
  - landscapes;
  - GRNs;
  - expression at unmeasured times;
  - in-silico perturbation screens.
- **Real lineage barcodes as validation:** only TIGON and PRESCIENT. All others rely on distributional distances or markers.
- **One-sided populations:**
  - Every discrete method above either fixes the marginals (PASTE, CINEMA-OT, CellOT) or lets mass vary softly (unbalanced, semi-relaxed).
  - Only uniPort and Pamona test held-out one-sided populations explicitly, and those tests are constructed and cross-modality.

## 3. Candidate applications

Fields:
- paper;
- question;
- data and download;
- size;
- original OT use;
- ConfidenceOT source / target;
- what rejection could mean;
- validation;
- pipeline reuse.

**A. LARRY in vitro haematopoiesis.**
- **Paper:** Weinreb et al., Science 2020; used by CellOT, PRESCIENT, TIGON, DiffusionOT and PI-SDE.
- **Question:** do day-2 progenitor states predict clonal fate?
- **Data:**
  - GSE140802 `stateFate_inVitro`, 130,887 cells, days 2 / 4 / 6;
  - 5,864 clones, 1,816 spanning day 2 and later;
  - counts recovered as integers (`larry_20261112/01_audit.py`).
- **Source / target:** day-2 cells -> day-4 or day-6 cells of the same starting population (LSK, LK).
- **Rejection:**
  - source: progenitors whose expression has no counterpart later;
  - target: later states with no day-2 counterpart.
  - A source clone with no sampled descendants is not proof of absent progeny (sampling).
- **Validation:** **clonal barcodes**, giving clone mass, fate prediction against a label-permutation null, and Neu vs Mono.
- **Reuse:** direct.
  - TACC run of 4 units × 2 configurations already submitted (`larry_20261112/`);
  - local 3,000-cell pilot here.

**B. Kang PBMC IFN-β.**
- **Paper:** Kang et al., Nat Biotechnol 2018; CellOT, scGen, CINEMA-OT.
- **Question:** cell-type-specific response to IFN-β.
- **Data:**
  - GSE96583 batch 2: `GSM2560248_2.1.mtx.gz` (ctrl) and `GSM2560249_2.2.mtx.gz` (stim), with barcodes, genes and `GSE96583_batch2.total.tsne.df.tsv.gz` labels;
  - 12,315 ctrl / 12,358 stim singlets, 8 cell types, 8 donors.
- **Source / target:** control -> stimulated (6 h).
- **Rejection:**
  - no cell type is gained or lost in 6 h, so rejection is false rejection (negative control);
  - with one cell type removed from one side (constructed, as in uniPort), rejection should find it.
- **Validation:**
  - **cell type is preserved per cell**, giving a true label-level coupling;
  - the removed type is a true one-sided population.
- **Reuse:** direct (manifest route).

**C. EMT time course, A549 TGF-β1.**
- **Paper:** Cook & Vanderhyden, Nat Commun 2020; used by TIGON and DiffusionOT.
- **Question:** context-specific EMT programmes; reversibility after withdrawal.
- **Data:**
  - GSE147405 `GSE147405_A549_TGFB1_TimeCourse_{UMI_matrix,metadata}.csv.gz` (10 MB);
  - 3,568 cells: 0d 577, 8h 885, 1d 788, 3d 754, 7d 129, 8h_rm 238, 1d_rm 155, 3d_rm 42.
- **Source / target:**
  - adjacent time points of induction;
  - withdrawal arms;
  - 0d vs 3d_rm.
- **Rejection:** cells whose state is left behind or newly entered at a step. Mesenchymal states appear; epithelial states disappear.
- **Validation:**
  - markers and the authors' pseudotime only;
  - no lineage.
  - 3d_rm has 42 cells.
- **Reuse:** direct.

**D. Reprogramming (WOT).**
- **Data:**
  - GSE122662;
  - hosted counts `https://exampledata.scverse.org/pertpy/schiebinger_2019_16day.h5ad` (379 MB; `age` D0–D16 + iPSCs; serum vs 2i from `GSM`, because the `perturbation` column mislabels serum as control).
- **Source / target:** adjacent days in the 2i branch.
- **Rejection:**
  - target: emerging iPSC / neural / trophoblast;
  - source: dying stromal cells.
- **Validation:** WOT cell sets (Google Drive) and marker programs; no lineage.
- **Reuse:** direct on TACC (size).

**E. sciPlex2 drug response.**
- **Paper:** Srivatsan et al., Science 2020.
- **Data:**
  - GSM4150377;
  - hosted counts `https://exampledata.scverse.org/pertpy/srivatsan_2020_sciplex2.h5ad` (145 MB);
  - A549 with Dex / Nutlin / SAHA / BMS at 0–100 µM; vehicle = dose 0.
- **Source / target:** vehicle -> high dose of one drug.
- **Rejection:** vehicle states depleted by the drug (cytostatic / cytotoxic), and drug-induced states.
- **Validation:** dose series (monotone with dose); perturbation identity.
- **Reuse:** direct.

**F. Lineage-truth candidates, not piloted here:**
- **C. elegans embryo, GSE126954:** exact invariant-lineage coupling.
- **Zebrafish heart regeneration, Hu et al. 2022:** scars plus injury-emergent fibroblasts.
- **SPLINTR AML chemotherapy, GSE161676 / GSE186305:** expressed barcodes; survival as source-side rejection.

These are the strongest for validating rejection against biological truth. Each needs a data-format audit first.
