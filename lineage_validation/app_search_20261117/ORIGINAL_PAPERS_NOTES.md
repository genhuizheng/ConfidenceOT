# Original data papers: aims, conclusions, evidence, limitations

**Reading:** full texts (PMC / Europe PMC), 2026-11-17.

**Purpose:** to decide which of each paper's biological conclusions a discrete source -> target matching (ConfidenceOT
with rejection) can test. Paraphrased; numbers are from the papers.

## 1. Weinreb et al., Science 2020 (LARRY, GSE140802)

**Citation:** doi:10.1126/science.aaw3381, PMC7608074.

**Aim:** link each progenitor's transcriptional state to its clonal fate; ask how much of fate a snapshot predicts and
whether trajectory inference recovers it.

### Design

- **Barcoding:**
  - lentiviral 28-mer barcode, read out by inDrops;
  - barcodes detected in 93% of GFP+ cells.
- **Sampling:**
  - cells are cultured 2 days (about 3 divisions);
  - half are profiled at day 2 and the rest later.
  - **The early state belongs to a sister of the cells whose fate is observed, never the same cell.**
- **In vitro:**
  - LK and LSK cells, day 2 -> day 4 / day 6;
  - 130,887 cells; 5,864 clones; 1,816 clones span early and late.
- **Other experiments:** in vivo transplant (week 1 / week 2), a cytokine split, and MDP / GMP profiling.

### Conclusions

1. **The landscape is continuous with extended bipotent boundaries.** Day-2 sisters are highly correlated (median
   R 0.846).
2. **Early fate-associated genes exist.** 447 DE genes in vitro (FDR 0.05); Ikzf2 marks the eos / mast fate.
3. **Fate is only partly predictable from state.**
   - The best day-2 classifier reaches 60% in vitro and 51% in vivo; the authors call these lower bounds.
   - Late sisters predict better: 76% vs 60%.
4. **Heritable information is hidden from scSeq.** Sisters in separate wells share their fate combination 70% of the
   time (22% by chance), and the authors infer "fate priming or pre-commitment".
5. **Monocytes arise by two routes:** a GMP-like (neutrophil-like) route and an MDP-like (DC-like) route.
6. **Trajectory methods miss early fate.**
   - For neutrophil vs monocyte fate, PBA, **WaddingtonOT** and FateID correlate with clonal fate bias at R < 0.26.
   - Held-out clones reach R = 0.5 (the upper bound).
   - "Fate choice occurs earlier than is detected by state-of-the-art algorithms."
7. **Pseudotime orders progression consistently** (sister R ≥ 0.89).
8. **State distance and clonal coupling agree** in vitro (r 0.93) and less in vivo (r 0.58).

### Validation

- held-out clones (the prediction ceiling);
- sisters split across wells and mice;
- MDP / GMP profiling;
- native barcoding.

### Limitations stated by the authors

- nothing faster than one cell cycle;
- culture or transplant only;
- chromatin / protein / niche hidden from scSeq;
- undersampling of clones.

### What matching can test against the paper

**(a)** Does the d2 -> d4/6 coupling put mass on clonal sisters above baselines? Compare with WOT's R < 0.26 and the
R = 0.5 ceiling for Neu vs Mono fate bias.

**(b)** Do rejected day-2 cells concentrate where fate is least predictable?

**(c)** Do the two monocyte routes (GMP-like vs MDP-like progenitors) appear as distinct couplings?

## 2. Kang et al., Nat Biotechnol 2018 (demuxlet, GSE96583)

**Citation:** doi:10.1038/nbt.4042, PMC5784859.

**Aim:** pool donors and demultiplex them by SNPs; shown on IFN-β responses and cell-type eQTLs.

### Design

- **Cells:** 8 lupus donors' PBMCs.
- **Treatment:** IFN-β 100 U/mL for 6 h.
- **Pools:** control 12,138 singlets; stimulated 12,167 singlets.
  - Control and stimulated are **separate pools of different cells (unpaired).**
- **Labels:** donor (genotype), condition, and cell type checked against sorted bulk and flow.

### Conclusions

1. **Demultiplexing is accurate:** 97% of singlets assigned, 92% of doublets detected.
2. **IFN-β drives cell-type-specific responses.**
   - 3,055 genes are DE in at least one type.
   - A 401-gene antiviral cluster rises in all leukocytes.
   - A 767-gene cluster is **myeloid-specific** (chemokine / SLE pathways).
   - Antibacterial and NK-cytotoxicity genes go down.
3. **Cell-type proportions do not change with stimulation** (flow R 0.88).
4. **Donors vary beyond composition.** Donor-variable genes are mostly in classical monocytes and CD4 T cells; in
   stimulated monocytes they are enriched for IFN response.
5. **Cell-type-specific eQTLs** (32 cis-eQTLs, 22 in a single type).

### Limitations stated by the authors

- one stimulus at one time point;
- optimal pool size about 20 donors;
- solid tissues are harder.

### What matching can test against the paper

**(a)** Does a ctrl -> stim coupling preserve cell identity, given that proportions are unchanged?

**(b)** Is the expression shift between matched cells largest in myeloid cells (the 767-gene myeloid programme) and
shared antiviral in all types?

**(c)** Is rejection near zero when no population is gained or lost?

## 3. Cook & Vanderhyden, Nat Commun 2020 (GSE147405)

**Citation:** doi:10.1038/s41467-020-16066-2, PMC7195456.

**Aim:** is there a shared EMT programme across cell types and inducers?

### Design

- **Cell lines:** A549, DU145, MCF7, OVCA420.
- **Inducers:** TGFB1, EGF, TNF, giving 12 time courses.
- **Time points:** 0d, 8h, 1d, 3d, 7d, then withdrawal at 8h_rm, 1d_rm, 3d_rm (labels confirmed in the metadata file).
- **Multiplexing:** MULTI-seq.
- **Size:** 103,999 cells including a 22-inhibitor screen.
- No lineage barcodes.

### Conclusions

1. **EMT is highly context-specific.** Mean Jaccard of response genes is 0.22; the same line under different inducers is
   more similar than different lines under one inducer.
2. **Hallmark EMT is inconsistent.** SNAI1, CDH1 and CDH2 rarely change; FN1 and CD44 change often.
3. **A small conserved programme exists.**
   - 86 genes up and 17 down (including KRT8 / 18 / 19) in at least 8 of 12 conditions.
   - Enriched for apoptosis inhibition, ECM disassembly and migration.
4. **TF activity is context-specific:** AP-1, RELB, ATF4, SOX4 and KLF6 are active; ELF3 and MYBL2 go down.
5. **EMT is multistep and reversible.**
   - Psupertime shows "combinations of discrete transcriptional events".
   - **Withdrawal gives "near-complete reversion after as few as 3 days".**
6. **Signalling dependencies differ by context.** TGFBR1 inhibition blocks TGFB1 EMT; MEK inhibition blocks it in A549
   and MCF7 but not DU145.

### Validation

- ATAC-seq (OVCA420);
- replicate concordance;
- the 86 genes checked in tumour scRNA-seq;
- the inhibitor screen.

### Limitations stated by the authors

- four lines only;
- transition may continue after day 7;
- intermediate and polar states need definition;
- non-responding subpopulations are not discussed.

### What matching can test against the paper

**(a)** Multistep ordering: which steps have cells with no counterpart, i.e. discrete events?

**(b)** "Near-complete reversion": do 3d_rm cells match back to 0d cells with little rejection? (3d_rm has 42 cells in
A549 TGFB1.)

**(c)** Is the matched shift carried by the conserved 86 / 17 genes?

## 4. Srivatsan et al., Science 2020 (sci-Plex, GSE139944)

**Citation:** doi:10.1126/science.aax6234, PMC7289078.

**Aim:** single-cell transcriptomes for chemical screens via nuclear hashing; HDAC inhibitor mechanism.

### Design

- **Pilot (sciPlex2):** A549 with Dex, Nutlin-3a, BMS-345541 and SAHA, 7 doses, 24 h.
- **Main screen (sciPlex3):**
  - 188 compounds × 4 doses × 3 cell lines (A549, K562, MCF7), 24 h;
  - 649,340 cells.
- **Time course:** 72 h for 48 compounds.
- **Labels:** compound, dose, line; per-well counts serve as viability.

### Conclusions

1. **Hashing is accurate.** Known targets are recovered: GR targets for Dex, p53 targets for Nutlin.
2. **The screen is reproducible and selective.**
   - Replicate r 0.99.
   - 52 compounds (27%) reduce viability by ≥ 50% at the top dose.
   - Bosutinib is selective for K562.
3. **Single cells separate composition from regulation.** Many drugs reduce the proliferating fraction.
4. **HDAC inhibitors give the strongest response, along a shared "pseudodose" trajectory.**
   - Some responses are homogeneous, others heterogeneous.
   - 942 pseudodose genes are shared across lines.
5. **Dose sets magnitude, not speed.** At 72 h most HDAC inhibitors do not push cells further along the trajectory.
6. **Acetate-reservoir model.** ACLY / ACSS2 and the metabolite-rescue experiments support it; the authors call it "an
   alternative model, albeit not a mutually exclusive one".

### Limitations stated by the authors

- a proof of concept;
- three lines;
- one main time point;
- low UMI counts.

### What matching can test against the paper

**(a)** Vehicle -> dose: rejection of vehicle states depleted by the drug (the proliferating fraction) vs drug-induced
states, increasing with dose.

**(b)** Homogeneous vs heterogeneous HDAC inhibitor responses.

## 5. Schiebinger et al., Cell 2019 (WOT reprogramming, GSE122662)

**Citation:** doi:10.1016/j.cell.2019.01.006, PMC6402800.

**Aim:** which cell classes arise during MEF -> iPSC reprogramming, the paths to iPSC and other fates, and the factors and
signals along them; tested experimentally.

### Design

- **Cells:** secondary MEFs, dox-inducible OKSM and an Oct4-EGFP reporter.
- **Phases:** dox from day 0 to 8, then 2i or serum.
- **Main series:**
  - 39 time points about 12 h apart (every 6 h on days 8–9) over 18 days;
  - 259,155 cells, 251,203 after QC.
- **Pilot:** 65,781 cells, 10 time points.
- No lineage barcodes: ancestry comes only from OT couplings.

### Conclusions

1. **Early split into stromal and MET fates.**
   - Ancestors of the two fates differ by 30% at day 3 and 60% at day 6; the divergence is gradual.
   - OKSM transgene level explains about 50% of the MET-vs-stromal fate ratio by day 2 and about 75% by day 5.
2. **Stromal cells look senescent.** They peak at days 10.5–11 and then decline through low proliferation and apoptosis.
3. **The iPSC path is a narrow bottleneck.**
   - iPSC ancestors: about 40% of cells at day 8.5; about 10% at day 10 in 2i; about 1% at day 11 in serum.
   - TF waves at days 9–10 (Nanog, Sox2, ...) and days 12–14 (Obox6, ...).
4. **Trophoblast-like cells** peak at about 20% of cells at day 12.5.
5. **Neural-like cells appear in serum only.** Their ancestors separate by day 9.
6. **Predicted paracrine signals:** Gdf9–Tdgf1 and Cxcl12–Dpp4 (stromal -> iPSC).
7. **Fates are indistinguishable until MET.** Ancestors of pluripotent, epithelial, trophoblast and neural cells cannot
   be told apart until after day 8.

### Validation

- held-out time points (about batch-to-batch level);
- **Obox6 / Zfp42 overexpression** raises efficiency about 2-fold;
- **recombinant GDF9** raises efficiency about 4–5-fold.
- Neural, trophoblast and the other paracrine claims were not tested experimentally.

### Limitations stated by the authors

- Markov model (no memory);
- cells assumed autonomous;
- lineage barcodes would help;
- fit all time points jointly.

**Later caveat:** Weinreb 2020 shows WOT misses the early Neu / Mo bifurcation (R < 0.26).

### What matching can test against the paper

**(a)** The stromal-vs-MET divergence (30% / 60%).

**(b)** Target-side rejection of iPSC / neural / trophoblast cells at their emergence.

**(c)** Source-side rejection of the declining stromal cells after day 11.

**(d)** The 2i vs serum difference in the iPSC bottleneck.

## 6. Packer et al., Science 2019 (C. elegans embryo, GSE126954)

**Citation:** doi:10.1126/science.aax1971, PMC7428862.

**Aim:** link transcriptomes to the invariant lineage; how well lineage predicts transcriptome; priming, TFs,
convergence.

### Design

- 10x; 86,024 cells.
- Embryo time estimated from bulk RNA-seq; 75% of cells fall in 210–510 min.
- 93% of cells annotated to 502 lineages / 112 terminal types.
- Lineage labels are **inferred annotations, not recorded barcodes.**

### Conclusions

1. **Lineage predicts transcriptome during gastrulation, then less so:** pseudo-R² 55% in AB7, 18% in AB9.
2. **Sisters diverge from primed parents:** 52% of parent–daughter trios show multilineage priming.
3. **Sister differences build up step by step:** TFs per cell rise from 1.5 to 14.
4. **Distant lineages converge,** e.g. IL1/2 neuroblasts from three symmetric lineage pairs.
5. **Body-wall muscle keeps its lineage history.**
6. **Some trajectories are discontinuous** at terminal divisions (glia, excretory).
7. **Left / right asymmetry is late:** ASEL / ASER diverge at about 650–700 min.

### Validation

Two independent annotation routes agree; EPiC reporters. No perturbation.

### Limitations stated by the authors

- after convergence, trajectories are hard to connect to progenitors, and naive UMAP misleads;
- continuous trajectory methods fail on abrupt transitions;
- pharyngeal neurons are missing.

### What matching can test against the paper

**(a)** Recovery of known parent–daughter relations between embryo-time bins, especially convergent (IL1/2) and
discontinuous (glia) cases where the authors say continuous methods mislead.

## 7. Hu et al., Nat Genet 2022 (zebrafish heart regeneration, LINNAEUS)

**Citation:** doi:10.1038/s41588-022-01129-5, PMC7613248; GSE159032 and GSE158919.

**Aim:** activated pro-regenerative fibroblast states after cryoinjury, their developmental origin, and Wnt control.

### Design

- uninjured, 3, 7 and 30 dpi;
- about 200k cells;
- LINNAEUS scars from early development (shared clone, not injury-time ancestry).

### Conclusions

1. **Three transient fibroblast states** (col11a1a, col12a1a, nppc) at 3 and 7 dpi, "virtually absent before injury and
   after regeneration".
2. **col12a1a ablation** gives a larger injury area at 30 dpi (P = 0.0027; n = 4 vs 3).
3. **col11a1a and col12a1a come from epicardium:** more than 90% of tree nodes containing them also contain epicardium.
4. **nppc and valve fibroblasts come from endocardium** (more than 80% of nodes); not full EndMT.
5. **Wnt inhibition** reduces the endocardial-derived states and enlarges the injury.

### Validation

ISH; Cre-lox (tcf21, fli1; n = 2–3); ablation; IWR-1; tree downsampling.

### Limitations stated by the authors

- col11a1a -> col12a1a rests on RNA velocity only;
- small n;
- fate after regeneration unknown.

### What matching can test against the paper

**(a)** Target-side rejection of the transient states at 3 dpi (no uninjured counterpart).

**(b)** Their re-matching or disappearance at 30 dpi.

**(c)** The epicardial vs endocardial source of the coupling vs the LINNAEUS node co-occurrence.

## 8. Fennell et al., Nature 2022 (SPLINTR AML)

**Citation:** doi:10.1038/s41586-021-04206-7. Paywalled, so this summary comes from the abstract and legends only.

**Aim:** non-genetic determinants of clonal fitness in AML; whether LSC clonal output shapes chemotherapy response.

### Design

- SPLINTR expressed barcodes;
- mouse MLL-AF9 models;
- serial transplants;
- chemotherapy vs vehicle;
- GSE161676 SuperSeries; chemo scRNA-seq GSE186305 (6 samples); chemo barcode-seq GSE186304.

### Conclusions

1. **Dominance is cell-intrinsic and heritable.**
2. **Dominant clones repress MHC-I / B2m and raise Slpi**, a signature present before transplant.
3. **Slpi is a functional regulator** (guides reduce fitness).
4. **Transcriptional heterogeneity goes with fitness.**
5. **LSC clonal output is heritable and sets chemosensitivity.** High- and low-output clones adapt differently, but both
   re-emerge from MRD with a higher LSC programme.

### What matching can test against the paper

(pending the rest of the report)

**Fennell, status:** the rest of the deep read was stopped by a safety classifier. The Discussion, limitations and the
chemotherapy regimen are not covered.

## 9. Bastidas-Ponce et al., Development 2019 (mouse pancreas endocrinogenesis, GSE132188)

**Citation:** doi:10.1242/dev.173849, read on the publisher's site.

**Aim:** an in vivo roadmap of endocrine-cell formation during the secondary transition.

### Design

- **Reporter:** Ngn3-Venus fusion knock-in, with no phenotype.
- **Sampling:** E12.5–E15.5, 36,351 cells (10,790 / 5,042 / 9,633 / 10,886).
- **Composition:** FACS-enriched NVF+ cells with NVF− epithelial cells added back 2:1, so the proportions are not tissue
  proportions.
- No lineage tracing: relationships come from clustering, PAGA and RNA velocity.

### Conclusions

1. **Ngn3 is transient** (Ngn3-low -> Ngn3-high -> Fev-high); 58 signature genes follow the same pattern.
2. **Ngn3-low progenitors come mainly from ductal cells,** and also from tip / trunk cells early (PAGA).
3. **Arx first appears in Fev-high cells.**
4. **α cells appear at E12.5; β, δ and ε at E13.5–14.5.** Velocity points toward α early and β late (no numbers given).
5. **Stage-specific Ngn3-high genes** suggest that subtype is specified at the Ngn3-high stage (qPCR, n = 1).

### Validation

Reporter fidelity only: no lineage tracing, perturbation or smFISH of the fate assignments.

### Limitations stated by the authors

- Do Ngn3-low cells revert?
- Fev-high unipotency is a hypothesis.
- Velocity assumptions have since been questioned (general reviews).

### What matching can test against the paper

**(a)** Stage -> stage: do progenitor and Fev-high cells map to α early and β late?

**(b)** Is there target-side rejection of newly appearing β / δ / ε cells?

**Caveat:** the FACS enrichment distorts mass.
