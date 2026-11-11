# RNA vs CRISPR lineage relationships: interim results (2026-11-11)

Plan: `PLAN_v2.md` (fixed 5a420df). 47 of 49 units compared; 3724 S1 and the 3724 pooled case are waiting for their
CRISPR-OT fits. Configuration: Splatter L5 best (logCPM + PCA in the tables, same picture). Tables are in
`results/rna_lineage_discordance_20261110/summary/` and `deg_gsea_v1{,_ds}/`; figures in `figures/l5best/`.

## QC

**Raw matrix (view C).** The allele-table states equal the authors' integer character matrices in all 3,849 columns
(`02_matrix_concordance.py`).

**Lineage resolution** (fixed rule):
- KP: 5 of 11 families pass. 3724 fails on distinct event sets (8%, only about 3 intBCs). 3454's L1 metastasis shares
  early edits, so founder exclusion leaves about 1 event per cell.
- Quinn: 51 of 148 clones pass.
- Every lineage is analysed, with its flag.

**CRISPR-OT (view B):**
- 47 fits; all converged.
- M4-E calibration validity is flagged False in 16, which are kept.
- Non-evaluable cross pairs are ≤ 0.1%. Quinn: 87% of cross pairs belong to different clones (D = 1).
- Same-family units are mostly retained, with exceptions:
  - 3457 T1 source 5%;
  - 3724 L1 pair source 35%;
  - 3460 N4 66%;
  - 3508 N5 67%.

**Gene identifiers.** Quinn genes are Ensembl ids, mapped to symbols with the GRCh38 table: 79-82% of ids, 97.5-98% of
hallmark genes.

## Observed

1. **The two OTs give different correspondences.**
   - Gate κ ≈ 0. Per-target total variation 0.71-0.83. Top-15 source overlap 0-2%.
   - RNA-OT top sources: edit agreement 0.10-0.34, RNA similarity 0.45-0.62.
   - CRISPR-OT top sources: edit agreement 0.47-0.86, RNA similarity 0.01-0.13.
2. **RNA-OT mass on RNA-close / lineage-far pairs is 0.23-0.40**, against 0.07-0.09 for the independent coupling. On
   lineage-close / RNA-far pairs it is 0.00-0.01, against 0.03-0.09. CRISPR-OT is the mirror: 0.10-0.16 on
   lineage-close / RNA-far pairs.
3. **RNA and lineage agree more than chance** in nearly every mouse (divergence and convergence below the
   within-site permutation null).

   | Mouse / experiment | Divergence, observed | Null |
   |---|---|---|
   | Quinn 100k | 0.17 | 0.51 |
   | Quinn 5k | 0.20 | 0.32 |
   | KP 3457 | 0.07 | 0.23 |
   | KP 3724 | 0.04 | 0.25 |

   Near-null mice are the low-resolution ones (3451, 3465, 3513). Cells exceeding the null by ≥ 0.25 are 0.2-7%.
4. **By KP state** (authors' annotation; unchanged without 3724):
   - strongest lineage–RNA coupling: Mesenchymal-1 / -2, Apc Early, Early EMT-1, Gastric-like;
   - near chance, i.e. occupied by many lineage branches: Pre-EMT (18.6k cells, 9 mice), Endoderm-like, Apc
     Mesenchymal-1 (Met), Early EMT-2, Late Gastric.
5. **Metastases are less lineage-coupled than primaries.** Divergence excess:
   - Mesenchymal-2: −0.20 primary, +0.01 metastasis;
   - Early EMT-1: −0.16 primary, −0.01 metastasis.
6. **Gates:**
   - RNA-OT-rejected primary cells have lower divergence (AUROC 0.28-0.37): lineage-and-state-coherent clones without a
     metastatic counterpart.
   - CRISPR-OT-rejected metastatic cells have high convergence (AUROC 0.93-0.95).
7. **Depth:** divergence is unrelated to it (ρ −0.01); convergence is weakly related (ρ −0.12).
8. **DEG / GSEA** (60 contrasts: discordant vs concordant cells of the same lineage, site and side; raw and
   depth-thinned counts):
   - Raw NES followed detected-gene differences (ρ 0.64); thinning reduced this to 0.34. The two versions agree:
     NES ρ 0.83, 81% same sign.
   - Under the strictest reading (all mice, both versions, |NES| ≥ 1), only KP convergent cells pass: apoptosis,
     IFN-α, apical junction / surface and E2F lower. That rests on 2 mice and 5 contrasts.
   - Partial trends:
     - KP divergent: OXPHOS, DNA repair, allograft, EMT higher (3 of 4 mice); bile acid, KRAS-DN lower.
     - Quinn divergent: EMT higher (3 of 4 mice, 85% of contrasts); E2F / MYC lower (3 of 4).
     - Quinn convergent: p53 pathway higher (4 of 4 mice, both versions); UPR lower (4 of 4).

## Model inference

- RNA-OT correspondences follow transcriptional state across lineage boundaries, so they cannot be read as lineage
  correspondences.
- The two gates select different cell kinds.

## Possible biology (not tested)

- Pre-EMT / Endoderm-like / Early EMT-2 / Apc Mesenchymal-1 (Met) may be states that many clones enter (plastic or
  convergent). This agrees in direction with Yang et al. 2022.
- Mesenchymal states look lineage-determined (heritable, clone-specific).
- Weaker coupling in metastases may reflect convergence during metastatic adaptation.
- Divergent cells lean towards EMT / OXPHOS; convergent Quinn cells lean towards p53 / ROS.

## Not yet supported

- Any program-level claim. The GSEA effects are small, few pass the strict reading, KP has few contrasts, and the
  cells are pseudo-replicates within one tumour per site.
- Direction or ancestry: the samples are taken at one time point.

## Pending

- 3724 S1 and the 3724 pooled case: CRISPR-OT, comparison and DEG.
- The lineage filter, to be decided after review.
- logCPM + PCA as a configuration check.
