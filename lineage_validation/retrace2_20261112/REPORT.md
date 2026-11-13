# RETrace2 MSH2 mouse: case report (2026-11-12)

Plan: `PLAN.md` (75bd6a5) plus its dated post-hoc coverage section. Code: `01_state_lineage.py`,
`02_coverage_and_ot.py`, `03_figures.py`. Outputs: `results/retrace2_20261112/{prep, analysis, ot, figures}`.

This is a **case report**:
- one mouse;
- 216 plate-sorted cells, 152 after the authors' QC;
- no RNA.

It tests whether our RNA-vs-lineage framework transfers to a homopolymer lineage with a methylation state. It does
not support general claims.

## Data (observed)

- **Cells:** 216 single cells, each with a homopolymer capture library and a methylation library (432 SRA
  libraries). They come from one female Msh2-deficient mouse: brain (anterior / posterior cortex), kidney (left /
  right) and liver (three lobes). 152 pass the authors' thresholds; liver and kidney lose about half (r1).
- **Lineage:**
  - 56,337 homopolymer loci; 7.6k shared per cell pair.
  - The authors' distance has a narrow range (median 0.121; 5-95%: 0.099-0.148).
  - Their tree has median bootstrap 0.29.
- **State:**
  - methylation is available only as each cell's dissimilarity to 24 brain cell-type references, with no per-cell
    CpG matrix;
  - brain cell types: 47 of 72 brain cells assigned, mostly P-Glut;
  - kidney and liver: no cell types.

## Results

1. **The lineage distance tracks coverage** (r2). Distance vs shared loci: Spearman −0.68. Brain cells capture about
   twice the loci (median 31.8k vs 15.0k kidney / 13.0k liver), so brain-brain pairs look closer.
   - After regressing on log(shared loci), the organ-level distance structure almost disappears (residual
     brain-brain −0.004, kidney / liver +0.003).
   - The coverage-residual tree has only one 10-20-cell clade under the authors' rule.
2. **Lineage vs organ, using the authors' clades** (r3, r4):
   - Cramér's V 0.45 against a label-permutation null of 0.24 (p = 0.001). Clades are enriched for brain (Hill z
     −5.5) and liver (z −4.4).
   - Within organ, tissue is at the null: V 0.34 vs 0.32 (p = 0.13).
   - Brain cell type is near the null: V 0.38 vs 0.34 (p = 0.13). The one concentrated type, NP-CT-L6b-Glut, has
     7 cells.
   - Because of result 1, the organ association cannot be separated from coverage with these data.
3. **Lineage vs methylation similarity:**
   - The Mantel-type Spearman correlation of the two distances is ≈ 0 within organs and within tissues (−0.07,
     null −0.11).
   - The view-C divergence is above its within-tissue null in brain (0.17 vs 0.12) and at the null elsewhere.
4. **Two OTs on the same pairs** (8 pairs; r6, r7):
   - **Gates:**
     - Homopolymer-OT rejects almost nothing, even across organs; retained fractions are 0.88-1.00 per side.
     - Methylation-OT also retains everything within organs, but rejects cells across organs: brain posterior →
       kidney right retains 0.64 / 0.87; brain anterior → liver left 0.89 / 1.00; liver left → kidney right
       0.94 / 0.93.
   - **Calibration:** 11 of 24 fits are flagged `m4e_inference_valid` False (marked in r6). With 6-36 cells per side
     the within-side split null is small.
   - **Coupling:**
     - Top-5 source overlap between homopolymer-OT and methylation-OT is at chance in every pair (e.g. brain
       anterior → posterior 0.13 vs 0.14).
     - Common-support TV is 0.47-0.68.
     - Homopolymer-OT before and after coverage correction agree closely (overlap 0.86-0.99).

## Reading

**Observed:**
- Methylation distinguishes organs and drives the only cross-organ rejections.
- The homopolymer distance does not separate organs once coverage is accounted for, and the two OTs choose unrelated
  partners.

**Possible interpretation (not tested):**
- At this resolution, homopolymer lineage within one Msh2 mouse carries little organ or cell-type information beyond
  coverage.
- Organs share early lineage, which is expected developmentally, and the remaining per-cell signal is mostly
  genotyping noise.

**Not supported:**
- Any claim of lineage–state coupling or decoupling, because there are 152 cells, one mouse and a weak tree.
- Any claim about brain cell types, because there are 47 assigned cells and a reference that covers brain only.

## Limits

- Coverage differs between organs. The authors' tree and clades inherit that difference.
- Methylation state is a projection onto brain references, so it is weak for kidney and liver.
- No RNA, so a methylation state stands in for the RNA state of our other analyses.
- The OT calibration uses split nulls of 3-18 cells per half.
