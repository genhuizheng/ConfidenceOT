# RETrace2 (MSH2 female mouse): methylation state vs homopolymer lineage (exploratory plan)

**Status:** fixed 2026-11-12, after the data audit below and before any association, concordance or OT was computed.

**Data:**
- Zenodo 17945194, `2025_RETrace2_MSH2_Female_Combined`.
- Read only. Nothing from the archive is executed, and the pickle is read with a restricted unpickler.
- Code: `lineage_validation/retrace2_20261112/`. Outputs: `results/retrace2_20261112/`.

**The analogue of our RNA vs CRISPR question.** RETrace2 has no RNA, so:
- the lineage view is the homopolymer character matrix;
- the state view is DNA methylation (the authors' per-cell dissimilarity to 24 brain cell-type references) plus the
  tissue and the authors' brain cell-type assignment.

## Audit (done before this plan)

**Cells:**
- 216 cells from 7 tissue samples:
  - brain: left anterior cortex 36, left posterior cortex 36;
  - kidney: left 18, right 36;
  - liver: left lobe 36, median lobe 36, right lobe 18.
- 152 pass the authors' `--min_targets 10000 --min_cpgs 3000` and are in their tree.

**Lineage evidence:**
- 56,337 homopolymer loci (HipSTR VCF).
- Mean 17.3k loci per cell; 7.6k shared per pair (minimum 1.6k).
- Authors' distance (`minComp_EqorNot`, `buildPhylo.distDict.pkl`): median 0.121, 5-95% range 0.099-0.148.
- Tree bootstrap support: mean 0.38, median 0.29. The lineage signal is weak relative to genotyping noise.

**State evidence:**
- **Methylation:** only the cell × 24-brain-reference dissimilarity matrix (`infer_celltype/pd_matrix`) and per-cell
  stats. No per-cell CpG matrix (the allc files are not in the record).
- **Brain cell types** (authors, brain-only rerun): 47 of 72 brain cells assigned, mostly P-Glut (30). Kidney and
  liver are not assigned.

## Definitions

**Cells.** The 152 tree cells. The attrition 216 → 152 is reported per tissue.

**Lineage distance D_L.**
- The authors' pairwise `minComp_EqorNot` distance, with its number of shared loci per pair.
- Our own character matrix is built from the VCF genotype bp-differences (GB field) for display and per-cell
  coverage only, never for the distance.

**Lineage clades.** The authors' rule: non-overlapping subtrees of 10-20 cells in their bootstrap tree, found by
recursive traversal from the root. Cells outside such a subtree are "unclustered".

**State:**
- **S1:** organ (brain / kidney / liver) and tissue sample (7).
- **S2:** brain cell type (authors' assignment, "Unassigned" kept), brain cells only.
- **S3, methylation profile:**
  - each cell's 24-vector of dissimilarities to the brain references;
  - each reference column z-scored across the 152 cells;
  - state distance D_M = 1 − Pearson correlation of two cells' vectors;
  - coverage (shared sites per reference) reported.

## Analyses

1. **Lineage–state association.**
   - Clade × organ, clade × tissue, and, within brain, clade × cell type.
   - Cramér's V and per-state clade Hill diversity / largest share.
   - Null:
     - labels permuted across all cells for organ;
     - within organ for tissue and for cell type;
     - 999 draws, seed 20261112.
   - Also the authors' hypergeometric clade-bias result, read from their notebook output, for comparison only.
2. **Lineage vs methylation concordance** (as view C).
   - Per cell, the share of its 10 lineage-nearest cells (smallest D_L) that are in the bottom quartile of methylation
     similarity, and the converse.
   - Null: within-tissue permutation of the methylation profiles, 99 draws.
   - Also the Mantel-type Spearman correlation of D_L with D_M, within and across organs.
3. **Two OTs on the same pairs.** ConfidenceOT M4-E / M4-R, exactly as `10_crispr_ot.py`:
   - cost = D / median cross-pair D;
   - within-side split null, 5 + 5 replicates, grid 5, acceptance 0.99;
   - ε 0.1, λ 1.

   Run once with D_L (homopolymer-OT) and once with D_M (methylation-OT), on these pairs:
   - **Within organ:** brain anterior → posterior; liver left → median, left → right, median → right; kidney left →
     right.
   - **Across organ:** brain anterior → liver left, liver left → kidney right, brain posterior → kidney right.

   Compared per pair:
   - gate agreement (κ, cross-table);
   - common-support conditional coupling TV and top-5 overlap with chance level;
   - clade-level Shared / D_L-enriched / D_M-enriched mass;
   - mass on same-clade pairs.

   With 18-36 cells per side the calibration is fragile; every validity flag is reported.

## Reading

- One mouse: no biological replication.
- The tree is weakly supported, so association with clades is limited by lineage resolution.
- The methylation state is a projection on brain references, which is weak for liver and kidney.

Observed associations, possible interpretation and unproven mechanism are kept apart. No threshold is changed after
results.
