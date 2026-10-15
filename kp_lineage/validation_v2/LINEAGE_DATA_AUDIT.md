# KP-Tracer lineage data audit (validation_v2)

Status: **complete, 2026-10-15.** Sections A and B were verified from the repository code and the files already inspected. Section C is the output of `03_audit_lineage.py`, run on the TACC login node on 2026-10-15 (a light read-only script; the gg queue was long) into `<root>/validation_v2/audit/`. No association between lineage and any gate outcome is computed in this audit.

## A. Verified from code

### A1. What acceptance 0.99 does: it calibrates the cost; it does not fix the retained mass

- **Where it acts.** `calibrate_confidence_cost` (`src/confidenceot/calibration.py:184-419`) fits M4-E on within-side split nulls, which contain no incompatible cells, across a cost grid. It then picks the smallest rejection cost `c` at which the mean null raw acceptance is at least 0.99 on both sides (`:288-353`), refines it by bisection, and checks it with M4-R on held-out nulls.
- **What the 0.99 is.** It is a property of the null, not of the observed pair. On the observed data the gate is `raw_gate = coeff < 0`, with `coeff = partner_mass × (decision_cost − c)` (`_cpu_uot.py:883-890`). The retained fraction is whatever falls below `c`.
- **Budget.** `02_run_pair.py:510` sets a default budget of 0.95. Since 2026-09 it is reported, not enforced: `--enforce-rejection-budget` is off by default. No `--*-rejection-bounds` were passed in this run, so no cardinality bound applies.
- **Consequence.** The observed source retained fraction is free to vary between pairs and is an informative quantity. Section C2 checks this on the outputs: the spread across pairs, active bounds and budget-overridden cells.

### A2. Gate-score convention

- `cell_confidence.csv` (`02_run_pair.py:201-219`) has one row per cell and method. Its columns are `retained`, `raw_retained`, `decision_cost`, `rejection_cost`, `signed_rejection_margin` (= `decision_cost − c`), `relative_rejection_margin`, `normalized_rejection_score` (a display-only logistic of the margin) and `budget_overridden`.
- **Direction.** A cell is retained when `signed_rejection_margin < 0`. Larger values mean less support. The continuous retention-confidence score used in validation_v2 is therefore `r = −signed_rejection_margin` of M4-E, so that larger values mean more confident retention.
- Section C2 measures how often the `retained` flag agrees with `margin < 0` in the outputs.

### A3. Lineage information is not used in the OT

- `01_prepare_kptracer_pairs.py` writes `raw/X` counts with five obs columns: `patient_id`, `site`, `sample_id`, `cell_type`, `malignant`. It also writes `met_family`, `sub_tumor`, `genotype` and `lane`.
- No lineage barcode, tree or character state is written into the h5ads.
- `02_run_pair.py` builds its representation from the counts alone. `met_family` is never read by the OT; it is used only by the readout.

### A4. Methods present in the outputs

- `02_run_pair.py:795` fits M4-E (exact) and M4-R (reversible), both on the UOT backbone.
- **There is no ordinary (balanced) OT baseline and no saved coupling** (`--save-coupling` and pairing edges were not set). Any Phase-4 baseline therefore needs new runs.

### A5. The cross-mouse control script

`cancer_metastasis/28_build_mismatched_manifest.py`:
- keeps every source side;
- replaces each target with another patient's lesion of the most similar `target_n` (absolute log ratio), drawn from the pooled lesions of every other patient;
- never gives one source sample the same donor twice;
- breaks ties with a seeded permutation (seed 20260915);
- keeps the row count, so the array indexes it unmodified.

**Two properties matter for KP-Tracer:**
- `target_n` counts all cells, normal ones included. Several metastases are mostly normal, e.g. 3508_Apc_N1 has 81 malignant of 704 cells. Matching on `target_n` therefore does not match the malignant population that is actually gated. Section B4 proposes the malignant count.
- Donors may come from any genotype (NT, Lkb1, Apc).

## B. Verified from files already inspected (2026-10-14)

### B1. Provenance

- **Source:** KP-Tracer, Yang, Jones et al., Cell 2022. Zenodo 5847462, `KPTracer-Data.tar.gz`, md5 `353d18a618f60e975089f2d278acaa88`, verified at download.
- **Expression:** `expression/adata_processed.combined.h5ad`, written by old anndata: categoricals are stored as codes with an attrs reference.
  - 104,197 cells;
  - `raw/X` holds integer counts for 27,998 genes.
- **Meta:** `KPTracer_meta.csv`, 106,684 cells. This is the union of the RNA and target-site libraries, so `TS_Present` is True for 76,294 cells.
  - The index format is `L10.AAACCTGAGACGCAAC-1`, the same as the h5ad obs names.
  - The README defines each column.
- **Lineage, per the README:**
  - Cassiopeia trees in newick, one per tumour, plus `_Fam` / `_All` trees for metastatic families;
  - one character matrix per tree, with priors;
  - neighbor-joining trees in `trees/neighbor_joining/`.
  - The README notes that "We do not report individual character matrices for each metastasis".
  - The trees were built with an older, deprecated Cassiopeia.

### B2. Family trees present (file names)

- **Cassiopeia `_Fam`/`_All`:** 3454_Lkb1_All, 3457_Apc_T1_Fam, 3457_Apc_T4_Fam, 3460_Lkb1_T1_Fam, 3508_Apc_All, 3508_Apc_T2_Fam, 3513_NT_T1_Fam, 3515_Lkb1_All, 3515_Lkb1_T1_Fam, 3519_Lkb1_All, 3519_Lkb1_T1_Fam, 3724_NT_All.
- **3513_NT_All and 3513_NT_Mets:** only priors and a neighbor-joining tree, with no Cassiopeia tree.
- **No family tree in the file list** for mice 3451, 3465, 3466 or 3467. Their metastases appear in no tree file, so their same-family pairs probably cannot enter a tree-based analysis. C3 confirms this from the leaves.

### B3. Pairs (`prepared/preparation_report.json`, `pair_manifest_all.csv`)

- 131 pairs, of which 104 are eligible and 40 eligible same-family. The preregistered tumour-level endpoint has 31 informative metastases in 8 mice.
- `same_met_family` compares the authors' `MetFamily` of the two tumours.
- OT outputs: 104/104 SUCCESS in both arms, `ranknm256_noscale_ds_cos` and `raw`.

### B4. Known biases and limitations, before any lineage computation

- **Terminal sampling.** Sampled primary cells are relatives of the seeding cells, not their ancestors. A tree-based association is evidence of lineage concordance or clade association, not of direct seeding by a sampled cell.
- **Target-site dropout.** Cells without a captured target site have no lineage position. If capture differs between retained and rejected cells (for example through depth), coverage itself is biased. C3 counts it per group.
- **Unequal metastasis sampling.** A metastasis with more sampled cells has more chances of a deep shared ancestor with any primary cell. Within one pair this affects retained and rejected cells equally. Across pairs it is one reason effects are aggregated, not pooled.
- **Old Cassiopeia.** The reconstruction is fixed and was not repeated here. Where both a `_Fam` and an `_All` tree, or a neighbor-joining tree, cover a pair, the other one is a sensitivity tree.
- **Mostly-normal metastases** (3508 lymph nodes, 3513_NT_N1): small malignant target sets.

## C. From `03_audit_lineage.py` (2026-10-15)

Files: `audit.json`, `audit.txt`, `trees.tsv`, `tree_tumours.tsv`, `character_matrices.tsv`, `pair_coverage.tsv`, `gate_semantics.tsv`, under `/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/validation_v2/audit/`.

### C1. Trees

- **Count.** 169 newick files: 85 Cassiopeia trees and 84 neighbor-joining trees.
- **Leaf ids.** In every tree, every leaf matches a `KPTracer_meta.csv` index exactly (format `L28.CTCTAATAGGAGTTTA-1`).
- **Branch lengths.** No tree has a branch length on any edge (`trees_with_branch_lengths` = 0). Under the preregistration's rule (1.6, item 3), the patristic-distance sensitivity analysis does not apply.
- **Unary internal nodes.** Present in most trees, for example 921 in Cassiopeia 3724_NT_All and 997 in 3508_Apc_All. Under the fixed rule (1.2), they are contracted before `s` is computed.
- **Polytomies.** Large and common. The largest internal node has 8,362 children in Cassiopeia 3724_NT_T1, 6,519 in 3724_NT_All, 1,033 in 3508_Apc_All and 552 in 3513_NT_T1_Fam. Many source cells therefore share the same `s`. Cliff's delta counts ties as neither greater nor smaller, so ties lower |δ| without biasing its sign.
- **Depth.** Median leaf depth (edges, before contraction) ranges from 4 to 25 in the Cassiopeia trees.
- **Family trees present** (Cassiopeia unless marked): 3454_Lkb1_All, 3457_Apc_T1_Fam, 3457_Apc_T4_Fam, 3460_Lkb1_T1_Fam, 3508_Apc_All, 3508_Apc_T2_Fam, 3513_NT_T1_Fam, 3513_NT_All (NJ only), 3513_NT_Mets (NJ only), 3515_Lkb1_All, 3515_Lkb1_T1_Fam, 3519_Lkb1_All, 3519_Lkb1_T1_Fam, 3724_NT_All. Neighbor-joining versions exist for all of them except 3457_Apc_T1_Fam and 3457_Apc_T4_Fam.

### C2. Gate semantics in the outputs

- **Retained flag.** `retained` agrees with `signed_rejection_margin < 0` in every pair (minimum agreement 1.0000).
- **Overrides and bounds.** No pair has a budget-overridden source cell, an active lower bound or an active upper bound.
- **Cell ids.** All 242,974 source-cell observations of the OT outputs are found in `KPTracer_meta.csv`.
- **Source retained fraction across the 104 pairs:**

| Label | min | q25 | median | q75 | max |
|---|---|---|---|---|---|
| ranknm256_noscale_ds_cos | 0.049 | 0.287 | 0.604 | 0.884 | 1.000 |
| raw | 0.827 | 0.974 | 0.991 | 0.998 | 1.000 |

In the raw arm almost every source cell is retained, so the rejected group is small in every pair.

### C3. Coverage of the 40 eligible same-family pairs

- **Covered: 28 of 40, the same in both arms.** These are in 7 mice: 3454 (1 pair), 3457 (3), 3460 (5), 3508 (8), 3513 (1), 3515 (6) and 3724 (4).
- **Not covered: 12 pairs.** No family tree holds both sides of:
  - 3451, all 5 pairs;
  - 3465, all 5 pairs;
  - 3466_Lkb1_T2 → N1;
  - 3454_Lkb1_T1 → L1 (3454_Lkb1_All does not contain L1).
- **Mapping.** In every covered pair, every source cell and every target cell of the OT is a leaf of the covering tree. Retained and rejected cells are mapped at the same rate (100%), so target-site dropout does not differ between the groups. Sensitivity analysis 4 (1.6) is therefore trivially satisfied.
- **Evaluability under the fixed rule (1.1):** at least 5 mapped retained cells, at least 5 mapped rejected cells and at least 1 mapped target cell.
  - **ranknm256_noscale_ds_cos: 26 of 28 evaluable, 7 mice.** Not evaluable: 3460 → N2 and 3460 → N4, with 0 rejected cells each. 3460 → N3 has exactly 5 and enters.
  - **raw: 21 of 28 evaluable, 6 mice.** Not evaluable: 3457 → L1 (2 rejected), 3460 → N3 (1), 3460 → N5 (3), 3508 → N3 (4), 3513 → N1 (2), 3515 → K1 (3) and 3515 → N1 (1). Mouse 3513 drops out.
- **Candidate trees.** Each covered pair can have several candidate trees (a `_Fam`, an `_All`, Cassiopeia or neighbor-joining), listed in `pair_coverage.tsv`. The primary tree is chosen by the fixed rule in 1.1; the others are sensitivity trees (1.6, item 2).

### C4. Character matrices

- **Count.** 85 matrices, one per Cassiopeia tree.
- **Size.** 5 to 73 characters; missing fraction 0.010 to 0.481.
- **Overlap.** Every row matches a meta id. Every matrix row is a leaf of the same-name tree, and every leaf of that tree is a matrix row.

### Blocker rule: not triggered

The leaves match the expression ids exactly, and 28 of 40 same-family pairs (70%) are covered, above the stated threshold of half. The cell-level analysis can proceed after the preregistration is frozen.
