# KP-Tracer lineage data audit (validation_v2)

Status: **partial.** Sections A and B were verified from the repository code and the files already inspected. Section C is filled from `03_audit_lineage.py` (job `audit`) and stays marked *pending* until that job's output has been read. No association between lineage and any gate outcome is computed in this audit.

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

## C. From `03_audit_lineage.py` — *pending*

- **C1.** Per tree: leaves, internal nodes, branch lengths (present, informative), polytomies, unary nodes, depth, the leaf-name format, and whether leaves match the expression ids exactly. Per tree, the tumours its leaves come from.
- **C2.** Per label: the spread of source retained fractions, active bounds, budget overrides, and agreement of `retained` with `margin < 0`.
- **C3.** Per eligible same-family pair: which family tree holds both sides, and how many source cells, target cells, retained cells and rejected cells are leaves of it.
- **C4.** Character matrices: cells, characters, missing fraction, and overlap with the same-name tree.
- **Blocker rule.** If the leaves do not match the expression ids, or fewer than half of the same-family pairs are covered, the cell-level analysis stops here. The report then states the blocker, and no substitute truth is built from expression.
