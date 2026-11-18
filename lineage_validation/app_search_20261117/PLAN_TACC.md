# Application search on TACC: real datasets, units and scoring (FROZEN 2026-11-18)

Written after the format audit (`05_audit_tacc_data.py`, `$SCRATCH/OT_project/app_search_20261117/audit/`) and before any
OT run on these data. Real data only: no simulation and no constructed one-sided units (the local Kang K1 / K2 removals
are not repeated). Approved by the user on 2026-11-18, with these decisions:
- LARRY in vivo: mature = every type except "Undifferentiated" and "Prog";
- C. elegans: the expression-estimated time bins are accepted as a stated limitation;
- SPLINTR and zebrafish: look for the authors' barcode / scar data.

Later changes go in dated post-hoc sections.

## Fixed for every dataset

- Pipeline: frozen calibration v3 (within_side_split + mnncovlocal, acceptance 0.99), one configuration
  `ranknm256-midrank|none|scale|cos|pca50` (L5 best). No grid, no solver change. M4-E primary, M4-R reported.
- Input: integer counts only. A dataset without recoverable integer counts is not run.
- Cell cap: at most 10,000 cells per side (seed 20261118), as in the LARRY TACC run. Where a truth label exists, cells
  carrying it are taken first; a capped side is reported as such.
- Labels, clones, lineages, times and doses stay in obs for scoring and are never OT input.
- Every unit, metric and null is reported, including negative results. No pair selection, no composite score, no
  PASS / FAIL.
- Rejection is not read as death, differentiation or ancestry. Each dataset says what its truth can and cannot support.
- Compute: preparation and scoring on the login node first (`/usr/bin/time -v`); gg or gh-dev if it does not fit. OT
  through the v3 worker's manifest route (`submit_v3.sh larry` with `V3_LARRY_MANIFEST` / `V3_LARRY_ROOT` set per
  dataset), submitted only with the user's go.
- Outputs: `$SCRATCH/OT_project/app_search_20261117/<dataset>/` (inputs, manifest.csv, v3/, score/).

## Audit summary

| Dataset | Cells | Counts | Truth / labels | Status |
|---|---|---|---|---|
| LARRY in vitro (GSE140802) | 130,887 | recovered integers (`larry_20261112`) | clones | done 2026-11-13, not rerun |
| LARRY in vivo (GSM4185643) | 182,172: d2 19,045, d9 80,614, d16 82,514; LSK 159,211, LK 22,962; 12 recipients | normalised only; integer recovery to be checked as for in vitro | clones, 10 cell types | ready after the recovery check |
| Kang PBMC (GSE96583 batch 2) | 12,315 ctrl / 12,358 stim singlets, 8 donors | integers | cell type, donor | ready |
| Cook A549 TGF-β1 (GSE147405) | 3,568 | integers | time, pseudotime, phase | done locally (all cells); not rerun |
| Schiebinger reprogramming (pertpy) | 68,339 (a 22-GSM subset of GSE122662, not the 251k) | integers | age D0–D16 + iPSCs; 2i 26,496, "control" 12,006 (serum), unlabelled 29,837 (Dox phase) | ready after the GSM → condition map |
| sciPlex2 (pertpy) | 24,262 A549 | integers | drug (Dex / Nutlin / SAHA / BMS), dose 0–100 µM, well | ready |
| C. elegans embryo (GSE126954) | 89,701 (86,024 pass QC) | integers (MatrixMarket integer) | embryo.time.bin (12 bins), lineage for 47,798 cells (515 names), cell type for 54,649 | ready |
| Mouse pancreas (GSE132188) | 36,351 after authors' QC; E12.5 10,790, E13.5 5,042, E14.5 9,633, E15.5 10,886 | h5ad is normalised; integers in RAW.tar per stage | 16 broad types (`clusters_fig6_broad_final`) incl. Ngn3 low / high EP, Fev+, Alpha, Beta, Delta, Epsilon | ready after barcode matching to the h5ad |
| SPLINTR AML (GSE161676 / GSE186305) | 10x matrices + HTO only | integers | **no SPLINTR barcode table, no cell metadata in GEO** | blocked |
| Zebrafish heart (GSE159032) | 35 libraries (h5 + loom) | integers | **no LINNAEUS scars, no cell types, no time labels in the files** | blocked |

## Per dataset

### 1. LARRY in vivo (Weinreb 2020)
- **Original conclusions:** in vivo, day-2 state predicts fate less well than in vitro (best classifier 51% vs 60%); fate
  bias is partly heritable within clones.
- **Units (4):** day 2 → day 9 and day 2 → day 16, per starting population (LSK, LK); recipients pooled on the target
  side (day 2 has no recipient).
- **Cap:** clonally informative cells first, as in `larry_20261112/02_prepare.py`.
- **Scoring:** `larry_20261112/03_score.py` unchanged: clone mass vs uniform and independent baselines, top-10 hit, fate TV
  vs a clone-label null, Neu vs Mono, gates.
  - **Decision needed:** in vivo has both "Undifferentiated" and "Prog". Proposed: mature = every type except these two.
- **Can support:** whether population-level fate recovery seen in vitro holds in vivo.
- **Cannot support:** death or engraftment failure. Day-2 cells without day-9/16 sisters are mostly unsampled.

### 2. Kang PBMC IFN-β (Kang 2018)
- **Original conclusions:** cell-type composition unchanged by 6 h IFN-β; a 767-gene programme is myeloid-specific.
- **Units (9):** ctrl → stim pooled (all singlets), and ctrl → stim per donor (8).
- **Scoring (`04_score_pilot.py`):**
  - rejection per cell type;
  - label × label coupling and same-label mass vs independent expectation;
  - depth AUROC.
  - Added, fixed now: per matched pair, the shift `log1p(CP10k)` of the target minus the source over the authors' 767
    myeloid genes, if the list is in their supplement. Otherwise over the hallmark IFN-α and IFN-γ response sets. Summary
    = mean per source cell type.
- **Can support:** whether rejection concentrates where the response is largest.
- **Cannot support:** absent populations; none exist here.

### 3. Schiebinger reprogramming (Schiebinger 2019)
- **Original conclusions:**
  - reprogramming branches into stromal, MET / epithelial, neural, trophoblast and iPSC fates;
  - many stromal cells die or do not continue;
  - iPSC ancestors are identifiable from about day 8;
  - 2i favours iPSC over serum.
- **Units:** adjacent ages within each condition.
  - Dox phase: D0 → D2 → D4 → D6 → D8.
  - Then 2i and serum separately from D8 to D16.
  - Last age → iPSCs where present.
  - The GSM → condition map comes from the GEO sample titles, because the pertpy `perturbation` column mislabels serum
    as control. Exact unit list after the map; every adjacent pair present is run.
- **Labels:**
  - the WOT published cell sets if their cell IDs match these obs names;
  - otherwise the WOT tutorial gene signatures (`gene_sets.gmt`), scored as mean `log1p(CP10k)`.
  - Both are the authors' own.
- **Scoring:** rejection per age and side; rejection per cell set / signature quartile; same-label mass.
- **Caveat:** the original conclusions come from OT themselves. Agreement is consistency, not validation.

### 4. sciPlex2 (Srivatsan 2020)
- **Original conclusions:** dose-dependent transcriptional responses and loss of cells with dose. Nutlin and SAHA give the
  strongest shifts, Dex induces glucocorticoid targets, and BMS-345541 reduces proliferation.
- **Units (28):** for each drug, vehicle (dose 0 of that drug) → each non-zero dose (0.1, 0.5, 1, 5, 10, 50, 100 µM).
  The 529 unassigned "control" cells are not used.
- **Scoring:**
  - rejection per side vs dose (Spearman across the 7 doses, per drug);
  - cells recovered per dose (the paper's viability proxy) beside the source rejection;
  - S and G2M scores (Tirosh 2016 lists) of rejected vs retained cells;
  - depth AUROC.
- **Can support:** whether source rejection grows with dose where cells are lost.
- **Cannot support:** death per cell; it is not observed.

### 5. C. elegans embryo (Packer 2019)
- **Original conclusions:**
  - the invariant lineage is resolved by expression;
  - sister lineages converge or diverge in expression;
  - cell types emerge at defined times.
- **Units (11):** adjacent `embryo.time.bin` (< 100 → 100–130 → … → > 650), QC-passed cells.
- **Truth:** for source and target cells that both carry a lineage name, the pair is lineage-compatible when the target
  name descends from or equals the source name. Matching is by Sulston prefix; `x` is a wildcard and `/`-separated
  alternatives are compatible if any is. The quantity is the share of retained coupling mass on compatible pairs vs the
  independent expectation over the same annotated cells.
  - Also reported: rejection per lineage depth and per cell type.
- **Caveats:**
  - `embryo.time` is estimated from expression, so the bins are not independent of the OT input;
  - lineage annotation covers 53% of cells;
  - programmed cell death is known biology here, but these data carry no death labels, so it is not scored.

### 6. Mouse pancreas endocrinogenesis (Bastidas-Ponce 2019)
- **Original conclusions:**
  - Ngn3 low EP → Ngn3 high EP → Fev+ → α / β / δ / ε;
  - Fev+ is an intermediate state;
  - α appears before β;
  - proportions shift toward endocrine and mature acinar with age.
- **Units (3):** E12.5 → E13.5, E13.5 → E14.5, E14.5 → E15.5. Cells are those in the authors' h5ad (their QC), with raw
  counts from RAW.tar matched by barcode.
- **Scoring:**
  - label × label coupling on `clusters_fig6_broad_final`: does Ngn3 high EP couple to Fev+ and endocrine types next day?
  - rejection per type and side;
  - target rejection of types absent or rare the day before;
  - same-label mass.
- **Can support:** consistency with the published order.
- **Cannot support:** lineage; there is no lineage truth.

### Blocked (not run)
- **SPLINTR AML:** GEO has expression and hashtag matrices only. Without the SPLINTR barcode calls and the sample →
  treatment map there is no clonal truth. Needs the authors' barcode data (paper data statement / Zenodo / GitHub).
- **Zebrafish heart:** GEO has expression only. Without the LINNAEUS scar trees and the authors' cell-type annotation the
  original conclusions cannot be checked. Needs the authors' processed data.

## Order
1. Preparation scripts and integer checks, on the login node:
   - LARRY in vivo recovery;
   - pancreas barcode match;
   - Schiebinger GSM map.
2. Unit counts reported to the user.
3. OT submission on the user's go.
4. Scoring with the definitions above.
5. Map each result to the original conclusion, both directions.
