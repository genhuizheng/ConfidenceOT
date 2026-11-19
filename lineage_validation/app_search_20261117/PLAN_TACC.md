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

## Post-hoc 2026-11-18: preparation outcome, SPLINTR and zebrafish (DRAFT, not yet approved)

Written after `06_prepare_tacc.py` ran on all six datasets and before any OT run. Nothing above is changed.

### Preparation outcome and deviations
- **Units formed:** Kang 9, sciPlex2 28, Schiebinger 14, C. elegans 11, pancreas 3, LARRY in vivo 3 (68 in all).
- **OT-ready:** all six manifests pass `build_worklist.py --dataset larry` (key `ranknm256-midrank|none`).
- **Inputs checked:** `07_verify_inputs.py` checked all 136 input files and found 0 problems. Checks:
  - integer, non-negative counts over the whole matrix;
  - no zero-total cells;
  - `sample_id` / `malignant` as the manifest says;
  - unique obs and var names;
  - identical var names on both sides;
  - no missing obs;
  - cell counts equal to `units.tsv`.
- **LARRY in vivo has 3 units, not 4.**
  - LK cells exist only at day 2 (222) and day 9 (22,740). There are no LK day-16 cells, so LK d2 → d16 does not exist.
  - It is listed in `units.tsv` as `formed = False` and is not in the manifest. The rule is only "a side with zero
    cells"; no threshold.
- **LK d2 → d9 has no clone shared between the two days.**
  - Its clone scores are undefined and reported as such. Fate scores (Neu vs Mono, fate TV) are reported.
  - LSK d2 → d9 and d2 → d16 share 508 and 245 clones.
- **LARRY in vivo integer recovery:** median share of integer entries 1.000, and every cell is at least 0.99.
  131,060 of 182,173 cells carry a clone; none carries more than one.
- **Pancreas:** every cell in the authors' h5ad was found in the raw barcodes of its stage. The raw totals equal the
  authors' `n_counts` (mean 15,668.9 for both).
- **Script fixes** were made before any output was used (`424ffd2`, `8e4c5f0`, `07acd7f`):
  - the in vivo clone matrix ships as clones × cells, so it is transposed;
  - the pancreas gene index carried an integer name;
  - manifest paths are now absolute.

### 7. SPLINTR AML chemotherapy (Fennell 2022), unblocked
- **Clone data found:**
  - The singletCode figshare (doi 10.6084/m9.figshare.25478680; md5 verified) holds
    `allBarcodeSheets/SPLINTR/stepFourStarcodeShavedReads50.csv`.
  - One row per barcode read; columns `cellID, barcode, sample`; 15 samples.
  - This is not the authors' own file: the authors released no per-cell barcode table.
- **Libraries match.** About 99% of each sample's cells fall in the expected GSE186305 library:

  | Samples | GSE186305 library |
  |---|---|
  | `chemoKRASBaseline_1` / `_2` | KF16_TP0_1 / _2 |
  | `chemoVehicle_*`, `chemoDay2_*` | KF16_POOL_1 |
  | `chemoDay5_*`, `chemoDay7_*` | KF16_POOL_2 |

  This matches the authors' HTO map (pool 1: vehicle × 2, day 2 × 2; pool 2: day 5 × 2, day 7 × 2).
- **Original conclusions:**
  - Malignant clonal dominance is cell-intrinsic and heritable (repressed antigen presentation, higher Slpi).
  - The two sources disagree on chemotherapy:
    - The published abstract says LSC clonal output dictates sensitivity, and that high- and low-output clones adapt
      differently yet both emerge from minimal residual disease.
    - The GEO series text says output does not dictate sensitivity.
  - The paper's text and figures are read before results are mapped to either statement.
- **Cells:**
  - Used: cells with a barcode call that are also in the GEX matrix of their library, with integer counts from the GEO
    matrices.
  - Not used: pool cells without a barcode, because this table gives them no mouse. We do not demultiplex HTOs
    ourselves.
  - Each sample's cells are checked against the HTO matrix: the argmax hashtag should agree with the authors' map.
    Disagreements are reported, not removed.
- **Units (16):** each baseline library (TP0_1, TP0_2) → each mouse sample (vehicle 1, 2; day 2 1, 2; day 5 1, 2;
  day 7 1, 2).
  - Baseline sides are capped at 10,000 cells at random. Every baseline cell carries a barcode, so no priority applies.
  - Baselines are not pooled; the two libraries are replicates.
- **Clone per cell (DECISION NEEDED):** 20–40% of cells carry more than one barcode.
  - Proposed: clone = the barcode with the most reads in that cell. A tie leaves the cell without a clone; it stays in
    the OT and is left out of clone scores.
  - Reported beside it: clone scores with every multi-barcode cell left out.
- **Scoring** uses the same definitions as LARRY `03_score.py` where they apply:
  - per source cell, truth = its clone has at least one cell in that target sample;
  - source rejection by truth (AUROC; rejected share in each class), with vehicle units beside chemo units;
  - clone mass per retained source cell vs uniform and independent baselines;
  - baseline clone size (cells per clone in that baseline library) vs source rejection (Spearman);
  - target rejection, and the share of target cells whose clone is absent from that baseline library.
- **Can support:** whether source rejection is higher, after chemotherapy than after vehicle, for baseline clones that
  are not recovered.
- **Cannot support:** death or resistance per clone. An absent clone may simply be unsampled: targets hold 389–2,619
  barcoded cells against about 3,900–4,600 clones per baseline library.

### 8. Zebrafish heart regeneration (Hu 2022, Nat Genet 54:1227), unblocked
- **Data found (moslin figshare):**
  - `Zebrafish_metadata.csv`: 202,075 cells after the authors' QC, with `Cell_type` (32 types), `time` (Ctrl / 3 / 7 /
    30 dpi), `morphine`, `inhib` and `AV`.
  - `trees.pkl`: 20 LINNAEUS trees as networkx DiGraphs, leaves = cell IDs. The pickle references only networkx
    DiGraph / NodeView and `dict`.
  - `Tree_times`.
- **Expression:** GSE159032 (filtered 10x h5 per library), plus GSE158919 for Hr8, Hr11, Hr15 and Hr23. Cell IDs are
  `<library>_<barcode>`, and every ID's prefix equals `orig.ident`.
- **Original conclusions:**
  - After cryoinjury, activated fibroblast states appear (col12a1a, col11a1a, nppc and others).
  - They derive from two sources, the epicardium and the endocardium (scar lineage; Cre-lox confirms this for
    col12a1a).
  - col12a1a fibroblasts are pro-regenerative.
  - Wnt signalling regulates the endocardial fibroblast response (IWR-1).
- **Conditions are kept apart, never pooled:**

  | Condition | Ctrl | 3 dpi | 7 dpi | 30 dpi |
  |---|---|---|---|---|
  | Untreated | H5, H6, H7, H8a/v | Hr11, 12, 24, 27 | Hr1, 2a, 2b, 6a/v, 7a/v, 8, 9, 13 | Hr3, 4, 19 |
  | Morphine | same Ctrl hearts | Hr10, 22, 23, 25, 26, 28 | Hr14, 15, 30 | Hr20, 21 (874 cells) |
  | Morphine + IWR-1 | none | Hr29, 34, 35 | Hr31, 32, 33 | none |

  The role of morphine is read from the paper before scoring.
- **Units (9):**
  - untreated: Ctrl → 3 dpi, 3 → 7 dpi, 7 → 30 dpi;
  - morphine: Ctrl → 3 dpi, 3 → 7 dpi, 7 → 30 dpi;
  - morphine → morphine + IWR-1, at 3 dpi and at 7 dpi (same day, inhibitor on the target side, as in Kang ctrl → stim);
  - untreated → morphine at 3 dpi.

  Hearts are pooled within a side and capped at 10,000 cells (seed 20261118), with cells that are tree leaves taken
  first.
- **Cells:** those in the authors' metadata, with raw counts from the GEO h5 matched by barcode (as for pancreas).
  Library names that differ between GEO and the metadata (Hr2 / Hr2_morecells vs Hr2a / Hr2b) are resolved by barcode
  overlap and reported in `prep/barcode_match.tsv`.
- **Truth:** none across time. Scars are not shared between animals, so no source-target pair has a lineage relation.
  Trees give within-heart relatedness only.
- **Scoring:**
  - label × label coupling on `Cell_type`, and same-label mass vs independent expectation;
  - in Ctrl → 3 dpi, target rejection of the activated fibroblast types that are near-absent in Ctrl (col11a1a 2 cells,
    col12a1a 9, nppc 8, proliferating fibroblasts 0);
  - which source types the retained coupling of activated fibroblasts lands on (epicardium, endocardium, constitutive
    fibroblasts, other);
  - tree relatedness, fixed now and parameter-free:
    - in each heart with a tree, a cell's clade is its parent node;
    - relatedness(a, b) = the share of clades holding type a that also hold type b, averaged over the hearts of that
      time point;
    - reported: the rank correlation, over source types, between the OT coupling of each activated fibroblast type and
      its tree relatedness at the target time point. This is consistency only, not lineage validation;
  - IWR-1 units: rejection and coupling of endocardium-derived vs epicardium-derived fibroblast types.
- **Cannot support:** the ancestry of any cell. Rejection is not read as death or as a cell-type origin.

### Content checks (`08_content_checks.py`, outputs in `$APP/content/`)

Checks that each label means what it says. Means are log1p(CP10k).

**Consistent with the original papers:**
- **Kang:** ISG15 / IFI6 / IFIT1 / MX1 are far higher in stim in all 8 types (CD14 Mono ISG15 0.69 → 6.06).
- **Schiebinger:**
  - Dppa5a and Nanog rise with age, and Col1a1 is high at D0.
  - 2i is above serum at D12 (Dppa5a 1.54 vs 0.53).
  - The pertpy age of every GSM equals its GEO title.
- **Pancreas:** every marker sits in its type: Neurog3 in Ngn3 high EP, Fev in Fev+, Ins1 in Beta, Gcg in Alpha, Sst in
  Delta, Ghrl in Epsilon, Cpa1 in acinar.
- **LARRY in vivo:** Ltf in Neu, Cd79a in B, Klf1 / Hba-a1 in Ery, Csf1r / Ly6c2 in Mono.
- **sciPlex2:**
  - Dex raises FKBP5 and TSC22D3.
  - Nutlin 5–50 µM raises CDKN1A and MDM2 and lowers MKI67 / TOP2A.
  - SAHA raises CDKN1A.
- **Zebrafish:** every metadata cell is found in the barcodes of exactly one GEO library (100% for all 39).
  - Hr2a = GEO Hr2, and Hr2b = GEO Hr2_morecells.
  - **GEO file names Hr22 and Hr23 are swapped relative to the metadata:** metadata Hr22 matches `GSE158919 Hr23`, and
    metadata Hr23 matches `GSE159032 Hr22`. Libraries are therefore taken by barcode match, never by file name. Both are
    3 dpi morphine hearts with trees, so no unit changes.

**Problems found:**
- **SPLINTR pool-2 day labels conflict.** The hashtags themselves are clean: each pool sample's cells take one hashtag at
  99–100%. Mouse identity is therefore certain; which pool-2 mice are day 5 and which day 7 is not:

  | Pool-2 hashtag | Authors' code | Supplementary Table 2 | singletCode sheet |
  |---|---|---|---|
  | A0301 (M07) | D5 | D5 | `chemoDay5_1` (D5) |
  | A0302 (M09) | **D7** | D5 | `chemoDay5_2` (D5) |
  | A0303 (M11) | **D5** | (not listed) | `chemoDay7_1` (D7) |
  | A0305 (M12) | D7 | D7 | `chemoDay7_2` (D7) |

  Supplementary Table 2 lists A0306, which has almost no counts. A second recode in the same Rmd labels M07 as D6. Pool
  1 agrees across all sources: A0301 / A0302 vehicle, A0303 / A0305 day 2. **Decision needed.**
- **C. elegans `< 100` bin is mostly germline.**
  - 368 of its 458 cells are annotated Z2 / Z3; the `> 650` bin holds 102 more.
  - Expression-estimated time places transcriptionally quiet germ cells early, so `< 100 → 100-130` is largely germline
    → early soma, and lineage compatibility there is not meaningful.
  - The unit stays (no pair selection) and is reported with this caveat; the Z2 / Z3 share is reported for every bin.
  - Other bins: estimated time rises monotonically and lineage names lengthen (median 2 → 11).
- **LARRY in vivo depth differs 3–5× between sides:** median total counts are d2 4,860 (LSK), d9 1,740, d16 920.
  Depth AUROC is reported with every rejection result.
- **sciPlex2 BMS 1 µM carries a p53-like signal in all 6 of its wells** (CDKN1A 0.69–1.43, MDM2 1.97–2.34, MKI67 about
  0.5). No other BMS dose shows it.
  - It is not a single-well error, and these data cannot say whether it is biology or a dose-level annotation problem.
    The unit stays and is reported with this note.
  - Nutlin 100 µM keeps 86 cells whose p53 targets are back near vehicle, so the survivors look like non-responders.

### Decisions needed before the SPLINTR and zebrafish preparation code is written
1. SPLINTR clone per cell: the dominant barcode with ties left unlabelled (as proposed above), or another rule.
2. SPLINTR cells: barcoded cells only (as proposed), or our own HTO demultiplexing so that unbarcoded pool cells can be
   used.
3. Zebrafish unit list: the 9 units above, in particular the treatment-contrast units.
4. SPLINTR pool-2 days. The options:
   - follow the authors' code;
   - follow Supplementary Table 2 / singletCode;
   - do not split day 5 from day 7, and treat the four pool-2 mice as "chemo day 5–7".
