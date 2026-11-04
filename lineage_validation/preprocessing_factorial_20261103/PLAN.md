# Preprocessing full factorial on CRISPR lineage truth (KP-Tracer, Quinn 2021): RNA only, L1-L3

Status: **PLAN, 2026-11-03, for the user's review. No code written for it yet, nothing run, nothing submitted.** Once
confirmed, this file is frozen; any later change is a dated section at the end, never an edit.

## 1. Question and scope

- **Question:** which RNA preprocessing gives ConfidenceOT the best agreement with CRISPR lineage? Measured at three
  levels: L1 (family), L2 (subclone) and L3 (cell-level tree distance).
- **Agreed with the user (2026-11-03):**
  - RNA only;
  - the full preprocessing factorial (§3);
  - no baselines in this round;
  - couplings saved for every fit, for later validation.
- **Selection use, option (b):** the lineage results are an independent check of the configurations Splatter favours.
  They are not a second selection. §6 gives the rule.
- **Untouched:** the frozen protocol (508609f), `SUBCLONE_RULES.md` (a545f1c), the Level 2 / Level 3 plans and every
  earlier result. Everything is written to a new versioned output root.
- **CRISPR data is used only for scoring.** Never in the cost, the calibration or any choice of configuration. No tree
  is re-inferred. No ancestry claim is made from a gate.

## 2. Data and units (fixed definitions, reused)

| Level | Data | Unit fitted | Truth |
|---|---|---|---|
| L1 | KP-Tracer: every metastasis × every primary of its mouse (`kp_lineage/01_prepare_kptracer_pairs.py`) | 131 pairs; 104 eligible (≥ 20 malignant cells per side) | the authors' `MetFamily` (`same_met_family`) |
| L2, L3 | 9 cases of `lineage_validation/level2_pairwise_ot` (below) | one plan per case | the frozen subclones (L2) and the authors' trees (L3, `level3_cell_lineage/ground_truth.py`) |

- **The 9 cases:**
  - KP 3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1: primary → its family's metastases, pooled;
  - Quinn 100k, 10k, 30k, 5k: LL → all other tissues, pooled.
- **Malignant cells:** `not is_normal` (KP). Quinn has only A549 cells.
- **QC:** as `27_downsample_counts.py` (1,000 counts, 500 genes, 20% mitochondrial) for every configuration, so every
  configuration gates the same cells.

## 3. Preprocessing grid (592 configurations)

**Fixed for every configuration:**
- HVG top 2,000 (rank-K and logcpm; rankhvg uses its H genes);
- PCA 50 on both sides stacked;
- cost divided by the median cross-side cost;
- entropic UOT: ε 0.1, λ_a = λ_b = 1, tolerance 1e-4.

**Varied:**

| Encoding | Levels | Count |
|---|---|---|
| logcpm (log1p of 1e4 × count / total) | DS {none, P10, P25, P50} × scaling {none, z-score} × cost {cos, sqeuc} | 16 |
| ranknm (Top-K, no gene-median division) | K {128, 256, 512, 1024, 2048} × ties {production, expected, midrank} × DS × scaling × cost | 240 |
| rank_value (Top-K, gene-median division) | same | 240 |
| rankhvg (ranks within H HVGs, no Top-K) | H {2000, 3000} × ties {production, expected, midrank} × DS × scaling × cost | 96 |

**Definitions:**
- **Depth equalisation (DS):** `confidenceot.equalise_depth`, seed 1, to the pooled quantile of the analysed cells.
  - Pooled over all prepared KP tumours.
  - For Quinn, pooled per experiment, as the frozen protocol's inputs.
- **Tie rules:**
  - production = the production stored order (`rb_common.rank_encode`);
  - expected = the average over random tie-breaking;
  - midrank = tied genes share their mean rank; a group straddling the Top-K cut takes the value at its mean rank,
    0 if beyond K.
- **rankhvg has no cut,** so expected and midrank are mathematically identical there. Both are fitted (brute force)
  and their equality is checked.

**Reproduction check (not part of the grid):** `label:ranknm256_noscale_ds_cos` at PCA 30 on the 104 L1 pairs must
reproduce the gates of the earlier KP run (E1) exactly.

## 4. OT

- **Primary method:** ConfidenceOT M4-E. M4-R is fitted and saved but not used to choose.
- **Calibration:** within-side split null (5 + 5 replicates, grid 5), acceptance 0.99, no per-side cell cap, no
  rejection-budget cap.
- **Flags as columns, never filters:**
  - calibration validity (`m4e_inference_valid`);
  - convergence;
  - depth correlation of the gate score.
- **Saved per fit:** gates (cell tables), both couplings (`--save-coupling`), joint coordinates, `run.json`.

## 5. Metrics (one primary per level; secondaries explain it)

**L1 (family):** the definitions of `kp_lineage/READOUT_PRESPECIFICATION_2026-10-14.md`.
- Per metastasis: d = same-family − mean other-family source retained fraction. That covers 31 metastases in 8 mice.
- **Primary:** the mean over mice of the per-mouse mean d.
- **Secondary:**
  - the number of mice with mean d > 0;
  - the median d over metastases;
  - the same contrast on the target side.

**L2 (subclone):** the M4-E coupling on its retained block, scored as `level2_pairwise_ot/PLAN.md`.
- Assignment rule: enrichment against the plan's own realised marginals, argmax over candidates.
- **Primary:** balanced accuracy − chance (1 / candidates), per family-level. Aggregated as the mean over mice of the
  per-mouse mean, over the evaluable family-levels.
- Scored over the **retained** target cells. Target **coverage** (the fraction retained) is reported beside it, never
  combined.
- **Secondary:**
  - the size-rule accuracy;
  - rare-subclone recall;
  - the all-cells variant, where a rejected target cell counts as unassigned.

**L3 (cell-level tree distance):** as `level3_cell_lineage/PLAN.md`, on the M4-E coupling of the retained block.
- The reference is the independent coupling with the method's own marginals.
- **Primary:** ΔD_norm per metastatic cell, as the median over cells within a family, then the mean over mice.
- **Secondary:**
  - the within-subclone decomposition;
  - coverage.

**No composite** across levels or metrics, and no PASS / FAIL.

## 6. How the results are read (fixed before any score is seen)

1. **Per level:** rank the configurations by the primary metric.
   - The **best set** is every configuration whose mouse-bootstrap 95% interval (10,000 resamples of mice) for its
     difference from the top configuration contains 0. The whole set is reported, not one winner.
2. **Leave-one-mouse-out:** for each configuration, its best-set membership with each mouse left out. Membership that
   depends on 3724 is flagged (3724 is depth-confounded, D1).
3. **Option (b):** report, for each configuration in the Splatter best set (Case 2 F1,
   `splatter_n5000_default_20261103`), its L1 / L2 / L3 rank and whether it falls in each lineage best set.
   - The recommendation is the overlap.
   - If the overlap is empty, that is reported as it is; no configuration is substituted.
4. **Every configuration's numbers are published:** all 592 × 3 levels, with flags.

## 7. Size, storage and jobs (estimates; the pilot measures them)

**Fits:**
- L1: 104 × 592 = 61,568, plus 104 reproduction fits.
- L2/L3: 9 × 592 = 5,328.
- **Total about 67,000.**

**Time:**
- KP tumours are mostly a few hundred to a few thousand cells per side, smaller than Splatter's 5,000.
- The pooled L2 / L3 cases are larger.
- **Pilot:** 1 gg node, about 100 rows drawn across the pair-size range and encodings, to measure time, memory and
  coupling size.
- The full run is sized from the pilot. Couplings are saved compressed (`np.savez_compressed`) for every fit.

**Jobs:**
- A Slurm array of gg nodes. Each node runs many workers; each worker claims the next unfinished row atomically.
- The largest rows go first.
- Arrays are kept under the 40-job cap.
- One chain: inputs → pilot → (user check) → full → collect → score L1 / L2 / L3 → summary.

**New code needed** (versioned, tests before any TACC job):
- registry v2;
- the midrank / rank_value / rankhvg-production encoders;
- PCA-50 wiring;
- the worker for these inputs;
- L1 / L2 / L3 scorers on M4-E couplings;
- the bootstrap / LOMO summary.

## 8. Not in this round

- Baselines: balanced OT, UOT, LineageOT, CloneOT, moslin.
- CRISPR-informed costs (designs A / B).
- CNV.
- Quinn L1: Quinn has clones, not families.
- Any configuration chosen after seeing lineage scores.

---

## 2026-11-03 amendment (user, before any score; replaces section 8's "CRISPR-informed costs" exclusion)

The factorial keeps four input settings. **RNA-only (R)** is unchanged: sections 1-7, ready to run. The three CRISPR
settings below are prepared, not run. Their open decisions are marked **DECIDE**.

| Setting | Cost | Evaluation status |
|---|---|---|
| R: RNA only | RNA block | L1-L3: valid. Lineage is used only for scoring |
| C: CRISPR only | CRISPR block | L1-L3 are circular (the truth is derived from the same characters): upper bound only, never validation |
| RC: RNA + full CRISPR | RNA block + CRISPR block | L1-L3 circular: upper bound only |
| RH: RNA + held-out intBCs | RNA block + CRISPR block over the input intBCs | L1-L3 partly circular (the authors' trees and families use every character); the clean readout is L3h below |

**CRISPR block (fixed here):**
- **Distance:** per cell pair, over characters observed in both cells (`-` = missing, never a state). It is Cassiopeia's
  weighted Hamming distance (Jones et al. 2020):
  - a character in the same state costs 0;
  - uncut (`0`) against an allele costs 1;
  - two different alleles cost 2;
  - with the authors' indel priors available, an allele mismatch is weighted by −log prior (rarer alleles weigh more);
  - the sum is divided by the number of shared observed characters.
- **Pairs with no shared observed character** get the unit's median distance.
- **Coordinates:** classical MDS of the unit's full distance matrix (source and target cells together), 30 dimensions.
  This lets the block enter the existing coordinate-based cost and the within-side calibration null unchanged.
- **Combined with RNA:** as `rna_cnv` (02_run_pair.py). Each block is divided by the square root of its median random
  source-target pair cost and the two are concatenated. The weight is label-free and never tuned on CRISPR.

**Held-out intBCs (RH):**
- The intBC is the independent unit: three consecutive characters r0-r2, r3-r5, … This is proven by shared
  missingness for 3454, 3457 T1, 3508 and 3515; for 3724 and 3457 T4 the allele table must confirm it.
- **Two fixed partitions, both reported:**
  - P1: even-indexed intBCs into the cost, odd-indexed held out;
  - P2: the reverse.
- **L3h (clean readout):** per metastatic cell, the coupling-weighted held-out-character distance to its matched
  primary cells, against the independent coupling with the method's own marginals. ΔDh_norm is as L3 with the tree
  distance replaced by the held-out character distance.
- **3724 and 3457 T4 have about 3 intBCs** (1-2 per half). Their RH result is reported, marked low-powered.

**DECIDE 1, scope:**
- (a) RC and RH over all 592 RNA configurations (brute force; about 3 x 67,000 more rows: RC, RH-P1, RH-P2);
- (b) RC and RH on the RNA-only best set plus `ranknm256_noscale_ds_cos`.

C is one configuration either way (no RNA preprocessing).

**DECIDE 2, Quinn:** its character matrices, trees and allele tables are on TACC only and not audited. Include Quinn
in C / RC / RH after an audit, or KP only.

**Needed from TACC before any CRISPR code runs:**
- KP allele tables (intBC × cut site) and indel priors from `KPTracer-Data`;
- Quinn's `converted/lineage/<tag>/` inventory (format, cell-id alignment, intBC structure).
