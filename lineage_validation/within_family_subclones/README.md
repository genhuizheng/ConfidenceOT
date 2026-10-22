# Within-family subclone analysis (exploratory biological application)

Status, 2026-10-22:
- **Feasibility audit and CRISPR subclone definitions:** complete for every downloaded tree.
- **OT classification:** complete for the 4 eligible families whose couplings are local (KP-Tracer 3515 T1, Quinn 10k LG2, LG3, LG5). The other 22 eligible families run on TACC with the same code (`submit_tacc.sh`), because their couplings are only there.

This analysis is separate from the pre-specified T1/T2 of `../VALIDATION_PROTOCOL.md`, which is not changed. The rules were fixed and committed (`SUBCLONE_RULES.md`, a545f1c) before any subclone-level OT quantity was computed. Nothing re-runs OT or re-infers a tree.

## Files

| File | What |
|---|---|
| `SUBCLONE_RULES.md` | the fixed rules: reliable cells, marker-edit support, level-1/level-2 partition, eligibility, outcome order |
| `define_subclones.py` | audit and subclone definitions from the authors' trees and character matrices only (reads no OT output) |
| `classify_subclones.py` | per metastatic cell, the existing coupling summed over primary subclones (q_ij of the protocol §6), within-family weights and the outcome; local export or TACC runs |
| `plot_subclones.py` | the five figures per family, reusing `../local_lineage_visualization/plot_lineage.py` |
| `submit_tacc.sh` | one TACC job: definitions + classification of every eligible family, packed for download |
| `output/` (not committed) | `feasibility.tsv`, `subclones.tsv`, `cells/<family>.csv.gz` (every leaf, in tree order, with L1/L2 label or the reason it has none), `classification_local/`, `figures_local/` |

## Ground truth available

- **Author annotations.** No dataset deposits lineage subclones.
  - KP-Tracer `SubTumor` (T1_1 … T1_5) is the physical tumour piece that was sequenced, a sampling label. It is drawn beside the subclones and never used to define them.
  - Quinn has LineageGroup and metastatic-rate scores only.
- **So subclones come from one rule** (SUBCLONE_RULES.md):
  - the authors' tree;
  - cells with ≤ 50% missing characters;
  - a branch is a subclone if it holds ≥ 20 such cells at the family's first split with ≥ 2 such branches, and carries ≥ 1 marker edit (in ≥ 80% of its observed cells, ≤ 20% of the rest, observed in ≥ 50%).

## Feasibility (all trees)

### KP-Tracer: 12 families on a Cassiopeia _Fam/_All tree

| Family (tree) | Primary / metastatic cells | Level-1 subclones | Eligible | Reason if not |
|---|---|---|---|---|
| 3454 T1 (_All) | 1,117 / 591 | 2 | yes | |
| 3457 T1 (_Fam) | 713 / 175 | 2 | yes | |
| 3457 T4 (_Fam) | 346 / 84 | 2 | yes | |
| 3515 T1 (_Fam) | 710 / 1,166 | 2 | yes | |
| 3724 T1 (_All) | 14,817 / 6,254 | 3 | yes | |
| 3460 T1 (_Fam) | 564 / 285 | 0 | no | both major branches lack a marker edit |
| 3508 T2 (_Fam) | 306 / 749 | 1 | no | the metastasis-rich branch has no marker edit |
| 3513 T1 (_Fam) | 969 / 243 | 1 | no | one supported branch |
| 3454 T2, 3508 T1, 3515 T2, 3515 T3 | primary only | – | no | no metastatic cell in the family |

Biology visible in the level-1 / level-2 subclones (counts: primary / metastatic cells):

| Family | Subclone | What it contains |
|---|---|---|
| 3457 T1 | L1.1.2 | **all 175 metastatic cells** (K1 137, N2 38) with 31 T1 cells |
| | L1.1.1 | 643 T1 cells and no metastatic cell |
| 3724 T1 | L1.1 | the S1 metastasis (3,781 cells) with 9,985 T1 cells |
| | L1.2 | the L1/L2/L3 metastases (2,456 cells) with 4,789 T1 cells, a distinct T1 branch |
| 3515 T1 | L1.1.2 | the L2 metastasis (241) and 44 N1 cells, with only 11 T1 cells |
| | L1.1.1 | the K1–K3 and L1 metastases with 433 T1 cells |
| 3454 T1 | L1.2 | a small branch: 9 T1 cells, 48 N1 and 9 Rib cells |

### Quinn 2021: 148 clones

- **Eligible:** 21 clones (100k: 4; 10k: 3; 30k: 2; 5k: 12).
- **Not eligible:**
  - 32 have no tree file;
  - most others have < 20 reliable LL cells, or no supported split.
- **The 21 eligible clones** have 107 level-1 subclones: 55 shared (≥ 5 LL and ≥ 5 metastatic cells), 52 with < 5 LL cells.
- **Subclones with no LL cell at all** (`unmatched_observed` for every one of their cells), for example:
  - 5k LG1 L1.9 (190 cells, 189 in Liv) and L1.16 (95, Liv);
  - 30k LG1 L1.7 (82, mostly LN);
  - 5k LG17 L1.2 (82).

## OT classification so far (4 of 26 eligible families; M4-E)

Level 1, accuracy = correct / (correct + incorrect); size reference = always predict the primary subclone with the most retained cells.

| Family | Strategy range of accuracy | Size reference | Minority subclone, correct / predicted |
|---|---|---|---|
| KP 3515 T1 (L1.1 vs L1.2) | 0.92–0.99 | 0.99 | L1.2: RNA 1/4, CNV 0/12 |
| Quinn 10k LG2 | 0.80–0.84 | 0.81–0.86 | L1.2: 0/20 (RNA), 0/39 (CNV) |
| Quinn 10k LG3 | 0.79–0.83 | 0.82–0.85 | L1.2: 0/24 |
| Quinn 10k LG5 | 0.71–0.75 | 0.73–0.80 | L1.2: 0 |

- **Level 2 in 3515 T1** gives the same picture: accuracy 0.72–0.76 against a size reference of 0.73–0.77.
- The L2-metastasis subclone L1.1.2 puts 0.01 of its within-family weight on its own primary branch, which is 0.01 of the retained source cells.
- **In every heatmap, each row repeats the source-share row.** The OT mass of a metastatic cell is spread over the primary subclones in proportion to their size, almost regardless of the cell's own CRISPR branch.
- **Weak exceptions:** 3515 L1.2 puts 0.29–0.34 on its own branch against a 0.25 share, never enough to change the prediction.

These four families are not yet the systematic answer. That needs the TACC run of all 26.
