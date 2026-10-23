# Level 3 results: cell-level CRISPR lineage correspondence of the 54 Level 2 couplings

Computed 2026-10-22, locally, from the 54 saved couplings and the authors' trees. The design is in `PLAN.md`, fixed before any coupling was scored; one post-hoc decomposition was added there with a date, before it was computed.

**Inputs.**
- Couplings `level3_couplings.tar`, md5 `e89f3a8fa35391be5d038b001b6cbcf7` (checked). Their column masses match the Level 2 tables to 1e-11.
- No OT was re-run. Preprocessing, solver, ε, trees and subclone definitions are unchanged. Level 1 and the frozen protocol are untouched.
- All 54 runs were scored (`results/scores/run_status.tsv`).

## Coverage and evaluability (`results/ground_truth/`)

**Evaluability.**
- 47,000+ metastatic cells in all 26 families are evaluable: reliable leaves with ≥ 5 comparable primary cells in the same family / clone tree.
- Not evaluable:
  - 18,919 cells lie in no family / clone tree of their case, mostly Quinn cells of other clones;
  - 7,944 have character missing share > 0.5.
- No cell is ambiguous or not informative under the fixed rules; the median tree-nearest share is < 5% in 21 of 26 families (5–29% in 3457 T4 and in Quinn 100k LG4, 5k LG6, LG12, LG17).

**Flags.**
- High missingness, reported separately: Quinn 30k LG1 (34% of leaves unreliable), 5k LG1, 5k LG15, 5k LG9.
- Low tree resolution. KP 3724's tree has a 6,519-child polytomy: a metastatic cell's tree-nearest set is typically 86 primary cells, and 8,466 primaries sit at one distance from S1 cells.

**Total-mass coverage.**
- KP-Tracer: every primary cell belongs to the family, so c_OT = c_ind = 1.0.
- Quinn: the LL side pools every clone. OT puts a median 0.09–0.20 of a metastatic cell's mass on primary cells of its own clone, against 0.05 under the independent coupling. That is a clone-level signal.
- All distances below are conditional on the comparable (same-tree) primaries, as fixed in PLAN.md.

## Primary results (evaluable cells; `results/summary/`)

**ΔD = D_ind − D_OT** compares the weighted tree distance with the independent coupling that has each method's actual marginals. ΔD_norm = ΔD / (D_ind − d_min).

| Strategy | Method | Families with ΔD > 0 | 95% CI above 0 / below 0 | Mice with ΔD > 0 | Median mouse ΔD_norm | Top-1 in tree-nearest set (median family) vs independent |
|---|---|---|---|---|---|---|
| RNA-only | A Standard OT | 23 / 26 | 15 / 0 | 8 / 8 | 0.029 | 0.053 vs 0.041 |
| RNA-only | B Our OT | 22 / 26 | 16 / 1 | 7 / 8 | 0.025 | 0.053 vs 0.040 |
| CNV-only | A | 24 / 26 | 13 / 1 | 8 / 8 | 0.021 | 0.051 vs 0.041 |
| CNV-only | B | 24 / 26 | 14 / 1 | 8 / 8 | 0.019 | 0.054 vs 0.042 |
| RNA+CNV | A | 25 / 26 | 15 / 1 | 8 / 8 | 0.029 | 0.051 vs 0.041 |
| RNA+CNV | B | 22 / 26 | 18 / 0 | 8 / 8 | 0.025 | 0.054 vs 0.042 |

**The direction is consistent; the size is small.**
- **In tree steps.** Median family values, all strategies and methods: D_OT 11.21–11.27 against D_ind 11.34–11.35. The achievable minimum d_min is 5.3.
- **Top-k.** The top-1 / 5 / 10 primaries are at a mean distance of 11.18–11.30, i.e. near the independent expectation and far from the nearest relatives. The top-10 mean is below D_ind in 19–24 of 26 families.
- **Close-relative mass.** OT puts 1.11–1.18 times the independent mass on the tree-nearest relatives (median family ratio of mean M_OT to mean M_ind).
  - The cell-level mean of log2(M_OT / M_ind) is negative (−0.9 to −1.5). That is a log mean over many cells with almost no mass on their few nearest relatives, so the ratio of means is the reportable summary.
- **One family carries most of the effect: KP 3724.**
  - ΔD_norm is 0.36 (A) / 0.32 (B) for RNA-only and 0.32 / 0.34 for CNV-only; top-1 in the nearest set is 9.4% against 2.9%.
  - Every other family is 0.00–0.13. Quinn 100k LG4 (0.13 / 0.06) and 5k LG6 (0.09 / 0.07) are next.

## Is there signal beyond the subclone? (post hoc, PLAN.md dated section; `results/within_subclone/`)

Primaries were restricted to the metastatic cell's own frozen subclone.

**KP 3724's effect is subclone-level, not cell-level.**
- OT puts 0.91 (A) / 0.87 (B) of a metastatic cell's mass on its own level-1 subclone, against 0.535 expected (S1 cells avoid the L1.2 primaries; Figure C).
- Within the subclone, ΔD_norm is 0.001 / 0.006.

**Within the subclone, across all families:**

| Level | Strategy | Method | Families with ΔD_norm > 0 | CI above / below 0 | Mice > 0 | Median family ΔD_norm |
|---|---|---|---|---|---|---|
| L1 | RNA | A | 18 / 25 | 10 / 1 | 8 / 8 | 0.010 |
| L1 | RNA | B | 21 / 25 | 12 / 0 | 7 / 8 | 0.015 |
| L1 | CNV | A | 17 / 25 | 10 / 2 | 8 / 8 | 0.007 |
| L1 | CNV | B | 17 / 25 | 9 / 2 | 8 / 8 | 0.008 |
| L1 | RNA+CNV | A | 18 / 25 | 11 / 1 | 8 / 8 | 0.009 |
| L1 | RNA+CNV | B | 21 / 25 | 10 / 1 | 8 / 8 | 0.010 |
| L2 | RNA | A / B | 10 / 15, 13 / 15 | 7 / 2, 7 / 1 | 5 / 7, 6 / 7 | 0.018, 0.023 |

- **A residual cell-level signal remains.** It is positive in most families and mice, but about 1–2% of the achievable improvement.
- **Exceptions:**
  - Quinn 30k LG2 L2 (22 cells): 0.60 / 0.61.
  - Quinn 5k LG15 L1 (87 cells): 0.19 / 0.19.
  - Quinn 5k LG4 is negative at L1 (−0.07 A) and L2 (−0.20 A / −0.06 B).

## Standard OT (A) vs Our OT (B), paired on the same cells

| | Families B higher / A higher | Median family B − A (ΔD_norm) | Mice B higher |
|---|---|---|---|
| RNA-only | 9 / 17 | −0.004 | 2 / 8 |
| CNV-only (same representation, solver only) | 16 / 10 | +0.001 | 3 / 8 |
| RNA+CNV | 10 / 16 | −0.002 | 2 / 8 |
| within subclone L1: RNA / CNV / RNA+CNV | 13 / 12, 12 / 13, 13 / 12 | +0.002, −0.000, +0.000 | — |

The differences are tiny compared with the effects (about 0.001–0.01 of normalised distance) and point no consistent way.

## Figures (`results/figures/`, rules fixed in PLAN.md)

- **A** (`A_lineage_matching_<strategy>_<family>.pdf`, 78 files).
  - Each file is a family's largest level-1 clade, with up to 40 primary and 40 metastatic leaves.
  - Every drawn metastatic cell is linked to its top-1 OT primary under A and under B, coloured by tree distance; top-1 primaries outside the clade go to a marker.
  - In most families the links cross the clade at large distances: top-1 is rarely in the tree-nearest set.
- **B** (`B_ot_vs_independent.pdf`). Per family, ΔD_norm with its 95% interval, A and B side by side, RNA / CNV / RNA+CNV, with mouse means.
- **C** (`C_examples_<strategy>.pdf`). The 90th- and 10th-percentile cells per strategy and method: mass at each tree distance for OT against independent, the number of primaries at each distance, d_min, the top-1, and c_OT.
  - The RNA-only B "success" is a 3724 S1 cell. Its gain comes from moving mass off the other subclone (distances 14–16) onto the 8,466-cell polytomy at distance 8, not onto the nearest relatives.

## Answers

1. **Does OT find closer CRISPR relatives than the independent coupling at the cell level?**
   - Yes, in direction: ΔD > 0 in 22–25 of 26 families and in 7–8 of 8 mice, for every strategy and method.
   - Barely, in size: the median mouse ΔD_norm is 0.02–0.03; the top-1 / 5 / 10 primaries sit at about the independent expected distance; the top-1 hit rate is 5.3% against 4.1% expected.
   - The only large effect, KP 3724, is the subclone-level separation of its sites.
   - For Quinn, OT also concentrates mass on the right clone: 0.09–0.20 against 0.05.
2. **Standard OT or Our OT?**
   - Neither is better at cell-level lineage correspondence.
   - Paired on the same cells the differences are ≤ 0.01 of normalised distance. RNA-only and RNA+CNV lean slightly to A (17 and 16 of 26 families); CNV-only, which differs only in the solver, leans slightly to B (16 of 26). Within the subclone they are even (12–13 each).
3. **Is there cell-level signal independent of the Level 2 subclone result?**
   - A small one. Within the cell's own frozen subclone, OT still prefers closer relatives in most families (17–21 of 25 at L1) and in nearly every mouse.
   - The effect is about 1–2% of the achievable improvement, with a few clones up to 0.2–0.6.
   - It does not reach the level where OT would identify a metastatic cell's closest primary relatives. The large effects seen at the family level are subclone-level, which Level 2 already captures.

Descriptive results; no test statistic is the result (PLAN.md).
