# Local Splatter benchmark: selection rules (fixed 2026-11-04, before the L−1 results)

User brief 2026-11-03 / 04.
- **L−1** selects the calibration method. It is a method-selection stage, not a Splatter difficulty level.
- **L1-L5** is the preprocessing grid. It runs with the calibration frozen in L−1, which is never changed inside the
  grid.

**Data:**
- Splatter 1.30.0 (local `D:/R-4.4.1`), defaults; batch 0.1 / 0.1;
- `--param-set splatter_default --pairing paired`;
- N = 200 per side;
- the existing levels (L0-L5, A4, A5) and Cases 1-3;
- no rejection cap in any arm.

**Metrics** (M4-E; per fit, then the mean over pairs and replicates):
- **Case 1 FRR:** false rejection rate, mean of the two sides.
- **Case 2 Rejection F1:** on the side holding the unmatched population (33_ convention), mean of the two directions.
- **Case 3 unmatched recall:** mean of the two sides' rejected fractions. The false asserted mass is reported beside
  it. A one-sided reject-all gives recall 0.5 with an empty asserted block, which is correct for the disjoint case.

## L−1: calibration selection

**Fixed preprocessing:** `ranknm256-production|P10s1|noscale|cos` (PCA 30).

**Arms:**
1. Raw PCA + `within_side_split`.
2. Raw PCA + `cross_side_rotation`.
3. Harmony PCA (harmonypy 0.0.10 defaults, batch = side) + `within_side_split`.
4. Harmony PCA + `cross_side_rotation`.

**Selection, replicates 1-2:**
1. For each arm, take the three metrics as the mean over the 8 levels, each level weighted equally. Also report them
   over the batch levels L3-L5 alone.
2. Remove every arm that another arm Pareto-dominates. Dominates means: no worse on all three metrics (lower FRR,
   higher F1, higher recall) and strictly better on one.
3. Among the remaining arms, choose the highest Case 2 F1. Splatter's primary metric is Case 2 F1.
4. If two arms are within one replicate-to-replicate SD of each other on F1, the lower Case 1 FRR wins.

**"Handles batch"**, checked at each of L3, L4 and L5, both conditions:
- Case 2 F1 is above the same level's one-side reject-all F1. That value is computed from the truth: reject the whole
  side that holds the unmatched population, keep the other.
- Case 1 FRR is below 0.5. A whole-side reject-all gives exactly 0.5.

**Exploration:** if no arm handles batch at all three levels, new batch-aware calibrations are explored under the same
rule.
- **The first candidates** are representation corrections before the within-side null:
  - each side's PCA mean removed;
  - the per-PC rank (quantile) map, which keeps each side's within-side order.
- **Each candidate also reports biological-structure checks,** before and after the transform: same-population
  retrieval among the 15 nearest neighbours, and the cross-side nearest-neighbour population agreement. A candidate
  that loses population structure is reported as such.

**Validation:** the chosen arm is re-checked on independent seeds (replicates 3-4) with the same rule.
- If the same arm wins, it is frozen in `FROZEN_CALIBRATION.json`.
- If not, both results are reported and the arm chosen on replicates 1-4 pooled is frozen. The disagreement is
  recorded.

## L1-L5: preprocessing grid

**Calibration:** the frozen L−1 calibration, fixed for the whole grid.

**Grid:**
- the 592 configurations of `factorial_common.grid()`: logcpm, ranknm K, rank_value K, rankhvg H; ties production /
  expected / midrank; DS none / P10 / P25 / P50; scaling; cos / sqeuc;
- × PCA {30, 50};
- plus the baseline `ranknm256-production|P10s1|noscale|cos` (PCA 30) through the production path.

**Runs:** levels L1-L5 × Cases 1-3 × replicates 1-2.

**Ranking:**
- Per level, by Case 2 F1, with Case 1 FRR and Case 3 recall beside it. There is no composite score.
- The best set at each level is every configuration within one replicate-to-replicate SD of the top Case 2 F1.
- Every configuration's numbers are kept.
