# Splatter N = 5,000: batch-effect audit and the default-parameter run (2026-11-03)

**Sources:**
- Root `splatter_n5000_default_20261103`, the `--param-set splatter_default` chain (TACC jobs 1119454-1119464, 864 / 864
  rows).
- `BATCH_EFFECT_AUDIT_20261103/` (job 1119458): 27_ tables and 28_ Splatter defaults / code from the generation
  container.
- The frozen run (`splatter_n5000_benchmark_20261101`) for comparison.

Downloaded to `results/splatter_n5000_default_20261103/`. Descriptive only: no threshold, filter or score is derived
from it.

## 1. Parameters actually used

The defaults come from `SPLATTER_DEFAULTS.tsv`, read from `newSplatParams()` of Splatter 1.30.0 in the container.

| Parameter | Splatter 1.30.0 default | Frozen run | Default run |
|---|---|---|---|
| batch.facLoc | 0.1 | 0 (L3-L5) | **0.1** (L3-L5), 0 below |
| batch.facScale | 0.1 | 0.0922 (L3-L5) | **0.1** (L3-L5), 0 below |
| dropout.type / mid / shape | none / 0 / -1 | experiment / 1.0 / -1, target only | **experiment / 0 / -1, both sides** (L2 on) |
| lib.loc / lib.scale | 11 / 0.2 | 11 / 0.2 | 11 / 0.2 |
| de.prob / downProb / facLoc / facScale | 0.1 / 0.5 / 0.1 / 0.4 | defaults | defaults |
| bcv.common / bcv.df, nGenes | 0.1 / 60, 10,000 | defaults | defaults |

**Kept as design factors** (Splatter has no parameter for them):
- target lib.loc + log 0.5 (L1);
- the Group1 target perturbation (L4, A4), drawn from the default DE parameters;
- target proportions 0.40 / 0.25 / 0.15 / 0.12 / 0.08 (L5, A5);
- 5 equal groups;
- N = 5,000 per side.

**How the sides are assigned:** source = Batch1 of the deep call, target = Batch2 of the shallow call. So every source
cell and every target cell are in different batches.

**Batch mechanism:** Splatter's own. One lognormal factor per gene and batch (`getLNormFactors`, every gene
selected). It is applied to the gene means before group DE, library size, BCV, counts and dropout
(`SPLATTER_BATCH_CODE.txt`).

## 2. Magnitude of the default batch effect

**True per-gene log2(BatchFacBatch2 / BatchFacBatch1)** (TRUE_BATCH_FACTORS, rep 1; reps 2-3 the same):
- every gene is shifted;
- SD 0.29; median |log2FC| 0.20; P90 0.48;
- 40% of genes above 0.25, 8.4% above 0.5, none above 1.
- In the frozen run the same SD was 0.19 (facLoc 0, facScale 0.0922). The default batch is about 1.5x larger in SD.

**Observed pseudobulk log2FC, same population, target vs source** (GENE_LOG2FC, genes with source CPM >= 10, mean
over populations and replicates):

| Level | Median abs log2FC | P90 | Share above 0.5 | Change vs L2: median / P90 | r with true batch log2 |
|---|---|---|---|---|---|
| L2 | 0.108 | 0.265 | 0.3% | | |
| L3 | 0.251 | 0.607 | 17.9% | 0.243 / 0.588 | 0.91 |
| L4 | 0.256 | 0.621 | 18.7% | 0.248 / 0.600 | 0.89 |

**Technical distributions** (TECHNICAL, medians): batch changes them very little.
- L2: source nCount 55,364 / target 26,561; nFeature 3,931 / 2,720; zero fraction 0.61 / 0.73.
- L3: nCount 55,337 / 26,556; nFeature 3,919 / 2,714.

**Side separation** (SEPARATION; side AUC of a cross-validated classifier on the 30 PCs, and the share of a cell's 15
same-population neighbours from the other side, 0.5 = mixed):

| Level | logcpm AUC / mix | ranknm256 AUC / mix | ranknm256 P10 AUC / mix |
|---|---|---|---|
| L0 | 0.50 / 0.50 | 0.51 / 0.50 | 0.51 / 0.50 |
| L1 | 0.98 / 0.20 | 0.89 / 0.38 | 0.64 / 0.49 |
| L2 | 0.99 / 0.11 | 0.92 / 0.36 | 0.65 / 0.49 |
| L3 | 1.00 / 0.00 | 1.00 / 0.00 | 1.00 / 0.00 |

The side explains about 30% of the within-population PC variance at L3 in every representation.

**Costs** (COSTS; cosine cost divided by the median cross-side cost; ranknm256 P10, the best configuration):

| Level | Same population, within side | Same population, across sides | Different populations, within side | Different populations, across sides |
|---|---|---|---|---|
| L2 | 0.49 | 0.52 | 1.05 | 1.05 |
| L3 | 0.28 | **0.74** | **0.59** | 1.03 |

- **At L3 the same population across sides costs more than different populations within one side.** This holds in
  88-97% of source cells, both for medians and for nearest neighbours, in all three representations.
- In **100%** of source cells, the nearest same-population target cell lies beyond the 99th percentile of the
  within-side nearest-neighbour costs. At L2 this share is 0.6% (ranknm256 P10).

## 3. L2 -> L3: what changes

From IDENTITY_VS_L2, per replicate.

**What is identical:**
- cell ids, group labels and expected library sizes;
- every generation-record field except the batch fields and the realised depth summaries.

**What is not (a defect):** **the group DE factors (DEFacGroup1-5) of L3, L4 and L5 differ from L2's.**
- **Cause:** R's `rnorm(n, mu, 0)` returns mu without drawing. At batch.facScale 0 (L2, A4, A5), Splatter's batch
  factors consume no random numbers, so the stream moves on differently and the next step, group DE, is drawn from a
  different state.
- **Consequence:** L3-L5 are the same populations in distribution but **a different draw of them** than L2 / A4 / A5.
  The generator comment "batch factors are drawn in every call whatever their scale" is wrong for scale 0.
- **The same applies to the frozen run,** whose L2 also has facScale 0.
- **Comparisons that are not paired:** L2 vs L3, A4 vs L4, A5 vs L5. Each confounds the batch with a new draw of group
  DE.
- **Comparisons that are paired:** L3 / L4 / L5 among themselves (identical gene truth), and L2 / A4 / A5 among
  themselves (identical gene truth; A4 and A5 source counts bit-identical to L2).
- **Cell-level noise:** the target is in any case a fresh realisation at every level, because the BCV draws depend on
  the means.

**Possible fix, not done:** draw the batch factors at the default scale in every level and neutralise them where the
level has no batch (a trace that sets them to 1 after drawing). The random stream then stays the same and only the
batch differs.

## 4. F1 (Case 2, M4-E; mean over 8 configurations and 3 replicates)

| Level | Frozen | Default |
|---|---|---|
| L0 | 0.988 | 0.988 |
| L1 | 0.935 | 0.935 |
| L2 | 0.448 | **0.796** |
| L3 | 0.242 | **0.167** |
| A4 | 0.400 | 0.654 |
| L4 | 0.239 | 0.167 |
| A5 | 0.354 | 0.565 |
| L5 | 0.193 | 0.167 |

**By configuration** (default run):

| Configuration | L2 | A4 | A5 | L3-L5 |
|---|---|---|---|---|
| ranknm256-production P10 | 0.989 | 0.832 | 0.819 | 0.167 |
| ranknm256-production none | 0.973 | 0.816 | 0.771 | 0.167 |
| ranknm1024-expected P10 | 0.957 | 0.759 | 0.706 | 0.167 |
| logcpm P10 | 0.790 | 0.602 | 0.508 | 0.167 |
| rankhvg2000 P10 | 0.692 | 0.590 | 0.455 | 0.167 |

- **L3-L5:** **every configuration is at 0.167 in every row, and Case 1 false rejection is exactly 0.5.** One side is
  rejected whole and the other kept whole.
- **Comparators:** IC-POT F1 is at most 0.05 and Vanilla UOT is 0.
- **Default dropout** (mid 0, both sides) is much milder than the frozen run's target-only mid 1.0: L2 rises from
  0.448 to 0.796. The ranking of configurations is unchanged: ranknm256 P10 is first.

## 5. Mechanism of L2 F1 0.99 -> L3 0.167

**The calibration cannot see the batch:**
- The rejection cost is calibrated on a within-side split null: two halves of one side, which share one batch.
- Under the default batch, every cross-side cost, including the same population's, exceeds the calibrated range
  (§2, 100% of cells).
- So M4-E rejects the whole unmatched side.

**The representation still holds the populations, but with a side offset:**
- Same-population retrieval stays high.
- The batch offset alone is larger than the difference between two populations within one side.

So no preprocessing in the grid separates the batch from biology. Removing it would need batch handling before OT
(not part of this benchmark).

## 6. Is a batch-strength sensitivity experiment needed?

Under the default batch, every configuration and both comparators fail completely. So the benchmark cannot say where
between L2 (no batch) and L3 (default batch) the method breaks.

A strength series would answer that, for example batch.facScale x {0.25, 0.5, 0.75} of the default with facLoc 0.
But it uses non-default batch parameters. Under the rule that Splatter stays at its defaults, it would have to be a
separate, labelled sensitivity experiment, never part of the L0-L5 levels.

**Not run.** Also not done: the fix of §3, and any regeneration.
