# Technical benchmark: the conditions

What is generated and what each condition means. The scoring is in
`BENCHMARK_METRICS.md`; the figures are in `figures/schematic/`.

## The three technical levels

Degradation arrives in stages rather than as independent axes, because nCount
and nFeature are not separable: moving the library size moves detection with
it, and dropout moves both again.

| level | target-side observation | expected readouts |
|---|---|---|
| `L0_matched` | none | nCount and nFeature match the source |
| `L1_depth` | reads thinned | nCount falls; nFeature follows it |
| `L2_depth_dropout` | reads thinned **and** Splatter dropout on the target batch | both fall further, nFeature further than nCount |

**Each level is its own realization.** The three draw independently from the
same Splatter biological parameters, each with its own seed:

```
seed = base + 1000 * replicate + 137 * level_index
```

No level reuses another's counts, and `L2_depth_dropout` is **not**
`L1_depth` with more dropout applied. It is a separate draw with a separate
observation setting.

This is a correction to an earlier design that derived all three levels by
transforming one realization. Under that design a difference between levels
was partly a difference in the same cells rather than in the measurement,
which is not what the benchmark claims to vary; and L2 was a link in a chain
rather than a condition.

The biological parameters are identical across levels and replicates:
`batchCells = c(N, N)`, `batch.facLoc = batch.facScale = 0`,
`group.prob = rep(1/K, K)` with `K = 5`. The batch effect is off so that the
two sides cannot differ biologically, which is what makes "every population is
on both sides" true by construction rather than by assumption.

## Where each operation lives

**Dropout is Splatter's own**, `dropout.type = "batch"`, with `dropout.mid`
set low on the source batch so its logistic sits at zero and at the requested
value on the target. It belongs to the simulation, not to a mask applied
afterwards.

**Thinning is not, and cannot be.** Splatter has one global library size and
no per-batch override, so the depth mismatch is a binomial thinning of the
target's counts in `20_build_conditions.py` — the exact likelihood of
sequencing the same library less deeply, every read surviving independently
with probability theta. It is the only post-hoc operation on counts, and it is
applied to that level's own draw.

**No level holds one readout while moving the other.** At a fixed library size
nothing can change how many genes are detected except changing what the counts
land on, and that is a change to the cell rather than to the measurement.

## The three biological cases

Built from each generated condition, so every case exists at every level.

| case | construction | ground truth |
|---|---|---|
| `all_shared` | both sides keep all K groups | reject nothing |
| `population_lost` | the last group is removed from the **target** | reject the **source** cells of that group |
| `population_emerged` | the last group is removed from the **source** | reject the **target** cells of that group |

The rejecting side is the side that still *has* the group, which is the
opposite of the side the removal was applied to. Both directions are built
because the gate is called on both sides, and a benchmark that only ever asks
one side to reject cannot see half of what it claims to measure.

Labels come from `colData(sim)$Group`, so the truth is what the simulator was
told to make, not something inferred afterwards.

## What is measured rather than set

`theta` and `dropout.mid` are nominal. The axis a result is plotted against is
the realised ratio, written into the manifest per condition:

```
R_nCount   = median(nCount_target)  / median(nCount_source)
R_nFeature = median(nFeature_target) / median(nFeature_source)
```

`min_nFeature_source` and `min_nFeature_target` are recorded too, with
`rank_cut_valid = min(...) > rank_top_n`. The rank encoding's fixed cut is the
only reason it is depth-invariant; once the cut exceeds the sparsest cell's
detected-gene count, detection breadth comes back in through it. A condition
with `rank_cut_valid = false` is outside the valid rank-encoding regime and
must not be read as a performance point.

## Layout

```
generated/N{N}/rep{r}/{level}/     counts.mtx  cells.csv  genes.csv  generation.json
conditions/N{N}/rep{r}/{level}/    manifest.csv
    {pair_id}/                     source.h5ad  target.h5ad  truth.csv.gz  readouts.json
```

`pair_id` is `N{N}_rep{r}_{level}_{case}`.

## Scale

`N x replicate x level x case x preprocessing`. Twelve preprocessing
configurations are expressible as labels; the other four cells of the
factorial put library-size normalisation and rank encoding in the same field
and are a bit-for-bit identity, asserted in
`tests/test_benchmark_identity.py`.
