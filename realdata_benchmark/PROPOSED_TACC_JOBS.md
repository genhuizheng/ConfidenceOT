# Proposed TACC jobs: joint Splatter + LTL331 benchmark

Nothing here has been submitted. Every stage refuses to overwrite and continues on resubmission. All new output
goes to `/scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031` (ROOT); frozen roots are read only.
Protocol: `realdata_benchmark/JOINT_BENCHMARK_PROTOCOL.md` (registered before any new fit).

## Validated locally before proposing

- Representations of every registered configuration come from the production code or from encoders checked against
  it bit for bit (`rb_common`; production ties replicated exactly).
- `05_ltl331_ot_pilot.py`: refitting the frozen `logcpm` configuration on 16 -> 20 weeks reproduces the frozen cost
  and the frozen M4-E and M4-R gates cell for cell (the `ranknm256_noscale_ds_cos` and precomputed-path checks are
  in `ot_pilot_check/`).
- `10_tacc_worker.py`, Splatter path: `tests/test_worker_splatter.py` 9 / 9 (label, precomputed expected-tie and
  Rank-HVG on a seeded P25 draw; fit, score with `benchmark/40_score.py`, resume).

## Stage 1. Worklists (one short job)

```bash
cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT && git pull && bash realdata_benchmark/tacc/submit_benchmark.sh --dry-run worklist
```

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh worklist
```

Builds ROOT/PREPROCESSING_CONFIG_REGISTRY.tsv from the code at this commit and the two worklists. Expected rows:

| worklist | rows | definition |
|---|---|---|
| ltl331 | 7 pairs x (24 historical + 40 registered main) = 448 | acceptance 0.99, LTL331 factorial arguments |
| splatter | 72 condition-pairs x 40 main at 0.99 + 72 x 8 R5 representatives at 0.90 = 3,456 | n_cells 1,000 and 5,000, replicates 1-3, every level and case |

Check ROOT/WORKLIST_SUMMARY.tsv; it prints how many Splatter rows lack a conditions_ds manifest (the frozen-P10 rows
need it).

## Stage 2. LTL331, all pairs

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh --dry-run --nodes 2 --hours 6 ltl331
```

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh --nodes 2 --hours 6 ltl331
```

Estimate: the frozen 2-arm run took 7-8 min per arm for all 7 pairs with 9 pairs in parallel; with balanced OT,
the matched-subset balanced OT and the encodings added, about 3-15 min per row (largest: 12,694 x 3,549 and
1,328 x 19,386 cells). 448 rows x ~8 min / 18 workers per node is about 3.3 node-hours: 2 nodes x 6 h leaves room.
Memory: a dense 12,694 x 3,549 float64 coupling is 0.36 GB; 18 concurrent rows stay well under a gg node's memory.

## Stage 3. Splatter Arm A

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh --dry-run --nodes 6 --hours 8 splatter
```

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh --nodes 6 --hours 8 splatter
```

Estimate: about 1 min per N = 1,000 row and 5 min per N = 5,000 row with 8 solver threads: (1,728 + 8,640) min / 18
workers per node is about 9.6 node-hours; 6 nodes x 8 h with resubmission for the remainder. Each row is fitted and
scored (`score.csv`) in its own directory.

## Progress

```bash
bash realdata_benchmark/tacc/submit_benchmark.sh status
```

## Stage 4. Splatter Arm B (PDX-informed), after the fidelity audit

1. Export (a short job; reads h5ad):

```bash
sbatch -p gg -N 1 -n 1 -t 00:30:00 -A MCB26031 -J rb_mtx -o /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/logs/mtx_%j.out --wrap "source /home1/10119/ghzheng/.bashrc; conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot; cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT; python realdata_benchmark/splatter/21_export_ltl331_mtx.py --run-root /scratch/10119/ghzheng/OT_project/ltl331_confidenceot --out /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/armB/mtx"
```

2. Estimate and simulate once per estimate in the Splatter container (after the export job):

```bash
sbatch -p gg -N 1 -n 1 -t 04:00:00 -A MCB26031 -J rb_splat -o /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/logs/splat_%j.out --wrap "cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT; bash benchmark/tacc/rscript.sh realdata_benchmark/splatter/22_estimate_ltl331.R /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/armB/mtx /scratch/10119/ghzheng/OT_project/realdata_benchmark_20261031/armB/estimates"
```

3. The fidelity comparison (`23_fidelity_check.py`, to be written against the estimates' actual layout) and only then
   the Arm B condition generator; the generator's factors are fixed in the protocol (section 4) and are not tuned on
   OT results.

## Stage 5. KP-Tracer / Quinn (versioned extension; designed, not yet scripted)

- Configuration x solver crossing on the frozen Level 2 cases: the existing runner pairs one preprocessing with one
  solver (`methods.METHODS`: A = `logcpm_noscale` + balanced OT, B = `ranknm256_noscale_ds_cos` + UOT without
  rejection), so the old A vs B contrast mixes the two. A wrapper supplying its own method table (never editing the
  frozen file) will cross {logcpm_noscale, logcpm_noscale_cos, ranknm256_noscale_ds_cos, ranknm256_noscale_cos,
  ranknm1024_noscale_cos} x {balanced OT, UOT without rejection, M4-E at 0.99}, scored by the frozen Level 2 / 3
  scorers into a new root.
- CRISPR editing-site input (exploratory): character-matrix (allele) distances as the cost, alone and with RNA. Its
  evaluation cannot use a tree built from the same sites. Held-out protocol: split each family's intBCs into two
  halves, fit on one half, evaluate on a tree rebuilt from the other half only; the halves still share the lineage, so
  the evaluation is not independent of the input and is reported with that dependence stated.
- Needs the allele tables and trees on TACC; scripts follow after Stages 1-3 are running.
