# Native simulation benchmark archive

**Archive location (TACC):** `/corral/utexas/MCB26031/genhui/Primary_Metastasis_archive/simulation_benchmark/`
- Written by `benchmark/tacc/archive_to_corral.sh`, one gg job (48 h limit).
- `benchmark_native/` is copied from `/scratch/10119/ghzheng/OT_project/` as it is: every file, nothing filtered, mode and timestamps kept.
- Only the native benchmark is archived (user, 2026-10-19). The superseded thinning benchmark, `benchmark_technical/`, is not.
- The scratch originals are not changed or removed by the archive.

**Checks written beside the copy** (`checks/`):
- the list of every file with its size, on scratch and on corral, and their diff;
- md5 of every file under 50 MB on both sides, and their diff;
- `STATUS`: OK or FAILED.

**Code:**
- `code_<commit>.tar.gz` is `git archive` of the repository at the commit that ran the archive.
- `run.json` does not record a commit, so the commits that produced each set of runs are listed below from the repository history and the experiment records.

## What the benchmark is

Design: `benchmark/DESIGN.md`; metrics: `BENCHMARK_METRICS.md`.

**Simulation:**
- Splatter's native generator, K = 5 groups, batch effect off.
- Three technical levels, each its own draw:
  - `L0_matched`;
  - `L1_depth`: target `lib.loc` + log 0.5;
  - `L2_depth_dropout`: depth as L1, plus Splatter batch dropout with target `dropout.mid` 1.0.
- Four biological cases: `all_shared`, `population_lost`, `population_emerged`, `populations_disjoint`.
- 3 sizes (1,000, 5,000, 10,000 cells per side) × 5 replicates, so 180 pairs per configuration.

**OT:**
- `cancer_metastasis/02_run_pair.py` via `benchmark/tacc/ot_array.slurm`.
- ConfidenceOT M4-E (and M4-R from the same run), ε 0.1, λ 1.
- Rejection cost from the within-side split null.

**Scoring:** `benchmark/40_score.py`, by rejected fraction against the simulator's own group labels.

## Where each reported result comes from (`benchmark_native/`)

| Result | Runs | Scored tables | Code and record |
|---|---|---|---|
| 32 preprocessing arms, acceptance 0.90 (arm heatmap) | `runs/<arm>/` | `metrics/`, `benchmark_metrics.csv` (local copy: `benchmark_results/native/benchmark_metrics.csv`) | generator 33fa384…0c99f46 (2026-09-30); scorer 505d286 (2026-10-01); heatmap e2fab5b…beeef1a |
| Acceptance 0.95 and 0.99, three arms (ranknm256_noscale_ds_cos, ranknm256_noscale_cos, raw) | `runs_acceptance_0.95/`, `runs_acceptance_0.99/` | `comparator_metrics_full/` (local copy: `benchmark_results/native/comparator_metrics_full/`) | 412bf4e (2026-10-05); the work list `worklist_3arms_balanced.csv` |
| Comparators: Vanilla UOT, Partial OT, Balanced OT; oracle-budget Partial OT; IC-POT at c = 0.5 | `solvers*/` (per solver and arm) | `comparator_metrics*/` | 5d658ef, ed92a35, 2c48479, a2081a3, f778bdd; `benchmark/EXPERIMENT_2026-10-06_IC-POT_COMPARATOR.md` |
| log1p + Pearson correlation over all genes, no HVG (`log1p_noscale_pcc`), acceptance 0.99 and 0.90 | `runs_acceptance_0.99/log1p_noscale_pcc/`, `runs_acceptance_0.90/log1p_noscale_pcc/` | `metrics_log1p_pcc/acceptance_<a>/summary_wide.tsv` (at 0.99 with the three arms scored beside it) | bdc9878 (2026-10-19); `benchmark/tacc/submit_log1p_pcc.sh`; work list `worklist_log1p_noscale_pcc.csv` |

**Inputs shared by all runs:** `generated/` (counts), `conditions/` (pairs, `manifest.csv`, `truth.csv.gz`), `conditions_ds/` (the equalised copies the `_ds` arms read), `worklist_*.csv`, `logs/`.

**Figures and value tables built from these:**
- `figures/slides_20261015/comparator/` (`method_comparison_heatmap_values.csv`, `method_comparison_f1.csv`);
- `figures/slides_20261015_v6/`;
- `benchmark_results/native/heatmaps/`.

## Not archived: `benchmark_technical/`

The earlier benchmark, superseded by the native generator (33fa384, 2026-09-30):
- the target's depth was a binomial thinning of Splatter's output;
- weaker L2 dropout;
- three cases, without `populations_disjoint`.

No reported result comes from it, and it is not in this archive.

## Caveats that travel with the results

- **No single best arm** over the heatmap's columns. Name the criterion with any ranking.
- **Acceptance 0.99 was chosen with the truth in view**, from three values. It is not confirmed on independent data.
- **Equalised L1 is not evidence about depth.** Splatter's composition does not depend on library size, so equalisation removes the L1 depth difference by construction. Equalisation can be judged only at L2 (`benchmark/DESIGN.md`).
- **Native L2 dropout** is `dropout.mid` 1.0, against −1.0 in the earlier thinning version.
- **M4-R cycled** at acceptance 0.90 in most pairs other than `populations_disjoint`; its numbers there are stopped oscillations.
- **Vanilla UOT on raw** has pairs whose plan mass diverges; its raw numbers are unreliable.

**Status:**
- Directory names above are from the run records.
- The authoritative list of what was copied is `checks/benchmark_native.corral.files.tsv`.
