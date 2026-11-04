# Frozen calibration v3 grid on TACC (2026-11-04)

The local L1-L5 grid (`splatter_local/`, `GRID_V3_REPORT.md`) at full size, on three datasets with one code path:
- Splatter N = 5,000;
- LTL331 (PDX time course);
- KP-Tracer (CRISPR lineage units of the lineage factorial).

## What is fixed

| Item | Value |
|---|---|
| Calibration | `within_side_split+mnncovlocal` (`CALIBRATION_V3.md`). `v3_common.py` imports `run_local_benchmark.correct`; it does not copy it. |
| Readout | Effective readout (`summarize_local.py`) |
| Acceptance | 0.99 |
| Rejection cap | None |
| Downsampling | None |
| Grid | 37 encoding keys × PCA {30, 50} × scaling × {cos, sqeuc} = 296 configurations |
| Baseline | `ranknm256-production\|none\|noscale\|cos\|pca30`. Locally it gave the same gates as the production-path baseline on all 160 rows. |
| Fit arguments | Splatter: the local benchmark's. LTL331: `05_` factorial arguments without `--cell-qc`. KP: `factorial_common.fit_command` without `--cell-qc`. `--cell-qc` filters nothing at thresholds 0 / 0 / 100, and it refuses coordinates. |

## Files

- `v3_common.py`: the grid keys, the representations (encoding once, then each PCA once per key), the correction, and
  the in-process `02_run_pair` fit on the corrected coordinates.
- `build_worklist.py`: one row per unit × key, 8 fits per row.
- `worker.py`: claim-based workers.
  - A claim left by an ended job is taken over.
  - Large units take a limited number of node-local slots.
  - Each configuration is resumable.
  - Outputs per configuration, every dataset:
    - the gates: `cell_confidence.csv.gz`, retained / rejected per cell for M4-E and M4-R;
    - the transition matrices: `coupling_m4e.npz` and `coupling_m4r.npz` (user, 2026-11-04). They are float32,
      compressed, with `source_ids` / `target_ids` in the gates' cell order.
  - Added per dataset: Splatter the scored rows; LTL331 cluster-level coupling mass.
- `collect.py`: `STATUS`, `FIT_METRICS`, Splatter `RESULTS_grid_v3tacc.tsv` (read by
  `summarize_local.py --stage grid`), and LTL331 / KP `CELL_GATES` per unit.
- `tacc/submit_v3.sh` with `tacc/v3_workers.slurm`. The chain is worklist → pilot (1 node) → run → collect.

## Checks done locally

**Splatter equivalence.**
- Setup: `build_worklist.py --conditions-subdir ""` on the local N = 200 root, then `worker.py` on 24 rows (192 fits:
  L3 and L5, replicate 1, keys ranknm256-production, logcpm and rankhvg2000-midrank).
- Compared with `RESULTS_grid_v3.tsv` on 768 rows: rejected counts, rejection cost, asserted and false-asserted mass and
  F1 are all identical (maximum absolute difference 0).
- This includes the row-blocked kernel step of `mnncovlocal`, added for N ≥ 5,000 with the same element-wise
  arithmetic.

## Sizes

| Dataset | Units | Rows | Fits |
|---|---|---|---|
| Splatter | L1-L5 × 4 pairs × replicates 1-3 = 60 | 2,220 | 17,760 |
| LTL331 | 7 adjacent pairs | 259 | 2,072 |
| KP | Every unit of `UNITS.tsv` | 37 per unit | |

Defaults are in `submit_v3.sh`; the pilot measures time and memory.

**Large-unit concurrency.**
- A number in `<dataset root>/MAX_LARGE_PER_NODE` replaces `--max-large` for every worker, running or queued, from its
  next large row.
- This changes only how many large rows run at once on a node, never a result.

**Pilot timing (2026-11-04).**
- No pilot fit failed.
- KP: per fit about 1e-5 s per source × target cell, from 1 s on small units to about 1,000 s on the largest unit
  (9.6k × 9.6k). The large rows (148, over 2e7) carry about 219 of 287 estimated worker-hours.
- LTL331: 60-490 s per fit; the pilot finished in 1 h 9 min.

**Coupling storage.** Compressed float32 takes about 3.4 bytes per entry (measured locally), for both methods:
- Splatter: about 3 TB;
- LTL331: about 0.27 TB;
- KP: as the lineage factorial planned.

**Local LTL331 check.** One fit (16→20 weeks, ranknm256-production noscale cos PCA30) wrote the gates and the cluster
coupling. The gate stayed closed (score 0), so the correction moved nothing for that pair.
