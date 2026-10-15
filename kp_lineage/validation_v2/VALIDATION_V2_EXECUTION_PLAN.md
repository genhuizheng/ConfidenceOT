# KP-Tracer validation_v2 execution plan

Outputs live under `/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/validation_v2/`: `audit/`, `lineage/`, `cross_mouse/`, `baselines/`, `readout/`, `figures/`, `logs/`.

- **Code** is in the repo under `kp_lineage/validation_v2/`.
- **Every job** runs on partition gg, account MCB26031, on CPU.
- **Submit cap.** Array tasks count one by one against the 40-job cap, and no step below exceeds 12 tasks at a time.

| Step | What | Job | Resources | Status |
|---|---|---|---|---|
| A | Lineage data audit, `03_audit_lineage.py` | `submit_v2.sh audit`, 1 job | 1 node, under 1 h (reads ~90 trees and 208 small CSVs) | done 2026-10-15, section C filled, e9e1a1f. Run twice, identical (see note) |
| B | Existing primary readout, `02_kptracer_readout.py`, unchanged | `submit_v2.sh readout`, 1 job | 1 node, under 30 min | done 2026-10-15; supported in the primary arm. Run twice, identical (see note) |
| C | Freeze the preregistration | none (a commit) | — | done: frozen 2026-10-15, text of b0817f1, freeze commit 5577e5f |
| D1 | E1 and E1b lineage concordance, `04_lineage_concordance.py` (no figures on TACC) | `submit_v2.sh lineage`, 1 job, together with E | 1 node, 2 h limit (expected minutes) | implemented, tested on synthetic data |
| E | Visualization-ready export, `05_export_package.py`: per-cell, per-pair, per-mouse tables, original trees, leaf map, metadata, configuration, MANIFEST.tsv, MD5SUMS; then one tarball and its md5 | same job as D1, runs only if D1 succeeded | minutes | implemented, tested on synthetic data |
| D2 | Cross-mouse manifests: `28_ --size-column target_malignant_n`, then path mapping | 1 job | minutes | to implement after C |
| D3 | Cross-mouse OT, 2 arms × 6 tasks | 2 arrays, 12 tasks | 72 cores per task. The matched runs took 0.5–15 min per task, so expect the same order | after D2 |
| D4 | E2 readout | 1 job | minutes | after D3 |
| D5 | E3 baselines | to be designed | — | exploratory, after D1 |
| F | Statistical report first, then figures, locally from the downloaded export | local | — | after the statistics are reviewed |

Every analysis script:
- takes explicit arguments;
- writes a provenance record (commit, inputs with md5, arguments);
- writes `SUCCESS` only after its outputs are complete and validated;
- exits nonzero on any error.

Nothing expensive is submitted before C.

**Note, 2026-10-15: steps A and B each ran twice.**
- The gg jobs 1074424 (audit, 15:36:56–15:37:24) and 1074425 (readout, 15:38:06–15:38:28) left the queue and completed first.
- The same scripts were then run on the login node at 15:42, so the files on disk are from that run.
- The code and inputs were identical, and so was the stdout: `diff readout/readout_login.log readout/readout_1074425.log` and `diff validation_v2/audit/audit_login.log validation_v2/logs/audit_1074424.out` are both empty. The primary readout is therefore unchanged.
- To prevent a recurrence, `submit_v2.sh` now checks for existing outputs again when each job starts, not only when it is submitted.
