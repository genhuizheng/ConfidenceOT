# KP-Tracer validation_v3: TACC execution plan

**Where things go.**
- Outputs: `/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/validation_v3/` (audit/, coupling_pilot/, coupling/, lineage_references/, metrics_level2/, metrics_level3/, baselines/, statistics/, export/, logs/).
- Code: `kp_lineage/validation_v3/`.
- Every job: partition gg, account MCB26031, CPU only.

**Queue cap.** Array tasks count one by one against the 40-job cap, and no stage below queues more than 16 at once.

**Guarantees.**
- The original experiment (`prepared/`, `ot/`, `readout/`, `validation_v2/`) is only read.
- Every phase refuses when its output holds SUCCESS, both at submission and again when the job starts.

| Stage | What | Jobs | Resources | Status |
|---|---|---|---|---|
| A | Audit v3 (`01_audit_v3.py`): inventory; prior, expansion and fitness files; 53 shared-tree pairs; L informativeness; Cassiopeia vs NJ; edit statistics | 1 | 1 node, 2 h limit (expected well under 1 h) | implemented, tested on synthetic data |
| B | Coupling pilot: 3 pairs × 2 representations rerun at the original cost (`02_rerun_with_coupling.py`), then `03_compare_reproduction.py` | 1 | 1 node, 2 h limit (largest: 3724 → L1, 14,775 × 90) | implemented, dry-run checked |
| C | Freeze `LEVEL3_VALIDATION_DESIGN.md` | — | — | after A, B, the literature review and the user's approval |
| D | Coupling reruns for the eligible shared-tree pairs: M4-E, M4-R, PCA coordinates; balanced OT on all cells and on the M4-E-retained cells (needs an opt-in balanced flag added to 02_run_pair.py, default unchanged) | 2 arrays × 8 tasks = 16 | 1 node per task, 4 h limit; about 7 pairs per task, the 4 pairs of mouse 3724 one per task | after C |
| E | Level 2 exploratory multi-metric | 1 | 1 node, 2 h | after A |
| F | Level 3 references, P1/P2(/R), decomposition, nulls, baselines (incl. kNN) | 16 (by pair blocks) | 1 node, 2 h each | after D |
| G | Mouse-level statistics, report tables, export package | 1 | 1 node, 1 h | last |

**Storage (scratch).**
- One coupling matrix for all 53 shared-tree pairs, in one representation, is 0.78 GB as float64 (mouse 3724: 0.73 GB).
- Four couplings × two representations come to about 6.3 GB uncompressed, written as compressed npz.
- The local export carries every coupling except mouse 3724's in full (about 0.4 GB). For 3724 it carries clade-aggregated matrices and per-cell summaries.
