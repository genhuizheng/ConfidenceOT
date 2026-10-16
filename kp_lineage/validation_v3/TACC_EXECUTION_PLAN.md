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
| A | Audit v3 (`01_audit_v3.py`): inventory; prior, expansion and fitness files; 53 shared-tree pairs; L informativeness; Cassiopeia vs NJ; edit statistics | 1 | 1 node, 2 h limit (expected well under 1 h) | **done**, job 1075005 (2026-10-15); see the audit notes below |
| B | Coupling pilot: 3 pairs × 2 representations rerun at the original cost (`02_rerun_with_coupling.py`), then `03_compare_reproduction.py` | 1 | 1 node, 2 h limit (largest: 3724 → L1, 14,775 × 90) | **passed**: refits in job 1075006 (all six exit 0); its comparison step crashed writing the summary when every row passed (fixed, e4a09d2) and was rerun alone with `submit_v3.sh compare`: 12 of 12 (3 pairs × 2 labels × M4-E/M4-R) reproduce exactly (rejection cost identical, 0 gate disagreements, margin and mass differences 0) |
| C | Freeze `LEVEL3_VALIDATION_DESIGN.md` | — | — | after A, B, the literature review and the user's approval |
| D | Coupling reruns for the eligible shared-tree pairs: M4-E, M4-R, PCA coordinates; balanced OT on all cells and on the M4-E-retained cells (needs an opt-in balanced flag added to 02_run_pair.py, default unchanged) | 2 arrays × 8 tasks = 16 | 1 node per task, 4 h limit; about 7 pairs per task, the 4 pairs of mouse 3724 one per task | after C |
| E | Level 2 exploratory multi-metric | 1 | 1 node, 2 h | after A |
| F | Level 3 references, P1/P2(/R), decomposition, nulls, baselines (incl. kNN) | 16 (by pair blocks) | 1 node, 2 h each | after D |
| G | Mouse-level statistics, report tables, export package | 1 | 1 node, 1 h | last |

**Storage (scratch).**
- One coupling matrix for all 53 shared-tree pairs, in one representation, is 0.78 GB as float64 (mouse 3724: 0.73 GB).
- Four couplings × two representations come to about 6.3 GB uncompressed, written as compressed npz.
- Measured in the pilot (whole pair directory, both couplings, PCA coordinates and edges, compressed): 3457 → L1 0.6 / 0.7 MB, 3515 → K2 2.0 / 4.6 MB, 3724 → L1 23 / 29 MB (ranknm256_noscale_ds_cos / raw). The compressed size is far below the float64 estimate.
- The local export carries every coupling except mouse 3724's in full (about 0.4 GB). For 3724 it carries clade-aggregated matrices and per-cell summaries.

**Audit notes (job 1075005, read 2026-10-15).**
- 104 eligible pairs: 53 on a shared tree (28 same-family in 7 mice, 25 different-family), 51 unsupported.
- No CNV file in the released data (0 of 500 inventory entries). raw arm cell sets identical to the main arm.
- Zero-variance pairs (one distinct L, design rule 3): 3454 T2 → N1, 3513 T3 → N1, 3513 T5 → N1. The job's summary listed only the first and the last because it tested `L_variance <= 0` on a float; 3513 T3 → N1 has one distinct L. The rule now tests the distinct count (01_audit_v3.py); the audit is not rerun, the per-pair table already carries `L_distinct`.
- Different-family pairs, by the low-information rule fixed in the design before the audit (modal L share >= 0.9), from the printed table (3 decimals): 3 zero variance, 14 flagged (3508 T1 × 8 at 0.982, all at the root; 3513 T2 and T4 → N1, NJ tree only; 3515 T3 → K2, K3, L1, L2), 8 not flagged (3515 T2 × 6, 3515 T3 → K1 and N1).
- Same-family pairs: flagged or at the threshold in the printed table: 3457 T1 → K1 and T1 → N2 (0.900), 3508 T2 → N2 (0.901). Whether the two 0.900 values reach 0.9 needs the unrounded column.
- Shared-edit share is high across families too (3515 T3 about 1.0, 3508 T1 about 0.4–0.6): identical edits recur in unrelated cells, so a raw shared-edit share is not lineage evidence.
- Cassiopeia vs neighbor-joining agreement on L (Spearman): high for 3454, 3515 T1, 3724 (0.80–0.98); low for the 3508 T2 family (about 0.0–0.26) and mixed for 3460 (−0.21 to 0.64). Results for 3508 and 3460 depend on the reconstruction.
