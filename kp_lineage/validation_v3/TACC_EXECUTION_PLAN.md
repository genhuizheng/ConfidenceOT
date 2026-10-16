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
| C | Freeze `LEVEL3_VALIDATION_DESIGN.md` and `METRIC_TAXONOMY_AND_FEASIBILITY.md` | — | — | after A, B, the literature review and the user's approval (given 2026-10-15 with refinements) |
| D | Coupling reruns (`submit_v3.sh couplings`): the 50 shared-tree pairs with more than one distinct L (design rules 1–3), both labels, M4-E and M4-R couplings and joint PCA coordinates; then `03_compare_reproduction.py` over all (rule 4). No lineage quantity. `02_run_pair.py` unchanged: every baseline is built in F from the saved coordinates and `cost_scale` | 2 arrays × 8 tasks + 1 dependent check = 17 | 1 node per task, 4 h limit (pilot: 3724 → L1 took 13 s; the largest, 3724 → S1, is 14,775 × 3,782); 6–7 pairs per task, round robin over the sorted list, which puts the 4 pairs of 3724 in 4 different tasks | after C; submitted only after the user sees the plan and the queue |
| E | Level 2 exploratory multi-metric | 1 | 1 node, 2 h | after A |
| F | Level 3: lineage-resolution diagnostics; P1, ΔS_norm, ΔR, P2; cost rebuild and scaling-form check; baselines EOT_C(a,b), balanced OT, kNN at the three comparison levels; decompositions; nulls with their constraint checks | 16 (by pair blocks) | 1 node, 2 h each | after D; code not yet written |
| G | Mouse-level statistics, report tables, export package | 1 | 1 node, 1 h | last |

**Storage (scratch).**
- One coupling matrix for all 53 shared-tree pairs, in one representation, is 0.78 GB as float64 (mouse 3724: 0.73 GB).
- Four couplings × two representations come to about 6.3 GB uncompressed, written as compressed npz.
- Expected for stage D (from the pilot, scaling with n × k, compressed): about 2–3 GB per label, mostly mouse 3724 (3724 → S1 alone is about 42 times 3724 → L1).
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
