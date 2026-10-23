# CRISPR lineage × site validation and biological application: plan

Started 2026-10-22. This is not a new OT design, and not a new subclone benchmark.
- **The thread:** CRISPR lineage × site ground truth → cell-level retain / reject validation → primary–metastasis correspondence → lineage-informed application.
- **Definitions:** `GROUND_TRUTH_DEFINITIONS.md`.
- **What is untouched:** the frozen protocol, the Level 1 runs, the Level 2 / 3 results and the subclone rules. No OT is re-run.
- **Archive:** everything that existed before this analysis is archived and verified at `F:\ConfidenceOT_Archive\20261022\` (ARCHIVE_MANIFEST.tsv, SHA-256, 7,091 files). Final results of this analysis go to a separate versioned folder there.

## Phases

0. **Archive.** Done: 7,091 files, 10.58 GB, 0 verification failures; git bundle verified.
1. **Lineage × site matrices** (`build_lineage_site_matrices.py`). Authors' annotations only; every KP-Tracer and Quinn mouse; family / clone level and subclone level; coverage reported.
2. **Cell-level gate validation** (`cell_level_gate_validation/`).
   - Inputs: the Level 1 runs (5 strategies × M4-E / M4-R, D_A / D_B / D_C3 / Quinn D_B).
   - Per cell: side, site, labels, comparison class, gate state, margin, effective matching mass.
   - Per run and side:
     - retention and rejection;
     - shared-lineage retention;
     - unmatched-lineage rejection;
     - false rejection among shared cells;
     - ambiguous / unknown coverage.
   - Aggregated cell → lineage × site group → run → mouse; no cell is a replicate.
3. **Figures:** 1 ground truth, 2 retain / reject by lineage × site, 3 rejection vs lineage support, 4 correspondence (gate-driven vs conditional coupling vs marginal composition).
4. **Systematic summary.** Every eligible mouse and comparison, failures and not-evaluable reasons. Checks:
   - multiple primaries;
   - Quinn LL sharing;
   - low abundance;
   - tree coverage;
   - preprocessing / depth;
   - source vs target.
5. **Application extension** (`application_extension/`): lineage-informed state remodeling, after 1–4. Replication is per mouse, and group sizes are checked first.
6. **Supplementary validation.** Level 2 / Level 3 are kept as they are and referenced.

## Data needed and where it is

| Need | Local | Only on TACC |
|---|---|---|
| Authors' annotations (KP MetFamily / Tumor, all 11 mice; Quinn obs, 4 mice), trees, matrices | yes | — |
| Level 1 run-level summaries (all 680 run × method rows) | yes (`summary/runs.tsv`) | — |
| Level 1 per-cell gates (`cell_confidence.csv`) | KP 3508, 3515; Quinn 5k, 10k, 30k, 100k | KP 3451, 3454, 3457, 3460, 3465, 3466, 3467, 3513, 3724 |
| Level 1 couplings (effective matching mass) | KP 3508, 3515; Quinn 10k (220 files) | the rest |
| Level 1 run sides (`inputs/sides.tsv`, `cell_sites.tsv.gz`) | — | yes |

Phase 2 across every mouse therefore needs a read-only export from TACC: the gates, the sides, and per-cell effective masses computed from the couplings. **It is not submitted without the user's go-ahead**, and nothing on TACC is re-run.
