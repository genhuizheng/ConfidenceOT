# L−1 calibration v3 (final, frozen 2026-11-04): `within_side_split+mnncovlocal`, no DS, effective readout

Supersedes v2 (`CALIBRATION_V2.md`). Frozen file: `results/splatter_local_20261103/FROZEN_CALIBRATION_v3.json`.

## What changed from v2

**1. Downsampling removed (user, 2026-11-04).**
- The fixed calibration representation is `ranknm256-production|none|noscale|cos`.
- The L1-L5 grid uses DS none only.
- Every other grid dimension is kept: encoding, K / H, ties, scaling, cost, PCA 30 / 50.

**2. Local credible correction (`mnncovlocal`).**
- **The gate is v2's:** credible-anchor coverage, normalised between the data's own random-half and k-means-half
  references; open only when the score is above 0.
- **When the gate is open,** each target cell moves by the Gaussian-kernel average of the credible-anchor displacements
  around it. The bandwidth is the median 20th same-side neighbour distance. The average is shrunk towards the global
  credible shift by one pseudo-anchor.
- **When the gate is closed** (nothing shared), nothing moves.

**3. Effective readout.**
- When one side retains no cell at all, the other side's retained cells have no possible partner: the asserted mass on
  them is exactly 0, as checked on a Case 3 fit. They count as unmatched. Nothing else changes.
- Without this, Case 3 recall is capped at 0.5 for every calibration, raw within_side_split included. The cap is not a
  calibration limit but a readout of "retained without a partner" as matched.

## Selection

| Seeds | mnncovlocal vs mnncovgate |
|---|---|
| Development, replicates 1-2 | No worse on any level; L5 F1 0.530 vs 0.518 |
| Never-used, replicates 7-8 | No worse on any level; L5 F1 0.698 vs 0.655 |

The gate and Case 3 behaviour are identical between the two.

**Never-used seeds 7-8, effective readout:**

| Metric | L0 | L1 | L2 | L3 | L4 | L5 | A4 | A5 |
|---|---|---|---|---|---|---|---|---|
| Case 1 FRR | 0.000 | 0.000 | 0.004 | 0.005 | 0.030 | 0.139 | 0.016 | 0.008 |
| Case 2 F1 | 1.000 | 0.913 | 0.961 | 0.957 | 0.920 | 0.698 | 0.945 | 0.954 |
| Case 3 recall | 1.000 | 0.938 | 0.858 | 1.000 | 1.000 | 1.000 | 0.858 | 0.906 |

Same seeds, raw within_side_split, effective readout:
- L3 / L4: Case 1 FRR 1.000 / 1.000, Case 2 F1 0.346 / 0.338. These are batch failures.
- Plain MNN: Case 3 recall 0.02 at L3.

No group label and no evaluation F1 enters the gate. The calibration is frozen and not changed inside the grid.
