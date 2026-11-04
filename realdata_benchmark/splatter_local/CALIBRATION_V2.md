# L−1 calibration v2: credible-coverage gated MNN (`within_side_split+mnncovgate`), 2026-11-04

Local Splatter benchmark: N = 200 per side, Splatter 1.30.0 defaults (batch 0.1 / 0.1), `--pairing paired`, no
rejection cap.
- Fixed representation: `ranknm256-production|P10s1|noscale|cos`. Method M4-E.
- Code: `run_local_benchmark.py` (correction `mnncovgate`).
- Outputs: `results/splatter_local_20261103/` (RESULTS_calibration_{devall,holdout78}.tsv, CALIBRATION_SELECTION_*,
  FROZEN_CALIBRATION_v2.json; v1 = MNN kept as FROZEN_CALIBRATION_v1_mnn_20261104.json).

## Why v1 (plain MNN + within_side_split) was not general

The MNN translation always finds mutual nearest neighbours, also between populations that share nothing. In Case 3
(nothing shared) the spurious shift pulls disjoint populations together:
- unmatched recall 0.02-0.14;
- false asserted mass about 0.5.

This happens with and without batch. The existence of MNN pairs is therefore no evidence of shared populations.

## Method

1. **Anchors and translation:** anchors are the cross-side mutual nearest neighbours (k = 20, fastMNN's default) of the
   joint PCA. The translation is their median displacement.
2. **Credible anchors:** pairs closer, after the translation, than both cells' own 20th same-side neighbour. The
   translation is re-estimated from the credible anchors (3 rounds).
3. **Evidence:** `cover_min`, the share of each side's cells covered by a credible anchor, taking the smaller side.
4. **Two references from the same data:** the same statistic between
   - two random halves of the source, which share everything, without batch;
   - the source's two k-means halves, which are two different population sets.

   Score = clip((observed − disjoint) / (shared − disjoint), 0, 1).
5. **Gate:** the target is translated by the full credible shift if the score > 0, i.e. coverage exceeds what two
   disjoint population sets give. Otherwise it is not moved. Every within-side distance is unchanged (a rigid
   translation).
6. **Calibration:** the within-side split null and acceptance 0.99, unchanged.

The gate uses no group label and no threshold fitted on F1. Its boundary (score 0) is the data's own disjoint
reference.

## How it was developed and checked

- **Development seeds (replicates 1-2):** the label-free statistics were compared (shared_evidence_v2.py).
  `cover_min` was the only one whose normalised score separated shared from none-shared pairs at every level:
  none-shared maximum 0.00 everywhere; shared minimum 0.32-0.71.
- **Candidates rejected on development:** genebatch (gene-wise robust batch), mnnc (consistency-weighted), mnncred3
  (regional maximum).
- **Validation on seeds never used in development** (replicates 7-8, generated afterwards; replicates 3-6 had been
  looked at). Every candidate and reference arm was run.

| Arm (replicates 7-8) | L3 FRR | L3 F1 | L3 Case 3 recall / FAM | L4 F1 / FRR | L5 F1 / FRR | L0 / L1 / L2 F1 | L1 / L2 Case 3 recall / FAM |
|---|---|---|---|---|---|---|---|
| Raw + within | 0.71 | 0.27 | 0.50 / 0.00 | 0.27 / 0.72 | 0.16 / 0.72 | 1.00 / 0.88 / 0.95 | 0.74 / 0.81, FAM 0.04 / 0.02 |
| Raw + rotation | 0.72 | 0.27 | 0.50 / 0.00 | 0.27 / 0.94 | 0.16 / 0.50 | 0.79 / 0.61 / 0.58 | 0.50 / 0.50 |
| Harmony + within | 0.66 | 0.33 | 0.50 / 0.00 | 0.32 / 0.64 | 0.28 / 0.78 | 1.00 / 0.40 / 0.74 | 0.05 / 0.20, FAM 0.60 / 0.42 |
| MNN + within (v1) | 0.001 | 0.88 | **0.02 / 0.63** | 0.87 / 0.02 | **0.67** / 0.11 | 1.00 / 0.87 / 0.94 | **0.003 / 0.10**, FAM 0.65 / 0.54 |
| **mnncovgate (v2)** | **0.005** | **0.93** | **0.50 / 0.00** | **0.88 / 0.04** | 0.63 / 0.15 | **1.00 / 0.87 / 0.93** | **0.74 / 0.81, FAM 0.04 / 0.02** |

- **In every Case 3 pair the gate stays closed,** so the arm equals raw within there. At no-batch levels it is identical
  to raw within.
- **Selection rule of SELECTION_RULES.md:** development chose mnncred, the independent seeds chose MNN; mnncovgate was
  within one replicate SD of the top on the independent seeds. Pooled as the rule prescribes, mnncovgate is frozen.
- **The user's explicit criteria** (L3 low FRR, L3 high F1, L3 Case 3 recall without forced alignment, no harm at
  L0-L2): mnncovgate is the only arm meeting all of them on the independent seeds.

## Limits

- **L5 (batch + perturbation + imbalance)** stays the hardest level: F1 0.63, FRR 0.15. Plain MNN reaches 0.67, but only
  by aligning Case 3 as well.
- **Case 3 recall of 0.5** is the one-sided reject-all of M4-E with an empty asserted block, which is the correct
  disjoint answer (project rule). The gate restores it; it does not raise it above raw within.
- **Scope of the evidence:** Splatter at N = 200 with five populations. The evidence statistic and its references
  assume each side holds more than one population. A single-population side has no meaningful k-means "disjoint"
  reference.
