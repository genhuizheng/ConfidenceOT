# The five strategies without rejection (Level 2), and the CNV baseline

Checks made 2026-10-22, before any Level 2 CNV coupling. Nothing is recomputed: the CNV profiles are the existing infercnvpy exports, the CNV cost is `02_run_pair.py:cnv_coordinates` as shipped. Level 1 and its results are untouched.

## CNV baseline (`check_cnv_baseline.py`)

- **What is stored.** `49_infercnvpy_unit.py:224` writes infercnvpy's `obsm["X_cnv"]` + 1 as float32, one row per cell, one column per window (100 genes, step 10; `genes.tsv` lists the windows). `X_cnv` is centred at 0 and sparse, so copy-neutral is exactly 1.0 in `cnv.f32`.
- **What the "− 1" does.** `cnv_coordinates` subtracts the 1 again (the export's own offset, kept for the R inferCNV file format) and runs a joint PCA (30 components, seed 20260831 + index). PCA centres every column, so no constant can change the coordinates.
- **Synthetic export-shaped unit** (1,200 cells × 900 windows, 80% of entries exactly 1):

  | Variant | max \|Δ coordinates\| (relative) | max \|Δ cost\| |
  |---|---|---|
  | values as stored, no shift | 4.6e-14 | 0 |
  | values − 1 (shipped) | 0 | 0 |
  | values − 1 + 7 (any constant) | 9.5e-11 | 0 |

- **Real profiles.** Every exported unit: KP-Tracer `infercnvpy_mouse` (11 mice) and Quinn `infercnvpy_experiment` (100k, 10k, 30k, 5k, LM0). Up to 4,000 cells per unit; TACC jobs 1091585 / 1091586 (identical reruns); output on TACC in `level2_pairwise_ot/strategy_check_cnv/`.
  - KP units have 2,269 windows, Quinn units 2,944 or 2,995.
  - Baseline: in every unit the median is 1.0000, and 86.8–89.2% of entries are exactly 1.0. No entry is within 1e-3 of 0. Means are 1.0005–1.0037, maxima 1.23–1.56.
  - Invariance: across the stored values, the shipped "− 1" and an arbitrary constant, the coordinates differ by at most 7.5e-9 (relative 4.2e-9; 3460), and in most units exactly 0. The costs differ by at most 2.4e-7, which is float32 rounding at a cost median near 1.
  - **The "− 1" only undoes the export's "+ 1" and changes neither the CNV PCA coordinates nor the cost.** The existing CNV implementation is used as it is.

## Behaviour of the strategies without rejection (`check_strategy_identities.py`, synthetic, ε 0.1)

**Two-step as defined in Level 1.** Step 1 gates the cells, step 2 runs on the retained cells.
- Without rejection step 1 retains every cell under both methods: BOT has no gate; our UOT at budget 0 retained 1.000 / 1.000.
- So step 2 sees exactly the single-step cost, and the plans are identical (max |Δ| = 0) for both methods and both orders.
- This is exact for any data: same cells, same representation, same seeds.

**A "soft" two-step that keeps step 1 without rejection.** Step 2 is regularised toward the step-1 plan P₁ instead of the product of the marginals: min ⟨C₂, P⟩ + ε KL(P ‖ P₁).
- **Standard (BOT):** it equals single-step OT on C_rna + C_cnv exactly (max |Δ| 4e-16), for either order.
  - P₁'s potentials are separable and cancel under fixed marginals, so the order of the steps is lost.
  - That plan is the shipped RNA+CNV cost (C_rna + C_cnv)/s at ε/s (s = its median): RNA+CNV at a different ε.
- **Ours (UOT):** the potentials do not cancel; they shift the relaxed marginals.
  - The plan differs from UOT on C_rna + C_cnv: total mass 0.51 vs 0.69, max |Δ| 7e-4.
  - The order matters slightly (mass 0.509 vs 0.502).
- It would be a new method. It is defined by neither the Level 1 strategies nor the existing code, and for Standard OT it collapses to RNA+CNV.

**A genuinely two-stage matching without rejection** would have to restrict step 2 by step 1 in a hard way. Examples are keeping only the pairs whose step-1 mass passes a threshold, or matching inside CNV-defined blocks. That needs a new parameter (threshold or block definition), which is a new method, not a re-run.

## Method × strategy

| Strategy | A, Standard OT workflow | B, Our OT workflow | Independent? |
|---|---|---|---|
| RNA-only | `logcpm_noscale` on raw counts → BOT | `ranknm256_noscale_ds_cos` on equalised counts → UOT, budget 0 | **yes** (running) |
| CNV-only | CNV PCA (shipped) → BOT | the **same** CNV PCA → UOT, budget 0 | **yes**, but A and B differ only in the solver: the CNV path does not use the RNA preprocessing label |
| RNA+CNV | [logcpm PCs ‖ CNV PCs], each block at median cost 1 → BOT | [ranknm256_ds_cos PCs ‖ CNV PCs] → UOT, budget 0 | **yes** |
| CNV→RNA | = RNA-only of A, exactly | = RNA-only of B, exactly | **duplicate**: not computed |
| RNA→CNV | = CNV-only of A, exactly | = CNV-only of B, exactly | **duplicate**: not computed |

A two-step strategy that differs from the single-step ones without rejection needs a new definition (above). With the Level 1 definitions it is a duplicate.
