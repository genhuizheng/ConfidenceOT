# ConfidenceOT, balanced vs unbalanced backbone over the v3 grid (Splatter N = 5,000; results 2026-11-23)

Runs: UOT backbone = the existing v3 grid (TACC `v3_grid_20261104`, 2220/2220 rows); balanced backbone (BOT) = the same
worklist rerun with `--backbone balanced` (`v3_grid_bot_20261118`, jobs 1152224-1152227, 2220/2220 rows SUCCESS, 0
FAILED). Everything else is the same: frozen calibration v3 (`within_side_split+mnncovlocal`), M4-E, acceptance 0.99,
effective readout, 296 configurations x L1-L5 x replicates 1-3. Comparison script `compare_bot_uot_grid.py`; tables and
panels in `results/ot_depth_benchmark_20261117/bot_uot_grid/{tables,figures}` (gitignored). Paired by configuration x
level; nothing is selected and no composite is formed. Case 2 Rejection F1 is the primary metric.

## 1 L1-L4: the two backbones give the same gate

Case 2 F1 of BOT and UOT correlate 0.997 over all configurations x levels. Median paired difference (BOT - UOT):

| Level | cosine configs: F1 UOT / BOT / diff | sqeuclidean configs: F1 UOT / BOT / diff |
|---|---|---|
| L1 depth | 0.977 / 0.981 / +0.001 | 0.887 / 0.894 / +0.005 |
| L2 depth + dropout | 0.961 / 0.961 / 0.000 | 0.793 / 0.801 / +0.007 |
| L3 batch | 0.976 / 0.977 / +0.001 | 0.798 / 0.802 / +0.005 |
| L4 batch + perturbation | 0.859 / 0.864 / +0.001 | 0.667 / 0.671 / +0.005 |
| L5 + imbalance | 0.551 / 0.522 / -0.025 | 0.464 / 0.418 / -0.024 |

Case 1 FRR differs by at most 0.002 (median) on L1-L4. On L4, BOT is a little better on the noscale cosine costs
(C3 F1 0.826 -> 0.858, C4 0.802 -> 0.852, FRR down 0.02).

## 2 L5 (population imbalance): the balanced backbone over-rejects shared cells

With unequal population proportions BOT lowers Case 2 F1 for 92% of cosine and 74% of sqeuclidean configurations
(difference > 0.005), and raises Case 1 FRR (median 0.252 -> 0.315 cosine, 0.213 -> 0.275 sqeuclidean). A balanced plan
has to move all mass, so the proportion difference itself pushes shared cells to poor partners and the gate rejects
them; the unbalanced plan can leave the excess untransported. The main configuration C1 (`ranknm256-midrank|none|scale|cos|pca50`):
L5 F1 0.675 -> 0.627, FRR 0.146 -> 0.211; its L1-L4 values are unchanged (within 0.007).

## 3 Case 3 (disjoint populations)

Cosine configurations assert no false mass with either backbone. On sqeuclidean configurations BOT asserts more false mass
on L1 / L2 (median 0.093 -> 0.141, 0.137 -> 0.157; e.g. `logcpm|none|scale|sqeuc|pca50` L1 0.374 -> 0.557), the same
reason as section 2: a balanced plan cannot leave the disjoint populations unmatched, and only the gate removes them.

## Named configurations (both backbones, every level)

`tables/NAMED_configs_bot_vs_uot.tsv`: C1 main, C3 (`ranknm512-expected|none|noscale|cos|pca30`), C4 (`logcpm|none|noscale|cos|pca30`),
the production baseline (`ranknm256-production|none|noscale|cos|pca30`) and `logcpm|none|scale|sqeuc|pca50`.
Over L1-L5 the configuration with the highest mean F1 is C1 for both backbones (UOT 0.902, BOT 0.894; worst level 0.675
vs 0.627); the order of configurations is nearly the same for both.

## Answer

On this grid the backbone matters only when population proportions differ (L5) or the cost is sqeuclidean with disjoint
populations, and in both places the unbalanced backbone is the better one. Where the earlier local tests favoured BOT
(C2 under a depth-identity conflict, RESULTS_SPLATTER.md section 4) the effect was specific to that conflict; it does not
appear in the L1-L5 grid. Panels: `figures/scatter_{case2_f1_eff,case1_frr_eff,case3_false_asserted_mass}_{level}`.

Observation, not investigated: on L3 and L4 the Case 3 values of `logcpm|none|scale|sqeuc|pca50` are identical for each
backbone (recall 0.794 / 0.787, false mass 0.054 / 0.083); the same holds in the UOT grid, so it is a property of the
L3/L4 disjoint units, not of the backbone.
