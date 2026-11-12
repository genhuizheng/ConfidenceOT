# State–lineage extension: four runs (2026-11-12)

Plan: `PLAN_v2.md`, "Extension 2026-11-11" (7ea2f8b), fixed before computing. Frozen A1–A4 (`REPORT_v2.md`) unchanged.

Code: `85_` (TACC export), `86_` (audit, harmonization), `70_` (runs), `88_` / `89_` (summaries, figures).

Tables: `results/rna_lineage_discordance_20261110/{kp_all_audit, state_lineage_ext/<run>, ext_summary}`.

## Audit (E2, `kp_all_audit/`)

**Eligible cells:** 74,399 of the 104,197 cells in the authors' object, in 32 mice.
- **Excluded:** 29,759 normal cells, and cells outside a tumour.
- **Allele table:** one malignant labelled tumour cell lacks an allele record. 17,917 normal cells do have allele
  records; they are excluded anyway.
- **Unfiltered table share** (cells absent from the FINAL table): 25% in 3454, 12% in 3515, 5% in 3724, 0-1%
  elsewhere.
- **MetFamily:** agrees between the expression object and the allele table for 100% of cells.
- **Sides:** 15,809 metastatic cells in 12 mice; 20 mice have primary tumours only.

**OT units:** all 34,820 OT-unit cells are inside the eligible set, and their state labels are identical. The
eligible set adds 39,579 cells and 22 mice.

**Primary vs metastasis:** the README rule and 70_'s regex agree on 100% of cells.

**Resolution** (frozen rule, recomputed on the analysed cells): 18 mice have at least one resolved family. On OT
cells it reproduces the frozen `lineages/resolution.tsv` exactly.

## E1 harmonization (`kp_all_audit/harmonize_*.tsv`)

Two candidates. Apc Mesenchymal-1 (Met) has no parent label, so it is not a candidate.

| Candidate → parent | Top-50 marker overlap with parent (best other) | Spearman with parent (best other) | Centroid cosine distance (best other) | Rule |
|---|---|---|---|---|
| Mesenchymal-1 (Met) → Mesenchymal-1 | 20 (3) | 0.962 (0.907) | 0.104 (0.379) | merge |
| Mesenchymal-2 (Met) → Mesenchymal-2 | 29 (9) | 0.975 (0.905) | 0.128 (0.426) | merge |

**Per mouse:** in 3451 and 3724, Mesenchymal-2 (Met) is closer to Pre-EMT (0.06) than to Mesenchymal-2 (0.11-0.12).
The pooled rule merges it; that per-mouse disagreement is a caveat on the merge.

## A2: does the lineage–state association hold in the fuller data?

**Mean over mice** (`ext_summary/mouse_cramer_pooled.tsv`), Cramér's V excess = V − within-site null:

| Run | Resolved scope | All lineages | All lineages without 3724 |
|---|---|---|---|
| OT units, original | 0.037 (3 mice) | 0.058 (7) | 0.021 (6) |
| OT units, harmonized | 0.037 (3) | 0.068 (7) | 0.019 (6) |
| All cells, original | **0.149 (18)** | 0.127 (23) | 0.120 (22) |
| All cells, harmonized | 0.149 (18) | 0.131 (23) | 0.120 (22) |

- **All cells:** most mice are above the null. The largest excesses are 3507 (0.63), 3765 (0.51), 3724 (0.28),
  3435 (0.27) and 3730 (0.27). Near 0: 3451, 3458, 3466, 3434 and 3467.
- **Annotation:** original vs harmonized changes nothing material.
- **Without 3724:** the all-cell result holds (0.120). The OT-unit result was mostly 3724.

**Per state across mice** (all cells, sensitivity scope; mice with Hill z ≤ −2 out of mice analysed; median z).
"Concentrated" means fewer clades than the null:

| Concentrated in few clades | Shared by many clades (at or above the null) |
|---|---|
| Early EMT-1: 8 of 11 mice (median z −4.3) | Pre-EMT: 4 of 15 (−1.2; 1 mouse above the null) |
| Apc Early: 5 of 6 (−6.1) | Early EMT-2: 2 of 6 (−1.5) |
| Mesenchymal-2: 4 of 6 (−5.4; 3 of 5 without 3724) | Endoderm-like: 4 of 9 (−1.8; 1 above) |
| Lung progenitor-like: 3 of 3 (−18) | High plasticity: 1 of 7 (−0.5; 1 above) |
| Gastric-like: 1 of 2 (−4.4) | AT2-like: 2 of 12, and **5 of 12 above the null** (+1.8) |

- **Mesenchymal-1:** it reaches 50 cells only in 3724 and 3434, so it cannot be read across mice.
- **The interim observation holds on 74k cells, without 3724 and in both annotations.** Mesenchymal / EMT-1 / Apc
  Early states sit in fewer lineages than chance. Pre-EMT, Early EMT-2, Endoderm-like and AT2-like are spread over
  many.

## A3: primary vs metastasis (within mouse)

**Which mice qualify:** 6 mice have at least 2 shared states with ≥ 20 cells on each side.

**Coupling beyond the null at equal n** (all cells, original annotation, sensitivity scope):

| Mouse | Primary | Metastasis | Identifiable share, primary / metastasis |
|---|---|---|---|
| 3451 | 0.019 | 0.003 | 0.78 / 0.97 |
| 3457 | 0.047 | 0.000 | 1.00 / **0.00** (not identifiable) |
| 3460 | −0.003 | 0.078 | 0.89 / 0.75 |
| 3467 | 0.007 | 0.150 | 1.00 / 0.94 |
| 3513 | 0.005 | −0.044 | 0.99 / 1.00 |
| 3724 | 0.198 | 0.006 | 1.00 / 0.69 |

**Means:** primary 0.046 vs metastasis 0.032; without 3724, 0.015 vs 0.037.

**Per state** (`state_side_runs.tsv`; largest-clade-share excess at equal n):
- 3724 Mesenchymal-1: primary 0.20, metastasis 0.005, but the metastatic side is 0.7% identifiable.
- 3724 Mesenchymal-2 (Met): primary 0.29, metastasis 0.003, and the metastatic side is identifiable.
- 3467 Early EMT-1: primary 0.02, metastasis 0.19.
- Most other states are near 0 on both sides.

**Verdict:** the interim statement "metastases are less lineage-coupled" does **not** generalise.
- It rests on 3724, where the metastatic Mesenchymal-2 (Met) cells are identifiable and uncoupled.
- In 3457 it cannot be tested.
- Two mice (3460, 3467) go the other way.
- It is reported as mouse-specific.

## What changes from REPORT_v2

1. The state-specific lineage dependence (concentrated vs shared states) is now supported across 18-23 mice and does
   not depend on 3724.
2. Primary vs metastasis: no consistent direction. 3724 shows weaker coupling in metastases; 3460 and 3467 show
   stronger.
3. Harmonizing the "(Met)" labels does not change any conclusion.
