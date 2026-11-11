# Matrix-based vs tree-based CRISPR validation of the v3 grid: results (2026-11-10)

Definitions: `MATRIX_METRICS.md`.
- Fixed at 2061c92, before any score.
- Clarifications 3d77805 and 8b54276 were added before any score.

Scoring run:
- TACC job 1130536: 49 units × 296 configurations × M4-E / M4-R = 29,008 fits, all scored.
- Output: `results/crispr_matrix_benchmark_20261108/tacc_scores/` (md5 b630e71c90f69b4114d46bf866d3eb0e).
- Summary: `results/crispr_matrix_benchmark_20261108/summary/` (`30_summarize.py`).

The old tree-based protocol and its results are unchanged. M4-E is reported throughout; M4-R is in the tables and
agrees closely.

## 1. Data quality and evaluability (audit)

**Source of the states.**
- KP: the allele table, FINAL for its cells and unfiltered for the 21,471 cells that FINAL leaves out. The two agree
  on every shared row.
- Quinn: one allele table per experiment.

**Coverage.** After this rule, every OT cell of the 49 units has CRISPR states.
- Comparable sites: ≈ 100% of cross-side pairs within a lineage.
- Shared informative edit: 50-93% of those pairs.
- KP 3724 L2 and 3515 K2 have 70% of their cells only in the unfiltered table. The results without those cells are
  close: 3724 median ΔS_norm 0.41-0.49 against 0.42-0.52.

**What limits resolution:**
- KP edits are near-saturated (uncut 0.1-14% of observed states).
- Many KP characters are founder (≥ 90% one allele), e.g. 3457 T4: 23 of 30; 3460: 20 of 30. Few informative edits
  remain per cell (median 1-2 in 3513 and 3457 T4).
- In 31% of KP allele-table rows one resection spans two cut sites; it is counted once.
- Quinn: uncut 28-65%.

## 2. Results (M4-E; value = mean over mice of the per-mouse value; [range over the 296 configurations])

| Level | Dataset / stratum | Mice | Median over configs [min, max] | Configs with bootstrap CI > 0 | Coverage (median) |
|---|---|---|---|---|---|
| L1 family d | KP pairs | 8 | 0.195 [−0.003, 0.259] | 274 / 296 | 1.00 |
| L1 own-clone mass gain | Quinn | 4 | 0.137 [0.106, 0.150] | 296 | 0.80 |
| **L3 matrix ΔS_norm** | KP cases | 4 | 0.129 [0.091, 0.167] | 216 | 0.91 |
| | KP cases without 3724 | 3 | 0.013 [−0.004, 0.032] | 158 | 0.91 |
| | KP same-family pairs | 10 | 0.004 [−0.005, 0.018] | 0 | 0.96 |
| | KP pairs without 3724 | 9 | 0.000 [−0.004, 0.003] | 0 | 0.97 |
| | Quinn | 4 | 0.022 [−0.005, 0.051] | 198 | 0.90 |
| L3 tree ΔD_norm (secondary) | KP cases | 4 | 0.083 [0.052, 0.118] | 228 | 0.91 |
| | KP cases without 3724 | 3 | 0.012 [−0.002, 0.033] | 179 | 0.91 |
| | Quinn | 4 | 0.032 [0.020, 0.044] | 293 | 0.92 |
| **L2 matrix, intBC allele groups** (BA − chance) | KP cases | 3 | 0.068 [0.029, 0.111] | 276 | 0.92 |
| | KP pairs | 7 | 0.023 [−0.005, 0.047] | 184 | 0.95 |
| | Quinn | 4 | 0.045 [0.027, 0.066] | 241 | 0.92 |
| L2 matrix, UPGMA | KP cases | 4 | 0.275 [0.223, 0.317] | 296 | 0.92 |
| | KP pairs | 9 | 0.051 [0.002, 0.086] | 61 | 0.98 |
| | Quinn | 4 | 0.042 [0.023, 0.061] | 284 | 0.91 |
| L2 tree subclones (secondary) | KP cases | 3 | 0.293 [0.071, 0.376] | 288 | 0.89 |
| | Quinn | 4 | −0.011 [−0.046, 0.018] | 0 | 0.54 |

- **Bootstrap caveat:** with 3-4 mice, a bootstrap interval is close to the range of the mouse values, so "CI > 0" is a
  weak statement.
- **Size rule:** the L2 differences from the size rule equal or exceed the differences from chance (BA_size = chance
  when every class is present).

**Per mouse** (median over configurations):
- **L3 matrix:** KP 3724 0.47, 3454 0.03, 3457 0.004, 3515 0.003. Same-family pairs −0.009 to 0.044. Quinn 100k 0.032,
  30k 0.019, 10k 0.016, 5k 0.010.
- **L3 tree:** 3724 0.29; the others 0.005-0.026; Quinn 0.009-0.048.

## 3. Gate vs coupling (edit agreement of the matched sources; `summary/gate_vs_coupling_L3.tsv`)

| Stratum | All sources (no gate) | Retained sources (independent) | OT coupling | Gate gain | Coupling gain |
|---|---|---|---|---|---|
| KP same-family pairs (40) | 0.292 | 0.424 | 0.424 | +0.132 | +0.001 |
| KP cases (5; target-weighted, 3724 dominates) | 0.346 | 0.348 | 0.589 | +0.002 | +0.241 |
| Quinn clones (113) | 0.098 | 0.098 | 0.108 | 0.000 | +0.010 |

**One metastasis at a time.** For each KP metastasis paired with its primary, the M4-E gate keeps the primary cells
whose edits match the metastasis. 3724: retained source 28-57%; edit agreement rises 0.20-0.52 over the whole
primary. Within the retained cells, the coupling is no better than independent.

**3724 pooled case.** When the metastases are pooled, both 3724 primary sub-populations are retained (68%) and the
coupling separates them by metastasis site. Median ΔS_norm per site:

| Site | Median ΔS_norm |
|---|---|
| S1 | 0.40-0.46 |
| L1 | 0.68-0.73 |
| L2 | 0.78-0.80 |
| L3 | 0.56-0.62 |

The pooled depth correlation, ρ(ΔS, depth) −0.35 to −0.44, comes from between sites. Within a site it is −0.26 to
+0.12.

## 4. Matrix vs tree

**Rank agreement over the 296 configurations (Spearman):**

| Pair of metrics | KP | Quinn |
|---|---|---|
| L3 matrix vs L3 tree | 0.94 (both carried by 3724) | 0.16 |
| L2 intBC vs L2 tree | 0.22 | 0.52 |
| L2 UPGMA vs L2 tree | 0.24 | 0.47 |

**Quinn.** Tree subclones show no gain (−0.011, no configuration above 0), but the matrix groups give a small positive
one (0.045 / 0.042). Tree-L2 also scores fewer cells: coverage 0.41-0.73 against 0.78-0.99.

**KP.** Tree subclones (0.29) and UPGMA groups (0.28) agree in size. IntBC allele groups are finer and smaller (0.07).

**Without 3724.** Both L3 versions are about 0.01: no reproducible cell-level signal remains.

## 5. Preprocessing

**Best sets are large.** At every level the paired mouse-bootstrap best set holds 21-296 configurations. Few of them
stay in under every leave-one-mouse-out: 1-147.

**Named configurations** (rank of 296; * = in the best set):

| Level | Splatter L5 best | Production baseline | logCPM + PCA only |
|---|---|---|---|
| L1 KP | #25* | #49* | #65* |
| L3 matrix, KP cases | #173 | #140 | #200 |
| L3 matrix, KP without 3724 | #248* | #288* | #41* |
| L3 matrix, Quinn | #226 | #208 | #164 |
| L2 intBC, KP cases | #192* | #293* | #27* |

**Splatter vs lineage rank (Spearman):**
- L1 0.44, L3 matrix KP 0.42, L3 tree KP 0.37, Quinn L3 tree 0.43;
- L2 levels −0.04 to 0.17.

**Reading:** no configuration is separated from the rest on CRISPR evidence. Lineage agrees moderately with the
Splatter ordering at L1 and L3, and not at L2.

## 6. Sequencing depth

Gate margin vs pre-equalisation depth, median over configurations:
- source ρ −0.11 to −0.19, target −0.17 to −0.32: deeper cells are retained more often;
- the range over configurations is −0.89 to +0.68.

ΔS vs depth within lineages: median ρ 0.04-0.06 (KP cases, Quinn), −0.003 (KP pairs).

## 7. Conclusions

1. **Family / clone level:** RNA-only ConfidenceOT agrees with independent CRISPR evidence.
   - In KP, same-family primaries are retained more (d 0.20, 274 of 296 configurations above 0).
   - In Quinn, retained targets put 0.14 more mass on their own clone than the independent coupling.
2. **Within a family:** the evidence sits in **which cells the gate keeps**, not in the pairing.
   - Gate gain +0.13 in edit agreement; coupling gain 0.001 in single primary-metastasis pairs.
3. **Cell level:** the only large edit-agreement gain is 3724's pooled case, where the coupling separates the primary
   sub-populations by metastasis site.
   - Without 3724, matrix and tree L3 are both ≈ 0.01.
   - In KP pairs the matrix L3 is 0.000.
4. **Matrix vs tree:** the matrix-based metrics reproduce the tree-based picture in KP. In Quinn they give a small
   positive subclone-level gain that the tree subclones do not show, at higher coverage.
5. **Preprocessing:** CRISPR evidence does not select a preprocessing. Best sets are large and unstable under
   leave-one-mouse-out.
