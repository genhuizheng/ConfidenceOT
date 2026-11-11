# RNA vs CRISPR lineage: state-lineage association, primary vs metastasis, common support (2026-11-11)

Plan: `PLAN_v2.md` amendment 2026-11-11 (024eb14, fixed before computing), plus the dated post-hoc section.
`REPORT_interim.md` stays as the record of sections 1-5.

Scope and status:
- RNA configuration: Splatter L5 best.
- 48 of 49 units have CRISPR-OT. `L23__KPTracer_3724__3724_NT_T1__family_metastases` is still fitting.
  - A2 and A3 do not need it: they use the raw matrix only.
  - A4 and the view-C tables do not yet include it.

Code: `70_state_lineage.py` (A2/A3), `72_lineage_flags.py` (A1), `75_common_support.py` (A4),
`78_amendment_summary.py`, `80_amendment_figures.py`.

Outputs: `results/rna_lineage_discordance_20261110/{state_lineage, lineages/flags.tsv, common_support, amendment_summary}`.
Figures: `figures/l5best/amendment/`.

## A1. Lineage flags (`lineages/flags.tsv`, every lineage)

| | KP (11 families) | Quinn (148 clones) |
|---|---|---|
| Resolved | 5 | 51 |
| Missing intBC share, median | 0.26 | 0.18 |
| Median depth | 7,022 | 7,020 |
| Tree polytomy share (> 50 children) | NA, trees not local | median 0, max 0.21 (116 trees) |

The primary analysis uses the resolved KP families: 3457 T1, 3460 T1, 3466 T2, 3508 T2 and 3515 T1. Only three of
these (3457, 3460, 3508) have two or more states with at least 50 cells; 3466 and 3515 are almost entirely Pre-EMT.
The sensitivity analysis uses every family.

## A2. KP state–lineage association (raw edits, view C)

**Cells.** 34,820 OT cells with an authors' state. Clades are UPGMA clusters on 1 − s cut at 0.5; clusters under 20
cells are pooled as "minor".

**What the null does.** It permutes states within site, so it keeps clade size, state abundance per site and site
composition. Any clade–state association that comes from clades and states sitting in different sites is therefore
already in the null.

### Observed

**Cramér's V of clade × state** (`mouse_cramer.tsv`):

| Mouse | Scope | V | Null V | z | p |
|---|---|---|---|---|---|
| 3457 | resolved | 0.555 | 0.491 | 9.7 | 0.001 |
| 3460 | resolved | 0.104 | 0.067 | 1.3 | 0.11 |
| 3508 | resolved | 0.906 | 0.896 | 2.2 | 0.02 |
| 3451 | all | 0.087 | 0.087 | 0.0 | 0.47 |
| 3465 | all | 0.102 | 0.071 | 2.6 | 0.02 |
| 3513 | all | 0.023 | 0.025 | −0.1 | 0.49 |
| **3724** | all | **0.540** | **0.257** | **218** | 0.001 |

- **Mean excess (V − null):**
  - resolved: 0.037 (3 mice);
  - all mice: 0.058, or 0.021 without 3724.
- **3508:** V is high, but almost all of it is site. The null is 0.896, and only 63% of its cells sit in sites the
  permutation can move.

**Per state** (`states.tsv`, the clade Hill diversity against the null; negative z = fewer clades than expected):
- **Concentrated in fewer clades than the null:**
  - 3457 Early EMT-1: Hill 1.09 vs 1.29, z −4.7; largest clade share 0.98.
  - 3457 Pre-EMT: z −4.5.
  - 3508 Apc Mesenchymal-2: z −2.7.
  - 3451 Mesenchymal-2 (Met): z −2.4.
  - 3724: every state.
    - Mesenchymal-2: Hill 1.01 vs 1.91; largest clade share 0.999 vs 0.67.
    - Mesenchymal-1: largest clade share 0.90 vs 0.52.
    - Early EMT-2 and Pre-EMT also z ≤ −4.
- **Spread over more clades than the null:** 3451 Endoderm-like (Hill 3.47 vs 2.68, z +2.9).
- **At the null:**
  - Pre-EMT in 3451, 3460, 3465 and 3513 (|z| < 1.8);
  - Early EMT-2 in 3451, 3460 and 3465;
  - Endoderm-like in 3513;
  - Apc Early and Apc Mesenchymal-1 (Met).
- **Absent:** Gastric-like has fewer than 50 cells in every mouse among the OT cells, so it is not analysed.

**3724** (`g1_residuals_3724`):
- Clade c4 holds 6,348 of 7,039 Mesenchymal-1 cells and all 807 Mesenchymal-1 (Met) cells.
- Clade c45 holds 8,880 of 8,886 Mesenchymal-2 cells and 3,570 of 3,606 Mesenchymal-2 (Met) cells.

**Representative family, 3457_Apc_T1** (876 cells; `g5_rep_3457_Apc_T1_*`):
- The edit matrix splits into clades c6, c1 and c12 on disjoint event blocks.
- c1 occupies the Pre-EMT RNA region; c12 sits at the AT2-like tip.
- c6 covers both the Apc Early region and the separate Early EMT-1 / Mesenchymal island.
- So one clade spans two distant RNA states, and the other clade stays in one region.

## A3. Primary vs metastasis (`primary_vs_met.tsv`, `g3_primary_vs_met`)

**Which mice qualify.** A mouse needs at least two states with at least 20 cells on both sides. Four mice qualify
(3451, 3457, 3513, 3724), all in the sensitivity scope; no resolved family qualifies. Two things limit this:
- 3508 labels its states by site (Apc Mesenchymal-1 (Met) vs Apc Mesenchymal-2);
- 3460, 3466 and 3515 are Pre-EMT on both sides.

**Coupling beyond the null at equal n** (all cells):

| Mouse | Primary | Metastasis | Movable share, metastasis |
|---|---|---|---|
| 3451 | 0.072 | 0.013 | 0.97 |
| 3457 | 0.177 | 0.000 | **0.00** (each metastatic site is one clade: the null cannot move) |
| 3513 | −0.006 | −0.036 | 1.00 |
| 3724 | 0.198 | 0.006 | 1.00 |

- **Means:** primary 0.110 vs metastasis −0.004; without 3724, 0.081 vs −0.008.
- **Controls:**
  - The direction (primary > metastasis) holds in every depth and events tertile of 3724.
  - It holds in every depth tertile of 3457, but only weakly in the low-events tertile (0.012 vs 0.000).
  - It holds in all 3 tertiles of 3451 that have enough cells.
  - 3513 has no coupling on either side.
- **3724 metastases:** V is 0.705 against a null of 0.699 while the null can move every cell. Metastatic clade–state
  structure is reproduced entirely by site: each metastasis carries its own clade and state mixture.
- **View C** (`a3_viewc_side_state.tsv`, `g3_viewc_side_state`): divergence excess per state is clearly negative in
  the primary tumours of 3457 (−0.12 to −0.23) and 3724 (−0.15 to −0.22). In metastases it is near 0 for 3457 and
  −0.04 to −0.15 for 3724. The other mice are near 0 on both sides.

## A4. RNA-OT vs CRISPR-OT on a common support (`common_support/`, `g4_*`)

**Gate disagreement** (unchanged; mean of mouse medians):
- KP κ: source 0.056, target 0.007. Without 3724: 0.014 / 0.004. The 3724 L1 units reach a source κ of 0.43.
- Quinn κ: 0.

**Conditional coupling disagreement** (sources and targets retained by both gates, columns renormalised, evaluable
pairs only):

| | TV, own blocks | TV, common block | Top-k overlap | Chance level (k / common sources) |
|---|---|---|---|---|
| KP | 0.72 | 0.68 (0.70 without 3724) | 0.10 | 0.046 |
| Quinn | 0.81 | 0.80 | 0.044 | 0.020 |

- **Mass on the common block:** usually 0.7-1.0 for both couplings. The exceptions are 3465 (0.17-0.29) and some 3724
  and 3457 units.
- **Overlap / chance ratio:** median 1.10 in KP units, 2.0 in Quinn (5.4 for 100k).
- **The KP units with a ratio above 2** are 3465 and 3451, the low-resolution mice. There only 17-77% of pairs are
  evaluable, so restricting to evaluable pairs alone concentrates the mass.

## Separation of claims

**Statistical association (observed):**
1. Beyond site, clade and state are associated:
   - strongly in 3724;
   - moderately in 3457;
   - weakly or not at all in 3451, 3460, 3465 and 3513.
   Few mice have resolved lineages and more than one state, so the per-mouse picture carries the result, not the mean.
2. EMT / mesenchymal states (3724 Mesenchymal-1/2, 3457 Early EMT-1, 3508 Apc Mesenchymal-2) sit in fewer clades
   than the null. Pre-EMT, Early EMT-2 and Endoderm-like sit at, or above, the null.
3. Where it can be tested (3451, 3457, 3724), clade–state coupling beyond site is present in the primary tumour and
   near 0 in metastases. In 3724 this is because metastatic structure is explained by site. In 3457 the null cannot
   test it.
4. On a common support, the two OTs still match targets to different sources (TV about 0.7-0.8). Their top-source
   overlap is at chance in most KP units and about twice chance in Quinn. The gate disagreement and the conditional
   disagreement are both large and are separate findings.

**Possible biological interpretation (not tested):**
- Mesenchymal / EMT states could be heritable clone programs: one clade enters them and stays.
- Pre-EMT / Endoderm-like could be states that many clades pass through or converge on.
- Metastases are each seeded by few clades. Within a metastasis, state then follows site (niche or seeding
  bottleneck) rather than clade identity. That would read as weaker lineage–state coupling, which matches the
  interim view-C result.

**Unproven mechanisms:**
- Direction (state → lineage expansion, or lineage → state) and timing: one sampling time.
- Whether site-specific state is niche-induced or seeded: site and clade are confounded in the metastases.
- Any pathway program. The GSEA of `deg_gsea_v1{,_ds}` is unchanged (A5). It supports no program beyond the KP
  convergent set and the partial trends listed in REPORT_interim.

## Limits
- Cells are only the OT cells of the units. In most mice they cover only Pre-EMT, and Gastric-like is absent.
- 3724 is a single mouse, unresolved by the fixed rule (distinct event sets 8%), and dominates every pooled
  number. Every pooled number is therefore also given without it.
- The authors' state labels include site-specific "(Met)" variants, which limits "comparable states" in A3.
  Merging them would be a new definition, so it was not done.
- KP tree resolution is not available locally.

## Pending
- The CRISPR-OT fit of the 3724 pooled L2/L3 case. Then rerun `30_compare_views.py` (that unit), `40_summarize.py`,
  `75_common_support.py`, `78_amendment_summary.py` and `80_amendment_figures.py`.
