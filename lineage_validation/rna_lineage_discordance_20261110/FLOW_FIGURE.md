# Tree-annotated OT difference flow (2026-11-12)

Script: `92_flow_figure.py`. **Update 2026-11-12:** clades are now the all-eligible-cell A2 clades, with one shared tree per family (`lineage_tree.py`); the numbers below are from that rerun. The OT-clade version is in `flow_v0_ot_clades/`.

Script: `92_flow_figure.py`. It reuses the existing couplings and gates (Splatter L5 best RNA-OT, M4-E; CRISPR-OT,
M4-E), the frozen A2 clades (`state_lineage/cells.tsv.gz`, original annotation) and the authors' states and sites.
Nothing was refitted.

## Outputs (`results/rna_lineage_discordance_20261110/`)

| File | Contents |
|---|---|
| `flow/units.tsv` | Per KP unit (45): common support, Shared / RNA-enriched / CRISPR-enriched totals, the three bars, same-clade mass of each coupling |
| `flow/flows/<unit>.tsv` | Clade × clade flow matrices: F_RNA, F_CRISPR, shared, rna_enriched, crispr_enriched |
| `flow/clades/<unit>.tsv` | Cells per clade per side, and the clades merged into Other |
| `flow/gates.tsv` | Per unit and side, cell fractions: both retained, RNA-only, CRISPR-only, both rejected |
| `flow/enriched_location.tsv` | Where the enriched mass sits: same-clade vs cross-clade blocks |
| `figures/flow/<unit>.{png,pdf,svg}` | The figures |

## Definitions

- **Common support:** sources retained by both gates × targets retained by both gates.
- **F_RNA, F_CRISPR:** each coupling on the common support, normalised to total mass 1, then summed into
  source-clade × target-clade blocks.
- **Per block:**
  - Shared = min(F_RNA, F_CRISPR);
  - RNA-enriched = max(F_RNA − F_CRISPR, 0);
  - CRISPR-enriched = max(F_CRISPR − F_RNA, 0).
- **Other:** a clade with less than 2% of the common-support cells on both sides. "minor" = the A2 pool of UPGMA
  clusters of fewer than 20 cells.
- **Drawing:** ribbons under 0.2% of the union mass are not drawn; the share left out is printed on each figure and is
  kept in the bars.
- **Bars:**
  1. RNA-OT: Shared + RNA-enriched = Σ F_RNA = 100%.
  2. CRISPR-OT: Shared + CRISPR-enriched = Σ F_CRISPR = 100%.
  3. Overall: Shared + RNA-enriched + CRISPR-enriched = Σ max(F_RNA, F_CRISPR) = 100%.
- **Bars 1 and 2 always have the same split.** Both F sum to 1, so total RNA-enriched = total CRISPR-enriched.
  This follows from the requested normalisation; it is not a result.
- **Tree:** average linkage on the mean 1 − s between clades (up to 300 cells per clade). It summarises the raw edits;
  it is not the authors' Cassiopeia tree.
- **Bar fill and order:** bar fill = the authors' RNA state of the clade's cells. Bar height = the clade's union mass.
  State colours and clade order are fixed in the script and the same in every figure.

## Selected figures

| Figure | Unit | Why chosen |
|---|---|---|
| 3724: lineage–state association | `L23__KPTracer_3724__3724_NT_T1__family_metastases` | The only 3724 unit with two named clades on both sides. c4 = Mesenchymal-1 → Mesenchymal-1 (Met); c45 = Mesenchymal-2 → Mesenchymal-2 (Met) |
| 3457: RNA vs CRISPR difference | `L23__KPTracer_3457__3457_Apc_T4__family_metastases` | The lowest Shared of the 3457 units (68%) |
| Supplementary: clearest difference | `L1__3460__3460_Lkb1_T1__3460_Lkb1_N6` | A resolved family, 5 clades, Shared 40% |

**Not chosen:**
- **3457 T1 → K1:** resolved, but its metastasis is one clade, so every flow ends in c1. The figure is drawn and
  still available.
- **3457 T1 units in general:** the resolved 3457 family has one named clade in every metastasis.

**Caveats on the chosen families:**
- 3724 is unresolved by the frozen rule (distinct event sets 8%).
- Its clades are confounded with site and state.
- Its median depth (11,180) is the highest of the KP mice in the units.
- 3457 T4 is unresolved (distinct event sets 8%).

## Caption

**Same Source → Target, two OTs, at clade level.**
- **Layout:** the source primary tumour is on the left and the target metastasis on the right. Bars are stacked by
  CRISPR clade (the tree on the left summarises edit similarity) and filled by RNA state.
- **Flow width:** clade-to-clade coupling mass on the cells both gates retain; each coupling is normalised to 1 there.
- **Colours:**
  - **Grey (Shared):** mass both OTs place on that clade pair.
  - **Blue (RNA-enriched):** extra mass RNA-OT places there.
  - **Orange (CRISPR-enriched):** extra mass CRISPR-OT places there.
- **Gate annotation:** the small bars above each side give, as cell fractions, which cells each gate retained. They
  are kept apart from coupling mass.
- **Bottom bars:** the same three colours summed over all blocks.
- **Level:** this is clade-level overlap of coupling mass, not agreement of single-cell matches.

## What the 45 KP units show (`flow/units.tsv`, `flow/enriched_location.tsv`)

**Overall overlap.** At clade level the two couplings mostly overlap: Shared has a median of 82% of the union
(range 34-100%). The cell-level total-variation distance is 0.5-0.8, so most of the disagreement lies within clades.

**Same-clade mass.** CRISPR-OT puts more mass on same-clade blocks than RNA-OT in **45 of 45** units:
median 0.675 vs 0.40.

**Where the enriched mass sits:**
- CRISPR-enriched mass is on same-clade blocks: median 87%.
- RNA-enriched mass is almost never on same-clade blocks: median 0%.
- RNA-enriched mass goes mostly to other named clades (median 70%); the rest goes to minor or Other.

**Units with the largest differences** (Shared 34-40%):
- 3513 N1;
- 3460 N2 / N6;
- 3454 N1 / L23;
- 3515 N1.

**3724** pair units have one named source clade on the common support, so their flows are trivial (Shared 97-100%).
Only the pooled case shows the two-clade structure.
