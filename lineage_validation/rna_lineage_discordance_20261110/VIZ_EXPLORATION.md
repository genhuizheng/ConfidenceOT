# Visualisation exploration: RNA vs CRISPR lineage (2026-11-12)

Real data only, no refits, no new thresholds.

**Scripts:**
- `92_flow_figure.py`: design A;
- `95_viz_explore.py`: designs B / C, validation panels, biology panels;
- `96_cell_tree.py`: cell-level trees;
- `lineage_tree.py`: one shared clade tree per family.

**Coverage:** every KP unit (45), every family with ≥ 2 named clades (55), every mouse (32).

**Consistency across figures:**
- clade ids come from the all-eligible-cell A2 clades;
- one tree per family, pruned per figure;
- state colours are fixed.

**Figures:** `results/rna_lineage_discordance_20261110/figures/`, in `flow/`, `explore/` and `cell_tree/`. Rejected
first versions are kept in `flow_v0_ot_clades/` and `explore_v0_first_layout/`.

## Three levels of agreement, kept apart

| Level | What it is | Where shown |
|---|---|---|
| Gate overlap | Cell fractions retained by both / one / neither gate | Small bars above the flow figure, `flow/gates.tsv` |
| Clade-level coupling overlap | min(F_RNA, F_CRISPR) summed over source clade × target clade blocks | Designs A, B, C |
| Exact cell-pair overlap | Per target, overlap of the two couplings' top-15 sources | `V_levels_<unit>` |

**Validation result 1: the levels disagree.**
- Per target clade, the median clade-level Shared is 0.83 of the RNA-OT mass.
- Over the same targets, the median cell-level top-15 overlap is 0.04 (97 target clades with ≥ 10 cells).
- High clade-level agreement must never be read as matched cells.

**Validation result 2: RNA-OT is close to clade-blind** (`V_baseline`).
- Clade-level Shared between RNA-OT and the **independent coupling** (mass placed by clade sizes alone) is a median
  of 0.98.
- RNA-OT is closer to it than CRISPR-OT in 44 of 45 units.
- Shared(RNA-OT, CRISPR-OT) is almost exactly Shared(CRISPR-OT, independent); the points lie on the diagonal.

**So the clade-level overlap mostly measures how far CRISPR-OT departs from clade sizes.** RNA-OT carries little
clade information of its own. That is a finding in itself, but it means the grey "Shared" area must be shown against
this baseline. It is not agreement between two lineage-aware methods.

## OT-difference designs

### A. Tree-annotated difference flow (`flow/<unit>`)

- **Seen in 5 seconds:** on the same Source → Target, orange (CRISPR-enriched) runs straight within a clade, and blue
  (RNA-enriched) crosses between clades.
  - Across 45 units, CRISPR-enriched mass is 87% same-clade (median).
  - RNA-enriched mass is 0% same-clade.
- **Best example:** 3460 N6 (resolved, 5 clades). 3724 pooled for the two-clade Mesenchymal-1 / -2 structure.
- **Limits:**
  - Bars 1 and 2 always have the same split under the requested normalisation.
  - Many thin crossing ribbons when a family has more than 6 clades (3454, 3513).
  - Units whose metastasis is one clade (most 3457 and 3724 pair units) are uninformative.
  - Must be shown with `V_baseline`.
- **Use:** main text, as the visual of "where the two OTs differ", with the baseline caveat.

### B. Difference heatmap (`explore/B_heatmap_<unit>`)

- **Seen:** the sign of F_RNA − F_CRISPR per block, beside Shared. The diagonal (same clade) is orange and the
  off-diagonal blue in nearly every unit.
- **Limits:**
  - Exact numbers are readable, but the reader must know both panels.
  - With 2-3 clades (3724), it carries less than the flow.
- **Use:** supplementary. It is the most precise of the three, and the easiest to check.

### C. Tree + overlap bars (`explore/C_bars_<unit>`)

- **Seen:** per source clade, how much of its mass goes to its own clade (solid) vs other clades (hatched), split
  into Shared / CRISPR / RNA. Large clades with mostly-hatched blue are the cross-lineage correspondences.
- **Limits:**
  - Hatching is hard to read at small size.
  - It shows the source side only.
- **Use:** supplementary, or as a compact alternative to A when a family has many clades.

## Biology designs

### Cell-level CRISPR tree with tracks (`cell_tree/<family>`): the strongest new view

**Layout:**
- leaves are cells, on the same UPGMA tree that defines the clades, with the cut at 0.5 drawn;
- aligned tracks: clade, RNA state, site;
- the informative-edit matrix (present / absent / intBC missing).

**What it shows:**
- **3457_Apc_T1 (resolved): one lineage → several states, organised below the clade cut.**
  - c6 splits into an Apc Early branch and an Early EMT-1 / Mesenchymal-1 branch, each with its own edits.
  - c1 is the metastasis-seeding branch, mostly Pre-EMT.
- **3724_NT_T1 (unresolved): the two clades carry the two mesenchymal programs.**
  - c5 = Mesenchymal-1 / -1 (Met).
  - c45 = Mesenchymal-2. Inside c45, the sub-branch with one extra edit is the metastatic Mesenchymal-2 (Met).
  - This is the clearest lineage → state → site picture in the data.
- **3460_Lkb1_T1 (resolved):** primary and metastatic cells are interleaved inside clades, almost all Pre-EMT.
  Metastases share edits with many branches.
- **3451_Lkb1_T1 (weak control, unresolved):** only 7 informative edits and heavy intBC missingness. States are
  interleaved along the tree. This is low resolution, not evidence of absent association.

**Limits:**
- Subsampled to ≤ 1,200 cells for large families (stated on each figure).
- The tree is ours (UPGMA on the edits), not the authors' Cassiopeia tree.
- The edit matrix is direct evidence only of edit sharing.

**Use:** main text for 3457 and 3724, with 3451 or 3460 as the contrast. Supplementary for every family.

### BA. Clade × edits × states (`explore/BA_lineage_states_<family>`)

- **Seen:** a compact summary of the cell tree, one row per clade: edit frequencies, state composition, primary /
  metastasis share.
- **Limits:** it hides structure below the clade (the 3457 c6 split). Use it after the cell tree, not instead of it.
- **Use:** supplementary.

### BB. Multiple lineages → one state (`explore/BB_state_families_<mouse>`)

**Layout:** per state, its cells split by independent family (colour) and clade (segments), with the null's expected
largest-clade share as a tick.

**Examples:**
- **3513:**
  - Gastric-like: 5 families / 22 clades, shared.
  - Endoderm-like: almost all in family T1, concentrated.
  - Early EMT-1: mostly in T4.
- **3457:**
  - Early EMT-1: 1 family / 1 clade.
  - Apc Early: 4 families.

**Use:** main text. It is the most direct picture of "shared vs lineage-restricted states", with no z-scores.

### E. State concentration across mice (`explore/E_state_concentration_{focus,all}`)

**Layout:** a dot per mouse × state; colour = largest clade share minus the null; area = cells.

**Seen:**
- Early EMT-1, Apc Early, Mesenchymal-2 and Lung progenitor-like are red in most mice.
- Pre-EMT and AT2-like are white or blue.

**Use:** main text. It is the across-mouse summary that BB illustrates for single mice.

### BC. Within-site mosaics (`explore/BC_sites_<mouse>`)

- **Layout:** per site, clades (column width = cells) and their states. Each site is labelled identifiable or not
  identifiable (fewer than 2 clades or states).
- **3724:**
  - The primary has two clades, c45 = Mesenchymal-2 and c5 = Mesenchymal-1.
  - L1, L2 and L3 are single-clade c5 (Mesenchymal-1 / -1 (Met)), so they are not identifiable.
  - S1 is c45, Mesenchymal-2 (Met).
- **3460:**
  - N1 is identifiable: several clades, all Pre-EMT dominated, with similar compositions.
  - N2 is identifiable: its second clade is Early EMT-2 rich.
  - N3-N6 are not identifiable.
- **Use:** main text for primary vs metastasis. It shows the change without a Cramér's V, and it never draws a
  single-clade site as "no association".

### BD. Discordant cells on the RNA UMAP (`explore/BD_discordance_<unit>`)

- **Seen:**
  - Divergent cells (lineage-close / RNA-far) are scattered.
  - Convergent cells (RNA-close / lineage-far) cluster where clades meet. In 3457 that is the AT2-like / c12 tip and
    the c1 / c6 boundary.
- **Limits:**
  - The top-10% flag is relative within each lineage.
  - UMAP distances are not the OT coordinates.
- **Use:** supplementary.

### A2 / A3 statistics panels (`figures/ext/`)

h1 residual heatmaps, h2 / h3 Hill z, h4 / h5 primary vs metastasis. These are for checking, not for the main text:
they need the reader to understand z-scores and permutation nulls.

## Not working well (kept)

- **Clade-level flow for single-clade metastases:** 3457 pair units, 3724 pair units. All flows converge on one bar.
- **Rotated clade labels in the first BC version:** illegible at 5 pt; replaced by horizontal labels.
- **First flow version** (OT clades, ad-hoc tree per figure): clade ids differed from the biology figures (3724 c4 vs
  c5). Replaced by the shared family tree.
- **Bars only (first version of the biology view):** they do not show what a clade is or how clades relate. Replaced
  by the cell-level tree.

## Recommended set (for discussion, not fixed)

1. **Lineage structure and states.**
   - Cell-level tree for 3724_NT_T1 and 3457_Apc_T1, with 3460_Lkb1_T1 as the contrast.
   - Shows: one lineage → several states, organised by sub-branch; Mesenchymal-1 / -2 each carried by one clade.
2. **Shared vs lineage-restricted states.**
   - E (focus states, all mice), with BB for 3513 and 3457 beside it.
   - Shows: EMT-1 / Mesenchymal-2 / Apc Early concentrated; Pre-EMT / AT2-like / Endoderm-like shared.
3. **Primary vs metastasis.**
   - BC for 3724, 3460 and 3467.
   - Shows: mouse-specific change. Metastatic sites with one clade are marked not identifiable.
4. **RNA-OT vs CRISPR-OT.**
   - Flow (A) for 3460 N6 and the 3724 pooled case, with `V_baseline` and `V_levels` as small insets.
   - Shows: CRISPR-OT adds same-lineage mass, RNA-OT adds cross-lineage mass. RNA-OT is close to clade-blind, and
     clade-level agreement ≠ cell-pair agreement.

**Supplementary:** B, C, BA and BD for every unit / family / mouse, and the statistics panels.

**Why this order:**
- Panels 1-3 answer the biology questions from the raw edits and states, which are independent of either OT.
- Panel 4 then shows how the two OTs relate to that structure.
- CRISPR-OT uses the same character matrix, so its agreement with the edit pattern is not an independent validation.
  It is not presented as one.
