# DEG / GSEA of the RNA-OT vs CRISPR-OT correspondence difference (plan v3)

**Status:** fixed 2026-11-12, before any group, contrast or DEG of this plan was computed. A later change is a dated
section at the end, marked post hoc.

**Reused unchanged:**
- the 49 RNA-OT / CRISPR-OT fits (M4-E);
- the common support (`75_` / `92_`);
- the all-eligible-cell A2 clades;
- the authors' states (`Cluster-Name`, original annotation);
- sites;
- the existing DEG / GSEA pipeline (`60_deg_gsea.py`);
- the earlier divergent / convergent contrasts (`deg_gsea_v1{,_ds}`), which are kept.

**Not done:** no OT refit, no new solver, no threshold changed after results.

## Scope

- **KP units:** every KP unit (45), RNA configuration L5 best.
- **Focus mice in the figures:** 3457, 3460, 3724, and **3451 as the weak-association control** (lowest Cramér's V
  excess among the mice with metastases).
- **Every evaluable contrast of every KP mouse** is computed and reported, so the focus does not select results.
- **Quinn:** not included. It has no authors' states and no within-LG clades in the frozen A2.

## Cells and coupling

**Common support.** Sources and targets retained by both gates. Each coupling is restricted to that block and
normalised to total mass 1 (as in the difference flow).

**Gate disagreement** stays outside every contrast and is reported per unit (`flow/gates.tsv`).

**Clade.** The all-eligible-cell A2 clade (`state_lineage_ext/all_cells__original/cells.tsv.gz`). "minor" cells are
never a contrast group.

## A. OT preference within a clade (main)

### A1, source side

**Blocks.** For a source clade a and the same target clade a (same lineage, a = b), within the source site.

**Preference score.** For source cell i in a:
- q_R(i) = P_R(i, targets in a) / P_R(sources in a, targets in a);
- q_C(i) the same for CRISPR-OT, so each sums to 1 over the clade's sources;
- d(i) = n_a · (q_R(i) − q_C(i)), where n_a = the clade's common-support sources. d = 0 when both OTs give the cell
  the same share; uniform = 1/n_a.

**Groups:**
- **RNA-preferred:** top quartile of d.
- **CRISPR-preferred:** bottom quartile of d.
- **Both-supported:** neither quartile, and q_R ≥ 1/n_a and q_C ≥ 1/n_a (each OT gives the cell at least the uniform
  share).

**Evaluability:** each group ≥ 20 cells, so blocks with n_a ≥ 80 only.

**Contrasts:** RNA-preferred vs CRISPR-preferred (main), RNA-preferred vs both, CRISPR-preferred vs both.

### A2, target side

**Cells.** Within a target clade b and one target site.

**Score.** s_R(j) and s_C(j) = the share of target j's column mass (RNA-OT / CRISPR-OT) that comes from sources of
clade b.

**Groups:**
- **RNA-cross:** bottom tertile of s_R (RNA-OT explains the cell from other lineages).
- **RNA-own:** top tertile of s_R.
- s_C is reported per group.

**Evaluability:** each group ≥ 20 cells.

**Contrast:** RNA-cross vs RNA-own.

### Controls for A (both fixed now)

1. **RNA-proximity baseline** (selection control). RNA-OT preference is defined from the same RNA, so expression
   differences are expected by construction. For every A1 block, source cells are ranked by their mean cosine
   similarity to the block's targets (L5-best coordinates). Top vs bottom quartile is the baseline contrast. For A2,
   targets are ranked by the mean similarity to other-clade common-support sources minus that to own-clade sources;
   top vs bottom tertile.
   - Reported: the Spearman correlation of the A-contrast NES with the baseline NES, and the overlap of their top-100
     DEG.
   - Only what differs from the baseline is read as OT-specific. What agrees with it is RNA proximity.
2. **Random-split null.** For every main contrast (A1 RNA- vs CRISPR-preferred; A2), the union of its two groups is
   split at random into groups of the same sizes, 5 draws (seeds 1-5 + 20261112), with the same DEG / GSEA. Reported:
   observed NES against the null NES range per pathway.
3. **CRISPR-preferred groups** are defined from edits, not RNA. Their differences from both-supported cells are not
   selection-induced in RNA, but they are confounded with RNA through lineage.

## B and C: cell set

B1 and C1 do not use OT. They use every eligible KP cell (E2) of the 10 KP mice in OT units, with the all-eligible-cell
clades; sites as in E2.

## B. Lineage-close / RNA-far (descriptive)

**B1, contrasts.** Within one named clade and one site, the two largest authors' states, each ≥ 20 cells: state X vs
state Y. This is the "one lineage, several states" contrast. The states are defined from RNA, so the DEG describes
the states. It is not independent evidence of divergence.

**B2.** The earlier view-C divergent vs concordant contrasts (`deg_gsea_v1`), reused where they exist.

## C. RNA-close / lineage-far

**C1, contrasts.** Within one authors' state and one site, the two largest named clades of the same family, each ≥ 20
cells: clade X vs clade Y. The groups are defined from edits, so a difference here is lineage-specific expression
within the same state. It is the one contrast type not selected on RNA.

**C2.** The earlier view-C convergent vs concordant contrasts, reused.

**Random-split null:** 5 draws for every B1 and C1 contrast, as in A.

## DEG / GSEA

**Counts.** Raw counts of the authors' object (`raw/X`; the counts the OT inputs were built from). They come from a TACC
export of every eligible cell of the KP mice in OT units, one h5ad per mouse.

**DEG and GSEA:** `60_deg_gsea.py` unchanged.
- normalize_total 1e4, log1p;
- genes detected in ≥ 10 cells of the contrast;
- Wilcoxon;
- hallmark prerank (MSigDB mouse hallmark 2025.1, all 50 sets), 1,000 permutations, seed 20261110, set size 15-500.

**Depth thinning:** the same run with `--downsample` (P10 thinning).

**Immediate-early genes:**
- **Set:** Fos, Fosb, Jun, Junb, Jund, Egr1, Egr2, Egr3, Ier2, Ier3, Ier5, Atf3, Dusp1, Nr4a1, Klf2, Klf4, Klf6, Zfp36,
  Btg2, Cyr61 / Ccn1, Hspa1a, Hspa1b.
- **Report:** their share of each contrast's top-50 genes (by |score|).
- **Sensitivity:** GSEA rerun with these genes removed from the ranking.

**Per contrast QC:**
- cells per group;
- median total counts and detected genes per group, with their ratio;
- median informative edits and missing-intBC share;
- clone size;
- site;
- state composition;
- the unit's gate fractions.

**Evaluable:** every group ≥ 20 cells. Depth imbalance is reported, not filtered on; the thinned run is the check.

## Reading

- **Replication.** Cells of one tumour are not independent, and there is one tumour per site, so pseudobulk
  replication is not available. DEG / GSEA is exploratory effect-size analysis.
- **What is read:**
  - logFC;
  - gene rank;
  - NES;
  - FDR, as given, not as evidence;
  - the direction of each hallmark set across mice (sign of NES per mouse, with and without 3724, raw and thinned).
- **A pathway is called consistent** only if all of the following hold. This is a reading rule, not a test:
  - same NES sign in ≥ 75% of evaluable mice;
  - in both the raw and the thinned run;
  - outside the random-split range in at least half of its contrasts;
  - not explained by the RNA-proximity baseline (for A).
- **3724** is reported separately; every pooled reading is also given without it.

## Figures (planned)

1. **Difference flow with DEG annotation:** a small table beside representative blue / orange ribbons. Only
   consistent pathways or genes are shown, each labelled with the contrast it comes from.
2. **Within-clade expression heatmap:** cells of one A1 block (RNA-preferred / both / CRISPR-preferred), the top
   genes of the main contrast. Tracks: state, site, depth, informative edits.
3. **Pathway NES heatmap:** consistent and focus pathways only, contrast type × mouse. The full hallmark tables go to
   the supplement.
4. **Across-mouse consistency:** per pathway, NES per mouse (dots) for each contrast type, raw and thinned.

Every figure labels whether its groups were chosen by OT difference, by CRISPR edits or by RNA state.
