# Does RNA-OT pair cells within the same CRISPR clade? (2026-11-19)

Question (user, 2026-11-19): with the clades derived from the CRISPR barcodes as ground truth, the primary and
metastatic cells of one clade should be matched to each other by RNA-OT, i.e. the OT pairs should overlap within clades.

Script `01_clade_overlap.py` (definitions in its docstring, fixed before any value was read). Read-only: existing
couplings and gates, no refit. Tables: `results/clade_overlap_20261119/{units.tsv, mice.tsv}` (gitignored).

## Inputs

- **RNA-OT:** the existing v3-grid M4-E fits, two configurations: `l5best` (ranknm256-midrank, scale, cos, PCA 50) and
  `logcpm_pca` (logCPM, PCA 30, squared Euclidean). No CRISPR information enters these fits.
- **CRISPR-OT:** the existing view-B fits (`rna_lineage_discordance_20261110/10_crispr_ot.py`). Its cost is built from
  the same edits that define the clades, so it is a **circular diagnostic** (what the edits themselves can do at this
  clade resolution), never validation.
- **Clades:**
  - KP: the all-eligible-cell A2 clades (UPGMA on 1 − s within the family, cut 0.5). "minor" and unassigned cells
    have no clade.
  - Quinn: the authors' LineageGroup.
- **Units:** 49. KP: 40 same-family primary → metastasis pairs (L1, 10 mice) and 5 pooled cases (L23, 4 mice). Quinn: 4
  pooled cases (L23, 4 experiments).
- **Coverage:** retained-target mass with a clade, median 0.98 (KP L1), 0.99 (KP L23), 1.00 (Quinn).

## Metric

On the M4-E retained block (retained sources × retained targets, renormalised):
- **O:** share of target mass placed on source cells of the target's own clade.
- **Baselines:**
  - I_ret: the independent coupling with the same marginals (clade shares of the retained sources).
  - I_all: no gate (clade shares of all analysed sources).
- **Decomposition:**
  - gate gain = I_ret − I_all (which sources survive);
  - coupling gain = O − I_ret (which surviving source each target is paired with).
- **O_norm** = (O − I_ret) / (U − I_ret), where U is the share of target mass whose clade has a retained source (the
  most O can reach).

Values are averaged per unit, then per mouse; tables give the median over mice.

## Result

| Stratum | Mice | Coupling | O | I_ret | Gate gain | Coupling gain (median, [min, max] over mice) | O_norm |
|---|---|---|---|---|---|---|---|
| KP L1 pairs | 10 | RNA l5best | 0.386 | 0.385 | 0.003 | **0.002** [0.000, 0.012] | 0.005 |
| | | RNA logcpm_pca | 0.389 | 0.386 | 0.002 | 0.003 [−0.001, 0.011] | 0.007 |
| | | CRISPR-OT (circular) | 0.589 | 0.466 | 0.032 | 0.069 [0.017, 0.493] | 0.240 |
| KP L23 pooled | 4 | RNA l5best | 0.301 | 0.294 | 0.008 | **0.008** [0.002, 0.429] | 0.011 |
| | | RNA logcpm_pca | 0.312 | 0.301 | 0.019 | 0.014 [0.000, 0.388] | 0.017 |
| | | CRISPR-OT (circular) | 0.768 | 0.390 | 0.062 | 0.246 [0.040, 0.482] | 0.380 |
| Quinn pooled | 4 | RNA l5best | 0.423 | 0.218 | 0.001 | **0.077** [0.011, 0.299] | 0.150 |
| | | RNA logcpm_pca | 0.481 | 0.240 | 0.007 | 0.099 [0.014, 0.322] | 0.187 |
| | | CRISPR-OT (circular) | 0.495 | 0.215 | 0.008 | 0.227 [0.044, 0.327] | 0.464 |

**1. The expectation does not hold for KP.**
- In the 40 KP primary → metastasis pairs, RNA-OT places target mass across clades exactly as an independent coupling
  would. The coupling gain is 0.000–0.012 in every one of the 10 mice, and no unit exceeds 0.05.
- Where O is high, the gate made it high, not the pairing. 3724: O 0.844 against I_all 0.410; the gate lifts it to
  0.832 and the coupling adds 0.012.
- 9 of the 40 pairs have a metastasis made of one clade, so there O is decided by the gate alone.

**2. Exceptions with a real pairing effect.**
- **KP 3724 pooled case:** coupling gain 0.43, O_norm 0.91 (logcpm_pca 0.39 / 0.94). This is the known 3724 structure:
  each metastatic site carries its own clade and state. RNA-OT recovers it, but clade, site and state are confounded
  there, and the family is unresolved by the frozen rule (about 3 intBCs).
- **Quinn 100k:** coupling gain 0.30, O_norm 0.41.
- **Quinn 10k:** coupling gain 0.11, O_norm 0.21.
- **Quinn 5k and 30k:** ≤ 0.04. 30k is effectively one clone (I_ret 0.88).

**3. Even the edits themselves do not reach full overlap.**
- CRISPR-OT, whose cost is the edits, has a median O_norm of 0.24 in KP pairs, 0.38 in KP pooled cases and 0.46 in
  Quinn.
- So "complete overlap" is not reachable at this clade resolution even with lineage as the cost. RNA-OT's 0.005 in KP
  pairs is far below that ceiling, not near it.

**4. The two RNA configurations agree** in every stratum, so this is not a preprocessing effect.

## Reading

- RNA-OT pairs by transcriptional state. Clades do not occupy separate states in most KP mice: many clades share
  Pre-EMT and Endoderm-like states (`REPORT_v2.md` of `rna_lineage_discordance_20261110`). So within a correct family,
  RNA-OT has no clade signal to follow.
- Where clade and state coincide, RNA-OT follows the clade: 3724 metastases, Quinn 100k / 10k.
- This agrees with the earlier gate-vs-coupling decomposition (`crispr_matrix_benchmark_20261108/REPORT.md` §3: gate
  +0.132, coupling +0.001 in KP same-family pairs). It does not replace it.

## The grey "Shared" of the flow figures (2026-11-23)

The user asked whether the 3724 pooled flow figure (`figures/flow/L23__KPTracer_3724__3724_NT_T1__family_metastases`) is representative, and whether grey dominates. Script: `02_flow_shared_baseline.py`, run on the existing clade × clade flows of `92_flow_figure.py` (45 KP units, common support). Table: `results/clade_overlap_20261119/flow_shared_baseline.tsv`.

**Grey dominates almost everywhere, but it is not evidence of matching.**
- Median Shared (grey) is 0.899 over the 45 units. Every unit has at least 0.50.
- A coupling with RNA-OT's clade marginals and no structure inside them also gets a median of 0.899 against CRISPR-OT.
- RNA-OT's own clade flow is 0.992 identical (median) to that structureless coupling.
- So the grey share is set by the clade marginals of the common support, not by the pairing. Example: 3515 T1 → L2 has 97% grey, while same-clade mass is 0.011 for RNA-OT and 0.029 for CRISPR-OT.

**The 3724 pooled figure is the one exception in 45.**
- Its grey is 0.926 against 0.518 for the structureless coupling, an excess of 0.41.
- The next largest excess is 0.021; 44 of 45 units have less than 0.05.
- It matches §1–2: this is the only KP unit where RNA-OT's coupling gain is large (0.43). There, the two clades c5 and c45 are also two states (Mesenchymal-1 vs Mesenchymal-2) and different metastatic sites.
- The figure was chosen as "the only 3724 unit with two named clades on both sides" (`FLOW_FIGURE.md`). It shows the best case and must not be read as typical.

**Implication for figures:** a flow figure should carry its structureless-coupling Shared beside the observed one, or be shown next to the distribution over all units.

## RNA-OT clade flow, every unit, CRISPR-OT removed (2026-11-23)

The user asked to show every result with the RNA-OT flow only. Script: `03_rna_flow_all.py`.

**Figures.** `results/clade_overlap_20261119/figures/rna_flow/`:
- 49 unit figures and one summary panel, one PDF+PNG per panel;
- `CAPTIONS.md` with every unit's numbers.

**How the figures differ from the `92_` flow figures.**
- The cells are RNA-OT's own M4-E retained block; there is no CRISPR gate.
- Ribbons are blue for the same CRISPR clade and grey for a different clade or no clade.
- Each title gives the same-clade mass, the mass expected without pairing (a coupling with the same clade totals and no structure inside them) and the excess.
- Quinn is included: its clades are LineageGroups, drawn without a tree.

**Excess** (same-clade mass − expected without pairing; median [range] over units):

| Stratum | Units | Median excess [range] |
|---|---|---|
| KP: primary → one metastasis | 40 | 0.000 [−0.009, 0.037] |
| KP: pooled metastases | 5 | 0.003 [−0.000, 0.422] |
| Quinn: pooled tissues | 4 | 0.074 [0.011, 0.292] |

- Only 3 of 49 units exceed 0.05: KP 3724 pooled (+0.42), Quinn 100k (+0.29) and Quinn 10k (+0.11).
- In the summary panel, every other unit lies on the diagonal.
- This agrees with §1–2. The numbers differ slightly from the `01_` table: here the denominator is all retained mass, with clades under 2% merged into Other; `01_` uses evaluable targets only.

The `92_` RNA-vs-CRISPR figures are kept unchanged, because decks `110_` and `113_` use them.

## Limits

- Clades are a summary of the edits (UPGMA cut 0.5), not the authors' Cassiopeia tree. A finer or coarser cut changes
  I_ret and O together; it was not varied here.
- KP aggregates are means over 1-8 pairs per mouse that share a primary tumour. There is no test statistic; with 4 KP
  pooled mice and 4 Quinn experiments, ranges are given instead.
- M4-R is not scored: the RNA rows saved only the M4-E coupling.
