# KP-Tracer validation_v2 preregistration

Status: **DRAFT, not frozen.**
- It is frozen when two conditions hold: (1) the audit (`LINEAGE_DATA_AUDIT.md`, section C) has been read, and (2) the user approves.
- Freezing is a commit that changes this line to `FROZEN <date> <commit>`. After that, changes go only into new dated sections.
- No result of any endpoint below has been computed. The existing tumour-level readout (`READOUT_PRESPECIFICATION_2026-10-14.md`) is unchanged and is reported separately whatever these endpoints show.
- Choices below that depend on facts the audit reports are written as explicit rules on those facts, never on any outcome.

## 0. Arms and units

- **Arms:** `ranknm256_noscale_ds_cos` is primary; `raw` is reported alongside. Both use the existing 104 completed pairs and gates, unchanged.
- **Statistical unit:** the mouse is the independent unit. Metastases are nested in mice, and cells are nested in pairs. No test treats cells as replicates.

## 1. Cell-level lineage concordance (new endpoint E1)

### 1.1 Pairs that can enter

- The 40 eligible same-family pairs. A metastasis that has a same-family primary enters, even when its mouse has no other-family primary, because the comparison is within the source tumour.
- **Tree.** A pair enters only if one Cassiopeia family tree holds both its source and its target cells (audit C3). The rule for which tree, fixed now:
  - use `<target MetFamily>_Fam` if that file exists;
  - otherwise use `<mouse>_<genotype>_All`;
  - if neither holds both sides, the pair is "not lineage-covered" and is listed.
- **Cells.** Source cells are the pair's gated source cells (M4-E, `side == source`) that are leaves of that tree. Target cells are the pair's gated target cells that are leaves of it.
- **Evaluable.** A pair is evaluable if it has at least 5 lineage-mapped retained source cells, at least 5 lineage-mapped rejected source cells, and at least 1 lineage-mapped target cell. Others are listed as "not evaluable" with their counts.

### 1.2 Lineage relatedness: primary definition

**Shared-ancestry depth.** For a source cell `i` and the metastasis `M`, the measure is:

`s(i, M)` = the number of edges from the root to the deepest ancestor of `i` whose clade contains at least one lineage-mapped cell of `M`.

Equivalently, it is the root depth of the deepest common ancestor that `i` shares with any cell of `M`. Larger `s` means more recent shared ancestry with the metastasis. `s = 0` means `i` shares only the root.

**Why this definition:**
- It uses only the topology of the authors' reconstruction, so it needs no branch lengths. The release trees may have no informative lengths, and edge counts are not read as time.
- In a tree whose mutationless edges are collapsed, each internal node on the path stands for at least one heritable edit. So `s` counts nested lineage-defining splits shared with the metastasis.
- Minimum pairwise tip distance mixes the shared ancestry with the depth of `i`'s own subtree below it, which depends on how deeply `i`'s own lineage was resolved. `s` does not.
- The number of sampled metastatic cells changes `s` for every source cell of a pair alike, so the within-pair contrast is not biased by it.

The audit reports unary nodes and polytomies (C1). If a tree carries unary nodes, they are contracted before `s` is computed. This rule is fixed now.

### 1.3 Effect size per pair, direction and aggregation

- **Per pair.** `δ` = Cliff's delta of `s` between lineage-mapped retained and rejected source cells: `P(s_ret > s_rej) − P(s_ret < s_rej)`. It lies in [−1, 1], is unaffected by unequal group sizes, and is positive when retained cells share deeper ancestry with the metastasis.
- **Per metastasis.** There is one same-family pair per metastasis in this data, so the metastasis δ is the pair δ. If there were more than one, the rule would be their mean.
- **Per mouse.** The mean δ of its evaluable metastases.

### 1.4 Primary test for E1

- **Test.** A one-sided Wilcoxon signed-rank test of the mouse means, with H1 that the median is above 0.
- **Reported with it:**
  - the number of mice with mean δ above 0;
  - a 95% percentile bootstrap interval of the mean of mouse means: 10,000 resamples of mice, seed 20261015.
- **Criterion for "supported".** The mean of mouse means is above 0 and the one-sided p < 0.05.
- **Small samples.** With few mice the test has little power. A non-significant result is reported as inconclusive, not as evidence of no effect.

### 1.5 Continuous gate score (E1b, secondary)

- **Per evaluable pair.** Spearman ρ between `r = −signed_rejection_margin` (M4-E) and `s` over the lineage-mapped source cells.
- **Aggregation and test.** The same as in 1.3–1.4.
- **Reported.** Direction and magnitude: the mean of mouse means and its bootstrap interval.

### 1.6 Sensitivity analyses for E1 (reported, none replaces the primary)

1. Minimum group sizes of 10 and 20 instead of 5.
2. The other tree where both a `_Fam` and an `_All` tree cover a pair, and the neighbor-joining tree where one exists.
3. Patristic distance to the nearest metastatic leaf, only if the audit finds informative branch lengths (more than one distinct non-zero length in the tree).
4. Lineage-tree coverage. The fraction of retained and of rejected source cells that are lineage-mapped is reported per pair. E1 is recomputed on pairs where the two fractions differ by under 0.1.
5. Leave-one-mouse-out on the mean of mouse means.
6. Depth and detected genes. The Spearman correlation of `s` with each source cell's pre-equalisation total counts (`predownsample_depth.csv.gz`) and with its detected-gene count, per pair, aggregated per mouse. This asks whether lineage relatedness and depth are confounded. It is not a correction.

## 2. Cross-mouse negative control (new endpoint E2)

### 2.1 Pairs, fixed rule

- **Builder.** `28_build_mismatched_manifest.py` on the 104-row eligible manifest. Each row keeps its source tumour and gets a metastasis from another mouse, at the closest malignant cell count (`target_malignant_n`; absolute log ratio).
- **Seed.** The existing default, 20260915.
- **Code change.** This needs one new option, `--size-column`, whose default `target_n` keeps the script's existing behaviour. The reason is in audit A5: most cells of some metastases are normal.
- **Genotype.** Donor genotype is not restricted: the NT mice have too few lesions for 3724_NT_T1's four rows. Genotype is reported for every row, and E2 is repeated on the genotype-matched rows as a sensitivity.
- **Both arms.** The same assignment is used in both arms. It is built once and the target paths are mapped to each arm's h5ads. The builder also refuses to proceed unless the two arms' donor assignments are identical.

### 2.2 Run

- The same `tacc_preprocessing_factorial.slurm` configuration as the matched runs: same labels, QC, null, acceptance 0.99, cap 0 and seed.
- Output goes to `validation_v2/cross_mouse/ot/`.

### 2.3 Endpoint and test

- **Per same-family row.** `Δ` = source retained fraction against its own metastasis minus source retained fraction against the cross-mouse metastasis. The source cells are identical in the two fits.
- **Aggregation.** Per source tumour, then per mouse, as means.
- **Test.** One-sided Wilcoxon signed-rank of the mouse means, H1 > 0. The mean and its bootstrap interval are reported as in 1.4.
- **Reported alongside:**
  - Δ on the other-family rows;
  - the target-side retained fraction;
  - the distribution of `decision_cost / rejection_cost`;
  - per-row cross-mouse retained fractions.
- **Interpretation.** Cross-mouse retention is not required to be 0, since unrelated tumours share malignant states. E2 asks only whether related pairs retain more than unrelated pairs.

## 3. Baselines (E3, exploratory)

- **Expression-only score.** For each source cell, the mean distance to its 15 nearest target cells, in the same joint representation the arm used. No transport and no gate.
- **Balanced OT score.** Entropic balanced OT (ε 0.1, budget 0), with the per-cell transported cost.
- **E1 and E1b on both.** E1b is computed with these scores in place of `r`, and E1 with their median split in place of the gate.
- **Status.** Both are exploratory. They need new runs, which are not yet implemented. No claim that rejection adds biological validity is made unless E1 or E1b differ between ConfidenceOT and a baseline on the same cells.

## 4. What is not done

- No lineage information enters any OT fit, representation, QC or calibration.
- No ground truth is inferred from expression.
- No definition above changes after any endpoint is seen; changes would be new dated sections, reported as post hoc.
- No pair, metastasis or mouse is removed except by the coverage and evaluability rules above. Every exclusion is listed.
