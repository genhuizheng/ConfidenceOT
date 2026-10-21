# Lineage ground-truth validation of ConfidenceOT: KP-Tracer and Quinn 2021 (protocol)

Status: **DRAFT, to be frozen.** Freezing is a commit that changes only this line to `FROZEN <date> <commit of this text>`. After that, any change is a new dated section at the end, marked post hoc. Nothing in it rewrites a definition above it.

## 0. Provenance

**Decisions taken by the user:**
- 2026-10-21:
  - KP-Tracer A and C2 are the pre-specified tests; everything else is exploratory.
  - Acceptance 0.99 only, with no cell cap.
  - Quinn target cells whose clone is absent from LL get a separate rejection test.
  - The protocol is frozen before any gate is compared with lineage.
- 2026-10-21, second message (the seven confirmed rules, all incorporated):
  - A/B are site-level retention rankings; C1/C2/C3 are cell-level coupling matching, where a rejected cell's matching score is 0.
  - Same-family / same-clone agreement is the main readout; tree distance is a subset analysis where the leaf mapping is reliable.
  - The label classes are matchable / unmatched_observed / ambiguous. Unobserved is not a confirmed negative, and a shared clone gives no direction.
  - Each score is reported end-to-end, conditional, with its coverage, and against the independent-coupling and lineage-shuffle nulls, with clone size controlled.
  - Five strategies, with the preprocessing, OT parameters and rejection settings fixed.
  - One evaluation code for both datasets; inapplicable experiments are not run.
  - The mouse is the unit.

**Seen before this protocol was written:**
- The read-only audit (`AUDIT_SUMMARY.md`, job 1087596): sites, cell counts, families, clones, tree files and leaf coverage. It contains no OT output.
- Earlier KP-Tracer work on one-primary/one-metastasis pairs: Level 1 (the tumour-level retention readout was supported in the primary arm) and E1 (inconclusive). A here is a pooled design, but its hypothesis is the Level 1 one, and that is stated.

**Not seen:** none of the runs in §4 existed when this text was frozen. No gate or coupling from them has been read against lineage. The Level 3 coupling analysis (`kp_lineage/validation_v3`, stage F) has not been run either.

## 1. Question and wording

- **Question.** ConfidenceOT is fitted on RNA and/or CNV only. Does it:
  - **(site)** keep the sites that share lineage with the other side and reject the ones that do not;
  - **(cell)** put a retained metastatic cell's coupling mass on source cells of the same family (KP-Tracer) or clone (Quinn);
  - **(tree, subset)** within a lineage, put more mass on cell pairs with deeper inferred shared ancestry?
- **Wording.** Results are stated as "same family", "same clone" or "deeper shared ancestry", never as "came from", "seeded" or a direction.
  - Both datasets sample primary and metastatic cells at one time point, so cells are relatives, not ancestor and descendant.
  - Quinn reports multidirectional seeding.

## 2. Leakage rule

- Lineage never enters an OT fit, representation, QC, gate, calibration or input file. This covers MetFamily, LineageGroup, trees, character matrices and the authors' metastatic scores.
- **What OT reads.** `02_run_pair.py` reads:
  - the counts;
  - `sample_id` (the side);
  - `malignant` (the scope);
  - for its output tables only, `cell_type` (Quinn: absent, written as `unannotated`) and `site`.
- **Inputs written for this validation.** The pooled H5ADs carry nothing else.
- **Stated exception.** A single KP-Tracer tumour side reuses the equalised file of the 104-pair runs. Its obs also holds `met_family`, a column `02_run_pair.py` never reads.
- **When labels join.** Labels are joined to OT outputs only by the scoring code, after the runs.

## 3. Data scope and eligibility

### 3.1 Cells and sites

| | KP-Tracer | Quinn 2021 |
|---|---|---|
| Mice | those in `kp_tracer_confidenceot/prepared/pair_manifest_eligible.csv` | 5k, 30k, 10k, 100k (LM0 is pre-injection, not a mouse with sites) |
| Site | tumour (`kptracer_tumours.csv`) | the authors' `sampleID` |
| Primary | tumour id starting with T (`kind == primary`) | LL (`30kLL` in 30k), the injection site, the only primary |
| Cells of a site | `malignant == "malignant"` (the authors' `is_normal` false) in the equalised per-tumour file of the 104-pair runs | cells with an author meta row (non-missing `sampleID`) in the converted per-mouse H5AD, all A549 |
| Depth equalisation | that file (27_ defaults, already done) | `27_downsample_counts.py` at its defaults on the pooled raw files, `--malignant-column malignant` |

### 3.2 Eligibility rules (fixed before the audit)

- **Site.** ≥ 20 cells as defined in §3.1, counted before Quinn's equalisation. Eligibility is never revised after the runs.
- **KP-Tracer designs, per mouse:**
  - D_A needs ≥ 2 eligible primaries and ≥ 1 eligible metastasis;
  - D_B needs ≥ 1 eligible primary and ≥ 2 eligible metastases;
  - D_C3 needs ≥ 2 of each.
- **Quinn:** D_B only, when LL and ≥ 2 other sites are eligible.

### 3.3 Expected runs (from the audit; the run list itself is `inputs/sides.tsv` and `inputs/manifests/`)

| Block | Runs | Mice |
|---|---|---|
| kp_D_A (one per eligible metastasis) | 35 | 3451 (5), 3454 (2), 3457 (3), 3460 (6), 3466 (1), 3467 (3), 3508 (8), 3513 (1), 3515 (6) |
| kp_D_B (one per eligible primary) | 22 | 3451 (3), 3454 (3), 3457 (5), 3460 (2), 3465 (1), 3467 (2), 3508 (2), 3515 (3), 3724 (1) |
| kp_D_C3 | 7 | 3451, 3454, 3457, 3460, 3467, 3508, 3515 |
| quinn_D_B | 4 | 5k, 30k, 10k, 100k |

- That is 68 pairs × 5 strategies = 340 runs.
- If `01_build_inputs.py` produces a different list, the difference is reported; the list is not edited by hand.

### 3.4 Analysed cells

- **The analysed cells of a run** are its M4-E rows in `cell_confidence.csv`.
- **Single strategies** (`rna`, `cnv`, `rna_cnv`): the analysed sets S⁰ and T⁰ are the pair's malignant cells. The pipeline's cell QC records and removes nothing.
- **Two-step strategies:** S⁰ and T⁰ are the step-1 run's analysed sets. Step 2 is fitted on S ⊆ S⁰ and T ⊆ T⁰, the cells step 1 retained.
- **Check.** S⁰ and T⁰ must be the same cells in all five strategies of a pair. A mismatch is reported, and each run is then scored on its own cells.

## 4. OT runs (fixed; through the existing pipeline)

| Design | Source | Target | Experiments |
|---|---|---|---|
| D_A | the mouse's eligible primaries, pooled | one eligible metastasis | A, C2 |
| D_B | one eligible primary | the mouse's eligible metastases, pooled | B, C1 |
| D_C3 | eligible primaries, pooled | eligible metastases, pooled | C3 |

- **Pooled side.** One H5AD whose cells share one `sample_id`; the original site is in obs `site`. It is the union of the sites' cells and nothing else.
- **Settings (the project's locked configuration, unchanged):**
  - `ranknm256_noscale_ds_cos`;
  - ConfidenceOT M4-E (scored) and M4-R (reported);
  - within-side split null, within-side acceptance minimum 0.99;
  - the pipeline's ε, λ and solver defaults as in the CNV/RNA strategy runs;
  - scope malignant;
  - `CONFIDENCEOT_MAX_OBSERVED_CELLS_PER_SIDE=0` (no cap);
  - `--save-coupling --save-pairing-edges`;
  - minimum 4 cells per scope.
- **Strategies:**
  1. `rna`;
  2. `cnv`: infercnvpy profile, KP-Tracer `infercnvpy_mouse` (reference = the authors' normal cells), Quinn `infercnvpy_experiment` (no reference);
  3. `rna_cnv`: cost on [RNA PCs / √med_rna, CNV PCs / √med_cnv]. RNA is 2 − 2cos on unit PCs and CNV is squared Euclidean (verified in `02_run_pair.py`);
  4. `cnv_then_rna`: RNA on the cells `cnv` retained on both sides;
  5. `rna_then_cnv`: CNV on the cells `rna` retained.
- Driven by `tacc_cnv_rna_strategies.slurm` (STRAT_STAGE=all), submitted by `submit_runs.sh`.
- **Failures.** A failed run is listed with its error and never replaced by another run, arm or cap.

## 5. Ground truth

### 5.1 Lineage label ℓ(x) of a cell

- **KP-Tracer:** ℓ(x) = `met_family` of x's tumour (`kptracer_tumours.csv`). A family ending in `_Unk` (3467) means "unknown".
- **Quinn:** ℓ(x) = the authors' `LineageGroup`. A missing value means unknown.
- **Related.** Two cells are related iff both labels are known and equal.
  - KP-Tracer: different primaries are separately initiated tumours, so source cells of different families are unrelated by design.
  - Quinn: different LineageGroups are different barcoded clones.

### 5.2 Label class of a cell (per run)

For a target cell j ∈ T⁰, with respect to the run's analysed source S⁰ (source cells are classed symmetrically against T⁰):

| Class | Rule | Used in |
|---|---|---|
| matchable | ℓ(j) known and ∃ i ∈ S⁰ with ℓ(i) = ℓ(j) | accuracy, mass, rejection test |
| unmatched_observed | ℓ(j) known and no i ∈ S⁰ with ℓ(i) = ℓ(j) | rejection test; mass (see below) |
| ambiguous | ℓ(j) unknown | nothing scored; counted |

**Unmatched_observed is read differently per dataset:**
- **KP-Tracer:** the family's primary is not in the source. Either it is another sampled primary, which is a different lineage by design, or the family has no sampled primary (3460 N1). Mass asserted to these cells counts as **false**.
- **Quinn:** the clone was not seen in the LL sample. That is **not a confirmed negative**: the clone may exist in LL unsampled. Mass asserted to these cells is reported separately as **unobserved-clone mass**, never as false. Their rejection is the separate rejection test the user asked for.
- **Sensitivity (Quinn).** Matchable with ≥ 5 LL cells of the clone (the audit's sensitivity rule); cells with 1–4 such cells are then left out of accuracy.

### 5.3 Site truth (A, B)

- **KP-Tracer.** Primary s and metastasis X are related iff ℓ(s) = ℓ(X). Every cell of a tumour carries its tumour's family.
  - A metastasis whose family is `_Unk` has no site truth.
  - In D_A, a metastasis whose family has no eligible primary (3460 N1) has no site truth.
- **Quinn.** There is no binary site truth: every tissue shares clones with LL. The site-level quantity is f_X (§7.3).

### 5.4 Tree labels (subset analysis only)

- **KP-Tracer.**
  - The tree of lineage λ (a family) is the Cassiopeia `<λ>_Fam`, else `<mouse>_<genotype>_All`; files `trees/<stem>_tree.nwk`, the E1/Level 3 rule.
  - A cell is mapped iff its name equals a leaf name, the rule of `validation_v2/04_lineage_concordance.Tree.leaf_of`.
  - Per-tumour trees are never used across tumours.
- **Quinn.**
  - The tree of clone c is `converted/lineage/<mouse>/trees/trees/m<mouse>_lg<c>_tree_hybrid_priors.alleleThresh.processed.txt`. Other files for the clone are listed, not used.
  - A cell is mapped iff its name equals the leaf name as written; in 10k, iff it equals `<sampleID>.<leaf name>`, the rule that joins 10k's meta table.
  - A clone without that file (100k LG1, its largest clone) has no tree and is listed.
- **Reliable mapping.** Exact name equality only. A leaf matching more than one analysed cell, or a cell matching more than one leaf, is unmapped. The mapped fraction of each side's cells of λ is reported with every tree result. No other threshold is used.
- **Depth.** L_ij = root depth of the MRCA of i and j, in edges after contracting unary nodes (Level 3 §1). Only cells of the same lineage λ, both mapped on λ's tree, have an L.

## 6. Notation for one run

| Symbol | Definition |
|---|---|
| S⁰, T⁰ | analysed source and target cells (§3.4), in `cell_confidence.csv` order |
| P | the saved M4-E coupling `coupling_m4e.npz["coupling"]`, \|S\| × \|T\|, rows and columns in `cell_confidence.csv` order. For two-step runs, P is extended by zero rows and columns to S⁰ × T⁰ |
| m, π | m = 1ᵀP1 (equals `transported_mass` in `pair_metrics.csv`, checked); π = P / m |
| ρᵢ, τⱼ | 1 if the cell is retained (`retained` column), else 0. Cells of S⁰ \ S and T⁰ \ T have ρ = 0, τ = 0 |
| sₓ | the cell's `signed_rejection_margin` (> 0 is the rejection side). Cells removed at step 1 get sₓ = +∞ |
| a, b | a = π1, b = πᵀ1 |
| Dⱼ | Σᵢ ρᵢ Pᵢⱼ, the retained-source mass in column j |
| qᵢⱼ | ρᵢ τⱼ Pᵢⱼ / Dⱼ if τⱼ = 1 and Dⱼ > 0, else 0 (the relative matching weight; a rejected cell scores 0) |
| groups | the distinct labels G = {ℓ(i) : i ∈ S⁰}, including groups whose cells were all rejected (W = 0). KP-Tracer: one family per source primary. Quinn: the clones present in LL |
| Wⱼ(g) | Σ_{i ∈ S⁰, ℓ(i) = g} qᵢⱼ |
| wⱼ | Wⱼ(ℓ(j)), the correct-group weight of target cell j |
| nₘ(g) | \|{i ∈ S⁰ : ℓ(i) = g}\| |
| J | matchable target cells (§5.2), a subset of T⁰ |
| J_R | {j ∈ J : τⱼ = 1, Dⱼ > 0} (retained matchable cells with retained-source mass) |
| Rᵢⱼ | 1 if ℓ(i) = ℓ(j), both known; else 0 |

## 7. Site-level metrics (A, B): retention ranking

The retention of site x is r_x = mean of ρ (source side) or τ (target side) over the analysed cells of site x (S⁰ or T⁰). Cells removed at step 1 count as rejected.

### 7.1 A (KP-Tracer, D_A run with target metastasis M that has site truth)

- P⁺ = the eligible primaries related to M (one in every audited case); P⁻ = the other primaries.
- **Retention enrichment:** e_M = mean_{s∈P⁺} r_s − mean_{s∈P⁻} r_s.
- **Top-1:** |argmax_s r_s ∩ P⁺| / |argmax_s r_s|. The argmax is a set; ties are counted as the expected value of a random order.
- **AUROC:** (1 / |P⁺||P⁻|) Σ_{s⁺∈P⁺, s⁻∈P⁻} [1(r_{s⁺} > r_{s⁻}) + ½·1(r_{s⁺} = r_{s⁻})].
- **Reported with it:**
  - r_M (target retention);
  - the asserted-mass share of each primary, Σ_{i∈s} Σ_j ρᵢ τⱼ πᵢⱼ / Σᵢⱼ ρᵢ τⱼ πᵢⱼ;
  - the asserted-mass quantities of §8.3.
- **Equal retention.** If every primary has the same r (for example all 1 because the target is fully rejected), then e_M = 0, Top-1 = 1/n and AUROC = ½.

### 7.2 B (KP-Tracer, D_B run with source primary s)

Each pooled target site X (each metastasis) is related or unrelated to s by §5.3; `_Unk` sites are left out.
- **Related and unrelated metastases both present** (audit: 3457 T1, T4; 3460 T1): e_s = mean t(related) − mean t(unrelated), and the AUROC as in §7.1 over target sites.
- **Only related metastases** (positive control): mean t over the sites, with the asserted precision of §8.3.
- **No related metastasis** (negative control): scored by the false asserted mass F (§8.3), whose ideal is 0.
  - Rejection rates are reported beside F, never scored alone. Rejecting all source cells or all target cells both empty the asserted block, and either is correct.

### 7.3 B (Quinn, D_B run, source LL)

- For each target site X: t_X, and f_X = |{j ∈ X : matchable}| / |{j ∈ X : ℓ(j) known}|.
- **Site-level readout:** Spearman correlation of t_X with f_X over the mouse's target sites, plus each site's (t_X, f_X).
  - Descriptive: 4 to 6 sites per mouse.
  - With 30k (one site has f < 1) and 10k (XM, f ≈ 0), it shows whether the low-f site is the low-retention one.

## 8. Cell-level metrics (C1 = D_B runs, C2 = D_A runs, C3 = D_C3 runs)

All metrics are computed per run on the cells of §6. Every one is reported together with its coverage (§8.4).

### 8.1 Correct-group weight

| Metric | Definition | Reference |
|---|---|---|
| End-to-end score | E2E = (1/\|J\|) Σ_{j∈J} wⱼ (a rejected j scores 0) | chance CH = (1/\|J\|) Σ_{j∈J} πⱼ*, where πⱼ* = nₘ(ℓ(j)) / \|S⁰\|; independent coupling E2E_ind = (1/\|J\|) Σ_{j∈J_R} wⱼ^ind |
| Conditional score | COND = (1/\|J_R\|) Σ_{j∈J_R} wⱼ | COND_ind = (1/\|J_R\|) Σ_{j∈J_R} wⱼ^ind, where wⱼ^ind = Σ_{i: ℓ(i)=ℓ(j)} ρᵢ aᵢ / Σᵢ ρᵢ aᵢ (the coupling (ρa) ⊗ b on the same retained cells, which removes the pairing and keeps which cells carry and receive mass) |

**What the references mean:**
- CH is what a method with no lineage information and no rejection gets: balanced OT with uniform marginals and a ⊗ b pairing gives wⱼ = πⱼ*.
- E2E − CH is the gain after rejection has taken its toll.
- COND − COND_ind is the gain from the pairing alone, given what the gate retained.

### 8.2 Top-k origin accuracy

- **For j ∈ J_R:**
  - n_>(j) = #{g ∈ G : Wⱼ(g) > wⱼ + 10⁻¹²};
  - n_=(j) = #{g ∈ G : |Wⱼ(g) − wⱼ| ≤ 10⁻¹²}, which includes ℓ(j);
  - hit_k(j) = min(1, max(0, (k − n_>(j)) / n_=(j))): the probability that ℓ(j) is among the top k under a random order of ties.
- **Scores:**
  - End-to-end Top-k = (1/|J|) Σ_{j∈J_R} hit_k(j);
  - conditional Top-k = mean over J_R.
- **k:** k = 1 everywhere; k = 3 also for Quinn.
- **When it is computed.** Only when |G| ≥ 2 (k = 3: when |G| ≥ 4). KP-Tracer D_B has one source group, so Top-k is N/A there.
- **References:**
  - Wⱼ^ind(g) = Σ_{i: ℓ(i)=g} ρᵢ aᵢ / Σᵢ ρᵢ aᵢ, the same for every j;
  - chance hit_k computed from the group sizes nₘ(g) / |S⁰| with the same tie rule;
  - the shuffle null (§10).

### 8.3 Asserted mass (the block ConfidenceOT asserts)

All quantities are fractions of π (total 1):

| Quantity | Definition |
|---|---|
| asserted mass A | Σᵢⱼ ρᵢ τⱼ πᵢⱼ |
| correct asserted mass C | Σ_{j∈J} Σᵢ ρᵢ τⱼ πᵢⱼ Rᵢⱼ |
| false asserted mass F | Σᵢⱼ ρᵢ τⱼ πᵢⱼ (1 − Rᵢⱼ) over j ∈ J, and also over KP-Tracer unmatched_observed j |
| unobserved-clone mass U (Quinn) | Σ_{j unmatched_observed} Σᵢ ρᵢ τⱼ πᵢⱼ |
| ambiguous mass | Σ over ambiguous j or ambiguous i |
| asserted precision | C / (C + F) |

- **Reference** for the precision: the same ratio under (ρa) ⊗ (τb).
- **Ideal.** F = 0 in every design. A one-sided reject-all is a correct answer when nothing is related (§7.2).

### 8.4 Coverage, coverage–accuracy and the rejection test

- **Coverage**, reported with every score:
  - cov_T = |J_R| / |J|;
  - the retained fraction of all T⁰;
  - cov_S = Σᵢ ρᵢ / |S⁰|;
  - the number of j with τⱼ = 1 and Dⱼ = 0.
- **Coverage–accuracy curve.**
  - Order J by sⱼ, ascending (most confidently retained first; step-1-removed cells last, in cell order).
  - Use the source-only weights q̃ᵢⱼ = ρᵢ Pᵢⱼ / Dⱼ (Dⱼ > 0, else hit = 0). These ignore τ, so the curve shows whether the gate's own order puts correct matches first.
  - For f ∈ {0.1, 0.2, …, 1.0}: Top-1 accuracy (hit_1 from q̃) among the first ⌈f |J|⌉ cells.
  - The gate's operating point (cov_T, conditional Top-1) is marked.
  - Computed only where |G| ≥ 2.
- **Rejection test**, on each side and in every run with both classes present:
  - the rejection rate of matchable and of unmatched_observed cells;
  - the AUROC of sₓ with unmatched_observed as positives (ties ½).
  - KP-Tracer: an expected direction. Quinn: the separate test of "an unobserved clone should be rejected", **not** a confirmed negative.
  - Never read alone: always beside F (KP-Tracer) or U (Quinn).

### 8.5 Clone and family size control

- CH and the shuffle null both keep the group sizes.
- **Group-balanced scores:** E2E_bal = mean over groups g with J_g ≠ ∅ of (1/|J_g|) Σ_{j∈J_g} (wⱼ − πⱼ*), where J_g = {j ∈ J : ℓ(j) = g}. COND_bal is the same over J_R ∩ J_g. One dominant clone therefore cannot carry the result.
- **Per-group table:** |J_g|, nₘ(g) / |S⁰|, retained fractions on both sides, E2E_g, COND_g, conditional Top-1.
- **Size bins** of πⱼ* = nₘ(ℓ(j)) / |S⁰|: [0, 0.01), [0.01, 0.05), [0.05, 0.20), [0.20, 1]. E2E, COND and CH are reported per bin.

## 9. Tree agreement (subset analysis)

- **Blocks.** One block per (run, lineage λ) with:
  - λ's tree available (§5.4);
  - ≥ 2 retained mapped source cells of λ and ≥ 2 retained mapped target cells of λ.
  - KP-Tracer: λ is a family; its source cells are the family's primary.
  - Quinn: λ is a clone.
- **Block coupling.** π^λ = P restricted to (retained mapped source cells of λ) × (retained mapped target cells of λ), divided by its sum.
- **Metrics:** ΔS_norm and ΔR exactly as in `kp_lineage/validation_v3/LEVEL3_VALIDATION_DESIGN.md` §3, with π^λ, its own marginals and L.
  - The block is reported but not scored if Var_{a⊗b}(L) = 0 or ΔS_max = 0.
  - The low-information flag (modal L share under u ⊗ v ≥ 0.9) is reported.
- **Nulls:** a ⊗ b (built in) and target permutation within the block (1,000 draws, Level 3 §5 construction).
- **Run value.** The mean over scored blocks, weighted by the block's target cell count.
- **Reported beside it:**
  - the mapped and retained fractions of λ's cells on each side;
  - the share of the run's matchable target cells that are in a scored block.
- **Status.** Exploratory in every dataset. It never replaces or rescues a family/clone result.

## 10. Nulls, baselines and controls

- **Independent coupling:** §8.1–8.3, on the same retained cells.
- **Lineage-label shuffle.**
  - A uniform permutation of ℓ over S⁰, which keeps every group size (KP-Tracer: across the pooled primaries; Quinn: within LL). T⁰'s labels and J stay as observed.
  - E2E, COND, Top-1 (Top-3 for Quinn) and asserted precision are recomputed.
  - 1,000 draws per run, `numpy.random.default_rng([20261021, crc32(pair_id + "/" + strategy)])`.
  - Run-level p = (1 + #{null ≥ observed}) / 1,001, descriptive.
  - Not computed when |G| = 1.
- **Baselines (exploratory),** for the single strategies `rna`, `cnv` and `rna_cnv`:
  - **The cost.** The cost is rebuilt from `joint_pca_coordinates.csv.gz` and `run.json`. The rebuild is accepted only if log P + C̃/ε is additively separable (fᵢ + gⱼ) to 10⁻⁶ on every entry, the Level 3 §4 check. A run that fails gets no baseline and is listed.
  - **Balanced entropic OT:** ε = 0.1, uniform marginals.
  - **kNN:** each source cell spreads its mass equally over its 15 nearest target cells.
  - **Scoring.** Both are scored by §8 with ρ = τ = 1, which is their coverage. They are compared with ConfidenceOT only beside coverage.
- **Depth control.** Spearman of sₓ with pre-equalisation total counts per site:
  - KP-Tracer: `downsampled/predownsample_depth.csv.gz`;
  - Quinn: 27_'s own `predownsample_depth.csv.gz`.
- **Numerical checks per run:**
  - P's shape equals the M4-E row counts;
  - m equals `transported_mass` to 10⁻⁹;
  - Σ_g Wⱼ(g) ∈ {0, 1} to 10⁻¹²;
  - every shuffle keeps group sizes;
  - tree S equals brute force on blocks ≤ 2,000 × 2,000;
  - ΔS ≤ ΔS_max + 10⁻¹².
  - A failed check stops that run's scores and is listed.

## 11. Aggregation and pre-specified tests

- **Units.** Run → metastasis (A, C2) or primary (B, C1) → mouse. The mouse is the only independent unit. There is never a p value over cells, cell pairs or sites.
- **Pre-specified tests** (KP-Tracer, strategy `rna`, M4-E, D_A runs with site truth: expected 31 metastases in 8 mice, 3451, 3454, 3457, 3460, 3466, 3508, 3513, 3515):
  - **T1 (A):** the mouse means of e_M.
  - **T2 (C2):** the mouse means of E2E − CH over the same runs.
  - **The test:** one-sided Wilcoxon signed-rank on mouse means, H1 > 0 (the smallest p with 8 mice is 1/256). Holm correction over {T1, T2}.
  - **Supported** if the mean of mouse means > 0 and the Holm-adjusted p < 0.05; otherwise **inconclusive**. Neither is ever called refuted.
  - A test runs only with ≥ 5 mice (5 is the smallest number of mice for which p < 0.05 is attainable). Missing mice are listed, never filled.
- **Reported with each test:**
  - the mean and median of mouse means;
  - a 95% percentile bootstrap interval over mice (10,000 resamples, seed 20261021);
  - the number of mice > 0;
  - every mouse's value;
  - leave-one-mouse-out (mean, interval, p);
  - T1 beside Top-1 and AUROC; T2 beside COND − COND_ind, conditional Top-1, coverage and E2E_bal − CH_bal.
- **Exploratory (estimates and bootstrap intervals over mice, no tests):**
  - the four other strategies, as per-mouse paired differences from `rna`;
  - M4-R;
  - B, C1, C3;
  - the Quinn rejection test and every Quinn quantity (4 mice);
  - tree agreement;
  - shuffle p values;
  - baselines;
  - the ≥ 5 LL cell sensitivity.

## 12. Not applicable (not run, or not scored)

| Experiment | Where | Reason |
|---|---|---|
| A, C2 | KP-Tracer 3465, 3724; all Quinn mice | one eligible primary (Quinn: one injection site) |
| A scoring | KP-Tracer 3467 (all three metastases), 3460 N1 | no site truth (`_Unk`; family without an eligible primary); run, reported under §8.3–8.4 only |
| B, C1 | KP-Tracer 3466, 3513 | one eligible metastasis |
| C3 | KP-Tracer 3465, 3466, 3513, 3724; all Quinn | needs ≥ 2 primaries and ≥ 2 metastases |
| site AUROC in B | KP-Tracer except 3457 T1, T4 and 3460 T1; all Quinn | no primary with both related and unrelated metastases; Quinn has no binary site truth |
| Top-k, coverage–accuracy, shuffle | KP-Tracer D_B | one source group |
| tree agreement | KP-Tracer metastases of 3451, 3465, 3466, 3467; 3454 L1; 3460 N1. Quinn 100k LG1. Any cell not mapped by §5.4 | no tree holds both cells |

## 13. Outputs and code (all in `lineage_validation/`; results in `/scratch/10119/ghzheng/OT_project/lineage_validation/`)

**Inputs and runs (written):**
- `01_build_inputs.py`, writing `inputs/`;
- `submit_runs.sh`, writing `runs/acc0.99/<strategy>/arm_A/<block>/ranknm256_noscale_ds_cos/<pair>/…`.

**Scoring (to be written against this text):**
- **labels:** cell → ℓ, label class per run, tree leaf;
- **per-run scores:** §7–§8 and the shuffle null;
- **tree blocks:** §9;
- **baselines:** §10;
- **aggregation and tests:** §11.

**Figures:** per mouse:
- A/B retention heatmaps and rankings, one quantity per heatmap (rejected fraction);
- coupling matrices at group level (family or clone × target site);
- coverage–accuracy curves;
- five-strategy panels;
- the fixed joint UMAP from `48_reference_umap.py`, coloured by site, lineage and retain/reject/exclude.

Every script writes a provenance file (commit, input md5, seeds) and SUCCESS only after its checks pass. Nothing outside the result root is written; no existing result is overwritten.

DEG/GSEA, through the existing downstream, only describes retained and rejected cells. It is never ground truth.
