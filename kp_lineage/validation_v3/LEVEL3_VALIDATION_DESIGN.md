# KP-Tracer Level 3 validation design: coupling against CRISPR lineage

Status: **DRAFT, not frozen.**
- It is frozen only after three things: the v3 audit (`01_audit_v3.py`) has been read, the literature review (`LINEAGE_VALIDATION_LITERATURE_REVIEW.md`, complete 2026-10-15) has checked every definition, and the user approves.
- 2026-10-15: audit read (job 1075005); coupling pilot passed, 12 of 12 (section 7); the user chose ΔS_norm as the across-pair primary and ΔR as the key secondary metric, and asked for the refinements now in sections 0, 2, 3, 4, 5, 8 and 9 before freezing.
- Freezing is a commit that changes this line to `FROZEN <date> <commit>`. The frozen definitions also cover `METRIC_TAXONOMY_AND_FEASIBILITY.md`.

## 0. The question

**Given a primary–metastasis pair, does ConfidenceOT assign more transport mass to cell pairs that are more closely related in the independently reconstructed CRISPR lineage tree than expected under an appropriate null?**

- ConfidenceOT sees RNA expression only. The lineage tree is built from CRISPR edits by the authors (Cassiopeia-Hybrid; Yang et al. 2022) and never enters any fit, representation, QC, gate or calibration.
- "Appropriate null" is, in order: the independent coupling with the same marginals (primary); target-label permutation; a sequencing-depth-preserving null; an expression-stratified permutation (section 5).

**What has and has not been seen.**
- Seen before this design: the Level 1 tumour-level readout (supported in the primary arm) and the cell-level E1 (inconclusive; carried by mouse 3724, where gate score and lineage depth both follow sequencing depth).
- Not seen: no lineage quantity has been computed from any coupling. The pilot recomputed three couplings only to check reproduction.
- **E1 stays inconclusive** (section 10). Level 3 is a new extension and every new Level 2 analysis stays exploratory.

## 1. Objects

For one primary-metastasis pair and one representation:

| Symbol | Meaning |
|---|---|
| i = 1..n | source (primary) cells: every cell the original M4-E gate scored, retained and rejected, in cell_confidence.csv order |
| j = 1..k | target (metastasis) cells, likewise |
| T (n x k, >= 0) | the coupling of a method, as the solver returns it |
| m = 1ᵀT1 | total transported mass, **reported** but divided out |
| π = T / m | the coupling as a probability matrix; every quantity below uses π |
| a = π1, b = πᵀ1 | its source and target marginals |
| u = 1/n, v = 1/k | uniform marginals over the scored cells |
| Lᵢⱼ | root depth, in edges after contracting unary nodes, of the MRCA of i and j in the pair's lineage tree (the E1 definition of depth) |
| hᵢ, hⱼ | contracted root depth of a cell's own leaf |
| S(ρ) = Σᵢⱼ ρᵢⱼ Lᵢⱼ | lineage relatedness of any probability matrix ρ |

**What L means biologically.** The recorder is switched on with Cas9 at tumour initiation. Each branching node of the tree is an inferred lineage split: a set of edits shared by the cells below it and absent from their relatives. Lᵢⱼ counts the inferred splits that cells i and j still share, i.e. how far down the tumour's family history they stay in one subclone. A larger L means a more recent common ancestor.

**Computation.** S is computed exactly on the tree, never by enumerating cell pairs.
- Lᵢⱼ is the number of branching ancestors c (parent with at least 2 children) whose clade holds both i and j.
- So S(ρ) = Σ_c ρ(clade c × clade c), where ρ(A × B) = Σ_{i∈A, j∈B} ρᵢⱼ.
- For a product ρ = a ⊗ b this is Σ_c a(c) b(c).
- Brute force must equal the tree computation to 1e-12 on every pair small enough to enumerate (a check, section 7).

## 2. What the tree can and cannot tell us

- **Cell-level lineage information exists**: L is graded, cell-pair-specific relatedness inferred from recorded edits.
- **Reliable cell-to-cell ancestry ground truth does not.**
  - Primary and metastatic cells are sampled at the same time. They are relatives through unobserved common ancestors, never ancestor and descendant. No primary cell is "the" origin of a metastatic cell.
  - L is inferred. It depends on the reconstruction (Cassiopeia vs neighbor-joining disagree for some families, section 6). Polytomies make many relationships unresolved: all cells under one polytomy tie.
  - Its resolution varies by pair, from many distinct depths (3724, 3515 T1 family) to almost none (3508 T1 against its metastases, 98% of pairs at the root).
- **Wording.** "Ground truth" is used only for the authors' tumour-level labels (MetFamily, Tumor, is_normal). At the cell level the tree is the *lineage reference*, and results are stated as "more mass on cell pairs with deeper inferred shared ancestry", never as recovering true ancestors or seeding cells.
- Pairs whose L is coarse (section 6) support clade-level statements only.

## 3. Principal metrics (fixed before any coupling is analysed)

### P1, coupling enrichment (primary)

ΔS = S(π) − S(a ⊗ b)

- a ⊗ b is the independent coupling with π's own marginals. It keeps which source cells carry mass and which target cells receive it, and removes only the pairing.
- ΔS > 0: given those marginals, mass flows preferentially between cells that share deeper ancestry.
- ΔS = 0 identically when L has zero variance under a ⊗ b. Such a pair is reported, not tested.

**Normalised P1, ΔS_norm: the across-pair primary (chosen by the user 2026-10-15).**
- ΔS is in edges of shared ancestry, and the trees differ in depth: 3724 reaches about 12 levels, 3508 mostly 2.
- The scale-free version is the share of the achievable enrichment: ΔS_norm = ΔS / ΔS_max. Here ΔS_max = S* − S(a ⊗ b), and S* = max S(ρ) over couplings ρ with marginals a and b.
- **How S\* is computed.** S(ρ) = ½[Σᵢ aᵢhᵢ + Σⱼ bⱼhⱼ − E_ρ(d)], so S* comes from the minimum expected tree cost W_tree(a, b).
  - W_tree(a, b) is the earth mover's distance on the tree, Σ over edges e of |a(below e) − b(below e)| (Evans & Matsen 2012, equal to weighted UniFrac). It takes one pass over the edges.
  - Hence ΔS_max = ½[E_{a⊗b}(d) − W_tree(a, b)].
- **Link to the literature.** 1 − ΔS_norm equals the normalised tree cost R_tree = (C(T) − W_tree)/(C(a⊗b) − W_tree) recommended by the literature review. That ratio follows moslin's normalisation by the independent coupling (Lange et al. 2024), where 1 means uninformative.
- ΔS_norm is 1 when T is as lineage-concentrated as any coupling with its marginals, 0 when it is no better than independence, and negative when it is worse.
- **Biological reading.** Of all the ways the same primary cells could send their mass to the same metastatic cells, how far toward the most lineage-faithful pairing does ConfidenceOT go? If heritable expression lets the RNA geometry find a cell's subclone, primary cells from the subclone that a metastatic population belongs to are paired with it, and ΔS_norm is positive.
- **Roles:** ΔS_norm for the across-pair test; raw ΔS and ΔS_max reported for every pair.
- Undefined when ΔS_max = 0, i.e. no coupling with these marginals can beat independence. Such a pair is reported, not tested.

### ΔR, row-wise lineage rank enrichment (key secondary; approved by the user 2026-10-15)

- R(π) = Σᵢ Σⱼ πᵢⱼ Fᵢ(Lᵢⱼ), with Fᵢ(x) = Σⱼ′ bⱼ′ [1(Lᵢⱼ′ < x) + ½·1(Lᵢⱼ′ = x)], the mid-rank distribution of source cell i's row under the target marginal b.
- Under a ⊗ b, R = ½ exactly, so ΔR = R(π) − ½.
- **Biological reading.** Take a primary cell and follow its mass to a metastatic cell. ΔR + ½ is the probability that this metastatic cell is more closely related to the primary cell than a metastatic cell drawn at random (with the same weights b), counting ties as one half. ΔR > 0: each primary cell's mass goes, within its own row, to its closer relatives.
- It is a per-cell AUROC of relatedness: invariant to each row's depth scale and to the tree's overall depth, with ties handled by mid-ranks. These are the two properties P1 lacks on trees as unresolved as these. It is not tied to P1 by an identity.
- Row bound (diagnostic): R_rowmax = Σᵢ aᵢ maxⱼ Fᵢ(Lᵢⱼ), an upper bound on R that ignores the column constraint. It shows how far ties limit ΔR.

### P2, transport-weighted tree path distance (sensitivity only, never confirmation)

ΔD = E_π[d(i,j)] − E_{a⊗b}[d(i,j)], where d(i,j) = path length in edges.

**Identity.** On any tree, d(i,j) = hᵢ + hⱼ − 2Lᵢⱼ. Hence

E_ρ[d] = Σᵢ ρᵢ. hᵢ + Σⱼ ρ.ⱼ hⱼ − 2 S(ρ).

The two couplings in ΔD have the same marginals, so the first two terms cancel and **ΔD = −2 ΔS exactly**, for any additive tree metric with the same edge weights.

**Consequence.**
- P2 is mathematically tied to P1 and can differ from it only through the edge weighting.
- It is computed with **raw (uncontracted) edge counts**, d_raw and L_raw, so it measures only how much P1 depends on contracting unary nodes.
- The identity is checked numerically on every pair (|ΔD + 2ΔS| ≤ 1e-10 on the same weighting).
- P2 is a sensitivity analysis. It is never counted as confirmation of P1, and it enters no test.

## 4. Baselines and fair comparison (P3)

### A fact about the M4-E coupling, which fixes what a fair comparison can show

The solver returns T = diag(e^f) · exp(−C̃/ε) · diag(e^g) (`traditional_ot/unbalanced.py`, `_solve_uot`), where C̃ = C on retained-source × retained-target entries and C̃ = c (the rejection cost) on every entry that touches a gated cell. Any matrix of that form is the unique entropic OT coupling, at that ε and cost, for its own marginals, whether or not the iterations converged. Therefore:
1. Given its marginals (a, b), ConfidenceOT pairs retained cells exactly as entropic OT on the original cost C does.
2. Gated cells see a constant cost, so their mass is paired lineage-blind (in proportion to the column scalings).
3. ConfidenceOT can differ from plain entropic OT only through its marginals (which cells carry and receive mass) and through point 2.

So a difference in ΔS between M4-E and balanced OT is not evidence of a better pairing rule. Section 9's bridge decomposition attributes it.

### Methods (same representation, same cell IDs, same tree, same cost matrix)

| Method | How the coupling is made | Its marginals |
|---|---|---|
| ConfidenceOT M4-E (primary) | rerun with the original calibrated cost (section 7) | a, b as the gate leaves them |
| ConfidenceOT M4-R | same run, reversible gate | its own |
| EOT with M4-E's marginals, **EOT_C(a, b)** | entropic OT on the ungated cost C, ε = 0.1, marginals fixed to M4-E's a and b | a, b (deviation reported) |
| Balanced OT | entropic OT on C, ε = 0.1, uniform marginals u, v | u, v (deviation reported) |
| kNN | each source cell spreads its mass equally over its 15 nearest target cells under the same cost | a = u; b = normalised in-degree |

- The cost C is rebuilt from the saved joint PCA coordinates and run.json's `cost_scale`. The rebuild is accepted only if log T + C̃/ε is additively separable (fᵢ + gⱼ) to 1e-6 on every entry, which checks the cost and fact 1 together. A pair that fails gets no baseline and is listed.
- No baseline needs a change to `02_run_pair.py`.

### Three comparison levels, kept apart

1. **Matched marginals (fair pairing comparison).** M4-E vs EOT_C(a, b). Same a, b, so the same a ⊗ b, the same ΔS_max and the same attainable range. The contrast isolates the effect of the gate's constant-cost treatment of gated cells on pairing. On the retained × retained block the two are identical by fact 1; that is a numerical check (section 7), not a result.
2. **Matched source coverage.** Only the source cells the M4-E gate retained (set Rs), for every method: M4-E's π restricted to Rs rows and renormalised; balanced OT refitted on Rs × all target cells; kNN on Rs. This removes the advantage of transporting fewer, easier cells. Target marginals still differ.
3. **Full cell.** All n source cells, each method with its own marginals. Descriptive only.

kNN has no marginal constraint, so it cannot be marginal-matched. It appears at levels 2 and 3 only, and this is stated with every kNN number.

### What is reported, and how it may be read

- For every method and level: ΔS, ΔS_max, ΔS_norm (each against the method's own a ⊗ b), ΔR, m, the retained source fraction |Rs|/n and the source-mass concentration exp(H(a))/n.
- **Attainable ranges differ** whenever marginals differ, because ΔS_max depends on a and b. A larger ΔS or ΔS_max from different marginals is **not** read as algorithmic superiority. Contrasts in ΔS_norm are interpreted as pairing quality only at level 1.
- Contrasts per pair, then aggregated by mouse (section 8): level 1, ΔS_norm(M4-E) − ΔS_norm(EOT_C(a,b)); level 2, ΔS_norm(M4-E | Rs) − ΔS_norm(balanced | Rs) and the same for kNN.

## 5. Nulls: construction, preserved constraints, reference marginals

Each null is a coupling (or a relabelling) whose enrichment is recomputed **against its own marginals**. Using π's a ⊗ b as the reference for a null whose marginals differ would be wrong, and is never done.

| Null | Construction | Preserves exactly | Does not preserve | Enrichment of the null recomputed as |
|---|---|---|---|---|
| Independent coupling (primary) | a ⊗ b | a, b | the pairing | ΔS = S(π) − S(a ⊗ b) |
| Target permutation | πσ,ᵢⱼ = πᵢ,σ(j), σ uniform over the pair's target cells; equivalent to permuting target leaf labels on the tree | a; the multiset of target masses; total mass | b's assignment to target cells (bσ = b∘σ) | ΔS(πσ) = S(πσ) − S(a ⊗ bσ); ΔS_norm with ΔS_max(a, bσ); ΔR with Fᵢ under bσ |
| Depth-adjusted | π0ᵢⱼ = π(Bs(i), Bt(j)) · aᵢ bⱼ / (a(Bs(i)) b(Bt(j))); Bs, Bt = quintiles of pre-equalisation total UMI counts among the pair's source and target cells, ties broken by cell order; empty strata give 0 | a, b; the mass between every pair of depth strata | the pairing within strata | ΔS_depth = S(π) − S(π0); normalised by ΔS_max(a, b) (same marginals) |
| Expression-stratified permutation | as target permutation, but σ permutes target cells only within Leiden clusters of the pair's joint PCA (resolution 1.0, 15 neighbours, the pair's cost metric, seed 20261016) | a; each source cell's mass into each cluster; the multiset of target masses within each cluster | b's assignment within clusters | as target permutation, with the permuted bσ |

- Permutations: 1,000 per null and pair, seed 20261016. Pair-level p = (1 + #{null ≥ observed}) / 1,001, descriptive, never pooled over cell pairs.
- **Centring.** For the global target permutation, E_σ ΔS(πσ) = 0: both S(πσ) and S(a ⊗ bσ) average to Σᵢ aᵢ L̄ᵢ, with L̄ᵢ the unweighted mean of row i's L. Its distribution is centred on the primary null.
- For the expression-stratified permutation, E_σ ΔS(πσ) = Σᵢ Σ_K [π(i, K) − aᵢ b(K)] L̄ᵢ(K) over clusters K, with L̄ᵢ(K) the mean of row i's L over the targets in K. This is the enrichment that cluster-level flow alone produces, i.e. what heritable expression states would give without within-state pairing. It is not 0, and it is reported per pair as the cluster-level part of ΔS. The observed ΔS minus this expectation is the within-cluster part.
- Reported with the stratified null: the number of clusters and the share of target cells in single-cell clusters (which that null cannot move).
- **Verification on every pair and every draw** (failures stop the pair's null and are listed):
  - row sums of every null equal a to 1e-12;
  - column sums equal b (a ⊗ b, π0) or exactly the permuted bσ (permutations);
  - π0 reproduces π's depth-stratum block masses to 1e-12;
  - the stratified permutation reproduces every source cell's per-cluster mass to 1e-12;
  - S(null) ≤ S*(the null's own marginals) + 1e-12.

## 6. Lineage-resolution diagnostics (every eligible pair, every tree)

Reported in one table per pair before any coupling quantity is read, and repeated beside every result:

| Diagnostic | Definition |
|---|---|
| Distinct relatedness values | number of distinct L values with mass under u ⊗ v |
| Modal relatedness fraction | largest u ⊗ v mass at a single L value (the low-information flag uses it, section 6a) |
| Tie probability | Σ_l p_l², the chance that two independent u ⊗ v cell pairs have equal L |
| Root fraction | u ⊗ v mass at L = 0 (no shared branching event) |
| Unresolved relationships | u ⊗ v mass whose MRCA is a polytomy (≥ 3 children); and mass whose MRCA is a terminal polytomy (all children leaves: cells the recorder cannot separate) |
| Attainable enrichment | ΔS_max under u ⊗ v, and under each method's a ⊗ b; R_rowmax |
| Recorder coverage | per-cell share of missing characters in the character matrix, summarised per side |
| Reconstruction stability | Spearman of L between the Cassiopeia and neighbor-joining trees (audit, on sampled cell pairs); normalised Robinson-Foulds distance between the two trees restricted to the pair's leaves; ΔS_norm and ΔR recomputed on the NJ tree, with sign agreement |

- Cassiopeia and NJ trees are built from the same characters. Their agreement shows robustness to the algorithm, not correctness, and is reported that way.
- The intBC-block bootstrap is not run this round (section 11).

### 6a. Eligibility (rules on audit facts only)

**Pair.** A pair enters Level 3 when all of the following hold:
1. Its source and target tumours are in one lineage tree:
   - same-family pairs: the E1 rule (Cassiopeia `<target MetFamily>_Fam`, else `<mouse>_<genotype>_All`);
   - different-family pairs: Cassiopeia `<mouse>_<genotype>_All`, else the neighbor-joining one (reported as such).
2. Every scored source and target cell is a leaf of that tree.
3. More than one distinct L value under u ⊗ v (Var > 0).
4. Its coupling rerun passes the reproduction check (section 7).

**Primary set.** The same-family pairs meeting rules 1–4, expected to be the 28 E1-covered pairs in 7 mice. 3460 → N2 and N4 enter, because Level 3 needs no rejected cell.

**Low-information flag.** A pair whose modal L share under u ⊗ v is ≥ 0.9 is flagged.
- Sensitivity: the primary analysis repeated without flagged pairs.
- The threshold was fixed before the audit was read.

**Different-family shared-tree pairs (25).**
- Not assumed to be informative negative controls.
- Those meeting rules 1–4 are analysed with the same metrics, **reported separately and never pooled** with same-family pairs.
- They enter no test.

**Audit facts (job 1075005, read 2026-10-15; the rules above were fixed before it).**
- 53 shared-tree pairs: 28 same-family in 7 mice (3454, 3457, 3460, 3508, 3513, 3515, 3724), 25 different-family.
- Rule 2 holds for all 53 (`LINEAGE_PAIR_ELIGIBILITY.tsv`, 0 pairs with unmapped cells).
- Rule 3 fails for 3 different-family pairs (one distinct L): 3454 T2 → N1, 3513 T3 → N1, 3513 T5 → N1.
- Low-information flag, different-family: 14 of the remaining 22 flagged (3508 T1 × 8, all at the root; 3513 T2 and T4 → N1, NJ only; 3515 T3 → K2, K3, L1, L2); 8 not flagged (3515 T2 × 6; 3515 T3 → K1, N1).
- Low-information flag, same-family, on the unrounded values: 3 of 28 flagged, 3457 T1 → K1 (0.900285), 3457 T1 → N2 (0.900285), 3508 T2 → N2 (0.900641). Just below the threshold: 3508 T2 → N6 (0.898152) and T2 → N1 (0.896288). The threshold stays 0.9.
- So the primary set is the 28 same-family pairs in 7 mice (rules 1–3 hold for all; rule 4 is checked per pair after the full rerun), with 3 flagged.
- Cassiopeia and neighbor-joining disagree on L for the 3508 T2 family (Spearman about 0.0–0.26) and partly for 3460 (−0.21 to 0.64); the NJ sensitivity matters most there.

## 7. Reproducibility and numerical checks

**Coupling reruns** (`02_rerun_with_coupling.py`):
- Same command line as the original.
- `--fixed-rejection-cost` set to the original calibrated cost.
- Saves the couplings and the joint PCA coordinates.

**A rerun reproduces the original when** (`03_compare_reproduction.py`):
- cell order is identical;
- 0 gate disagreements (retained and raw_retained);
- the same rejection cost;
- max |Δ margin| ≤ 1e-8;
- |Δ transported mass| ≤ 1e-9;
- the coupling has shape n x k, and its mass equals transported_mass.

A pair that fails is excluded from Level 3 and listed with its discrepancies. Iteration counts and convergence flags are reported.

**Pilot.** 3 pairs × 2 representations, chosen by size only: 3457 → L1 (344 × 69), 3515 → K2 (710 × 362), 3724 → L1 (14,775 × 90). **Result (2026-10-15): passed.** Job 1075006 refit all six; its comparison crashed writing the summary when every row passed (fixed, e4a09d2) and was rerun alone: 12 of 12 (pairs × labels × M4-E/M4-R) reproduce with identical rejection cost, 0 gate disagreements and zero margin and mass differences.

**Numerical checks on every pair:**
- tree S equals brute force (enumerable pairs);
- W_tree equals the exact LP (POT `emd2` on the tree-distance matrix) on enumerable pairs, to 1e-10;
- ΔS ≤ ΔS_max + 1e-12 for every coupling and null;
- the decomposition sums to E_tot, and the bridge decomposition (section 9) sums to its total, to 1e-10;
- ΔD + 2ΔS = 0 to 1e-10;
- the cost rebuild and fact 1 (section 4);
- M4-E and EOT_C(a, b) agree on the retained × retained block (renormalised) to 1e-8;
- balanced-OT and EOT_C(a, b) marginal deviations;
- every null's constraints (section 5).

**Provenance.** Every analysis script writes a provenance file (commit, input md5s, software versions, seeds) and a SUCCESS marker only after its outputs validate.

## 8. Statistical aggregation

- **Units.** Pair effect, then mean per metastasis (one same-family pair per metastasis), then mean per mouse. **The mouse is the independent biological unit.** Never a p value over cells, cell pairs or transport edges.
- **Primary test (ΔS_norm, M4-E, `ranknm256_noscale_ds_cos`, primary set):**
  - one-sided Wilcoxon signed-rank of the mouse means, H1: median > 0;
  - "supported" if the mean of mouse means > 0 and p < 0.05; otherwise "inconclusive". With 7 mice the smallest attainable one-sided p is 1/128.
- **Effect sizes and uncertainty, reported for every test:** the mean and the median of mouse means; a 95% percentile bootstrap interval of the mean of mouse means (10,000 resamples of mice, seed 20261016); the number of mice > 0; every mouse's value.
- **Leave one mouse out:** the primary and the secondary recomputed seven times, each without one mouse (mean, interval, p). **Without 3724** is named in the report next to the primary result, with 3724's own ΔS_norm, ΔS_depth and depth diagnostics, because E1 was carried by 3724 and there gate score and lineage depth both follow sequencing depth. With 6 mice the smallest one-sided p is 1/64.
- **Secondary (ΔR):** the same test, effect sizes, interval and leave-one-mouse-out on the mouse means of ΔR. Reported beside the primary; it never replaces or rescues it.
- **P3 contrasts (section 4):** the level 1 contrast and the level 2 balanced-OT contrast, each one-sided H1 > 0 on mouse means, Holm correction within this family of two. The level 2 kNN contrast and every level 3 comparison are descriptive (estimate and interval, no test).
- **Effect scale.** ΔS is in edges of shared ancestry and not on a common scale across trees. The across-pair test uses ΔS_norm. A pair with ΔS_max = 0 has no ΔS_norm; it is listed and leaves its metastasis mean. Raw ΔS and the standardised ΔS / SD_{a⊗b}(L) are sensitivity analyses.
- **Sensitivity (reported, none replaces the primary):**
  - the raw arm;
  - M4-R;
  - neighbor-joining trees (not an independent measurement);
  - the matched-retained set;
  - without low-information pairs;
  - ΔS_depth (depth-adjusted null);
  - the expression-stratified null;
  - P2 (raw edges).
- **Exploratory.** Every other metric in `METRIC_TAXONOMY_AND_FEASIBILITY.md` is exploratory: corrected within its family (Benjamini-Hochberg) and never used to restate the primary result.

## 9. Decompositions

### Total effect of one method

E_tot = S(π) − S(u ⊗ v), the total lineage concordance relative to the sampled cells. It splits into three terms that **sum to E_tot exactly**. The two marginal steps are averaged over both orders (Shapley):

| Term | Formula | Question it answers |
|---|---|---|
| Source selection | ½ [S(a⊗v) − S(u⊗v) + S(a⊗b) − S(u⊗b)] | do the source cells that carry mass share deeper ancestry with the metastasis? |
| Target reweighting | ½ [S(u⊗b) − S(u⊗v) + S(a⊗b) − S(a⊗v)] | do the target cells that receive mass share deeper ancestry with the primary? |
| Coupling enrichment | S(π) − S(a⊗b) = ΔS (P1) | given both, does mass pair related cells? |

- Both single-order splits are reported alongside the Shapley split.
- For balanced OT, a = u and b = v up to the tolerance, so its first two terms are about 0 by construction. That is reported, not interpreted.
- The source-selection term is mass-weighted and coupling-based. It is a different quantity from E1 (the gate score's association with lineage depth) and is **not a re-test of E1** (section 10).

### Bridge from M4-E to balanced OT

By fact 1 (section 4), M4-E → EOT_C(a, b) → balanced OT changes one thing per step:

E_tot(M4-E) − E_tot(balanced) = [S(π) − S(EOT_C(a,b))] + [E_tot(EOT_C(a,b)) − E_tot(balanced)]

- The first bracket is the gate's constant-cost pairing of gated cells at fixed marginals (equal to the level 1 ΔS contrast, since a ⊗ b is shared).
- The second is the effect of the marginals, itself split into the source-selection, target-reweighting and enrichment terms of the two couplings.
- Reported per pair and by mouse, descriptively.

## 10. E1 is not revisited

- E1 (validation_v2, preregistered and frozen at 5577e5f) is **inconclusive**, and stays so whatever Level 3 shows.
- Level 3 estimates different quantities (couplings, not gate scores) on a design fixed after E1 was seen. No Level 3 result is described as confirming, rescuing, explaining or reinterpreting E1, and E1 is never cited as support for a Level 3 result.
- Where the report places them side by side, E1 is quoted as preregistered with its own numbers.

## 11. Literature status of each element (from `LINEAGE_VALIDATION_LITERATURE_REVIEW.md`)

**Established and used directly:**
- the independent coupling a ⊗ b as the uninformative reference (Peyré & Cuturi 2019; moslin, Lange et al. 2024);
- the earth mover's distance on a tree, which equals weighted UniFrac (Evans & Matsen 2012);
- tip-label permutation nulls (Wang 2001; Parker 2008; Yang 2022);
- Cassiopeia's prior-weighted Hamming distance;
- the Cassiopeia vs NJ comparison and Robinson-Foulds distance (Yang 2022, with the caveat that both trees are built from the same characters).

**Our adaptations, with no established source:**
- every evaluation of a coupling against a lineage tree for same-time primary and metastatic cells. The published coupling benchmarks (LineageOT, moslin, CoSpar, LARRY) assume ancestor-descendant ground truth across time points, which does not exist here;
- P1, ΔS_norm and ΔR;
- the decompositions and the marginal-matched EOT baseline;
- the depth-adjusted and expression-stratified nulls;
- the use of the gate label as a trait on the tree.

**Not available this round:**
- branch-length statistics (Fritz-Purvis D, Pagel's λ, Blomberg's K);
- the intBC-block bootstrap. The allele tables are in KPTracer-Data (audit v3: `KPTracer.alleleTable.FINAL.txt` and `.unfiltered.txt`), but each replicate needs a Cassiopeia reconstruction, which is not planned this round;
- Yang's mutation-count distance, which needs Camin-Sokal ancestral reconstruction (Cassiopeia), unless Cassiopeia is available in the analysis environment.

## 12. Not done

- No lineage, tree or edit information enters any OT fit, representation, QC, gate or calibration.
- No lineage reference is built from expression.
- Retrieval metrics (precision@k, MRR, AUROC, AUPRC) are computed only where a positive/negative split is fixed by rule; unresolved ancestry is never turned into binary truth.
- No definition changes after any Level 3 outcome is seen. Any change is a new dated section, reported as post hoc.
