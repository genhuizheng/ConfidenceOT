# KP-Tracer Level 3 validation design: coupling against CRISPR lineage

Status: **DRAFT, not frozen.**
- It is frozen only after three things: the v3 audit (`01_audit_v3.py`) has been read, the literature review (`LINEAGE_VALIDATION_LITERATURE_REVIEW.md`) has checked every definition, and the user approves.
- Freezing is a commit that changes this line to `FROZEN <date> <commit>`.

**What has and has not been seen.**
- Seen before this design: the Level 1 tumour-level readout (supported in the primary arm) and the cell-level E1 (inconclusive; carried by mouse 3724, where gate score and lineage depth both follow sequencing depth).
- Not seen: no coupling exists for any pair. The pilot (below) recomputes three couplings only to check reproduction, and computes no lineage quantity from them.
- Level 3 is a new extension. It does not change Level 1 or E1, and every new Level 2 analysis stays exploratory.

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

**Computation.** S is computed exactly on the tree, never by enumerating cell pairs.
- Lᵢⱼ is the number of branching ancestors c (parent with at least 2 children) whose clade holds both i and j.
- So S(ρ) = Σ_c ρ(clade c × clade c), where ρ(A × B) = Σ_{i∈A, j∈B} ρᵢⱼ.
- For a product ρ = a ⊗ b this is Σ_c a(c) b(c).
- Brute force must equal the tree computation to 1e-12 on every pair small enough to enumerate (a check, below).

## 2. Principal metrics (fixed before any coupling is analysed)

### P1, coupling enrichment (primary)

ΔS = S(π) − S(a ⊗ b)

- a ⊗ b is the independent coupling with π's own marginals. It keeps which source cells carry mass and which target cells receive it, and removes only the pairing.
- ΔS > 0: given those marginals, mass flows preferentially between cells that share deeper ancestry.
- ΔS = 0 identically when L has zero variance under a ⊗ b. Such a pair is reported, not tested.

### P2, transport-weighted tree path distance (sensitivity of P1, not independent)

ΔD = E_π[d(i,j)] − E_{a⊗b}[d(i,j)], where d(i,j) = path length in edges.

**Identity.** On any tree, d(i,j) = hᵢ + hⱼ − 2Lᵢⱼ. Hence

E_ρ[d] = Σᵢ ρᵢ. hᵢ + Σⱼ ρ.ⱼ hⱼ − 2 S(ρ).

The two couplings in ΔD have the same marginals, so the first two terms cancel and **ΔD = −2 ΔS exactly**, for any additive tree metric with the same edge weights.

**Consequence.**
- P2 can differ from P1 only through the edge weighting.
- It is computed with **raw (uncontracted) edge counts**, d_raw and L_raw, so it measures how much P1 depends on contracting unary nodes.
- The identity is checked numerically on every pair (|ΔD + 2ΔS| ≤ 1e-10 on the same weighting).
- P2 is never counted as confirmation of P1.

**Decision needed.** A non-additive complementary metric that is not tied to P1 by an identity, proposed for the user's approval:

- **R, row-wise rank enrichment:** R(π) = Σᵢ aᵢ Σⱼ (πᵢⱼ / aᵢ) Fᵢ(Lᵢⱼ), with Fᵢ(x) = Σⱼ′ bⱼ′ [1(Lᵢⱼ′ < x) + ½·1(Lᵢⱼ′ = x)], the mid-rank distribution of source cell i's row under the target marginal b.
- Under a ⊗ b, R = ½ exactly, so ΔR = R(π) − ½.
- It asks whether, within each source cell's row, mass goes to its more related target cells.
- It is invariant to each row's depth scale and handles ties by mid-ranks. These are the two properties P1 lacks on trees as unresolved as these.

### P3, baselines (same representation, same cell IDs, same tree)

| Method | How the coupling is made | Its marginals |
|---|---|---|
| ConfidenceOT M4-E (primary) | rerun with the original calibrated cost (section 6) | a, b as the gate leaves them |
| ConfidenceOT M4-R | same run, reversible gate | its own |
| Balanced OT | entropic balanced OT on the same cost matrix, ε = 0.1 as in M4-E, uniform marginals u, v | u, v up to the Sinkhorn tolerance (deviation reported) |
| kNN | each source cell spreads 1/n equally over its 15 nearest target cells in the same joint PCA coordinates under the same cosine cost | a = u; b = normalised in-degree |

- **Fairness.** Each method's P1 is measured against its own a ⊗ b, so ΔS compares how each method pairs cells given its own marginals. kNN and OT do not share marginal constraints, and none is imposed on them.
- **Contrasts.** P3 = ΔS_M4-E − ΔS_balanced and ΔS_M4-E − ΔS_kNN, per pair.

### Two comparison sets, kept apart

1. **Full-cell.** All n source cells, as above.
2. **Matched retained cells.** Only the source cells the M4-E gate retained (set R), for every method:
   - M4-E: π restricted to the rows in R, renormalised;
   - balanced OT: refitted on R × all target cells;
   - kNN: on R.

   This removes the advantage of transporting fewer, easier cells.

**Reported with every quality number:** m, the retained source fraction |R|/n, and the source-mass concentration exp(H(a))/n.

## 3. Decomposition of the total effect

Total lineage concordance of a method relative to the sampled cells:

E_tot = S(π) − S(u ⊗ v)

It splits into three terms that **sum to E_tot exactly**. The two marginal steps are averaged over both orders (Shapley), so the split does not depend on which marginal is changed first:

| Term | Formula | Question it answers |
|---|---|---|
| Source selection | ½ [S(a⊗v) − S(u⊗v) + S(a⊗b) − S(u⊗b)] | do the source cells that carry mass share deeper ancestry with the metastasis? (the coupling-level analogue of E1) |
| Target reweighting | ½ [S(u⊗b) − S(u⊗v) + S(a⊗b) − S(a⊗v)] | do the target cells that receive mass share deeper ancestry with the primary? |
| Coupling enrichment | S(π) − S(a⊗b) = ΔS (P1) | given both, does mass pair related cells? |

- Both single-order splits are reported alongside the Shapley split.
- The sum is checked to 1e-10 on every pair.
- For balanced OT, a = u and b = v up to the tolerance, so its first two terms are about 0 by construction. That is reported, not interpreted.

## 4. Nulls and confounders

- **Principal null:** a ⊗ b, deterministic and exact (P1).
- **Per-pair permutation reference (descriptive).** Target cells are permuted (π_σ,ij = π_i,σ(j)); 1,000 permutations, seed 20261016. Each permutation is a valid coupling with a permuted target marginal. It gives each pair's ΔS a reference distribution. Pair-level p values are descriptive and are never pooled across cell pairs.
- **Depth-adjusted null:**
  - π0_ij = π(Bs(i), Bt(j)) · aᵢ bⱼ / (a(Bs(i)) b(Bt(j))), with Bs and Bt the quintiles of pre-equalisation total counts among the pair's source and target cells.
  - It keeps a, b and the mass flowing between depth strata. ΔS_depth = S(π) − S(π0) is the enrichment beyond depth-to-depth flow.
  - Reported for every pair, and singled out for mouse 3724.
- **Diagnostics per pair:**
  - Spearman of hᵢ and of the gate score with pre-equalisation counts and with detected genes;
  - source and target cell counts;
  - the share of cell pairs at the modal L, and the number of distinct L values;
  - m.

## 5. Eligibility (rules on audit facts only)

**Pair.** A pair enters Level 3 when all of the following hold:
1. Its source and target tumours are in one lineage tree:
   - same-family pairs: the E1 rule (Cassiopeia `<target MetFamily>_Fam`, else `<mouse>_<genotype>_All`);
   - different-family pairs: Cassiopeia `<mouse>_<genotype>_All`, else the neighbor-joining one (reported as such).
2. Every scored source and target cell is a leaf of that tree.
3. Var_{u⊗v}(L) > 0.
4. Its coupling rerun passes the reproduction check (section 6).

**Primary set.** The same-family pairs meeting rules 1–4, expected to be the 28 E1-covered pairs in 7 mice. 3460 → N2 and N4 enter, because Level 3 needs no rejected cell.

**Low-information flag.** A pair whose modal L share under u ⊗ v is ≥ 0.9 is flagged.
- Sensitivity: the primary analysis repeated without flagged pairs.
- The threshold is fixed here, before the audit is read.

**Different-family shared-tree pairs (25 expected).**
- Not assumed to be informative negative controls.
- Their L variance, ties and modal share are reported first (audit).
- Those meeting rules 1–4 are analysed with the same metrics, **reported separately and never pooled** with same-family pairs.
- They enter no primary test.

## 6. Reproducibility

**Coupling reruns** (`02_rerun_with_coupling.py`):
- Same command line as the original.
- `--fixed-rejection-cost` set to the original calibrated cost.
- Saves the coupling.

**A rerun reproduces the original when** (`03_compare_reproduction.py`):
- cell order is identical;
- 0 gate disagreements (retained and raw_retained);
- the same rejection cost;
- max |Δ margin| ≤ 1e-8;
- |Δ transported mass| ≤ 1e-9;
- the coupling has shape n x k, and its mass equals transported_mass.

A pair that fails is excluded from Level 3 and listed with its discrepancies. Iteration counts and convergence flags are reported.

**Pilot first.** 3 pairs × 2 representations, chosen by size only:
- 3457 → L1 (344 × 69);
- 3515 → K2 (710 × 362);
- 3724 → L1 (14,775 × 90).

Nothing larger is submitted unless all six pass.

**Numerical checks on every pair:**
- tree S equals brute force (enumerable pairs);
- the decomposition sums to E_tot;
- ΔD + 2ΔS = 0;
- balanced-OT marginal deviation.

**Provenance.** Every analysis script writes a provenance file (commit, input md5s, software versions, seeds) and a SUCCESS marker only after its outputs validate.

## 7. Statistical aggregation

- **Units.** Pair effect, then mean per metastasis (one same-family pair per metastasis), then mean per mouse. The mouse is the independent unit.
- **Primary test (P1, M4-E, `ranknm256_noscale_ds_cos`):**
  - one-sided Wilcoxon signed-rank of the mouse means, H1: median > 0;
  - reported with mice > 0, and a 95% percentile bootstrap of the mean of mouse means (10,000 resamples of mice, seed 20261016);
  - "supported" if the mean > 0 and p < 0.05; otherwise "inconclusive". With 7 mice the smallest attainable one-sided p is 1/128.
- **P3:** two paired contrasts (vs balanced OT, vs kNN), each one-sided H1 > 0 on mouse means; Holm correction within this family of two.
- **Effect scale.** ΔS is in edges of shared ancestry, so it is not on a common scale across trees. The standardised ΔS / SD_{a⊗b}(L) is reported as a sensitivity analysis.
- **Sensitivity (reported, none replaces the primary):**
  - the raw arm;
  - M4-R;
  - neighbor-joining trees where one holds both sides (not an independent measurement);
  - the matched-retained set;
  - without low-information pairs;
  - leave one mouse out;
  - ΔS_depth;
  - P2 (raw edges);
  - R, if approved.
- **Exploratory.** Every other Level 3 metric in `METRIC_TAXONOMY_AND_FEASIBILITY.md` is exploratory: corrected within its family (Benjamini-Hochberg) and never used to restate the primary result.
- **Never** a p value over cell pairs or transport edges.

## 8. Not done

- No lineage, tree or edit information enters any OT fit, representation, QC, gate or calibration.
- No lineage reference is built from expression.
- Retrieval metrics (precision@k, MRR, AUROC, AUPRC) are computed only where the audit finds a defensible positive/negative split fixed by rule; unresolved ancestry is never turned into binary truth.
- No definition changes after any Level 3 outcome is seen. Any change is a new dated section, reported as post hoc.
