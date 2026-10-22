# Rejection–coupling mass audit of the within-family subclone analysis (2026-10-22)

**Scope.**
- Run locally, read only, on every coupling available locally: 22 pairs × 5 strategies × M4-E / M4-R = **220 runs** (110 `coupling_m4e.npz` + 110 `coupling_m4r.npz`, byte copies of the TACC files, md5-checked).
  - KP-Tracer 3508: D_A × 8, D_B × 2, D_C3;
  - KP-Tracer 3515: D_A × 6, D_B × 3, D_C3;
  - Quinn 10k: D_B.
- Families with CRISPR subclones in those runs: KP 3508 T2, KP 3515 T1, Quinn 10k LG1–LG5 and LG7.
- Of these, classifiable (≥ 2 level-1 subclones): **3515 T1, 10k LG2, LG3, LG5**.
- Nothing was re-run. No TACC job was submitted. The subclone definitions (SUBCLONE_RULES.md), OT parameters and the frozen protocol are unchanged.

## The four answers

1. **Is there a rejected/excluded-mass or cell-ID alignment bug? No.**
   - In all 220 runs: excluded cells carry exactly 0 mass; the four gate blocks sum to the total to 2e-15; q columns of retained targets sum to 1 to 6e-15; q of every rejected or excluded target is exactly 0.
   - The gated block of the saved coupling is exactly rank one on the rows/columns flagged rejected in `cell_confidence.csv` (log-residual ≤ 8e-13). On the same number of random retained rows/columns it is not (≥ 1.5). This proves the flags sit on the right rows and columns.
   - Three defects were found in the classifier's **outputs**, not in the mass. They are fixed below; no prediction, outcome, accuracy, coverage or size reference changed:
     - A: weights were stored for non-retained targets;
     - B: the reference share left out the family's unlabelled cells;
     - C: there was no duplicate-id guard.
2. **Are lineage-supported primary subclones retained preferentially? Not in the data available locally, and the test is weak there.**
   - 3515 T1: both level-1 subclones carry metastatic cells, so there is no unsupported subclone to compare against. They are retained at the same rate (RNA-type strategies 0.29–0.34; CNV-type 0.92–0.96).
   - Its one sharply metastasis-associated branch goes the other way. Level-2 L1.1.2 holds the L2 metastasis (241 cells) and 44 N1 cells, with only 11 T1 cells. It is retained **less**: RNA-type 0–2 of 11 cells; CNV-type 6–7 of 11, against 0.92–0.98 for its sister branch.
   - 3508 T2: every metastatic cell is ambiguous, because the metastasis branch has no marker edit, so no subclone is lineage-supported.
   - Quinn 10k: every subclone is supported; retention again shows no consistent difference.
3. **Is the subclone-level raw coupling determined by the retained source marginal composition? Yes, in every classifiable family available locally.**
   - The mean within-family weight of a metastatic subclone on each primary subclone equals that subclone's share of retained effective transport mass to within 0.01–0.07.
   - The raw-mass argmax equals the independent-coupling argmax for 89–100% of the cells.
   - Rejection barely changes the composition: eligible-cell, retained-cell and retained-mass shares agree to within 0.01–0.05.
4. **Is there within-family lineage discrimination beyond rejection? At most a small signal, in one family, on 6–12 cells. Not demonstrated.**
   - In 3515 T1, the 6–12 retained metastatic cells of subclone L1.2 put more weight on primary L1.2 than its mass share in every strategy and both runs: enrichment 1.23–1.46, against 0.74–0.92 on L1.1. The other subclone's cells show E ≈ 1.
   - In Quinn 10k LG2, LG3 and LG5 the enrichments are 0.73–1.07 with no diagonal pattern.
   - The signal is never large enough to change the raw-mass argmax.

**H1 vs H2 vs H3.**
- **H1 (implementation issue) is rejected** for the mass computation.
- **H2 (rejection-driven identification)** explains Level 1 in 3508 but **not** Level 2. Rejection selects families (3508: T1 retained 2.7%, T2 81%), but within a family it does not select subclones.
- **H3 (within-family coupling ≈ independent coupling of the retained block)** describes all four classifiable families. The exception is the weak 3515 L1.2 enrichment.

**Not verified** (couplings only on TACC):
- the other 22 of the 26 eligible families: KP 3454 T1, 3457 T1, 3457 T4, 3724 T1; Quinn 100k LG2–LG5; 30k LG1, LG2; 5k × 12;
- every other KP-Tracer mouse at Level 1.

No conclusion here extends to them.

## Part A — implementation, read from code and outputs

| # | Question | Finding (code location; data) |
|---|---|---|
| 1 | What is saved? | `02_run_pair.py` saves `result.coupling` (`np.savez_compressed(... coupling_m4e/m4r.npz)`). That is `ConfidenceOTResult.coupling` (`src/confidenceot/api.py:377`) = `BidirectionalUOTResult.coupling` (`src/confidenceot/_cpu_uot.py:914`): the full plan of the terminal fixed-gate KL-UOT solve, `solve_fixed_bidirectional_uot` (`_cpu_uot.py:374-413`, re-solved at `:844` when the gate moved). Its cost is `np.where(gate_s ⊗ gate_t, C, c)` (`:400`, as `filtered_cost` `:358-371`). **No dummy row or column, no masking:** gated rows and columns carry mass at the constant cost c. Every local run used the numpy backend (`pair_metrics.csv`: backend numpy, device cpu). |
| 2 | How is rejection implemented? | Only through the cost and the separately saved gate (`cell_confidence.csv` `retained` = `fitted.source_gate` / `target_gate`, the gate the terminal plan was solved with). Rejected rows and columns are **not** zero. In the 220 runs, rows of rejected sources hold 0–96% of the plan's mass (M4-E median 33%), columns of rejected targets 1–37%. This is the model's definition: the disowned mass of `target-side-cannot-reject-without-retained-source`, not a format or scorer bug. |
| 3 | Rejected sources in the subclone aggregation? | No. `classify_subclones.classify()` builds `GP = onehot(group, weights=rho) @ P`, so rejected rows weigh 0 before any sum. Same in the audit's own code (`audit_core.effective`). |
| 4 | Rejected targets predicted? | No: their outcome is `abstain`. **Defect A:** their `S__*`, `family_weight` and `top_weight` columns held the unmasked weights (summing to 1). Fixed. |
| 5 | Two-step | `load_export_run` and `run_audit.load` index the step-2 coupling into the step-1 cell set by cell id. Step-1-excluded cells get zero rows/columns and gate 0 (3515: 61 source / 20 target cells; 10k: 10 / 366; 0 in the step-2 gate; 0 mass). Step-2 rejected cells are masked as in (3). |
| 6 | Alignment | Rows/columns follow `cell_confidence.csv` order, since both come from the same AnnData order in `02_run_pair.py`. Tested from the coupling itself by the rank-one structure of the gated block (above). Every downstream join is by cell id (dict / `get_indexer`), never by row number. Tests with shuffled step-2 file order recover the id-ordered matrix exactly. |
| 7 | Exclusive subclones | One label per leaf (`cells/<family>.csv.gz`, 0 duplicated ids in all 7 families). Level-1 subclones are disjoint clades by construction, tested on nested synthetic clades. `audit_core.group_sum` refuses a label outside the exclusive list. **Defect C:** the classifier lacked a duplicate-id check. Added. |
| 8 | Zero-mass / unnormalised / double-normalised columns | 0 retained targets with zero effective mass in the 220 runs. Normalisation happens once, per target column after masking. The within-family weight is one further, explicit conditioning, reported separately. |
| 9 | M4-E vs M4-R | Same coupling form (filtered cost; rank-one gated block holds for M4-R too). They differ in the gate score (`_cpu_uot.py:880-888`): exact = partner-restricted loss, reversible = counterfactual loss. M4-R retains more mass in the asserted block (effective share median 0.63 vs 0.59). Several M4-R fits did not converge in the outer loop (gate cycling; e.g. 3515 D_C3 rna `outer_converged` False); they are used as saved. |

## Part B — mass conservation (`mass_conservation.tsv`, 220 runs)

- **Input measure.** 1 per side, uniform over the analysed cells (`_solve_uot`, `normalize_weights=True`); KL-UOT then gives a plan of total mass 0.68–0.74.
- **Units.**
  - Cell counts: `n_*`.
  - Transport mass: `mass_*`, `block_*`, `effective_mass`, in plan units.
  - Matching weights: q, S, dimensionless, per target.
- **Effective coupling.** P̃ = P · r^S r^Tᵀ, the retained × retained block. Its share of the plan, by strategy (median of M4-E / M4-R): rna 0.32 / 0.48, cnv 0.89 / 0.90, rna_cnv 0.71 / 0.74, cnv_then_rna 0.35 / 0.47, rna_then_cnv 0.98 / 0.98.
- **The audit's masking is the classifier's masking.** Same weights on real data, to 2e-16 over 240 (family × run × true × source) cells. It is applied once, to the saved plan.

## Part C — subclone-level rejection (`subclone_rejection.tsv`, `subclone_mass_decomposition.tsv`, Figures A, B)

Retention rate R_g (cells) and retained-mass share π_g^F (transport mass) are both in the table.

| Family (run), M4-E | Group (lineage support) | R_g: RNA / CNV / CNV+RNA / CNV→RNA / RNA→CNV | π_g^F (RNA) |
|---|---|---|---|
| 3515 T1 (D_B) | L1.1 (supported; 453 cells) | 0.32 / 0.96 / 0.92 / 0.32 / 0.32 | 0.685 |
| | L1.2 (supported; 154) | 0.32 / 0.95 / 0.93 / 0.34 / 0.32 | 0.231 |
| | L2 L1.1.2 (supported: L2 metastasis + N1; 11) | 0.18 / 0.64 / 0.55 / 0.09 / 0.18 | 0.009 |
| | unlabelled (103) | 0.18 / 0.63 / 0.52 / 0.19 / 0.18 | 0.083 |
| 3508 T2 (D_B) | L1.1 (no observed metastatic counterpart; 262) | 0.20 / 1.00 / 0.99 / 0.20 / 0.20 | 0.778 |
| | unlabelled (50) | 0.32 / 0.98 / 0.94 / 0.32 / 0.30 | 0.222 |
| 10k LG2 (D_B) | L1.1 / L1.2 (both supported) | 0.68 / 0.98 / 0.95 / 0.92 / 0.68 and 0.82 / 0.95 / 0.91 / 0.95 / 0.82 | 0.46 / 0.13 |

- Within a family, rejection removes cells of every subclone at about the same rate.
- The one strong lineage contrast available (3515 L1.1.2) is retained less, not more.
- `no observed metastatic counterpart` (3508 T2 L1.1) is a sampling statement, not a lack of metastatic capacity.

## Part D — OT vs independent coupling (`ot_vs_independent.tsv`, `classifier_comparison.tsv`, Figure C)

For a retained metastatic cell of true subclone t: S̃_gj is the within-family weight, π_g^F is the retained effective-mass share (the independent coupling of P̃ gives exactly π_g^F to every target), and E = mean S̃ / π_g^F. E is left undefined where π < 0.01 or the group has < 5 retained cells.

| Family (D_B), M4-E | Raw argmax acc. | Independent argmax | Size argmax | Enrichment argmax, balanced acc. (chance 0.5) | E on own subclone, minority subclone |
|---|---|---|---|---|---|
| 3515 T1 L1 | 0.90–0.98 | 0.99 | 0.99 | 0.61–0.69 | L1.2: 1.23–1.46 (n = 6–12) |
| 10k LG2 L1 | 0.81–0.82 | 0.82–0.82 | same | 0.41–0.48 | L1.2: 0.97–1.01 |
| 10k LG3 L1 | 0.80–0.83 | 0.83–0.85 | same | 0.47–0.54 | L1.2: 0.73–0.84 |
| 10k LG5 L1 | 0.67–0.73 | 0.74–0.75 | same | 0.50 | L1.2 has 1 primary cell: undefined |

- **Why raw mass picks the largest subclone.** Every metastatic subclone receives the same profile, so the argmax is the largest share. The independent coupling of the retained block already does exactly that, and rejection leaves the composition almost unchanged (Figure B).
- **What enrichment changes.** It removes the size effect but finds no consistent diagonal, except 3515 L1.2. It is a diagnostic, not a calibrated origin probability.

## Part E — Level 1 vs Level 2 in the same run (Figure D)

| Mouse, run | Level 1 (family / clone) | Level 2 (subclones in the correct family) |
|---|---|---|
| 3508, D_C3 | **Rejection-driven.** RNA: T1 retained 101/3,719 (0.027), T2 253/312 (0.81), so T2 holds 0.72 of the retained mass despite 12× fewer cells. Raw accuracy 0.93, independent 1.00: the retained block alone identifies T2. CNV keeps 30% of T1, gives T1 0.77 of the mass, and fails (raw 0.08, independent 0.00). | Not classifiable: one level-1 subclone (primary only), metastases without a supported branch. |
| 3515, D_C3 | Uninformative: every metastasis is T1 family and T1 is the largest primary. The size and independent references are already 1.00; retention T1 0.28, T2 0.25, T3 0.13 (RNA). | Raw 0.89–0.98 ≈ independent 0.99. Subclone retention flat; L1.1.2 (the L2 metastasis's branch) mostly rejected. |
| 10k, D_B | **Coupling beyond the retained composition.** Raw argmax 0.56–0.64 vs independent 0.39–0.41 over 12 LL clones (balanced 0.28–0.44 vs chance 0.08); retention per clone 0.57–1.00, without a lineage pattern. | Raw ≈ independent (LG2, LG3, LG5); no diagonal enrichment. |

So the same machinery that identifies the family (rejection in 3508, coupling geometry across Quinn clones) does not, in these data, resolve subclones inside a family.

## Part G — fixes and tests

Fixes, all in `classify_subclones.py` (and the matching heatmap row in `plot_subclones.py`):

| Defect | Before | After |
|---|---|---|
| A | non-retained targets: `S__*` summed to 1.000, `family_weight` 1.000 | 0 (3515 rna: 720 targets; 10k LG2: 12) |
| B | reference share and chance over labelled subclones' retained **cells**; unlabelled left out | `source_effective_mass` (retained × retained transport mass, unlabelled included); `chance_accuracy_by_mass`; size reference by mass (identical choice). Chance changes: 3515 rna 0.747 → 0.683; 10k LG2 0.702 → 0.407; LG3 0.710 → 0.658; LG5 0.737 → 0.581. The old value is kept as `chance_accuracy_by_labelled_cells` |
| C | no duplicate-id check | gate and subclone-table ids must be unique (ValueError) |

- **Unchanged by the fixes:** outcome counts, accuracy, coverage, size-reference accuracy, retained targets' weights (max change 0 over all local families and runs).

**Tests:**
- `audit/test_audit.py`: 15/15 pass (`test_audit_result.txt`). Covers rejected source, rejected target, two-stage excluded, zero-mass column, shuffled ids, overlapping assignment, incomplete annotation, sparse vs dense, and audit = classifier.
- `test_subclone_mass.py`:
  - before the fixes, 16 passed and 3 failed (commit 3e4786b; the failures were A and B);
  - after the fixes, 20/20 pass (`test_subclone_mass_result_after_fix.txt`). Two tests were updated to the fixed specification: a step-2 rejected target must have weight 0, and the share must be by effective mass.

## Files

- **Tables:** `mass_conservation.tsv`, `subclone_rejection.tsv`, `subclone_mass_decomposition.tsv`, `ot_vs_independent.tsv`, `classifier_comparison.tsv`.
- **Figures:** `figures/A_rejection_*`, `B_mass_*`, `C_ot_vs_independent_*`, `D_level1_vs_level2_{3508,3515,10k}` (PNG + PDF).
- **Code:** `audit_core.py` (independent implementation), `run_audit.py`, `plot_audit.py`, `test_audit.py`.
