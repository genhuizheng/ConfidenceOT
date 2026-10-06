# IC-POT as a comparator on the native benchmark, 2026-10-06

What was implemented, how it was configured, what ran, and what it returned.
Nothing here changes a setting or a number. Every value below is read from
outputs that already exist, and the files are named where each one comes from.
The benchmark itself, meaning the conditions, levels and cases, is described in
`DESIGN.md`.

In one paragraph: IC-POT (Tripathi et al., arXiv:2605.20030) was added to the
solver comparison at the constant unmatched cost the paper reports for its
open-partial domain adaptation runs, c_s = c_t = 0.5. It was run on all 540
pairs of the three preprocessing arms. On both cosine arms it transported
essentially all mass in every case. At the fixed cutoff it rejected no cell in
all_shared or population_emerged, and at most 0.9% of a side in any other
pair. On raw it rejected 28-30% of cells in every case and side, matched or
not. Its
untransported mass carries no ranking information about the truth (ROC-AUC
0.50 to 0.53). The truth-tuned global threshold therefore selected `-inf`,
which rejects every cell. That post-processing result, not the solver, is what
gives the directional F1 of 0.56, identical on every arm.

---

## 1. Implementation

### 1a. The reference implementation

`src/traditional_ot/icpot.py`, `intent_controlled_partial_ot`, is an
independent implementation of Eq. (2) of the paper, the slack formulation:

```
min_{P,u,v}  <C, P> + <c_s, u> + <c_t, v>
s.t.         P 1 + u = mu,   P^T 1 + v = nu,   P, u, v >= 0
```

It solves the equivalent reduced problem of Proposition 1. That problem
minimises sum_ij (C_ij - c_s(i) - c_t(j)) P_ij over sub-couplings with
P 1 <= mu and P^T 1 <= nu. It is restricted to the admissible pairs of
Proposition 5 and Appendix E, and solved as one exact linear program by
scipy's HiGHS. No entropic term is added; the paper's Appendix L shows that
entropic regularisation of the augmented problem is a different objective.
The slacks are recovered as u = mu - P 1 and v = nu - P^T 1, and the Eq. (2)
objective is reported alongside the reduced one.

Two details differ from the paper's text without changing the model:

- **Admissible pairs.** The code keeps pairs with C_ij < c_s(i) + c_t(j)
  strictly; the paper's sparse LP uses <=. The optimal value is the same, by
  Proposition 5: an edge on the equality can be emptied without changing the
  objective. When optima are not unique, the code returns the optimum that
  puts no mass on equality edges. A positive `admissibility_tolerance` would
  drop further edges and make the result approximate. The default is 0, which
  is exact.
- **Weights.** `normalize_weights=True`, the default, rescales each side to
  unit mass. The paper allows any non-negative masses, and
  `normalize_weights=False` keeps them.

### 1b. The solver used for the benchmark runs

The 540 benchmark runs were **not** solved by `icpot.py`. They were solved by
`constant_cost_icpot` in `benchmark/50_solver_comparison.py`, the paper's
Proposition 6 written out.

**The construction.** It is balanced OT on supports augmented by one dummy
point each:

- the marginals are mu_bar = (a, sum b) and nu_bar = (b, sum a);
- the cost is C_bar = [[C, c_s], [c_t^T, 0]];
- the solver is POT's exact network simplex, `ot.emd`;
- the real block of the plan is the IC-POT coupling.

**Why not `icpot.py`.** Its LP has one variable per admissible pair, which is
tens of millions at N = 10,000. In addition, the scipy 1.15.1 HiGHS wrapper
used locally copies the solution once per column. That makes its run time
quadratic in the number of edges: about 8 s at 60 x 60 cells, where HiGHS
itself took 0.05 s. The network simplex is the one the benchmark already uses
for Traditional OT and Partial OT at that size.

### 1c. Validation

All checks below were run before the benchmark, as ad hoc scripts outside the
test suite; those scripts are not in the repository. `tests/test_icpot.py`
holds the unit tests of `icpot.py`: accept or reject on a 1 x 1 problem, the
slack constraints and the Proposition 1 objective identity, the exactness of
the admissible support, recovery of the balanced assignment at high uniform
cost, and input validation.

**`icpot.py` against the paper's equivalent formulations.** The test set was
27 random instances: pointwise costs, constant costs, and integer costs that
put many pairs exactly on C_ij = c_s(i) + c_t(j). The sizes were 5 x 7,
30 x 20 and 60 x 40, with unequal total masses (1 and 1.3) and
`normalize_weights=False`. Each instance was solved four ways:

- `icpot.py`;
- Eq. (2) directly, with P on every pair and explicit slacks;
- the Appendix E sparse LP, on the admissible set with <=;
- Proposition 6 with `ot.emd`.

| check | largest gap over the 27 instances |
|---|---|
| optimal value, `icpot.py` against Eq. (2) | 9.2e-17 |
| optimal value, Appendix E sparse LP against Eq. (2) | 0 |
| optimal value, Proposition 6 (`ot.emd`) against Eq. (2) | 3.3e-16 |
| Proposition 1 identity, objective = <c_s, mu> + <c_t, nu> + reduced objective | 4.4e-16 |
| feasibility of the `icpot.py` plan and slacks | 2.8e-17 |
| Proposition 5, mass on pairs with C_ij > c_s(i) + c_t(j) | 0 |
| Proposition 3, dual caps f <= c_s and g <= c_t, and f_i + g_j <= C_ij | 5.6e-17 |
| Proposition 3, strong duality | 1.1e-16 |
| Proposition 4, complementary slackness | 5.6e-17 |

The plans themselves were identical except on the integer-cost instances,
where they differed by up to 0.114. Those instances have many optimal plans
with the same value.

**The paper's counterexample.** The two-source, one-target example of
Appendix D was reproduced with `normalize_weights=False`: P = (1, 0),
u = (0, 1), objective 0.3, the unique optimum the paper states. With the
default normalisation the masses become (0.5, 0.5) against 1, a different
problem, and the solution is P = (0.5, 0.5).

**The benchmark solver against `icpot.py`.** `constant_cost_icpot` was
compared with `icpot.py` on four instances (20 x 25, 40 x 32, 40 x 50 and
30 x 30, at c = 0.5, 0.15 and 2.0):

- the objectives agreed to 2.8e-17;
- the transported masses were identical;
- no mass sat on pairs costing more than lambda.

---

## 2. Benchmark configuration

**Unmatched costs.** c_s(i) = c_t(j) = 0.5 for every cell on both sides. This
is the constant A = 0.5 of the paper's Appendix H, the value of its
open-partial domain adaptation runs, where it is both the constant-cost
baseline (c_s = c_t = A) and the base of the two pointwise variants. That task
is chosen because its two kinds of unmatched class match this benchmark's
cases:

- source-private classes correspond to population_lost;
- target-private classes correspond to population_emerged.

The paper's PU experiment uses A = 0.15 instead, and is not the analogue here.

**Scale.** The value is used as reported and is not rescaled. The paper's cost
in that setting is not this benchmark's. Here the cost is median-normalised,
so the price lambda = c_s + c_t = 1.0 sits at the cost's median. About half of
all pairs therefore fall below it by construction.

**Rejection amount.** Nothing is read off the simulation's truth. How much
mass is left unmatched is decided by the optimisation, at that price. This is
the difference from the benchmark's "Partial OT (truth-derived m)". That
method is POT's `partial_wasserstein`, Chapel et al.'s fixed-mass partial-W,
and it is handed its transported mass from each pair's truth.

**Terminology.** With constant costs, IC-POT is the constant-cost
specialisation of the paper's model. By its Proposition 2, it is equivalent to
constant-rebate partial transport with rebate lambda = c_s + c_t: only the sum
matters, and splitting it between the sides changes nothing. In the paper's
own experiments this setting is the "partial-W" baseline.

It belongs to the same family as fixed-mass partial-W, parametrised by a price
instead of a mass. On a 50 x 50 instance, constant-cost IC-POT at
lambda = 0.2, 0.5, 1, 2 and 5 transported 0.72, 0.76, 0.78, 0.80 and 0.88 of
the mass. POT's `partial_wasserstein`, given each of those masses, returned
the same transport cost to 10 decimals.

**Not run: the pointwise variants.** The paper's two pointwise variants for
the open-partial task are entropy-based pricing and prototype-support pricing.
Both need calibrated classifier posteriors over source prototypes, and the
second also needs neighbourhoods in that classifier's feature space. This
benchmark has no classifier and no prototypes, so neither can be built
without inventing the side information.

The other pointwise profiles in the paper are tied to their own experiments:

- the PU profile prices rejection from a known covariate of the selection
  mechanism;
- the SWIM/SAR profiles are built from sensor diagnostic maps.

Neither has a counterpart here.

**The paper's code.** The paper's code link (anonymous.4open.science,
IC-POT-68F4) had expired when checked on 2026-10-05. arXiv lists only v1 and
no other repository. No comparison with the authors' own implementation was
possible.

---

## 3. Experimental coverage

| | |
|---|---|
| preprocessing arms | `ranknm256_noscale_ds_cos`, `ranknm256_noscale_cos`, `raw` |
| sizes | N = 1000, 5000, 10000 |
| cases | `all_shared`, `population_lost`, `population_emerged`, `populations_disjoint` |
| technical levels | `L0_matched`, `L1_depth`, `L2_depth_dropout` |
| replicates | 5 |
| pairs | 180 per arm, 540 in all |
| completed | 540 of 540 (`solver_comparison.csv` present for every pair, checked on TACC 2026-10-06) |

**The solve.** Every completed pair passed two checks, because `50_` refuses
to write a pair that fails either one:

- `ot.emd` reported the optimum (`result_code` 1);
- the cost matched the ConfidenceOT run's recorded geometry.

Each pair's `provenance.json` also records two further quantities, which have
not been aggregated across the 540 pairs here:

- the share of pairs below lambda (`admissible_pair_fraction`);
- the mass on pairs above it (`mass_on_pairs_above_lambda`).

**The test pair.** Before submission, a 360 / 480-cell `population_emerged`
test pair was run locally on the ds_cos arm. It transported mass 1.0 with an
admissible pair fraction of 0.500 and rejected no cell on either side.

---

## 4. Observed rejection behaviour

Two different things are reported for IC-POT, and they must not be read as one.

### 4a. The solver's own result, no truth used

**The score.** IC-POT's plan gives each cell an untransported share u, defined
as clip(1 - transported mass / nominal mass, 0, 1). At the fixed cutoff, a
cell with u > 0.5 is rejected. This is the solver's native behaviour.

Untransported nominal mass fraction, mean over pairs, identical on the two
sides (`main_score_quality.csv`):

| arm | all_shared | population_lost | population_emerged | populations_disjoint |
|---|---:|---:|---:|---:|
| `ranknm256_noscale_ds_cos` | 0.000 | 0.000 | 0.000 | 0.001 |
| `ranknm256_noscale_cos` | 0.000 | 0.000 | 0.000 | 0.001 |
| `raw` | 0.282 | 0.287 | 0.287 | 0.304 |

Fixed cutoff u > 0.5, rejected fraction per case and side, and directional F1
(`main_decisions_per_pair.csv`):

| arm | all_shared src / tgt | lost src / tgt | emerged src / tgt | disjoint src / tgt | directional F1 |
|---|---|---|---|---|---:|
| `ranknm256_noscale_ds_cos` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 |
| `ranknm256_noscale_cos` | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 |
| `raw` | 0.28 / 0.28 | 0.29 / 0.29 | 0.29 / 0.29 | 0.30 / 0.30 | 0.30 |

Rounded to two places, the cosine rows are zero. Pair by pair, the fixed
cutoff on the cosine arms rejected:

- no cell in all_shared or population_emerged;
- at most 0.13% of a side in population_lost (cos arm only);
- at most 0.93% of a side in populations_disjoint.

Score quality on the sides that hold both classes (`main_score_quality.csv`;
per case in `set2_oracle_threshold.csv`):

| arm | ROC-AUC | PR-AUC | fixed-cutoff F1 |
|---|---:|---:|---:|
| `ranknm256_noscale_ds_cos` | 0.501 | 0.201 | 0.000 |
| `ranknm256_noscale_cos` | 0.501 | 0.202 | 0.000 |
| `raw` | 0.530 | 0.221 | 0.239 |

The PR-AUC sits at the prevalence of the unmatched cells, about 0.20.

**Cosine arms.** IC-POT transports essentially all mass in every case,
including the unmatched populations in lost, emerged and disjoint. Both sides
carry unit mass, so the side with an extra population can place it on the
other side's cells at a cost below lambda. That costs less than leaving it
unmatched.

**Raw.** About 29% of each side is left untransported whatever the case. That
includes all_shared, where nothing should be rejected. It is unrelated to
which cells are unmatched: on the source side of population_emerged, where no
cell should be rejected, it rejected 0.286, ranging over pairs from 0.004 to
0.439 (`audit_emerged_source.csv`).

### 4b. The truth-tuned global threshold, an upper bound

The global-optimal threshold is a post-processing rule applied to u:

- it is chosen once per method and arm, with the truth;
- it maximises the mean pair F1 over the sides holding both classes, the
  source in population_lost and the target in population_emerged, 90 pairs per
  arm;
- it is then frozen and applied to every pair, case and side.

For IC-POT it selected `-inf` on all three arms, with macro F1 0.3328
(`main_global_thresholds.csv`). Every cell of every pair is rejected: all 1080
pair-side decisions of that strategy reject every cell.

**Why `-inf`.** With a score that does not rank the unmatched cells,
rejecting a share q of cells gives an expected F1 of 2pq / (p + q), where p is
the true unmatched share. That rises with q, so the maximum is at rejecting
everything. The other candidates are the observed values of u on those
sides. On the cosine arms almost all of them are zero, and none gave a higher
mean F1 than rejecting everything.

**Why it is identical across arms.** Under reject-all, F1 = 2p / (1 + p)
depends only on the truth of the side. All three arms are scored on the same
pairs and the same cells, so the result cannot differ between them:

| case (scored side) | true unmatched share | F1 when every cell is rejected |
|---|---:|---:|
| population_lost (source) | ~0.20 | 0.31 to 0.36 per pair |
| population_emerged (target) | ~0.20 | 0.31 to 0.36 per pair |
| populations_disjoint (source) | 1.00 | 1.00 |

Directional F1 over the 135 pairs is 0.5552 ± 0.3158 on each of the three
arms. The local-optimal F1, a per-pair truth-tuned cutoff, is 0.333 on every
arm for the same reason.

### 4c. How to read the two

On the cosine arms the solver's native result rejects almost nothing. The
reject-all result is produced entirely by the truth-tuned threshold. Its F1 of 0.56 is
what any method scores on this benchmark by rejecting every cell. It is not
evidence that IC-POT separates matched from unmatched cells.

The method comparison heatmap shows the cost of that rule in the columns
whose ground truth is 0: all_shared and the matched side of the one-population
cases are fully rejected there. Directional F1 reads only the sides holding
cells to reject, so it does not count that cost and must be read with the
heatmap.

**What this result is about.** It is about c_s = c_t = 0.5 on this
benchmark's median-normalised cost. It does not test IC-POT with pointwise
costs, and it does not test other constant values. Choosing another value
would be choosing it with the truth in view.

---

## 5. Reproducibility

### 5a. Reference

> Salil Parth Tripathi, Bertrand Chapron, Fabrice Collard, Nicolas Courty,
> Ronan Fablet. *Take It or Leave It: Intent-Controlled Partial Optimal
> Transport.* arXiv:2605.20030v1 [cs.LG], 19 May 2026.
> https://arxiv.org/abs/2605.20030

Equations and results used: Eq. (2), Propositions 1-7, and Appendices C, D, E,
H and L.

### 5b. Exact parameters

| | |
|---|---|
| unmatched costs | c_s(i) = c_t(j) = 0.5 for every cell |
| price | lambda = c_s + c_t = 1.0 |
| marginals | a_i = 1/n, b_j = 1/m; each side has unit mass |
| cost | the ConfidenceOT run's own (squared Euclidean on the joint PCA coordinates, read back as float32 from the acceptance-0.90 run's `joint_pca_coordinates.csv.gz`, divided by `median_pair_scale` with `default_rng(seed + index * 104729)`) |
| geometry check | cost_scale, cost_median and cost_max against the run's `run.json`, rtol 1e-6 |
| seed | 20260925 |
| solver | POT 0.9.7.post1 `ot.emd` on the (n+1) x (m+1) augmented support of Proposition 6; `numItermax` 1e9; `result_code` must be 1 |
| regularisation | none |
| score | u_i = clip(1 - sum_j P_ij / a_i, 0, 1), each side (`benchmark/comparator_rejection.py`, `unmatchedness`) |

### 5c. Evaluation protocol

The protocol is `benchmark/56_comparator_metrics.py`. It is unchanged from the
other comparators.

- **Fixed cutoff.** u > 0.5. This uses no truth.
- **Global-optimal threshold.** One per method and arm. Candidates are every
  distinct u on the sides holding both classes, plus `-inf`. The chosen
  threshold maximises the mean of the per-pair F1 over those sides. Ties go to
  the larger cutoff. It is then frozen and applied to every pair, case and
  side. It is an upper bound.
- **Local-optimal F1.** A per-pair max-F1 cutoff, chosen with the truth. It is
  an upper bound.
- **Directional F1.** 2TP / (2TP + FP + FN), on population_lost on the source,
  population_emerged on the target, and populations_disjoint on the source.
  all_shared has nothing to reject and enters only the heatmap.
- **ROC-AUC and PR-AUC.** Of u, on the sides holding both classes.
- **Refusals.** `56_` refuses an IC-POT run whose `provenance.json` records
  any cost other than 0.5 on either side. It also refuses one whose score
  files do not reproduce the stored fixed-cutoff counts.

### 5d. Code

| | |
|---|---|
| reference implementation | `src/traditional_ot/icpot.py` (since `ce5cc2f`), unit tests `tests/test_icpot.py` |
| benchmark solver | `benchmark/50_solver_comparison.py`, `constant_cost_icpot`, mode `--icpot-cost 0.5` |
| array job | `benchmark/tacc/solvers.slurm`, switch `CONFIDENCEOT_SOLVER_ICPOT_COST` |
| scoring | `benchmark/56_comparator_metrics.py`, strategies "IC-POT (c_s = c_t = 0.5): fixed cutoff (u > 0.5)" and "... : global-optimal threshold" |
| audit | `benchmark/57_audit_main_decisions.py` |
| figures | `benchmark/55_plot_method_comparison.py` |
| commits | `f778bdd` IC-POT in the comparison; `412bf4e` acceptance minimums for ConfidenceOT and the renamed ConfidenceOT strategies; `680442b` F1 whiskers cut at 0 and 1 |

### 5e. How it was run

On TACC Vista, queue `gg`. The work list is the native benchmark's, filtered
to the three arms and dealt round-robin by size into blocks of 45, so that
each array task holds 15 pairs of each size. Neither the order nor the
chunking changes any result.

```bash
# the work list, 540 units
source /home1/10119/ghzheng/.bashrc; conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot; python - <<'EOF'
import pandas as pd
source = "/scratch/10119/ghzheng/OT_project/benchmark_native/worklist_1000_5000_10000.csv"
out = "/scratch/10119/ghzheng/OT_project/benchmark_native/worklist_3arms_balanced.csv"
arms = ["ranknm256_noscale_ds_cos", "ranknm256_noscale_cos", "raw"]
w = pd.read_csv(source)
w = w[w["preprocessing"].isin(arms)].copy()
w["size"] = w["pair_id"].str.extract(r"^N(\d+)_", expand=False).astype(int)
w = w.sort_values(["size", "preprocessing", "pair_id"], kind="stable").reset_index(drop=True)
blocks = 12
w = w.iloc[[i for k in range(blocks) for i in range(k, len(w), blocks)]].reset_index(drop=True)
w.drop(columns="size").to_csv(out, index=False)
EOF

# IC-POT, 4 array tasks per arm
for l in ranknm256_noscale_ds_cos ranknm256_noscale_cos raw; do CONFIDENCEOT_BENCH_ROOT=/scratch/10119/ghzheng/OT_project/benchmark_native CONFIDENCEOT_BENCH_WORKLIST=/scratch/10119/ghzheng/OT_project/benchmark_native/worklist_3arms_balanced.csv CONFIDENCEOT_SOLVER_LABEL=$l CONFIDENCEOT_SOLVER_ICPOT_COST=0.5 CONFIDENCEOT_SOLVER_CHUNK=45 sbatch --array=0-3 --time=12:00:00 -o /scratch/10119/ghzheng/OT_project/benchmark_native/logs/solv_%A_%a.out -e /scratch/10119/ghzheng/OT_project/benchmark_native/logs/solv_%A_%a.err /scratch/10119/ghzheng/OT_project/code/ConfidenceOT/benchmark/tacc/solvers.slurm; done

# scoring and audit, once every run tree is complete
sbatch -p gg -N 1 -t 02:00:00 -A MCB26031 -J cot_metrics -o /scratch/10119/ghzheng/OT_project/benchmark_native/logs/metrics_%j.out --wrap "source /home1/10119/ghzheng/.bashrc; conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot; cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT; python benchmark/56_comparator_metrics.py /scratch/10119/ghzheng/OT_project/benchmark_native /scratch/10119/ghzheng/OT_project/benchmark_native/comparator_metrics_full && python benchmark/57_audit_main_decisions.py /scratch/10119/ghzheng/OT_project/benchmark_native/comparator_metrics_full/main_decisions_per_pair.csv /scratch/10119/ghzheng/OT_project/benchmark_native > /scratch/10119/ghzheng/OT_project/benchmark_native/comparator_metrics_full/audit_57.txt && cd /scratch/10119/ghzheng/OT_project/benchmark_native && tar czf comparator_metrics_full.tgz comparator_metrics_full"
```

The figures were drawn locally from the downloaded table:

```bash
python benchmark/55_plot_method_comparison.py benchmark_results/native/comparator_metrics_full/main_decisions_per_pair.csv benchmark_results/native/method_comparison_full --check-against benchmark_results/native/benchmark_metrics.csv
```

### 5f. Outputs

All under `/scratch/10119/ghzheng/OT_project/benchmark_native/` on TACC.

| path | contents |
|---|---|
| `solvers_icpot/<arm>/<pair_id>/` | `solver_comparison.csv`; `provenance.json`, whose `icpot` block holds the costs, lambda, `result_code`, transported mass, admissible pair fraction and mass above lambda, beside the geometry check and the POT version; `score_IC-POT_source.npy`, `score_IC-POT_target.npy` |
| `comparator_metrics_full/` | `main_decisions_per_pair.csv`, `main_scores_per_pair.csv`, `main_global_thresholds.csv`, `main_score_quality.csv`, `set1_fixed_operating_point.csv`, `set1_mass_rejection.csv`, `set2_oracle_threshold.csv`, `set3_partial_ot_budget.csv`, `audit_emerged_source.csv`, `audit_57.txt` |
| `comparator_metrics_full.tgz` | the directory above, as downloaded |

The same tables were extracted locally to
`benchmark_results/native/comparator_metrics_full/`. The figures are in
`benchmark_results/native/method_comparison_full/`: the heatmap and the
directional F1 bars, as PNG and PDF, with the CSVs behind them.
`benchmark_results/` is not tracked by git.
