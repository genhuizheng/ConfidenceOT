# OT depth benchmark: do OT couplings follow biology or sequencing depth? (PLAN, fixed 2026-11-17)

This plan is written before any result of this study is computed. Changes after the first result go into dated
post-hoc sections at the end, never into the text above them.

## Core question

When OT is applied to two single-cell populations, does the coupling preserve biological relationships (population
identity, perturbation state), or do sequencing depth (nCount, total UMI) and detection breadth (nFeature, detected
genes) dominate the matching?

Three quantities are kept apart throughout:

1. **Coupling–depth association**: how strongly the coupling links cells of similar nCount / nFeature.
2. **Harm**: whether that association moves mass away from the biologically correct partner.
3. **Depth–biology association**: how far depth itself tracks biological state in the data. When it does, depth-similar
   matching can be correct, and it is not called an error.

## Motivation (prior findings, not re-tested here)

- KP-Tracer: RNA-OT-preferred cells within a clade were deeper (median depth ratio x1.40) and closer in RNA (NES
  Spearman with an RNA-proximity baseline 0.77).
- LTL331 PDX: rejected-vs-retained GSEA followed depth.

## Locations

- Code and documents: `ot_depth_benchmark_20261117/` (tracked; `.md` files added with `git add -f` if needed).
- Outputs: `results/ot_depth_benchmark_20261117/` (gitignored).
- Local data: `D:/Xia_lab/CellOT_data/norman2019/` (Dataset B local). The Arc Virtual Cell Challenge data are used on
  TACC only (user, 2026-11-17: large data go to TACC).

## Fixed settings

### Representations (cost families)

The cost is part of the comparison condition, never of the method score.

| Cost | Definition | Role |
|---|---|---|
| **C1 main** | `ranknm256-midrank\|none\|scale\|cos\|pca50` + frozen calibration v3 correction `within_side_split+mnncovlocal` (v3_common.py, imported, not copied); cost = squared Euclidean distance of the L2-normalised corrected coordinates (= 2 - 2 cos) divided by its median sampled cross-side pair cost (02_run_pair.py) | ConfidenceOT main configuration. Chosen on Splatter L5 Case 2 F1, not on any dataset of this study; not reselected |
| **C2 standard** | scanpy defaults as used by WOT / moscot / CINEMA-OT tutorials: `normalize_total(target_sum=1e4)`, `log1p`, 2,000 HVG (`seurat` flavour, joint), `scale(max_value=10)`, PCA 50 (arpack, seed), squared Euclidean, divided by its median cross-side pair cost; no correction | The common default of published OT tools |

### Solvers on a fixed cost (comparison A)

Same cells, same cost matrix C (median-scaled), uniform marginals a = 1/n, b = 1/m.

| Solver | Implementation | Parameters (fixed now) |
|---|---|---|
| Independent coupling | a ⊗ b | none (baseline: no information) |
| Exact balanced OT | `ot.emd` (POT 0.9.7) | none |
| Entropic balanced OT | `ot.sinkhorn` (log domain, `sinkhorn_log`) | ε = 0.1 (the ConfidenceOT backbone value) |
| Entropic unbalanced OT | `ot.unbalanced.sinkhorn_unbalanced` (KL), log domain | ε = 0.1, λ (reg_m) = 1.0 (the ConfidenceOT backbone without the gate) |
| Partial OT | `ot.partial.partial_wasserstein` | transported mass m = 0.8 (fixed; not tuned) |
| ConfidenceOT M4-E | `02_run_pair.py` in process (v3_common.fit), the Splatter fit arguments of the local benchmark (`FIT_ARGS`: ε 0.1, λ 1, acceptance 0.99, within-side split null) | as frozen |

ConfidenceOT on C2 uses the same fit arguments and the within-side split null, without the mnncovlocal correction
(the correction belongs to C1's representation).

### Learned-map / original methods (comparison B, reported separately)

Run with original implementations and recommended settings, on TACC for final numbers (user, 2026-11-17); local runs
are tests only. A learned map is not a coupling: it is scored only on held-out, distribution-level and depth-leakage
metrics, never on coupling overlap. Candidates: CellOT (ICNN), CINEMA-OT, moscot (as a solver it equals entropic UOT on
C2, so it is a check of comparison A, not a separate method). The final list is in `METHODS.md`.

## Dataset A: Splatter (existing simulations only)

Data: local N = 200 per side, `results/splatter_local_20261103/{generated,conditions}`, replicates 1-8, levels L0-L5,
A4, A5, cases all_shared / population_lost / population_emerged / populations_disjoint, truth from
`population_truth.tsv`. No new simulation is run. The N = 5,000 TACC grid kept no couplings locally (3 TB on TACC);
its gate tables are not used for coupling metrics.

### S1 Native scenarios

Every level x case x replicate (8 x 4 x 8 = 256 pairs) x {C1, C2} x every solver above.

| Scenario asked | Where it is read |
|---|---|
| Same population, different depth | L1_depth all_shared (target depth about 0.52x) |
| Different populations, similar depth | L0_matched populations_disjoint |
| Population imbalance | A5_imbalance_only all_shared |
| Unmatched populations | population_lost / emerged / disjoint at every level |
| Depth and dropout | L2_depth_dropout, all cases |
| Batch, perturbation, imbalance (reference) | L3, L4, L5, A4 |

### S2 Depth-identity conflict (constructed from existing counts)

Built from L0_matched all_shared (no depth difference, no dropout), replicates 1-8. This is a data manipulation for
the experiment, not a preprocessing step.

- **Conflict arm**: on the target side only, the cells of Group1 and Group2 are binomially thinned,
  X' ~ Binomial(X, f), f ∈ {1, 0.5, 0.25, 0.1}; seed 20261117 + 100 x replicate + index of f. The other target groups
  and the whole source are untouched. For a source cell of Group1/Group2 the correct partner is now shallower, and the
  wrong groups Group3-5 have source-like depth (the "technical lure").
- **Aligned arm**: the same thinning applied to Group1/Group2 on both sides (independent draws). Depth now tracks
  biology on both sides, so depth-similar matching is correct matching.
- **Global arm**: the whole target thinned by f (same population, different depth, continuous version of L1).

Read for source cells of Group1/Group2: mass share to the correct group, to the wrong-unthinned groups (similar depth,
wrong biology), to the other thinned group (dissimilar depth, wrong biology), each against the independent coupling's
expectation.

### Splatter metrics (per fit; P is the coupling, normalised to total mass 1 unless stated)

| Metric | Definition |
|---|---|
| `correct_mass` | Σ P_ij 1[g_i = g_j] / Σ P_ij |
| `correct_mass_indep` | the same under a ⊗ b (Σ_g a_g b_g) |
| `wrong_simdepth_mass` | wrong-group mass on pairs whose \|Δ log10 nCount\| is below the median \|Δ log10 nCount\| of all wrong-group pairs, divided by all wrong-group mass (independent coupling: about 0.5) |
| `depth_corr_count` / `depth_corr_feature` | coupling-weighted Pearson correlation between log10 nCount (nFeature) of the source cell and of the target cell over all pairs (independent coupling: 0). Invariant to a global side shift |
| `depth_corr_count_within` | the same restricted to correct-group pairs: depth preference among biologically equivalent partners |
| `depth_gap_ratio` | coupling-weighted mean \|Δ log10 nCount\| divided by the independent coupling's value (< 1: matched pairs closer in depth than chance) |
| `transported_share_unmatched` | for cases with an unmatched population: Σ of P rows (columns) of the unmatched population / its marginal mass, before normalisation; balanced solvers give 1 by construction |
| `rejected_fraction` | gate methods only, per side, separately for should-reject and shared cells (one quantity per heatmap) |
| `rejection_depth_corr` | gate methods only: Spearman between log10 nCount and rejection among shared cells, per side |
| `false_asserted_share` | gate methods: wrong-group mass in the retained x retained block / asserted mass (as run_local_benchmark.score) |

Depth–biology association in the data: η² of log10 nCount by group, per side (Splatter: about 0 by construction except
in S2's conflict and aligned arms).

### Splatter key figure

What OT matches when biology and technical similarity conflict: S2 conflict arm, x = thinning factor f, y = mass share
of Group1/2 source cells, three lines (correct group, wrong similar-depth groups, wrong dissimilar-depth group),
panels by solver, C1 and C2 side by side, independent-coupling expectation as a dashed line; aligned arm as the
companion panel.

## Dataset B: perturbation scRNA-seq used for virtual-cell training

### Local: Norman et al. 2019 (scPerturb v1.4)

- File `NormanWeissman2019_filtered.h5ad`, Zenodo record 13350497 (scPerturb v1.4, CC-BY-4.0), md5
  c870e6967d91c017d9da827bab183cd6, 0.70 GB. K562 CRISPRa Perturb-seq, single and double perturbations; the
  training set of GEARS, CPA, scGPT and others.
- `01_audit_norman.py` records before any fit: matrix kind (integer counts or not), perturbation / control labels,
  cells per perturbation, nCount / nFeature, batch (GEM group) and QC columns. If a field is missing it is reported,
  not imputed.
- Training split: Norman has no split in the deposit. GEARS defines perturbation-level splits for unseen-perturbation
  prediction, which is not this question. We follow CellOT: a cell-level split per condition, 80% train / 20% held
  out, seed 20261117.

### TACC: Arc Virtual Cell Challenge 2025 (H1 hESC CRISPRi)

- `gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/train/adata_Training.h5ad` (15.48 GB, updated
  2025-12-16) and `pert_counts_Training.csv` (150 target genes; e.g. TMSB4X 4,760 cells, median UMI 53,652). The
  official split is train (150 genes) / validation (50) / test (100); only the training file is used.
- The same audit runs first on TACC; the pipeline runs only after the audit confirms counts and labels. Datasets are
  never pooled.

### Units and fits

- One unit = one perturbation (Norman: every single and every double perturbation, none filtered; VCC: every training
  target gene).
- Source = control (non-targeting) cells, target = perturbed cells. One fixed control reference: a seeded random
  subsample of 2,000 control cells (all if fewer), split 80/20 like the perturbed cells. Fits use train cells only.
- Negative control unit: control vs control (two disjoint control subsamples of 2,000) with every solver: the pure
  technical baseline of coupling–depth association and of rejection.
- Replicate consistency: a second control reference (seed + 1) for every unit.
- Solvers: comparison A on C1 and C2; comparison B separately.

### Perturbation metrics (no one-to-one truth; coupling overlap is never reported as accuracy)

| Metric | Definition |
|---|---|
| `depth_corr_count` / `depth_corr_feature` | as Splatter, between control and perturbed cells under P |
| `depth_shift` | log2 median nCount perturbed / control (depth–biology association at the population level) |
| `depth_state_corr` | among perturbed cells: Spearman between log10 nCount and the perturbation score (projection of the cell's C2 PCA coordinates on the train mean shift perturbed − control) (depth–biology association at the cell level) |
| `ite_depth_r2` | per-cell treatment effect ITE_i = barycentric target of control cell i minus x_i (C2 PCA space, as CINEMA-OT); R² of the ITE projection on the mean shift explained by the paired cells' Δ log10 nCount and Δ log10 nFeature (technical leakage into effects) |
| `heldout_energy` | energy distance in C2 PCA space between the predicted perturbed population (held-out control cells mapped by the 1-NN train-control barycentric target; learned maps: their own T) and held-out perturbed cells |
| `heldout_de_corr` | Pearson correlation over the top 50 DE genes (train perturbed vs train control, log1p CP10k) between predicted and observed held-out mean shifts |
| `rejected_fraction` | gate only, per side |
| `rejection_score_corr` / `rejection_depth_corr` | gate only, perturbed side: Spearman of rejection with the perturbation score and with log10 nCount |

Baselines for the held-out metrics: identity (predict control), mean shift (control + train mean shift), random
train-perturbed sample (a distribution reference that ignores control state).

## Fair-comparison rules

- Comparison A shares cells and cost; comparison B is reported separately.
- No preprocessing is reselected; C1 is the main configuration, C2 is the community default, both are reported.
- No PASS/FAIL, no composite score, no unit filtering, no pair selection. Missing fits are listed with their reason.
- One quantity per heatmap; separate panels in nbt style (`realdata_benchmark/v3_grid/figures/nbt_style.py`).

## Deliverables

`LITERATURE.md`, `METHODS.md`, Splatter tables + figures, Norman tables + figures, TACC scripts for VCC and for the
original methods, `CONTROLS_AND_LIMITATIONS.md`, a short English deck, `SUMMARY_zh.md`.

## Post-hoc implementation note, 2026-11-17 (before any result was read)

- Smoke test (rep 1, L0, 4 pairs): POT `sinkhorn_unbalanced(method="sinkhorn_stabilized")` did not converge on the C2
  cost (final error 0.997, transported mass 0.12). POT `method="sinkhorn"` gives 0.6548, equal to ConfidenceOT's own
  UOT backbone with an unreachable rejection cost (0.6548) on the same cost. The UOT solver is therefore POT
  `method="sinkhorn"`; ε and λ are unchanged. Balanced solvers record their largest relative marginal error.
- `check_against_02.py`: the direct ConfidenceOT call equals the production path (v3_common.fit -> 02_run_pair.py) on
  three pairs: identical gates on both sides, coupling max |difference| about 1e-9 (float32 storage).
- 2026-11-17, after the first S1/S2 run (its numbers are discarded, not reported): POT `method="sinkhorn"` (not log
  domain) underflowed in S2 conflict f = 0.1 on C2 (transported mass about 0 for every group, while ConfidenceOT's own
  log-domain UOT backbone on the same cost transports about 0.5-0.66). The UOT solver is replaced by the repository's
  log-domain KL-UOT (`confidenceot._cpu_uot.solve_fixed_bidirectional_uot` with every cell open, threshold 1e-4), i.e.
  exactly the ConfidenceOT backbone without the gate; ε, λ unchanged. S1 and S2 are rerun in full.
