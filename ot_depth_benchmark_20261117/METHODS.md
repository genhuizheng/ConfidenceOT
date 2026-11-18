# Methods compared (reproducible list)

Fixed in PLAN.md before results; implementation notes in its dated post-hoc sections. `LITERATURE.md` has the wider
survey. Code: `otdb_common.py` (costs, solvers, metrics), `run_splatter.py`, `run_perturb.py`, `run_learned.py`.

## Comparison A: one fixed cost, same cells

Uniform marginals a = 1/n, b = 1/m. The cost C is the median-scaled squared Euclidean distance of the cost family's
coordinates (C1: L2-normalised, so 2 - 2 cos).

| # | Method | Paper | Code | Parameters | Input | Output |
|---|---|---|---|---|---|---|
| 0 | Independent coupling | (null) | numpy `outer(a, b)` | none | marginals only | coupling with no information |
| 1 | Exact balanced OT | Kantorovich; POT: Flamary et al., JMLR 22(78) 2021 | POT 0.9.7 `ot.emd` | `numItermax=1e7` | C | sparse coupling |
| 2 | Entropic balanced OT | Cuturi, NeurIPS 2013 | POT `ot.sinkhorn(method="sinkhorn_log")` | ε = 0.1, `stopThr=1e-9`, `numItermax=20000` | C | dense coupling |
| 3 | Entropic unbalanced OT (KL) | Chizat et al., Math. Comp. 2018 | repository log-domain KL-UOT (`confidenceot._cpu_uot.solve_fixed_bidirectional_uot`, every cell open) | ε = 0.1, λ_a = λ_b = 1, threshold 1e-4 | C | coupling with soft mass loss |
| 4 | Partial OT | Figalli 2010; Chapel et al., NeurIPS 2020 | POT `ot.partial.partial_wasserstein` | m = 0.8 (always passed) | C | coupling of total mass 0.8; zero rows = unmatched |
| 5 | ConfidenceOT M4-E | this project | `02_run_pair.py` path (equivalence-checked) | ε = 0.1, λ = 1, within-side split null, acceptance 0.99, no rejection cap | C + the cost family's coordinates for the null | coupling + per-cell retain / reject on both sides |

Methods 3 and 5 share one backbone; 5 adds the calibrated gate. Literature defaults (WOT ε = 0.05, λ2 = 50; moscot
ε = 1e-3) are listed in LITERATURE.md and were not run.

### Cost families

| Cost | Steps |
|---|---|
| C1 main | rank top-256, midrank ties (`factorial_common.encode_sides`, `ranknm256-midrank\|none`), 2,000 top-variance encoded genes, gene scaling, PCA 50 (seed), unit rows, `mnncovlocal` correction (v3, imported), unit rows, cost |
| C2 standard | `normalize_total(1e4)`, `log1p`, 2,000 HVG (seurat, joint), `scale(max_value=10)`, PCA 50, cost |

## Comparison B: original implementations, reported separately

| Method | Paper | Code | Settings | Input | Output | Scored on |
|---|---|---|---|---|---|---|
| CINEMA-OT | Dong et al., Nat. Methods 2023 | PyPI `cinemaot` 0.0.5 (authors' package), `cinemaot_unweighted` | package defaults: dim 20, thres 0.15, smoothness 1e-4, eps 1e-3, mode parametric | log1p CP10k, 2,000 HVG, PCA (`X_pca`) of train cells | matching matrix (control x perturbed) + per-cell treatment effects | coupling-depth association, ITE depth leakage (as comparison A), never coupling overlap |
| CellOT | Bunne et al., Nat. Methods 2023 | github.com/bunnech/cellot (BSD-3), scRNA route: scGen autoencoder (latent 50) then ICNN in the latent | the repository's `configs/models/scgen.yaml` + `cellot.yaml`: ICNN [64, 64, 64, 64], Adam lr 1e-4, β (0.5, 0.9), 100,000 iterations, 10 inner iterations, batch 256 | top-1,000 HVG log1p CP10k of train cells (the authors' `hvg-top1k-train-only`) | map T(x) for any control cell | held-out energy distance and DE-shift correlation against identity / mean-shift / random-target baselines in its own feature space; depth leakage of T(x) − x |
| moscot `SinkhornProblem` | Klein et al., Nature 2025 | not run | as a solver it is entropic OT on a fixed cost (comparison A #2/#3) | | | |

The learned maps run on TACC (gg CPU); local runs are smoke tests only (user, 2026-11-17).
