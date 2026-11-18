# Controls and limitations

## Controls used

| Control | Splatter | Perturbation data (Norman local, VCC TACC) |
|---|---|---|
| Biological identity | population labels (truth) | perturbation labels; per-cell perturbation score (projection on the train mean shift); CRISPRa / CRISPRi target-gene expression |
| nCount / nFeature | from the counts after any thinning | from the raw counts |
| Depth mismatch | L1 (0.52x), S2 thinning f 1-0.1 (conflict, aligned, global) | natural depth differences between control and perturbed cells (`depth_shift_log2`) |
| Population proportions | A5 / L5 imbalance | perturbed vs control cell counts differ per unit |
| Independent coupling | a ⊗ b in every fit | a ⊗ b in every fit |
| Technical-only reference | L0 all_shared; S2 global arm | control vs control unit (two disjoint control subsamples) |
| Gate behaviour | rejected fraction of unmatched and shared cells; Spearman of rejection with depth | rejected fraction; Spearman of rejection with perturbation score and with depth |
| Replicate consistency | 8 Splatter replicates (SE in every figure) | control references A and B for every perturbation: rank agreement of per-unit metrics, Jaccard of rejected perturbed cells |
| Held-out evaluation | not needed (truth) | 20% cells per condition held out; identity, mean-shift and random-target baselines |

## How the three quantities are kept apart

1. **Coupling–depth association**: coupling-weighted correlation of log nCount / log nFeature between matched cells,
   against 0 for the independent coupling and against the control-vs-control unit.
2. **Harm**: Splatter: correct-group mass and the share going to wrong, depth-similar groups (S2). Perturbation data:
   no one-to-one truth, so harm is read indirectly: does depth association rise above the control-vs-control reference,
   does it leak into per-cell treatment effects (ITE R² on depth terms), does it go with worse held-out prediction.
3. **Depth–biology association**: Splatter η² of depth by group (≈ 0 natively, up to 0.97 in S2); perturbation data:
   depth shift per perturbation, Spearman of depth with the perturbation score and with target-gene expression.
   Where depth tracks state, depth-similar matching can be correct (S2 aligned arm) and is not called an error.

## Limitations

- **Splatter scale.** The local run is N = 200 per side (8 replicates). The N = 5,000 run is prepared for TACC
  (`submit_otdb.sh splatter_s1|s2`); gate recall is known to be lower at N = 200.
- **S2 is constructed.** Binomial thinning of existing counts creates the conflict; it is a controlled manipulation of
  depth, not a model of biological depth differences (e.g. cell size). Thinning lowers nCount and nFeature together.
- **Splatter depth is independent of population by construction**, so native Splatter cannot show depth-biology
  confounding; S2 aligned supplies it.
- **No cell-level truth in perturbation data.** Coupling overlap is never reported as accuracy. Held-out energy
  distance rewards matching the perturbed distribution, which a random sample of perturbed train cells already does
  (the random-target baseline); balanced couplings give the same population mean as the mean-shift baseline, so
  population-level DE metrics separate methods only through mass loss or rejection.
- **Fixed solver parameters.** ε = 0.1, λ = 1, m = 0.8 were fixed in advance; partial OT's behaviour depends on m and
  entropic couplings on ε. Literature defaults (WOT, moscot) were not run.
- **Two cost families only.** C1 is the ConfidenceOT main configuration (chosen on Splatter L5 Case 2 F1, not here), C2
  the community default. Other preprocessing (depth equalisation, Pearson residuals) was not tested here; the 296-
  configuration grid was out of scope.
- **Solver implementation.** POT's unbalanced Sinkhorn failed on C2 (non-convergence in the stabilised variant,
  underflow in the plain variant); the repository's log-domain KL-UOT is used instead (PLAN post-hoc note). Any study
  that reports UOT on unscaled log1p-PCA costs with small ε should check transported mass for this failure.
- **Learned maps.** CellOT is scored only on distribution-level held-out metrics in its own feature space; CINEMA-OT's
  matching is read as a coupling on its own preprocessing. Neither is comparable cell-for-cell with comparison A.
- **ConfidenceOT on CPU.** The local torch has no CUDA; full perturbation runs go to TACC.
