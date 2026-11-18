# Dataset B, Norman 2019 local sample (results 2026-11-18)

Rule fixed before running (PLAN.md, 2026-11-18): every 12th perturbation in sorted order (20 perturbations; 80-962
train cells each), control references A and B (1,600 train control cells each), and the control-vs-control unit; cost
families C1-C4; every solver including ConfidenceOT with the UOT and the balanced backbone. 1,271 metric rows, 0
failures. Tables `results/ot_depth_benchmark_20261117/norman/tables/`, figures `.../norman/figures/`. The full Norman and
VCC runs are for TACC. No coupling overlap is reported as accuracy.

## 1 Depth tracks perturbation state in these data (depth-biology association)

Median over the 20 perturbations: log2 depth ratio perturbed / control -0.27 (range -1.02 to 0.20); within perturbed
cells, Spearman of log nCount with the perturbation score -0.40 (range -0.76 to 0.44); CRISPRa target-gene log2FC 2.0
(target-gene expression vs depth -0.07). Strongly perturbed cells are shallower, so depth-similar matching is partly
state matching here and is not an error by itself.

## 2 Coupling-depth association is mostly the technical baseline

Median coupling-weighted correlation of log nCount (control -> perturbed, reference A) and the control-vs-control unit:

| Cost | Exact | Entropic bal. | UOT | Partial | ConfidenceOT (UOT) | ConfidenceOT (bal.) | control vs control (UOT) |
|---|---|---|---|---|---|---|---|
| C1 | 0.46 | 0.31 | 0.36 | 0.52 | 0.35 | 0.29 | 0.46 |
| C2 | 0.49 | 0.34 | 0.25 | 0.43 | 0.25 | 0.36 | 0.40 |
| C3 | 0.65 | 0.44 | 0.46 | 0.70 | 0.42 | 0.37 | 0.53 |
| C4 | 0.53 | 0.30 | 0.34 | 0.57 | 0.30 | 0.24 | 0.43 |

- Every solver links cells of similar depth (independent coupling 0), and at least as strongly between two halves of
  the same control population as from control to perturbed: the association is a property of the cell geometry in every cost
  family, present without any biological difference.
- Sparse solvers (exact, partial) show the most; entropic ones less. C3, the most depth-robust cost on Splatter, shows
  the most here: on real data rank encodings do not remove depth-linked state.
- Reproducible across the two control references (Spearman of per-perturbation values 0.91-0.99; partial OT
  0.77-0.85).

## 3 Technical leakage into per-cell treatment effects

R² of the per-cell treatment effect (along the mean shift) explained by own and partner nCount / nFeature: 0.11-0.46
for couplings vs 0.05 for the independent coupling. Per-cell effects inherit depth, which a population-level DE does not
show (held-out DE-shift correlation 0.92-0.99 for every solver and for the mean-shift baseline).

## 4 Held-out prediction (distribution level)

Median held-out energy distance: identity baseline 3.20, mean shift 0.46, random train-perturbed sample 0.73; exact OT
0.42-0.57; entropic couplings 1.7-3.0 (the barycentric map averages partners and shrinks variance). Couplings add
nothing over the mean-shift baseline at the population level here.

## 5 Gate behaviour (ConfidenceOT)

Rejected fraction of perturbed cells 1-7% (control vs control ≤ 0.9%); rejection leans to shallow cells (Spearman with
log nCount -0.09 to -0.21) and to strongly perturbed cells (with the perturbation score +0.08 to +0.18), which in these
data are the same cells. Jaccard of the rejected perturbed cells between control references A and B 0.33-0.69. The
balanced backbone rejects about as many cells as the UOT backbone (C1 0.026 vs 0.011, C3 0.067 vs 0.029, C4 0.071 vs
0.036).

## Answer for this sample

OT couplings on Norman link cells of similar depth, but no more than between two control halves; depth here also
tracks perturbation strength, so the association cannot be split into harm and biology from coupling statistics alone.
What is clearly technical is the leakage of depth into per-cell effect estimates. These numbers are a 20-perturbation
local sample; the full run is on TACC.
