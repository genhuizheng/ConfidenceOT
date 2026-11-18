# External OT methods "as their authors run them" (checked 2026-11-18)

Task: Splatter, 10,000 genes, raw integer UMI counts, source and target of 5,000 cells each. No time course and no
labels; the source/target flag is the only covariate. For each method, this file gives the authors' scRNA
preprocessing and solver settings, the output, and a tier:

- **Tier 1**: run it.
- **Tier 2**: run it if the install works. These need their own environment or are slow on CPU.
- **Tier 3**: not applicable, with the reason given.

Sources were read on 2026-11-18: raw GitHub files, PyPI JSON, tutorial notebooks, and PMC/arXiv full texts. If a
statement could not be confirmed it is marked **UNVERIFIED**, with what was checked. Where an item is inferred from code
or data metadata rather than stated in a paper, it is marked "(inferred)". CPU times are estimates; none were measured.

Shared facts used below:

- `log1p CP10k` means `sc.pp.normalize_total(target_sum=1e4)` followed by `sc.pp.log1p`.
- tau = lambda/(lambda+epsilon), the same as in LITERATURE.md.
- Splatter gene names are `Gene1...`. Gene-set growth priors (WOT, moscot `score_genes_for_marginals`, PRESCIENT KEGG
  sets) therefore cannot be computed, and every growth prior falls back to uniform.

## A. Discrete couplings on the observed cells

| Method | Normalisation | Feature selection | Embedding / cost | Solver settings | Output | "Unmatched" |
|---|---|---|---|---|---|---|
| **WOT** | Paper (STAR Methods, PMC6402800): UMI / total x 10,000 after downsampling each cell to 15,000 UMI; log(E+1). Tutorial: "units are log-tpm". | Seurat v2 MeanVarPlot: 20 bins, z-scored dispersion > 1.0 (1,479 genes) | `compute_pca(p0.X, p1.X, 30)`, a joint PCA of the pair with no transform, fit as `pca.fit(x.T)`. Coordinates are scaled by the singular values; sqeuclidean cost, **divided by its median** | `OTModel` defaults: `epsilon=0.05, lambda1=1, lambda2=50, growth_iters=1, local_pca=30, epsilon0=1, tau=10000, solver='duality_gap'`, `tolerance=1e-8`, `batch_size=5`. The paper and tutorial batch run use `growth_iters=3`. Implied tau_a = 1/1.05 = 0.952, tau_b = 50/50.05 = 0.999 | `compute_transport_map(t0, t1)` returns an AnnData: X = coupling (source x target); obs `g0..g{k}` = learned growth (row sums)^(1/dt) | Not defined. Row sums are read as growth. Target marginal q = mean(G), fixed by lambda2=50 |
| **moscot SinkhornProblem**, balanced and unbalanced | Tutorial 700 (sciplex): log1p CP10k | None in the tutorials read (partial check) | Tutorial: `sc.pp.pca()` (50) and `joint_attr="X_pca"`. Default `joint_attr=None` instead runs a joint local PCA, `n_comps=30`, on `adata.X` with no transform. Cost `sq_euclidean`, `scale_cost="mean"` | Defaults `solve(epsilon=1e-3, tau_a=1.0, tau_b=1.0, rank=-1, scale_cost="mean", threshold=1e-3, lse_mode=True)`. Tutorial 700 unbalanced: `solve(1e-2, 0.95, 0.95)` (epsilon, tau_a, tau_b) | `solutions[(s,t)].transport_matrix` (dense), plus `push` / `pull`, and `.a` / `.b` = row / column sums | None defined. With tau < 1, low row or column mass is the only signal |
| **moscot TemporalProblem**, 2 time points | Tutorials: log1p CP10k (200 pancreas, 100 lineage) | None seen | `prepare(time_key, joint_attr=None, policy="sequential")`: local PCA 30 on `adata.X`, sq. Euclidean, `scale_cost="mean"` | Same `solve` defaults. Tutorial 200: `epsilon=1e-3, scale_cost="mean", max_iterations=1e7`, balanced. Tutorial 100 (`LineageProblem`): `tau_a=0.99`. Marginals are uniform unless `score_genes_for_marginals` (human/mouse sets) is run; see `estimate_marginals` (beta_max 1.7, beta_min 0.3, delta_max 1.7, delta_min 0.3) | As above. `posterior_growth_rates` comes from `.a` | As above |
| **CINEMA-OT** (unweighted) | Paper / tutorial: `normalize_total`, log1p, then scale before PCA (inferred from `uns['log1p']` and the var columns of the tutorial h5ad) | Interferon data: seurat-flavour HVG `min_mean=0.0125, max_mean=3, min_disp=0.5`. Alzheimer's data: 2,000 Seurat-v3 HVG | `sc.pp.pca` (50), then FastICA (`dim` components, `whiten="arbitrary-variance"`, `random_state=0`) on `X_pca[:, :dim]`. ICs with Chatterjee xi(IC, group) < `thres` are kept as confounders. Kernel `exp(-d^2/(smoothness * n_cf))` on **unscaled** distances | `cinemaot_unweighted(adata, obs_label, ref_label, expr_label, dim=20, thres=0.15, smoothness=1e-4, eps=1e-3, mode='parametric', preweight_label=None)`. Own balanced Sinkhorn-Knopp, `max_iter=1000`, `eps` = marginal tolerance. Authors' tutorial: `thres=0.5, smoothness=1e-5, eps=1e-3` | `cf, ot_matrix, TE`. `ot_matrix` is ref (source) x expr (target) | None (balanced). Raises "No confounder components identified" if every xi >= thres, which is plausible for disjoint populations |
| **CINEMA-OTW** (weighted) | As above | As above | As above, plus a kNN alignment of expr to ref (k=10), then Leiden (`resolution=1`) | `cinemaot_weighted(..., use_rep=None, dim=20, thres=0.75, smoothness=1e-4, eps=1e-3, k=10, resolution=1)`. pertpy `causaleffect_weighted` uses k=20 and thres=0.15 | `cf, ot, te2, weights` | **Yes, implicitly.** Within each Leiden cluster the larger side is subsampled to the smaller side's count, so a cluster with no partner loses all its cells (dropped, absent from the output) |

Sources:

- WOT
  - Code: https://raw.githubusercontent.com/broadinstitute/wot/master/wot/ot/ot_model.py, which contains `OTModel.__init__`, `compute_single_transport_map` and `compute_default_cost_matrix`.
  - Solver: `wot/ot/optimal_transport.py`. `optimal_transport_duality_gap` sets p = G and q = mean(G), with exponent lambda/(lambda+epsilon_i).
  - PCA: `wot/ot/util.py::compute_pca`.
  - Tutorial notebook: https://github.com/broadinstitute/wot/blob/master/notebooks/Notebook-2-compute-transport-maps.ipynb, which uses `OTModel(adata, epsilon=0.05, lambda1=1, lambda2=50)` and says "We recommend sticking with the default value of epsilon = 0.05". With no growth file, `G = np.ones`.
  - Tutorial page: https://broadinstitute.github.io/wot/tutorial/.
  - Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC6402800/ (STAR Methods: downsampling, MeanVarPlot, "30 dimensional local PCA", epsilon 0.05, lambda1 1, lambda2 50, growth_iters 3).
  - **UNVERIFIED:** the paper does not say in words whether the local PCA input is log or linear. The code applies no transform, and the tutorial input is log-normalised, so the authors' input is log.
- moscot
  - https://raw.githubusercontent.com/theislab/moscot/main/src/moscot/problems/generic/_generic.py
  - `.../problems/time/_lineage.py`, `.../base/problems/problem.py` (`_local_pca_callback`, `n_comps=30`), `.../base/problems/birth_death.py`
  - Tutorials: https://moscot.readthedocs.io/en/latest/notebooks/tutorials/700_tutorial_perturbation_with_Sinkhorn.html, `.../200_temporal_problem.html`, `.../100_lineage.html`
- CINEMA-OT
  - Code: https://raw.githubusercontent.com/vandijklab/CINEMA-OT/main/cinemaot/cinemaot.py
  - Authors' tutorial: `cinemaot_tutorial.ipynb` in the same repository.
  - pertpy wrapper: https://raw.githubusercontent.com/scverse/pertpy/main/src/pertpy/tools/_cinemaot.py
  - Paper: PMC10630139.
  - **UNVERIFIED:** whether the unscaled kernel underflows on Splatter ICs.

| Method | Two-population, no covariates | CPU at 5k x 5k | Install | Tier |
|---|---|---|---|---|
| WOT | Yes. Set `obs['day']` = 0.0 / 1.0 and call `compute_transport_map(0, 1)`; dt = 1 | Yes (dense 5k^2, minutes) | `pip install wot==1.0.8.post2` (2020; deps unpinned: numpy, pandas, h5py, anndata, scikit-learn, scipy, matplotlib, POT) | 1 |
| moscot Sinkhorn balanced | Yes, `key="side"` with 2 categories | Yes; check `.converged` at eps 1e-3 | See section E | 1 |
| moscot Sinkhorn unbalanced | Yes | Yes | See section E | 1 |
| moscot Temporal (2 points) | Yes, `time_key` = 0 / 1 | Yes | See section E | 1 |
| CINEMA-OT | Yes. ref = source, expr = target | Yes ("~1 min for 50,000 cells", paper) | `pip install cinemaot==0.0.5` (py >= 3.7) | 1 |
| CINEMA-OTW | Yes | Yes | Same | 1 |

## B. Gromov-Wasserstein / integration methods (cross-modality by design)

| Method | Normalisation | Feature selection | Embedding / cost | Solver settings | Output | "Unmatched" |
|---|---|---|---|---|---|---|
| **SCOT v1** | Authors feed reduced data, e.g. SNARE RNA = PCA 10 (`replication/SNAREseq_SCOT_alignment.ipynb`); in code, `normalize=True, norm="l2"` | None in the repo | kNN graph (forced `mode="connectivity", metric="correlation"`), Dijkstra, divided by max. Intra-domain only; the shared gene space is not used | `align(k=None→min(0.2n,50), e=1e-3, normalize=True, norm="l2")`, `ot.gromov.entropic_gromov_wasserstein`. Values used: SNARE k=110, e=1e-3 | `scot.coupling`, plus a barycentric projection | None (balanced). If the plan sum is < 0.95 it reports "CONVERGENCE ERROR" |
| **SCOT v2** (unbalanced GW) | As v1 | None | As v1 | `align(normalize=True, norm="l2", k=20, eps=0.01, rho=1.0, projMethod="embedding", out_dim=10)`. Unbalanced example: `k=110, eps=5e-3, rho=0.01, projMethod="barycentric"` | `couplings[0]`, plus a co-embedding | Implicit (marginal shortfall); no rule |
| **Pamona** (partial GW) | Reduced inputs: PBMC = Seurat scaledata (50), then `zscore_standardize`; SNARE RNA = PCA 10 | None | kNN (`mode="distance"`, minkowski), geodesic, divided by max | `Pamona(n_shared=None→min(n1,n2), n_neighbors=10 (README says 30), epsilon=1e-3, Lambda=1.0, output_dim=30, max_iter=1000, tol=1e-9, manual_seed=666)` | `integrated, T = run_Pamona([X, Y])`. T includes a virtual row and column | **Yes.** Mass sent to the virtual cell. `n_shared` is the user's choice via the paper's scree-plot (SPL) procedure. With the default (5,000 / 5,000) it reduces to balanced GW |
| **uniPort** (minibatch UOT in a VAE latent) | log1p CP10k (`normalize_total` default) → HVG 2,000 (`batch_key='domain_id'`, subset) → `up.batch_scale` (MaxAbs per dataset) | 2,000 HVG on the common genes, plus per-dataset HVG | Coupled VAE with a 16-d latent; UOT cost between latent Gaussians, scaled by max | `Run(mode='h', lambda_s=0.5, lambda_ot=1.0, iteration=30000, reg=0.1, reg_m=1.0, batch_size=256, lr=2e-4, save_OT=False, seed=124)`. The paper uses eps 0.1 and rho 1 for every experiment | `obsm['latent']`, plus a dense plan (query x ref, stitched from minibatches, not globally normalised) when `save_OT=True` | None defined. Relaxed marginals only |

Sources:

- SCOT: https://raw.githubusercontent.com/rsinghlab/SCOT/master/src/scotv1.py and `.../src/scotv2.py`
- Pamona: https://raw.githubusercontent.com/caokai1073/Pamona/master/Pamona.py; bioRxiv 10.1101/2020.11.03.366146
- uniPort: https://raw.githubusercontent.com/caokai1073/uniPort/main/uniport/function.py; tutorial `docs/source/examples/PDAC/PDAC_integration.ipynb`; paper PMC9715710
- Code quirk: in uniPort mode 'd', `reg` and `reg_m` are not passed through. Mode 'h' passes them.
- Repository note: GitHub `main` of uniPort was refactored on 2026-08-23. Pin PyPI 1.3 (2024-09-28).

| Method | Two-population | CPU at 5k x 5k | Install | Tier |
|---|---|---|---|---|
| SCOT v1 | Runs, but uses only within-side geometry, so it is a weak fit for a same-modality task | Hours per solve (O(n^3) GW); do not self-tune | git clone; `requirements.txt` unpinned (POT, torch) | 2 |
| SCOT v2 | Same | Hours (3,000 outer x 3,000 Sinkhorn) | Same | 2 |
| Pamona | Same. `n_shared` must be chosen without ground truth (SPL) | Hours, a few GB | `pip install pamona==0.1.0` (examples use `np.int`, which needs numpy < 1.24). PyPI vs GitHub identity is UNVERIFIED | 2 |
| uniPort | Yes: mode 'h', `obs['source']` and `obs['domain_id']` = side. The paper only applied it across modality or technology | Tens of minutes (estimate) | `pip install uniport==1.3` (py > 3.8, torch >= 1.0) | 2 |

## C. Learned maps / generators (no coupling on the observed cells; score at the distribution level only)

| Method | Normalisation | Features | Embedding | Settings | Output | "Unmatched" |
|---|---|---|---|---|---|---|
| **CellOT** | Paper: library-size normalisation, log1p (`target_sum` UNVERIFIED), no scaling stated | Top 1,000 HVG computed on the train split only (`hvg-top1k-train-only.h5ad`). The repo has no scRNA preprocessing script | scGen AE latent 50 (`scgen.yaml`: hidden [512,512], lr 1e-3, `n_iters 250000`) | `cellot.yaml`: ICNN [64]x4, `latent_dim 50`, `fnorm_penalty 1`, `kernel_init uniform b=0.1`, Adam lr 1e-4, betas (0.5, 0.9), `n_iters 100000` (the paper says 250k), `n_inner_iters 10`. sciplex3: batch 256, `test_size 1000`, `eval_size 1000`. lupus: batch 64, `test_size 0.2` stratified, seed 0 | Map T(x) = grad g(x) | None |
| **CondOT** | From CPA's preprocessed sciplex3 | Top 1k HVG | Precomputed `X_pca` 50 | `condot.yaml`: PICNN 4x64, lr 1e-4, 100k iterations, 10 inner, batch 256; balanced W2 dual | `transport(x, condition)` | None |
| **UOT neural Monge** (Eyring et al., ICLR 2024) | "standard" (not stated) | Not stated | Pancreas: 50 PCs. sciPlex: PCA 30 | Code `TemporalNeuralProblem.solve(batch_size=1024, tau_a=1, tau_b=1, epsilon=0.1, dim_hidden=(64,)*4, iterations=25000, inner_iters=10, pretrain_iters=15001)`. Paper: tau 0.85 (pancreas), ε = 0.01·mean(C) | ICNN map, plus rescaling u, v (paper) | Paper only says UOT "implicitly removes outliers"; low u is our reading. **UNVERIFIED** that the released code learns u and v at all (none found in `_neuraldual.py`) |
| **GENOT** (Klein et al., NeurIPS 2024) | "log-transformed counts" | Not stated | Pancreas: PCA 30. sciplex: PCA 50 | ε 1e-2, tau 1 (U-GENOT 0.8–0.9), batch 1,024, 10k iterations, AdamW lr 1e-4; vector field 8x256 | Conditional generator `transport(source)` | Not in ott's GENOT (no tau, no eta/xi) |
| **CellFlow** | Tutorial 100: log1p CP10k | None seen | `centered_pca(n_comps=100)` (tutorial 300: 30) | `match_linear(epsilon=1.0 default; tutorials 0.5, scale_cost="mean", tau 1.0 or 0.99)`, `hidden_dims=(2048,)*3`, 500k iterations, batch 1,024 | Generator | None |
| **Meta Flow Matching** | Not applicable (mass cytometry, 43 markers) | n/a | GNN population embedding | Needs hundreds of training populations | Generator | None |
| **OT-CFM** (torchcfm) | Paper reuses TrajectoryNet's EB PCs (library-size normalise, sqrt, PHATE-internal PCA; inferred). Cite / Multi: Kaggle data | None (an HVG option exists in the runner) | First **5 PCs, whitened** (App. E.5; `time_dist.yaml max_dim 5, whiten True`) | `ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)` default; runner `otcfm.yaml` `sigma_min 0.1`, lr 1e-3. Paper E.2: MLP 3x64 SELU, AdamW wd 1e-5, batch 128, 1,000 epochs, early stop (patience 30) | Velocity field; map by ODE (`NeuralODE(..., solver="dopri5")`) | None (balanced minibatch EMD) |
| **SB-CFM** | As OT-CFM | As OT-CFM | As OT-CFM | `SchrodingerBridgeConditionalFlowMatcher(sigma=1.0, ot_method="exact")`; runner `sbcfm.yaml`: `ot_sampler "sinkhorn", sigma_min 1.0` (reg = 2·sigma^2). Single-cell sigma is not stated in the paper (UNVERIFIED) | Velocity (+ SDE) | None |

Sources:

- CellOT: https://raw.githubusercontent.com/bunnech/cellot/main/configs/models/cellot.yaml, `.../scgen.yaml`, `.../configs/tasks/{sciplex3,lupuspatients}.yaml`; paper PMC10630137
- CondOT: https://github.com/bunnech/condot (`configs/condot.yaml`, `condot/data/cell.py`)
- UOT-ICNN: https://raw.githubusercontent.com/theislab/moscot_not/main/src/moscot/problems/time/_neural.py; arXiv:2311.15100
- GENOT: https://raw.githubusercontent.com/ott-jax/ott/0.5.2/src/ott/neural/methods/flows/genot.py; https://github.com/MUCDK/genot; arXiv:2310.09254
- CellFlow: https://cellflow.readthedocs.io/en/latest/notebooks/100_pbmc.html; `src/cellflow/utils.py`
- Meta Flow Matching: arXiv:2408.14608; https://github.com/lazaratan/meta-flow-matching
- torchcfm: https://raw.githubusercontent.com/atong01/conditional-flow-matching/main/torchcfm/conditional_flow_matching.py, `.../torchcfm/optimal_transport.py`, `examples/single_cell/single-cell_example.ipynb`; arXiv:2302.00482 App. E

| Method | Two-population | CPU | Install | Tier |
|---|---|---|---|---|
| CellOT | Yes (source / target in one obs column) | Hours (AE 250k + ICNN 100k iterations) | git; py 3.9.5, `torch==1.11.0`, `numpy==1.19.5`, `scanpy==1.8.1`, `anndata==0.7.6`. The project already runs it via `cellot_shim/` and `run_learned_splatter.py` | 1 |
| CondOT | Needs a condition column. With one dummy level it becomes CellOT | Hours | git; py 3.9.7, torch 2.0.1, jax 0.4.13, ott-jax 0.4.2 | 3 (duplicates CellOT) |
| UOT-ICNN | Yes, as 2 "time points" | Hours | git `theislab/moscot_not`; `ott-jax==0.4.4`, `jax<=0.4.23`; separate env | 2 (only if the rescaling nets exist; otherwise it is OT-ICNN = CellOT-like) |
| GENOT | Yes (`condition_dim=None`) | Hours (10k iterations) | ott-jax 0.4.6–0.5.2 only (removed in 0.6.0) | 2 |
| CellFlow | Needs a dummy perturbation covariate plus an embedding | Not at the authors' 500k iterations, 2048-wide networks | `pip install cellflow-tools==0.0.9`, **py >= 3.11** | 2 (separate py3.11 env, reduced iterations = deviation) |
| Meta FM | No: with one population pair it reduces to plain FM | n/a | git, py 3.9/3.10, torch_geometric | 3 |
| OT-CFM | Yes | Minutes | `pip install torchcfm==1.0.7` (torch >= 1.11, torchdyn >= 1.0.6, POT) | 1 |
| SB-CFM | Yes | Minutes | Same | 1 |

## D. Time-series dynamics (two snapshots = one interval)

| Method | Authors' preprocessing | Settings (code defaults) | Output | "Unmatched" | 2-pop | CPU | Install | Tier |
|---|---|---|---|---|---|---|---|---|
| **TrajectoryNet** | scprep library-size (median) normalise, sqrt, PHATE-internal PCA; **5 PCs**, whitened (`--max_dim 5`) | `parse.py`: dims 64-64-64, dopri5, `niters 10000`, batch 1,000, lr 1e-3, `top_k_reg 0.1` (paper command) | CNF velocity + density | None (growth nets are hard-coded for EB) | Yes | Hours | `pip install TrajectoryNet==0.2.4` (py 3.7/3.8, `torchdiffeq==0.0.1`) | 2 (low) |
| **MIOFlow** | Library-size filter, `normalize_total(median)`, sqrt; PCA 50, then PHATE / GAGA 2-d latent | `MIOFlow(hidden_dim=64, n_epochs=100, lambda_ot=1.0, learning_rate=1e-3)`; balanced `ot.emd2` loss | ODE / SDE trajectories | Optional growth model; source mass <= 0.05 is masked as "dead" | Yes | Feasible | `pip install mioflow==0.1.14` (py >= 3.8; torch, phate, POT). The tutorial 02 API is outdated | 2 |
| **JKOnet** | normalize_total, log, 4,000 HVG (cell_ranger), PCA 20 | batch 250, 10k iterations, Sinkhorn eps 1.0 | Energy + ICNN maps | None | Algorithmically yes, but the repo has no scRNA loader (2-D only) | Slow | jax 0.3.15, ott-jax 0.2.2 | 3 |
| **JKOnet\*** | TrajectoryNet EB PCs, `StandardScaler`, 5 components | Exact EMD couplings, 100 epochs, Adam 1e-3, batch 250 | Potential / interaction energies | None | Yes | Fast | git; py 3.12, jax 0.4.26, ott-jax 0.4.6, POT 0.9.3 | 2 |
| **TIGON** | Seurat log-normalise, 3,000 HVG, AE 10-d latent scaled to [-2, 2] (or 3-d UMAP) | `niters 5000`, lr 3e-3, hidden 16 x 4 layers, Dopri5 | Velocity + growth g (WFR) | Mass only through g; with 5,000 / 5,000 the total mass is fixed at 1 | Yes | Hours (Python-loop GMM density) | git only; torch 1.13.1, TorchDiffEqPack 1.0.1 | 2 (low; subsample or GPU) |
| **PRESCIENT** | Normalised, not scaled; inside `process_data`: StandardScaler, PCA 50 (paper: 30 / 50) | `train_model`: `k_dim 500`, `train_epochs 2500`, `train_lr 0.01`, `train_tau 1e-6`; geomloss Sinkhorn `blur 0.1` | SDE generator | None | Yes, with a dummy cell-type column and a zero-growth `.pt` file | Slow ("GPU ... necessary", paper) | `pip install prescient==0.1.0` (py 3.7; `geomloss==0.2.3`, pykeops) | 2 (low) |

Sources:

- TrajectoryNet: https://raw.githubusercontent.com/KrishnaswamyLab/TrajectoryNet/master/TrajectoryNet/parse.py
- MIOFlow: https://raw.githubusercontent.com/KrishnaswamyLab/MIOFlow/main/mioflow/mioflow.py
- JKOnet: https://raw.githubusercontent.com/bunnech/jkonet/main/configs/config_base_jko.yaml
- JKOnet\*: https://raw.githubusercontent.com/antonioterpin/jkonet-star/main/preprocess_rna_seq.py
- TIGON: https://raw.githubusercontent.com/yutongo/TIGON/main/utility.py; PMC10805654
- PRESCIENT: https://raw.githubusercontent.com/gifford-lab/prescient/master/prescient/commands/train_model.py; Nat Commun 2021

Not covered: scOT. No installable same-modality two-population package was identified, and it was not searched further.

## E. Recommended runnable set for Splatter N = 5,000 on TACC CPU nodes

Environments. Linux, CPU only. Each line was checked against PyPI `requires_python` / `requires_dist`; **none of these
environments has been built yet**.

- `ot-moscot` (py3.10)
  - Install: `pip install "moscot==0.4.3" "jax[cpu]==0.6.2" "ott-jax[neural]==0.5.2" scanpy`.
  - Version constraints:
    - moscot 0.5.x needs py >= 3.11 and ott-jax >= 0.6.0.
    - moscot 0.4.3 needs py >= 3.10, jax >= 0.6.1 and ott-jax >= 0.5.0.
    - jax 0.6.2 is the last jax release for py3.10; 0.7.0 needs py >= 3.11.
    - ott-jax 0.5.2 (2025-10) still has GENOT.
  - Run `import moscot, ott` once first. Whether jax 0.6.2 works with ott 0.5.2 is UNVERIFIED.
- `ot-wot` (py3.10)
  - Install: `pip install wot==1.0.8.post2 scanpy POT`.
  - wot is from 2020 and its deps are unpinned. A scan of the source found no `np.float` / `np.int` in `ot/`, but compatibility with numpy 2 / pandas 2 is UNVERIFIED. If the import fails, pin `numpy<2 pandas<2`.
- `ot-torch` (py3.10)
  - Install: `pip install torch --index-url https://download.pytorch.org/whl/cpu` then `pip install torchcfm==1.0.7 cinemaot==0.0.5 uniport==1.3 mioflow==0.1.14 scanpy POT`.
- `cellot`: the existing project route (`cellot_shim/`, `run_learned_splatter.py`).
- Tier 2 git repos: SCOT, Pamona, moscot_not, jkonet-star, TIGON, TrajectoryNet, PRESCIENT. Each gets its own environment with the pins in its row above.

Input for every call below: `ad` is the AnnData of raw counts with source cells first. Set
`ad.obs["side"] = pd.Categorical(["src"]*5000 + ["tgt"]*5000, categories=["src","tgt"])` and
`ad.obs["day"] = [0.0]*5000 + [1.0]*5000`.

```python
import numpy as np, scanpy as sc

# --- WOT (Tier 1) -- paper: downsample to 15,000 UMI, CP10k, log1p, MeanVarPlot HVG, local PCA 30
w = ad.copy()                      # if any cell has > 15,000 UMI, multinomially thin it to 15,000 first
                                   # (the paper's step; OTModel(ncounts=) would apply it to log data, so do it here)
sc.pp.normalize_total(w, target_sum=1e4); sc.pp.log1p(w)
sc.pp.highly_variable_genes(w, flavor="seurat", n_bins=20, min_disp=1.0)   # approximates Seurat v2 MeanVarPlot z>1
w = w[:, w.var.highly_variable].copy()
import wot
m = wot.ot.OTModel(w, day_field="day", epsilon=0.05, lambda1=1, lambda2=50, growth_iters=3, local_pca=30)
tm = m.compute_transport_map(0, 1)   # tm.X: 5000x5000 coupling; tm.obs['g0'..'g3']: growth (row sums)

# --- moscot (Tier 1) -- tutorial 700 preprocessing
from moscot.problems.generic import SinkhornProblem
from moscot.problems.time import TemporalProblem
g = ad.copy(); sc.pp.normalize_total(g, target_sum=1e4); sc.pp.log1p(g); sc.pp.pca(g)        # 50 PCs
sp = SinkhornProblem(g).prepare(key="side", joint_attr="X_pca", policy="explicit", subset=[("src", "tgt")])
sp = sp.solve(epsilon=1e-3, tau_a=1.0, tau_b=1.0, scale_cost="mean")      # balanced (package defaults)
P_bal = np.asarray(sp.solutions[("src", "tgt")].transport_matrix)
sp_u = SinkhornProblem(g).prepare(key="side", joint_attr="X_pca", policy="explicit", subset=[("src", "tgt")])
sp_u = sp_u.solve(epsilon=1e-2, tau_a=0.95, tau_b=0.95)                  # tutorial-700 unbalanced values
P_unb = np.asarray(sp_u.solutions[("src", "tgt")].transport_matrix)       # row / col mass = only "unmatched" signal
tp = TemporalProblem(g).prepare(time_key="day")                           # joint_attr=None -> local PCA 30 on log X
tp = tp.solve(epsilon=1e-3, tau_a=1.0, tau_b=1.0, scale_cost="mean")      # uniform marginals (no gene-set prior)
P_tmp = np.asarray(tp.solutions[(0.0, 1.0)].transport_matrix)

# --- CINEMA-OT / CINEMA-OTW (Tier 1) -- authors' tutorial values
import cinemaot as co
c = ad.copy(); sc.pp.normalize_total(c); sc.pp.log1p(c)
sc.pp.highly_variable_genes(c, min_mean=0.0125, max_mean=3, min_disp=0.5)
c = c[:, c.var.highly_variable].copy(); sc.pp.scale(c); sc.pp.pca(c)     # X_pca, 50
cf, ot_cin, te = co.cinemaot.cinemaot_unweighted(c, obs_label="side", ref_label="src", expr_label="tgt",
                                                 dim=20, thres=0.5, smoothness=1e-5, eps=1e-3, mode="parametric")
cf_w, ot_w, te_w, w_src = co.cinemaot.cinemaot_weighted(c, obs_label="side", ref_label="src", expr_label="tgt",
                                                        dim=20, thres=0.75, smoothness=1e-4, eps=1e-3, k=10, resolution=1)
# also run package defaults (thres 0.15, smoothness 1e-4) as a second arm. Expect ValueError if no IC has xi < thres

# --- OT-CFM / SB-CFM (Tier 1) -- paper App. E: 5 whitened PCs, MLP 3x64, AdamW wd 1e-5, batch 128
import torch
from torchcfm.conditional_flow_matching import (ExactOptimalTransportConditionalFlowMatcher,
                                                SchrodingerBridgeConditionalFlowMatcher)
from torchcfm.models.models import MLP
from torchcfm.utils import torch_wrapper
from torchdyn.core import NeuralODE
f = ad.copy(); sc.pp.normalize_total(f); sc.pp.sqrt(f); sc.pp.pca(f, n_comps=5)   # library-size + sqrt (inferred)
Z = (f.obsm["X_pca"] - f.obsm["X_pca"].mean(0)) / f.obsm["X_pca"].std(0)                   # whiten
x0_all, x1_all = map(torch.tensor, (Z[:5000].astype("f4"), Z[5000:].astype("f4")))
for FM in (ExactOptimalTransportConditionalFlowMatcher(sigma=0.1),                   # runner otcfm.yaml
           SchrodingerBridgeConditionalFlowMatcher(sigma=1.0, ot_method="exact")):   # class default
    net = MLP(dim=5, time_varying=True, w=64)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    # loop: x0, x1 = 128 random rows each; t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
    #       loss = ((net(torch.cat([xt, t[:, None]], -1)) - ut) ** 2).mean(); early stop on held-out W2
    T_x0 = NeuralODE(torch_wrapper(net), solver="dopri5", sensitivity="adjoint").trajectory(
        x0_all, t_span=torch.linspace(0, 1, 101))[-1]                        # mapped source cells

# --- CellOT (Tier 1): existing route, train-only top-1k HVG of log1p CP10k, scGen AE latent 50,
#     configs/models/scgen.yaml then cellot.yaml (see METHODS.md comparison B).
```

Tier 2 calls, written from the code read but not run:

- **uniPort.** Set `src.obs["source"] = "src"` (and `"tgt"`) and `domain_id` per side. Preprocess `adata_cm` (common genes) with log1p CP10k, `highly_variable_genes(n_top_genes=2000, batch_key="domain_id", subset=True)` and `up.batch_scale`, and treat each side the same way. Then run `up.Run(adatas=[src, tgt], adata_cm=adata_cm, mode="h", lambda_s=0.5, iteration=30000, reg=0.1, reg_m=1.0, batch_size=256, lr=2e-4, save_OT=True)`. The return form with `save_OT=True` follows the PDAC tutorial (`adata, OT = up.Run(...)`): UNVERIFIED.
- **Pamona.** `Pamona.Pamona(n_shared=[s], n_neighbors=10, epsilon=1e-3, Lambda=1.0, output_dim=30).run_Pamona([X_src, X_tgt])`. Inputs are z-scored PCA 50 of log1p CP10k. Choose `s` by the authors' SPL histogram (`utils.Interval_Estimation`), never from the truth.
- **SCOT v2.** `SCOTv2([X_src, X_tgt]).align(normalize=True, norm="l2", k=20, eps=0.01, rho=1.0, projMethod="embedding", out_dim=10)`. Inputs are PCA 10 of log1p CP10k, as for the SNARE RNA data.
- **GENOT.** Use ott 0.5.2 `ott.neural.methods.flows.genot.GENOT(vf, flow, data_match_fn, source_dim=30, target_dim=30)` on PCA 30 of log1p CP10k, with epsilon 1e-2, batch 1,024, 10k iterations and AdamW 1e-4. The exact wiring of `vf` / `flow` / `data_match_fn` follows the ott 0.5.2 GENOT tutorial and is UNVERIFIED here.
- **UOT-ICNN.** `TemporalNeuralProblem(g).prepare(time_key="day", joint_attr="X_pca").solve(tau_a=0.85, tau_b=0.85)`, with the remaining code defaults. Check first that rescaling networks exist.
- **CellFlow** (py3.11 environment). Add a dummy perturbation covariate, then `CellFlow(adata, solver="otfm")` on `centered_pca(n_comps=100)` of log1p CP10k, with `match_linear(epsilon=0.5, tau_a=tau_b=1.0)`. The iterations must be cut from 500k, and this is a stated deviation from the authors.
- **MIOFlow, JKOnet\*, TIGON, PRESCIENT, TrajectoryNet.** Use each one's preprocessing from table D, with source = t0 and target = t1.

How to read the outputs:

- **Couplings** come from WOT, moscot (x3), CINEMA-OT, SCOT v1/v2, Pamona and uniPort. Score them as comparison A.
- **"Unmatched" signals** exist only in these:
  - WOT: row growth
  - moscot: row / column mass when tau < 1
  - CINEMA-OTW: dropped cells
  - Pamona: virtual-cell mass
  - SCOT v2: marginal shortfall

  None of them has a built-in decision rule.
- **Maps and generators** come from CellOT, OT-CFM / SB-CFM, GENOT, UOT-ICNN, CellFlow and the section-D methods. Score them only on distribution-level metrics and on depth leakage (LITERATURE.md group B).
