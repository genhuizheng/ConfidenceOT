# Lineage-aware OT baselines: compatibility with the KP-Tracer / Quinn primary -> metastasis benchmark

**Status: 2026-10-31.** Desk audit. I read the code and docs but ran nothing: no package was installed and no downloaded code was executed. Versions and commits read:

- LineageOT: `aforr/LineageOT` HEAD `6081b402074f7e5934e729e81669aef430219da8` (2023-07-28). `setup.py` says 0.2.0, and PyPI `lineageot` 0.2.0 was uploaded 2022-01-12.
- CloneOT: `poseidonchan/CloneOT` HEAD `cbb02d947ef4c1a6a326979d1fa17e7a338a4de9` (2026-10-12). `pyproject.toml` says 0.1.0, and the package is not on PyPI. Paper: arXiv 2610.16093v1 (submitted 2026-10-12).
- moscot: `theislab/moscot` HEAD `d6e1bb10398cb9ad1e8577552439a6c571a4903a` (2026-10-14). PyPI `moscot` 0.5.2 was uploaded 2026-08-25, and conda-forge also has 0.5.2. PyPI `ott-jax` is at 0.6.0.
- moslin: `theislab/moslin` HEAD `484e1416a29d437a9690ac6ff2fffcec613da8a5` (2024-10-05). This is a reproducibility repo plus the `moslin_utils` helper. The method itself is `moscot.problems.time.LineageProblem`.

PyPI and conda versions come from `https://pypi.org/pypi/<name>/json` and `https://api.anaconda.org/package/conda-forge/moscot`, queried 2026-10-31.

**URL prefixes** (all line numbers refer to these commits):

- `[L]` = `https://github.com/aforr/LineageOT/blob/6081b402074f7e5934e729e81669aef430219da8/`
- `[C]` = `https://github.com/poseidonchan/CloneOT/blob/cbb02d947ef4c1a6a326979d1fa17e7a338a4de9/`
- `[M]` = `https://github.com/theislab/moscot/blob/d6e1bb10398cb9ad1e8577552439a6c571a4903a/`
- `[S]` = `https://github.com/theislab/moslin/blob/484e1416a29d437a9690ac6ff2fffcec613da8a5/`
- `[P]` = `D:\Xia_lab\CellOT\` (this project)

**A project rule applies to all three methods.** `[P]lineage_validation/VALIDATION_PROTOCOL.md:40` says "Lineage never enters an OT fit, representation, QC, gate, calibration or input file". Every method below needs lineage as an input, so none of them can be a baseline under the frozen protocol. Running them means a separate comparison, labelled as **lineage-informed**: these methods see information that ConfidenceOT never sees.

**What our evaluation levels read** (from the protocol and the subclone rules):

- **L1 (site).** On KP-Tracer, a primary and a metastasis are related iff their MetFamily labels are equal (`VALIDATION_PROTOCOL.md` §5.3, lines 142-147). On Quinn, the site-level quantity is f_X.
- **L2 (subclone).** Subclones are CRISPR-supported branches of the authors' trees, defined from the deposited character matrices (`within_family_subclones/SUBCLONE_RULES.md` §1-4).
- **L3 (tree distance).** L_ij is the root depth of the MRCA of i and j, in edges, after contracting unary nodes. It exists only for cells of the same lineage (`VALIDATION_PROTOCOL.md:160`).
- **What the KP trees look like.** I counted the 10 local trees in `[P]results/kp_tracer_lineage_validation/export/trees/cassiopeia/*.nwk` with a crude Newick counter of my own (`scratchpad/scripts/treestats.py`, not project code). They have no `:` branch lengths and many unary nodes (for example 1,322 of 2,061 internal nodes in `3515_Lkb1_All`). The largest polytomy has 6,519 children and the largest tree has 21,108 leaves (both `3724_NT_All`).

---

## 1. LineageOT

**Identity.** Forrow A. & Schiebinger G., "LineageOT is a unified framework for lineage tracing and trajectory inference", *Nat Commun* 12, 4940 (2021), doi:10.1038/s41467-021-25133-1. Read at https://pmc.ncbi.nlm.nih.gov/articles/PMC8367995/. Code: https://github.com/aforr/LineageOT.

**Lineage input**
- **What it takes.** A lineage tree of the cells at the **later** time point only. Earlier cells are not in the tree and enter only as states (paper, Methods "Ancestor inference with lineage information"). The API is `fit_lineage_coupling(adata, time_1, time_2, lineage_tree_t2, ...)` (`[L]lineageot/core.py#L108`).
- **Tree format.** A networkx DiGraph whose nodes carry `time` and `time_to_parent`. It can be built three ways:
  - `fit_tree` from barcodes by neighbor joining (`core.py:11`); the example calls that step "not optimized" (`[L]examples/pipeline.py`);
  - from clone labels (`method='non-nested clones'` or `'clones'`, `core.py:62-76`);
  - from a Newick file with `read_newick` (`core.py:82`).
- **Branch lengths are needed.** Edge times are read from the Newick branch lengths (`[L]lineageot/inference.py#L1370`). If a leaf time is given, all leaves must have consistent times, otherwise `ValueError("Edge length annotations ... not consistent with a single sampling time")` is raised (`inference.py:1391`). A leaf shallower than `time_1` raises "Tree has no ancestor of cell" (`inference.py:869`).
- **When lineage is needed.** At fit time, for every coupling. There is no trained model to reuse.
- **Without lineage.** The code has no lineage-free mode. The paper's lineage-free counterpart is Waddington-OT, i.e. entropic OT on expression (paper, Results "LineageOT outperforms Waddington-OT"). The paper gives no formal reduction, so what the method degenerates to with an uninformative tree is not verified.

**Assumptions**
- **Time series.** Two time points, earlier to later (t1 < t2). Earlier and later samples are independent draws from separate populations (paper, Methods "Developmental stochastic processes" and "Optimal transport as maximum-likelihood estimate").
- **State model.** Cell state is Brownian along tree edges, with variance proportional to edge time. The ancestor state of each later cell at t1 is the Gaussian posterior mean and variance given all observed leaves of its connected component (`inference.py:811-856`).
- **Cost.** The cost is `||x_i - mu_j(t1)||^2 / sigma_j^2(t1)`. The code is `ot.utils.dist(early, ancestor_means) @ diag(1/ancestor_variance)`, then divided by its median (`core.py:185-188`; paper Eq. 9).
- **Fit to primary -> metastasis.** Poor in principle. Our two sides are sampled at the same time, so primary cells are relatives of metastatic cells, not their ancestors at an earlier time (`VALIDATION_PROTOCOL.md` lines 34-36 say the same of our data).
  - `time_1` has no biological meaning here and has to be invented. It must be strictly earlier than every leaf: at `time_1 = time_2` the ancestor variance is 0 and the cost is divided by zero (my reading of `core.py:185` with `inference.py:858-872`).
- **Tree scope.** LineageOT does **not** require both sides to be leaves of one tree. It uses only the later side's tree, so with KP-Tracer the family tree has to be pruned to its metastatic leaves.

**Marginals, growth, rejection**
- Balanced by default (`balance_reg = np.inf`, `ot.sinkhorn`, `core.py:191-192`). A finite `balance_reg` calls `ot.unbalanced.sinkhorn_unbalanced` (`core.py:193-194`; CHANGELOG 0.2.0 "allows unbalanced transport").
- Marginals are uniform unless `marginal_1` and `marginal_2` (relative growth rates) are passed (`core.py` docstring, lines 138-141).
- No rejection or unmatched-mass output. Only the row and column sums of an unbalanced coupling.

**Preprocessing defaults.** The package does no preprocessing: states come from `adata.X` or `adata.obsm[state_key]` (`core.py:153-158`).
- `epsilon = 0.05` on the median-normalised cost, `normalize_cost = True`, `ot_method = 'sinkhorn'` (`core.py:108`).
- Paper, C. elegans: the first 50 PCs of normalised, log-transformed counts; epsilon chosen from parameter scans (Fig. 3 caption, as read at PMC8367995).

**Output.** A coupling AnnData: early cells in `obs`, late cells in `var`, the matrix in `X` (`core.py:198-201`). `save_coupling_as_tmap` writes it for Waddington-OT downstream (`core.py:209`). It gives no fate probabilities itself.

**Installation and scale**
- `pip install lineageot` (0.2.0, `python_requires >= 3.6`), or a conda env from `conda_requirements.yml`. Dependencies: anndata, cvxopt, ete3, networkx, newick, numpy, POT (`[L]pip_requirements.txt`).
- CPU only. No GPU code.
- Untested with current numpy/POT/newick releases: the last release was 2022 and the repo was last committed 2023. Compatibility is not verified.
- **Scaling.** The cost matrix is dense (n_early x n_late). Ancestor inference calls `np.linalg.inv` on the dense precision matrix of all unobserved nodes of each tree component (`inference.py:839`).
  - For the 21,108-leaf `3724_NT_All` pruned to its metastatic leaves, the dense inverse can be several GB. I did not measure this.
  - The cell limit is not stated in the docs. The paper's C. elegans analysis used 6 time groups of a 7,087-cell sublineage and 5,123 fully annotated cells.

**Leakage**
- **What it reads.** Only the metastatic side's tree. No primary-side lineage and no cross-side lineage enters.
  - **L1: not circular**, provided the metastatic tree comes from a file chosen without the MetFamily label, i.e. the induced subtree of the mouse-level `<mouse>_<genotype>_All` tree.
  - Using `<family>_Fam` means picking the file by the L1 truth label. That leaks no primary identity, but it should be avoided.
- **L2 and L3: partly circular.** The metastasis-side half of both truths enters the cost:
  - metastatic subclone membership, because cells of one branch share inferred ancestors;
  - metastasis-metastasis tree depths.
  - The primary-side half and every cross-side MRCA depth do not enter.
  - A coupling that is homogeneous within metastatic subclones is rewarded by group-based scores without any expression signal. So L2 and L3 results would be "target-side lineage-informed", not independent.
- **A non-circular variant for L2 and L3.** Build the input tree from a disjoint subset of target sites/intBCs, and define the subclones and L3 from the complementary subset.
  - This needs re-inferred trees for both halves, which conflicts with `SUBCLONE_RULES.md` §1 ("authors' trees only, never re-inferred"). It would need a dated rule change.
  - Otherwise, use LineageOT on L1 only.
- `time_1` must be fixed before any scoring. It must not be tuned on L1-L3.

**Verdict: adaptable, with heavy assumptions.** Five adaptations, all fixed a priori:
1. Prune the tree to metastatic leaves.
2. Assign branch lengths and make the tree ultrametric. The trees have none; unit edges plus extended terminal edges is one choice.
3. Choose an artificial `time_1` < leaf time.
4. Treat the primary cells as the "earlier" population. The model's ancestor interpretation is false for same-time sampling.
5. Pass our representation via `state_key`. Squared Euclidean on L2-normalised vectors equals 2 - 2cos, so it is compatible with the cosine arm.

**Risks.**
- The invented time axis.
- No branch lengths.
- Dense inverse on large trees.
- Stale dependencies.
- No rejection. Balanced by default, so mass from KP metastases whose primary is absent is forced onto unrelated primaries.

---

## 2. "CloneOT"

**Identity: a published method with this exact name exists.**

> Chen Y., Chen W., Huang H., "CloneOT: predicting clone-specific perturbation responses with lineage-supervised neural inverse optimal transport", arXiv:2610.16093v1, submitted 2026-10-12 (https://arxiv.org/abs/2610.16093). Code: https://github.com/poseidonchan/CloneOT. Docs: https://poseidonchan.github.io/CloneOT/ (not opened; I read the repo's `docs/` sources). GPL-3.0 (`[C]LICENSE`). Preprint only, not peer reviewed.

- It is a **perturbation-response** method, not a metastasis or trajectory method.
  - The arXiv abstract mentions neither metastasis nor primary tumour.
  - Its 14 real scenarios are Watermelon (PC9 + osimertinib, 12 scenarios), ClonMapper (MDA-MB-231 + doxorubicin) and ReSisTrace (Kuramochi + olaparib) (arXiv HTML v1, Sections 2.4 and 3.2).
- I found this repo with a web search for "CloneOT". If the user meant a different method, the clone-aware OT methods the name could be confused with are listed below. I am not choosing among them.
  - Super-OT (Prasad, Yang & Uhler, arXiv:2007.12098): appeared in search results, not read.
  - CoSpar's clone-aware transition maps: not read, not verified.
  - moscot's `LineageProblem` (section 3).
  - LineageOT's clone-label tree option (`fit_tree(method='non-nested clones')`).

**Lineage input**
- **What it takes.** **Flat clone labels** on both sides, for a subset of cells: an `obs` column, NaN for unlabeled cells (`[C]README.md`, "Quick start"; `[C]cloneot/api.py#L44-L66`).
  - No tree is used. The paper describes only flat clone labels with a learned clone embedding (arXiv HTML Section 2.2).
  - Clones seen on one side only are treated as unlabeled (`[C]cloneot/model.py#L82-L84`).
- **Training only.** The labels are used to learn the cost.
  - The learned cost minimises the excess cost OT^A - OT of the clone-constrained over the unconstrained entropic OT (`[C]docs/method.md`).
  - The prediction map T(x) = x + F(x) needs expression only. The repo says it "uses lineage only for training" (README, Highlights).
  - The unconstrained coupling over training cells can be read with `CloneOT.conditional()` (`model.py:387`).
- **Without lineage.** "Without clone labels, CloneOT is entropic optimal transport with the squared Euclidean cost, the solver underlying CellOT" (README). In code, the cost is learned only if `cost == 'neural'` and at least one clone is labelled (`model.py`, `fit`, `learn_cost = ...`).

**Assumptions**
- **Unpaired before/after populations.** "a target cell of clone k may only descend from source cells of clone k" (`docs/method.md`). The Watermelon tutorial is a time series, day 0 to day 3 (`[C]docs/tutorials/watermelon.ipynb`).
- **No tree.** Both sides do not have to be leaves of one tree. Shared clone labels across the two sides are the requirement.
- **Fit to primary -> metastasis.** The data structure fits: source = primary, target = metastasis, clone = KP MetFamily or Quinn LineageGroup. The descent semantics does not: our cells are relatives.

**Marginals, growth, rejection**
- **Balanced** entropic OT (arXiv Section 2.2). The target is uniform: `g.mean()` in `_semidual`, `model.py:157-164`.
- **Source weights come from the clone labels.** Each labelled clone's source mass is set to its share of target cells, split equally within the clone (`model.py:223-243`, line 238). These weights are used in the unconstrained problem as well.
- If any target cell is unlabeled and no source cell is, it raises `ValueError` (`model.py:235`).
- No growth model, no rejection, no unmatched mass.

**Preprocessing defaults** (the defaults are "the settings of the paper", `model.py:35-37`):
- **Genes.** 2,000 HVGs: seurat_v3 on counts, with the condition as batch (`api.py:69, 91-94`).
- **Normalisation.** log1p of CP10k (`api.py:107`).
- **VAE.** Hidden [256, 128], latent 50, KL weight 1e-4, 25,000 steps (`[C]cloneot/vae.py#L19-L29`).
- **Entropic temperature.** `eps = 0.1 x m`, where m is the median of 1/2|y-x|^2 over random pairs (`model.py:38, 218-219`).
- **Cost.** Neural cost with ICNN h and neural ODE flows (`model.py:47-55`).
- **Schedule.** 500 warmup, 2,000 cost updates, 2,000 polish steps, 3,000 map steps (`model.py:62-68`).
- **Evaluation genes.** 50 response genes (`api.py:121`).

**Output.**
- Predicted expression of source cells (`CloneOTModel.predict`, `api.py:200`).
- The conditional coupling pi(y_j | x_i) over training target cells (`model.py:387-390`).
- The barycentric projection (`model.py:392`) and the learned cost matrix (`model.py:399`).
- Clone-level metrics (`cloneot/metrics.py`).

**Installation and scale**
- Not on PyPI. Install with `pip install git+https://github.com/poseidonchan/CloneOT.git` after PyTorch (`[C]docs/installation.md`).
- Python >= 3.10. Dependencies: torch >= 2.0, scanpy, scikit-misc and others (`[C]pyproject.toml`).
- "CloneOT runs on a CPU, but a GPU makes training much faster" (`docs/installation.md`). The paper reports a median 1.6 min per model on one RTX 5090 (arXiv Section 2.3). The tutorial says "A GPU is recommended. With about 20,000 cells, each model trains in several minutes."
- CPU runtime: not verified.
- Memory: the full n_source x n_target learned-cost matrix is held in memory (`model.py:141-147`).

**Leakage**
- **Seen clones (default use).** The clone labels are the L1 truth: KP site truth is MetFamily equality, and Quinn's clone is the LineageGroup.
  - **L1 is fully circular.** The constrained problem and the source weights are built from those labels.
  - **L2 and L3 are not given directly**, because flat family labels do not encode within-family subclones or MRCA depths. But any score that counts cross-family mass as error is inflated.
  - Only within-family, group-normalised L2 and L3 scores are not circular by construction. Even those are not independent of the training signal.
- **Non-circular variant.** The paper's own unseen-clone (OOD) split: "entire clones are held out" (arXiv Section 2.4; tutorial "Unseen clones").
  - Train on some families or LineageGroups, then score L1, L2 and L3 only on families held out of training. Leave them out entirely, or keep them as unlabeled cells.
  - Fix the split a priori, for example by a seeded family-level split. Do not tune on held-out labels.
- **Feasibility of that split.** On KP-Tracer, each mouse has few families, so the labelled training set per mouse is small. Quinn has more LineageGroups per mouse.

**Verdict: adaptable** as a held-out-clone, lineage-supervised baseline, with four adaptations:
1. Source = primary, target = metastasis.
2. Clone = MetFamily (KP) or LineageGroup (Quinn).
3. A held-out-family split fixed in advance.
4. Either its VAE representation, or ours through the array API `CloneOT().fit(source, target, source_clones, target_clones)` (README, "The solver also works on any arrays").

It is **incompatible** with any seen-clone evaluation at L1.

**Risks.**
- Three-week-old preprint, single version, no peer review.
- Balanced, with no rejection. KP metastases without their primary are forced onto other primaries. With held-out families on both sides, the "unlabeled" block must hold the unmatched mass.
- GPU recommended.
- The perturbation framing is not our biology.
- Few families per KP mouse.

---

## 3. moscot / moslin

**Identity.**
- moscot: Klein D. ... Theis F.J., "Mapping cells through time and space with moscot", *Nature* 638, 1065-1075 (2025), doi:10.1038/s41586-024-08453-2 (Europe PMC record; PMC11864987). Code: https://github.com/theislab/moscot.
- moslin: Lange M., Piran Z., Klein M., Spanjaard B., Klein D., Junker J.P., Theis F.J., Nitzan M., "Mapping lineage-traced cells across time points with moslin", *Genome Biology* 25, 277 (2024), doi:10.1186/s13059-024-03422-4. Read at https://pmc.ncbi.nlm.nih.gov/articles/PMC11492637/; I read only the first ~100k characters of the page.
- moslin is implemented as `moscot.problems.time.LineageProblem`. The class docstring cites `lange-moslin:23` (`[M]src/moscot/problems/time/_lineage.py#L282-L291`, and `[S]README.rst`).

**Lineage input**
- **Within-time-point lineage cost matrices**, one per side, as the quadratic (GW) term:
  - precomputed in `adata.obsp['cost_matrices']` (the default: `_lineage.py:397-421`);
  - `BarcodeDistance`: scaled Hamming on `obsm` barcodes, a dense O(n^2) Python loop (`[M]src/moscot/costs/_costs.py#L16-L60`, L131-143);
  - `LeafDistance`: shortest path on a networkx tree in `uns`, with edge attribute `weight` (`_costs.py:63-128`). Leaves are nodes of degree 1 and must match `obs_names`, or be given through `cell_to_leaf` (`_costs.py:117-124`).
- **Lineage is not compared across sides.** The paper says lineage "is not directly comparable across time points" (moslin paper, as read; also the moscot tutorial `100_lineage`).
  - In the zebrafish script only the within-time blocks of `obsp['cost_matrices']` are filled (`[S]analysis/hu_zebrafish_linnaeus/hu_moslin_couplings_transient_fibro_prob.py#L381-L389`).
  - No shared tree is needed. Early and late cells are typically different individuals.
- **When lineage is needed.** At fit time, for every coupling.
- **Without lineage.** `alpha -> 0` is the pure linear, expression-only problem (`_lineage.py:471-474`). The zebrafish script switches to `TemporalProblem` at `alpha == 0` (`hu_moslin_...py:393-405`).

**Assumptions**
- **Time series.** `TemporalProblem` and `LineageProblem` are for "time-series single cell data" (`_lineage.py:29, 283`). Subproblems are built from a time key with policy `sequential`, `triu`, `tril` or `explicit` (`_lineage.py:298, 345-352`).
- **Lineage concordance** across time: related early cells map to related late cells (moslin paper, as read).
- Only one replicate per time point is supported, and FGW is non-convex (paper limitations, as read).
- **Fit to primary -> metastasis.** Mechanically possible: code primary = 0 and metastasis = 1, or use the time-free `moscot.problems.generic.FGWProblem` (`[M]src/moscot/problems/generic/_generic.py#L496`).
  - Growth-rate marginals (`BirthDeathMixin`) do not apply to same-time sampling. Use uniform `a` and `b`, which is the default unless proliferation/apoptosis keys are set (`_lineage.py:152-153`).
- **Tree scope.** Both sides need **not** be leaves of one tree. When they are (KP `_Fam`, Quinn LineageGroup trees), moslin throws away the cross-side part of the tree. It uses only the within-primary and within-metastasis distance submatrices.
  - With a pooled multi-family primary side (our D_A design), primaries from different family trees have no defined distance. A constant has to be chosen a priori.

**Marginals, growth, rejection**
- Balanced by default (`tau_a = tau_b = 1`, `_lineage.py:447-448`). `tau < 1` gives KL-unbalanced marginals (`_lineage.py:203-208`).
- Marginals are uniform unless given or estimated from proliferation/apoptosis scores (`_lineage.py:366-381`; `[M]src/moscot/base/problems/birth_death.py#L128-L142`).
- No rejection output. The unbalanced row and column sums could be scored with the same unmatchedness rule as our Vanilla UOT.

**Preprocessing defaults**
- If `joint_attr = None`, moscot runs a local PCA with `n_comps = 30` and `scale = False` on `adata.X` of the two sides (`[M]src/moscot/problems/_utils.py#L13-L18`; `[M]src/moscot/base/problems/problem.py#L657-L700`).
- Linear cost `sq_euclidean` (`_lineage.py:299`).
- **API solve defaults** (`_lineage.py:443-458`): `alpha = 0.5`, `epsilon = 1e-3`, `tau_a = tau_b = 1`, `rank = -1` (full rank), `scale_cost = 'mean'`, `threshold = 1e-3`.
- **Tutorial values** (https://moscot.readthedocs.io/en/latest/notebooks/tutorials/100_lineage.html): `normalize_total`, `log1p`, scanpy `pca` (components not stated), `joint_attr = 'X_pca'`, precomputed barcode cost matrices, and `solve(alpha = 0.99, epsilon = 1e-3, tau_a = 0.99)`.
- **moslin paper.** No single recommended default (as read).
  - C. elegans: 3k Seurat HVGs and 50 PCs. alpha and epsilon were grid-searched against ground truth (`[S]analysis/packer_c_elegans/run_gridsearch/moslin.yaml`: alpha 0 to 1, epsilon 1e-3 to 0.5, `tau_a = 1`, `scale_cost = mean`, 30k inner iterations).
  - Zebrafish: scVI latent space; script defaults `alpha = 0.5`, `epsilon = 0.05`, `tau_a = 0.9`, `tau_b = 1.0` (`hu_moslin_...py:623-627`), tuned by a proxy "cell type persistency".

**Output.**
- A coupling per subproblem: `problem.solutions[(t0, t1)]`.
- `push` and `pull` (descendant/ancestor distributions), `cell_transition` and `sankey` (tutorial `100_lineage`).
- Fate probabilities through CellRank 2 downstream (moslin paper, as read).

**Installation and scale**
- `pip install moscot` (0.5.2) or conda-forge `moscot` 0.5.2. Python >= 3.11. Dependencies: jax >= 0.6.1, ott-jax, flax, optax, diffrax (`[M]pyproject.toml`).
- CPU works through JAX's CPU backend, and GPU is optional (README: "GPU acceleration"). JAX/ott-jax wheels on TACC's ARM (Grace) nodes: not verified.
- `moslin` itself is not a pip package. `moslin_utils` (`[S]pyproject.toml`) is only an analysis helper.
- **Scaling.**
  - The linear (TemporalProblem) problem scales to 1.7 M cells (moscot paper abstract).
  - The lineage GW term needs dense n x n cost matrices per side, plus low-rank `rank > 0` or `batch_size` options (`_lineage.py:449-451`).
  - `LeafDistance` runs one Python Dijkstra per leaf into a dense matrix (`_costs.py:101-115`). A side with as many cells as `3724_NT_All` has leaves (21,108) would need about 3.6 GB for that matrix in float64.
  - GW runtime on CPU at these sizes: not verified.

**Leakage**
- **What it reads.** Within-primary and within-metastasis lineage distances. No cross-side lineage.
- **L1.** Not circular in the strict sense: no cross-side identity enters.
  - But with a pooled multi-family primary side, the source matrix encodes which primary cells share a family. GW matches block structure. Report L1 as lineage-informed.
- **L2 and L3: partly circular, more than LineageOT.** Both sides' halves of the truth enter: within-side subclone block structure and within-side path distances.
  - On a unit-edge tree, path distance = depth(i) + depth(j) - 2 depth(MRCA), so the within-side inputs are the same quantity L3 measures, restricted to same-side pairs.
  - The GW term rewards couplings that preserve that structure across sides.
- **Non-circular variant for L2 and L3.** Compute the input distances from a disjoint subset of target sites/intBCs (for example `BarcodeDistance` on half the characters), and the subclones/L3 from the other half.
  - This needs re-inferred evaluation trees, which conflicts with `SUBCLONE_RULES.md` §1, as for LineageOT.
  - Otherwise, use moslin on L1 only.
- **Hyperparameters.** The paper tuned alpha and epsilon against ground truth. We must not do that: fix the API defaults or the tutorial values before scoring.

**Verdict: adaptable** as a time-free FGW with within-side lineage distances, with these adaptations, all fixed a priori:
1. Code the two sides as 0/1, or use `FGWProblem`.
2. Uniform marginals.
3. Per-side distances from induced subtrees with unit edge weights. The trees have no lengths. `LeafDistance` uses `weight`; whether networkx treats a missing weight as 1 is not verified here.
4. A constant for cross-family pairs within one side.
5. Our representation through `joint_attr`.

**Risks.**
- O(n^2) dense lineage matrices for the largest trees.
- FGW non-convexity.
- Hyperparameters chosen by grid search in the paper, with no default intended for our setting.
- JAX on ARM is unknown.
- No rejection.

---

## Summary table

| Method | Lineage input | Needed at inference | Same-time primary -> metastasis | Non-circular levels as run | Non-circular variant | Install on CPU cluster | Verdict |
|---|---|---|---|---|---|---|---|
| LineageOT | metastasis-side tree with branch times | yes (every fit) | no time axis: `time_1` must be invented; ancestor model false | L1 only (L2/L3 see metastasis-side structure) | disjoint-character trees (needs rule change) | pip 0.2.0, CPU only, stale dependencies | adaptable, heavy assumptions |
| CloneOT (arXiv 2610.16093) | flat clone labels, both sides | no (training only) | data structure fits, descent semantics does not | none with seen clones (L1 circular; L2/L3 inflated) | held-out families/clones, then L1-L3 on held-out only | git install, torch; CPU possible, GPU recommended; CPU runtime not verified | adaptable (held-out clones only) |
| moscot LineageProblem / moslin | within-side lineage distance matrices | yes (every fit) | mechanically yes (0/1 key or FGWProblem); growth priors not applicable | L1 strictly (lineage-informed); L2/L3 partly circular | disjoint-character distances (needs rule change) | pip/conda 0.5.2, JAX CPU; ARM wheels not verified | adaptable |

Our own Balanced OT and Vanilla UOT defaults are in `DEFAULT_SETTINGS.tsv`, for reference.
