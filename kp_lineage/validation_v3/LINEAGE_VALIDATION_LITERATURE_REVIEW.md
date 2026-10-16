# Lineage-based validation of ConfidenceOT on KP-Tracer: literature review of candidate metrics

Compiled 2026-10-15. Every citation below was checked against Crossref or a resolving URL. The reference list says what was actually read for each one. Formulas from sources not read in full are marked as coming from package documentation or code, or as unverified.

## 1. Scope

**What we validate.** ConfidenceOT works on expression only and produces two outputs:

- (a) a gate that rejects some primary-tumour cells, the source set S, as having no counterpart in the metastasis, the target set K;
- (b) a coupling T in R^{|S| x |K|}, with marginals a over sources and b over targets.

**Independent evidence.** KP-Tracer (Yang et al. 2022 [1]) provides CRISPR-Cas9 lineage evidence:

- Cassiopeia trees for each tumour and each tumour-metastasis family, in newick. They have no branch lengths, many polytomies and many unary nodes.
- Neighbor-joining (NJ) trees.
- Character matrices (0 = unedited; -1 or '-' = missing).
- Indel priors.
- MetFamily and Tumor labels.

**Scope of the review.** For each candidate metric family, the review gives the source, the formula as the source uses it, and whether applying it to KP-Tracer source/target cells is a direct use or our adaptation. It then recommends a small primary set.

**Notation used throughout.**

- d(i,j): a leaf-to-leaf distance on the tree.
- LCA(i,j): the most recent common ancestor of i and j.
- Leaf labels g(i): the gate label, retained or rejected, for i in S.

**Three facts about the KP-Tracer methods that shape everything below.** All are from the STAR Methods of [1], read in full.

1. Each MetFamily tree was reconstructed jointly over all primary and metastatic cells: "we first reconstructed trees encompassing all cells in the primary and metastatic tumors". Source and target cells therefore sit on one tree, and between-set tree distances are defined directly.
2. Every cell in a MetFamily descends from one transformed founder. Clone identity therefore carries no information inside a family; only subclonal structure does.
3. The terms "TreeMetRate" and "phylogenetic metastatic score" do not appear in Yang et al. 2022. TreeMetRate comes from Quinn et al. 2021 [4]. The metastasis analyses in Yang et al. use two quantities instead:
   - Evolutionary Coupling;
   - a "modified phylogenetic distance". This is the sum of branch lengths, where each branch length is the number of mutations on the edge, inferred by Camin-Sokal parsimony. It was averaged from each metastatic cell to all primary cells, stratified by expansion, and tested with a one-sided Mann-Whitney U test.

   [1] defines a tumour-metastasis family only informally: "a tumor containing both a primary tumor and a large enough metastatic population". We found no numeric threshold.

## 2. Metric families

Columns:

1. Name
2. Formula
3. Original publication
4. DOI or URL
5. Original use
6. KP-Tracer use: direct or adaptation
7. Assumptions
8. Limitations on unresolved trees without branch lengths
9. Implementation
10. Recommendation

Bracketed numbers point to the reference list.

### 2.1 Tree topology: distances and relatedness

| 1 Name | 2 Formula | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits (polytomies, no BL) | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| MRCA depth (shared-ancestry depth) | depth(LCA(i,j)) = edges from root to LCA(i,j). LineageOT uses the MRCA *time*: "A complete lineage tree T ... encodes the time t_{j1,j2} of the most recent common ancestor of each pair" | No single originating paper. Used by Forrow & Schiebinger 2021 [2], and by Jones 2020 [3] to stratify triplets by LCA depth | 10.1038/s41467-021-25133-1; 10.1186/s13059-020-02000-8 | LineageOT: builds the Gaussian model of ancestor states. Cassiopeia: depth stratification in tree evaluation | Adaptation. Pairs (i in S, j in K) are scored by how deep they coalesce | Depth tracks time since divergence | Without branch lengths, depth = edge count, which unary nodes inflate and polytomies compress. All cells under one polytomy tie | Cassiopeia `CassiopeiaTree.find_lca`, `get_time`; ete3 `get_common_ancestor` | Sensitivity. Only after collapsing unifurcations (`collapse_unifurcations`) |
| Normalised MRCA relatedness | e.g. depth(LCA)/mean(depth(i), depth(j)) | No established source found; our proposal | n/a | n/a | Adaptation | Leaf depths are comparable | Same as above, plus ratio instability for shallow leaves | Custom | Not recommended. Unsourced, and no better than raw depth |
| Topological patristic distance | d(i,j) = number of edges on the path i-j | Path length in a graph; no single original paper. Lange 2024 (moslin) uses the "shortest path distance along reconstructed trees" built by Cassiopeia or LINNAEUS [6] | 10.1186/s13059-024-03422-4 | moslin: lineage cost matrices C_{t1}, C_{t2} in a Gromov-Wasserstein term | Direct as a distance; adaptation as a validation target | Edges stand for comparable amounts of divergence | Unary nodes add spurious length. A polytomy gives d = 2 between any two children, however divergent they are | Cassiopeia `get_distance`, `get_distances`, `compute_phylogenetic_weight_matrix`; ete3 `get_distance(topology_only=True)` | Primary, as one of two tree metrics (after collapsing unifurcations) |
| Mutation-count ("modified phylogenetic") distance | d(i,j) = sum over path edges of the number of mutations on the edge, from Camin-Sokal (irreversible) ancestral reconstruction [1] | Yang et al. 2022 [1] | 10.1016/j.cell.2022.04.015 | Mean distance from each metastatic cell to all primary cells, stratified by expansion; one-sided Mann-Whitney U | Direct metric, applied to the source side | Irreversible edits; correct ancestral states | Reconstruction of ancestral states depends on the tree. Missing data hide mutations | Cassiopeia `reconstruct_ancestral_characters`, `get_mutations_along_edge`, then set the edge lengths | Primary (the second tree metric). This is the authors' own distance |
| Evolutionary Coupling / mean between-group distance | D(M,K) = (1/(n_M n_K)) sum_{m in M, k in K} d(m,k); z-score against 2000 shuffles of leaf labels; d = "number of mutated edges" [1] | Yang 2022 [1]; related to the net relatedness index (Webb et al. 2002 [19]) | 10.1016/j.cell.2022.04.015; 10.1146/annurev.ecolsys.33.010802.150448 | Phylogenetic proximity between transcriptional states, and between primary pieces and metastases (3724_NT_T1) | Direct, with M = retained or rejected sources and K = targets | Labels are exchangeable under the null | Mean distance is dominated by deep splits. Polytomies create ties | Cassiopeia `compute_evolutionary_coupling`, `data.utilities.net_relatedness_index` | Sensitivity. Group-level, and redundant with the nearest-relative distance below |
| Nearest-relative (minimum) distance to the target set | m(i) = min_{k in K} d(i,k), for each i in S | No source defines it as named; our adaptation. Closest established analyses: the per-metastatic-cell mean distance to primary cells in [1], and the closest-relative statistics in [1,4] (section 2.2) | n/a | n/a | Adaptation. Compare m(i) between rejected and retained sources | A source cell with a metastatic counterpart has a close metastatic relative | Ties at polytomies (fix with ranks). Edge counts are coarse | Custom on Cassiopeia `get_distances` | Primary for (a) |
| Cophenetic correlation | r = Pearson(cophenetic tree distances, original dissimilarities) over all pairs | Sokal & Rohlf 1962 [9] (not read; bibliographic record verified). Cassiopeia defines it as "the Pearson correlation between the phylogenetic distance and dissimilarity between characters" [5] | 10.2307/1217208 | Comparing a dendrogram with its input dissimilarity | Direct, but only as a tree-QC statistic (tree vs character matrix) | Pairs are not independent, so r has no simple p-value | Ties from polytomies lower r | Cassiopeia `tl.compute_cophenetic_correlation`; scipy `cluster.hierarchy.cophenet` | Sensitivity, as a QC covariate per MetFamily. It does not validate ConfidenceOT |

### 2.2 Clustering of gate labels on the tree (trait-clade association)

| 1 Name | 2 Formula | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Fitch-Hartigan parsimony score, normalised | PS = minimum number of label changes needed to explain the leaf labels. Normalised by the number of edges: EffectivePlasticity [1] and TreeMetRate ("minimal number of metastatic transitions ... divided by ... edges in the tree") [4] | Fitch 1971 [10]; Hartigan 1973 [11] (multistate, multifurcating); Slatkin & Maddison 1989 [12] (migrations counted as parsimony changes); used as the PS statistic in BaTS [14] | 10.2307/2412116; 10.2307/2529676; 10.1093/genetics/123.3.603 | Ancestral character fitting; gene flow; tissue transitions and plasticity in [1,4]; viral trait-phylogeny association [14] | Direct statistic. Labels are adapted: g in {retained, rejected} on the source leaves of the pruned family tree | Unweighted state changes; the tree topology is correct | Handles polytomies: Cassiopeia's Hartigan bottom-up step takes the most frequent child states. Normalising by edges depends on whether unary nodes were collapsed | Cassiopeia `tl.fitch_hartigan`, `tl.score_small_parsimony`, `tl.fitch_count`; phangorn `parsimony` | Primary for "is rejection clade-structured". Use a label-permutation null |
| Single-cell version (scEffectivePlasticity, scMetRate) | Mean of the normalised score over every subtree on the root-to-leaf path [1,4] | Yang 2022 [1]; Quinn 2021 [4] | as above; 10.1126/science.abc1944 | Per-cell plasticity; per-cell metastatic rate regressed on expression [4] | Adaptation | as above | Neighbouring cells share most subtrees, so values are strongly autocorrelated | Custom | Not recommended. It cannot be tested independently cell by cell |
| Association index (AI) | Per node, d = (1-f)/2^{n-1}, with n = leaves below the node and f = frequency of the most common label; summed over nodes. AI = ratio to the value under label reassignment (10 shuffles, 100 bootstrap trees) [13] | Wang et al. 2001 J Virol [13] | 10.1128/JVI.75.23.11686-11699.2001 | Compartmentalisation of HIV-1 by tissue on NJ trees | Adaptation (labels) | Bifurcating nodes ("each bifurcating node") | The 2^{n-1} weight assumes bifurcation and is ill-defined at large polytomies | BaTS [14] (standalone) | Sensitivity only |
| BaTS: PS, AI and maximum monophyletic clade (MC) over posterior trees | Statistics averaged over a posterior sample of trees, with a null from tip-label randomisation [14] | Parker, Rambaut & Pybus 2008 [14] | 10.1016/j.meegid.2007.08.001 | Viral phenotype-phylogeny correlation, "account[ing] for uncertainty arising from phylogenetic error" | Adaptation. A set of bootstrap trees would stand in for the posterior | A tree sample exists | MC is very sensitive to polytomies. Formulas were not read (abstract only) | BaTS | Sensitivity. Use the principle (integrate over trees), not the software |
| Fritz-Purvis D | D = (Σd_obs - mean Σd_Brownian) / (mean Σd_random - mean Σd_Brownian), where Σd sums sister-clade differences in a binary trait. D = 1 means random; D = 0 means a Brownian threshold model (caper documentation [15]) | Fritz & Purvis 2010 [15] | 10.1111/j.1523-1739.2010.01455.x | Phylogenetic signal in extinction-risk status | Adaptation | Brownian latent liability; branch lengths; robust above about 50 tips (abstract) | The Brownian null needs meaningful branch lengths. Behaviour at polytomies was not verified | R caper `phylo.d` | Not recommended. The null depends on branch lengths we do not have |
| Pagel's lambda; Blomberg's K | lambda rescales the internal branches of the Brownian covariance [16]; K = observed/expected ratio of the mean squared error under Brownian motion [17] | Pagel 1999 [16]; Blomberg, Garland & Ives 2003 [17] | 10.1038/44766; 10.1111/j.0014-3820.2003.tb00285.x | Phylogenetic signal in continuous traits | Adaptation (continuous per-cell retained mass) | Brownian motion; branch lengths | Both are undefined or arbitrary with unit lengths, and assume a dichotomous tree | phytools `phylosig`; phylosignal | Not recommended |
| Phylogenetic Moran's I | I = z^T W_n z, with z standardised, W_ij = 1/d(i,j), W_n = W/ΣW (Cassiopeia docstring) | Gittleman & Kot 1990 [18] (not read). The Cassiopeia docstring cites Wartenberg 1985 (unverified) | 10.2307/2992183 | Phylogenetic autocorrelation of traits | Adaptation, on the continuous per-source retained mass or score | The weight function is chosen; requires a permutation test | Inverse edge-count weights tie at polytomies. Leaves at distance 0 (after collapsing) must be handled | Cassiopeia `tl.compute_morans_i` (no p-value; permute yourself); ape `Moran.I`; phylosignal | Sensitivity, for the continuous score |
| Nearest-relative label concordance | a(i) = (1/abs(K_i)) Σ_{k in K_i} 1[label(k) = label(i)], with K_i = the closest relatives by modified edit distance; ties share a vote [1]. Variants: Yang's CNV "nearest neighbor probability" on the Cassiopeia tree with a permutation test [1]; Quinn's AlleleMetRate, "the probability that a cell's closest relative by allele state is in a different tissue" [4] | Yang 2022 [1]; Quinn 2021 [4] | 10.1016/j.cell.2022.04.015; 10.1126/science.abc1944 | Tree-agnostic plasticity; agreement between CNV clusters and the tree; metastatic rate | Direct statistic, adapted labels (gate label among source cells; or source vs target among closest relatives) | The closest relatives are informative | Ties at polytomies are handled by fractional votes. Each version can be computed with the tree or from alleles alone | Custom on Cassiopeia `compute_dissimilarity_map` | Primary. Its allele-only version does not depend on tree reconstruction |
| Hotspot local autocorrelation | Moran-like local autocorrelation on a cell-cell similarity graph; the graph can be "clonal relatedness in a cell lineage tree" [20] | DeTomaso & Yosef 2021 [20] | 10.1016/j.cels.2021.04.005 | Heritable gene modules (used on CP007 in [4]) | Adaptation (the gate score as the feature) | kNN graph from the tree | Inherits the ties | `hotspot` (Python) | Sensitivity |

### 2.3 CRISPR edits: priors, dissimilarities, missing data, clones

| 1 Name | 2 Formula | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Indel prior | q_s = f(s)/abs(I), with f(s) = number of intBCs that carry s in at least one cell and abs(I) = number of intBCs [3]. Weight w = -log q (default `negative_log`; `inverse` and `square_root_inverse` also exist) [5] | Jones 2020 [3] | 10.1186/s13059-020-02000-8 | Choosing the greedy split argmin p_i(s0,s_j)^{f(n_ij)} with f(n) = n; addressing homoplasy | Direct. KP-Tracer ships the priors used for its trees [1] | Indel formation is independent of intBC | None (does not depend on the tree) | Cassiopeia `solver_utilities.transform_priors` | Use the priors as given |
| Weighted Hamming dissimilarity | Over characters where neither cell is missing: add 0 if equal; w(non-zero state) if one state is 0; w(a)+w(b) if both are non-zero and differ (unweighted: 1 and 2). Divide by the number of shared non-missing characters [5] | Cassiopeia software [5]. Related scorings: the allelic distance in [3] (1 if one state is 0 or missing, 2 otherwise, /2M) and Yang's modified edit distance h' (2, 1, 0) [1] | https://github.com/YosefLab/Cassiopeia | Dissimilarity for NJ/UPGMA solvers; allelic coupling and plasticity [1] | Direct, as a tree-free relatedness between source and target cells | Irreversible edits; missing entries carry no information | None (tree-free). But characters within an intBC are linked | `solver.dissimilarity_functions.weighted_hamming_distance`; `CassiopeiaTree.compute_dissimilarity_map` | Primary, as the tree-free twin of each tree metric |
| Shared informative edits | Σ over characters where both cells carry the same non-zero state (optionally weighted); missing and 0 are ignored [5] | Cassiopeia `hamming_similarity_without_missing` [5]. The idea of shared rare indels as evidence of ancestry: Jones 2020 [3] | as above | Similarity for solvers | Adaptation, as a "relatedness evidence" count per (i,j) | Rare shared indels are rarely recurrent | Not normalised, so it scales with coverage | Cassiopeia (as named) | Sensitivity |
| Missing-data handling | Weighted Hamming skips missing characters [5]. Greedy assigns cells with missing values by the average shared mutated states; ILP imputes "considering all possible indels" [3]. Yang removes characters with more than 80% missing and ignores missing data in the compatibility graph [1] | [1,3,5] | as above | Reconstruction | Direct | Missing at random | Dropout is often whole-intBC, so it is correlated within cassettes. moslin simulates stochastic silencing for this reason [6] | as above | Report per-cell coverage; stratify metrics by coverage |
| Homoplasy (recurrent indels) | Priors down-weight common indels [3]. Star-homoplasy model [39]. CoSpar robustness to "barcode homoplasy" [7] | Jones 2020 [3]; Sashittal 2023 [39] (not read) | 10.1016/j.cels.2023.11.005 | Reconstruction under recurrence | Direct caveat | n/a | A shared high-prior indel is weak evidence | Startle (separate software) | Weight shared edits by -log q |
| Clone / clonal-group assignment | Non-overlapping intBC sets (Cassiopeia `call-lineages`) plus MULTI-seq and Lenti-Cre-BC filtering; consensus intBC sets per mESC clone [1] | Yang 2022 [1] | 10.1016/j.cell.2022.04.015 | Defining tumours and MetFamilies | Direct (the MetFamily labels) | n/a | Inside a MetFamily every cell is one clone, so the clone label is constant | n/a | Use only as a stratum (permute within it), never as a validation label |

### 2.4 Evaluating couplings against lineage

| 1 Name | 2 Formula | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| LineageOT descendant and ancestor errors | L_D(γ) = Σ_i W_2²(γ*_{i,:}, γ_{i,:}); L_A(γ) = Σ_j W_2²(γ*_{:,j}, γ_{:,j}), with W_2 taken in expression space. Ground truth: γ*_ij = (abs({x_i}) n_{d,i})^{-1} if i is an ancestor of j, else 0 [2] | Forrow & Schiebinger 2021 [2] | 10.1038/s41467-021-25133-1 | Comparing fitted couplings with the known C. elegans ancestor-descendant coupling and with simulations | Adaptation. KP-Tracer has no ancestor-descendant ground truth: primary and metastatic cells are contemporaneous relatives, not ancestors and descendants | A true coupling exists | Without γ*, it needs a lineage-implied surrogate, which is circular | `lineageot` (github.com/aforr/LineageOT) | Not recommended in its original form. See the tree-cost adaptation below |
| How lineage enters LineageOT (for context) | Gaussian graphical model on the fitted tree, with precision Λ_uv = 1/(D abs(t_u - t_v)) on edges. Ancestor means μ_U = -Λ_UU^{-1} Λ_UO (y_O - y_0 1). Cost c(i,j) = abs(x_i - μ_j)²/σ_j². NJ on barcode Hamming distances, made ultrametric by a QP [2] | [2] | as above | Coupling estimation | n/a (ConfidenceOT does not use lineage) | Branch lengths (times) | Needs branch lengths. KP-Tracer trees have none | `lineageot` | Context only. Do not use as a validator, since it builds lineage into the coupling |
| moslin normalised error and top-ancestor accuracy | The LineageOT errors divided by the error of the outer-product coupling a b^T, "such that a value of one corresponds to an uninformative result". For emergent states, check that "the most probable ancestor corresponds to its ancestor in the ground truth tree". Robustness: randomly permute a fraction of the lineage-distance entries [6] | Lange et al. 2024, Genome Biology (not Nat Methods) [6] | 10.1186/s13059-024-03422-4 | Benchmarking moslin, LineageOT and CoSpar on simulations and C. elegans | Adaptation. The normalisation by the independent coupling, the top-1 criterion and the perturbation design carry over directly | Ground truth (moslin); the tree metric (ours) | as above | moscot / moslin | Use the normalisation in primary metric 2 |
| Fate-bias correlation | Pearson correlation between predicted fate bias and fate bias observed from clones [7,8] | Weinreb et al. 2020 [8]; Wang et al. 2022 (CoSpar) [7] | 10.1126/science.aaw3381; 10.1038/s41587-022-01209-1 | Benchmarking fate predictors in LARRY haematopoiesis and reprogramming | Adaptation. Per primary subclade (for example Yang's expansions), compare T-predicted target mass with the observed share of targets whose closest relatives fall in that subclade | Clades are defined before ConfidenceOT runs | Depends on the clade cut; few clades per family | Custom | Sensitivity |
| Retrieval metrics on rows or columns of T | precision@k; mean reciprocal rank (MRR = mean of 1/rank of the first relevant item) [45]; AUROC [42]; AUPRC [43,44]; FOSCTTM (fraction of samples closer than the true match) [40,41] | Voorhees 1999 for MRR [45] (not read); Hanley & McNeil 1982 [42]; Davis & Goadrich 2006 [43]; Saito & Rehmsmeier 2015 [44]; Liu et al. 2019 [40]; SCOT [41] | see references | Ranking; classifier evaluation; alignment of multi-omic cells (SCOT computes FOSCTTM after barycentric projection of the coupling) | Adaptation. Rank sources by T_ij for each target j. "Relevant" = sources at minimum tree (or allele) distance to j | A relevance set can be defined | Polytomies make relevance sets large, which changes the AUPRC baseline. Always report against the permutation baseline | sklearn `roc_auc_score`, `average_precision_score` | Primary (one retrieval metric: MRR or AUPRC with its baseline) |
| Expected tree cost of the coupling | C(T) = Σ_ij T_ij d(i,j)/Σ_ij T_ij; normalised R = (C(T) - W_tree(a,b))/(C(a b^T) - W_tree(a,b)), where W_tree is the earth mover's distance on the tree (section 2.5) | No established source found; our proposal. It combines Evolutionary Coupling [1], the moslin normalisation [6] and Evans & Matsen 2012 [23] | n/a | n/a | Adaptation | d is a tree metric (true for edge counts and mutation counts) | Unary nodes and polytomies change d, so use both distance definitions | Custom: POT `ot.emd2` with the tree-distance matrix, or the Evans-Matsen formula | Primary for (b) |

### 2.5 Tree-aware distributional distances and coupling descriptors

| 1 Name | 2 Formula | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Unweighted UniFrac | "fraction of the branch length of the tree that leads to descendants from either one environment or the other, but not both" | Lozupone & Knight 2005 [21] | 10.1128/AEM.71.12.8228-8235.2005 | Comparing microbial communities | Adaptation (communities = retained sources, rejected sources, targets) | Presence or absence | Every leaf is a distinct "taxon", so the value is almost always maximal | scikit-bio `unweighted_unifrac` | Not recommended |
| Weighted UniFrac | u = Σ_i b_i abs(A_i/A_T - B_i/B_T) over branches, with b_i = branch length; normalised by D = Σ_j d_j (A_j/A_T + B_j/B_T) [22] | Lozupone, Hamady, Kelley & Knight 2007 [22] | 10.1128/AEM.01996-06 | Abundance-weighted community differences | Adaptation | Branch lengths | Unit lengths make it an edge-count earth mover's distance. Polytomies merge mass | scikit-bio `weighted_unifrac` (rooted tree; normalisation optional) | Sensitivity |
| Kantorovich-Rubinstein (earth mover's) distance on a tree | Z(P,Q) = ∫_T abs(P{τ(y)} - Q{τ(y)}) λ(dy), with τ(y) = the subtree below y. "The weighted UniFrac distance ... is just the classical Kantorovich-Rubinstein, or earth mover's, distance". Z_p generalisation; permutation and Gaussian-process nulls [23] | Evans & Matsen 2012 [23] | 10.1111/j.1467-9868.2011.01018.x | Comparing environmental sequence samples | Adaptation. It gives the tree-optimal transport cost W_tree(a,b) for any marginals, in linear time, as the lower bound in primary metric 2 | Path metric with edge lengths | As for weighted UniFrac | guppy (pplacer suite) [23]; one pass over edges by hand | Primary, as the normaliser (not on its own) |
| Jensen-Shannon divergence | JS(P,Q) = H((P+Q)/2) - (H(P)+H(Q))/2 | Lin 1991 [24] (not read) | 10.1109/18.61115 | Divergence measures based on Shannon entropy | Adaptation. Compare clade-composition vectors (predicted vs lineage-implied) | Discrete supports | Ignores tree geometry. Depends on the clade cut | scipy `spatial.distance.jensenshannon` (returns the square root) | Sensitivity, alongside the fate-bias correlation |
| Conditional entropy of T | H(K given S) = -Σ_ij T_ij log(T_ij/a_i) | Textbook definition: Cover & Thomas [25] (not read) | 10.1002/047174882X | n/a | Our adaptation, as a coupling diagnostic | n/a | Not a lineage metric. Entropic blur moves it directly | numpy | Report as a descriptor only. Not recommended as validation |

### 2.6 Null models for couplings and gate labels

| 1 Name | 2 Formula or procedure | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Tip-label permutation | Shuffle labels over leaves and keep the label counts and the tree: "random reassignment ... keeping the relative numbers of sequences ... and the tree structure" [13]; 2000 shuffles [1]; tip randomisation in BaTS [14] | Wang 2001 [13]; Parker 2008 [14]; Yang 2022 [1] | see above | Null for trait-clade association | Direct. Permute gate labels among source cells within each MetFamily | Exchangeable labels | Ignores the fact that expression is heritable (below) | numpy | Primary null for (a) |
| Independence coupling a b^T | Proposition 4.1: P_ε → a ⊗ b = a b^T as ε → ∞ [26]; used by moslin as the "uninformative coupling" normaliser [6] | Peyré & Cuturi 2019 [26]; Lange 2024 [6] | 10.1561/2200000073; 10.1186/s13059-024-03422-4 | The limit of entropic OT; normalising errors | Direct (the same marginals as T) | Marginals fixed | Weak null: almost any structured T beats it | `numpy.outer` | Primary null for (b), as the floor reference |
| Permuted-identity coupling | T_perm = Π_S T Π_K: permute source and/or target identities inside the family, preserving the multiset of values | No established source found; our proposal | n/a | n/a | Adaptation | Exchangeability within a stratum | Breaks the expression structure too | numpy | Primary empirical null for (b), stratified as below |
| Expression-stratified permutation | Permute only within transcriptional clusters (for example Leiden) of the source and target | No established source found; our proposal. Motivated by heritable expression [1,4] | n/a | n/a | Adaptation | Clusters capture the expression confound | Its power depends on cluster granularity | numpy | Primary stringent null. Without it, lineage agreement may only restate that expression is heritable |
| Random fixed-margin tables | Patefield AS 159 [29]; Diaconis-Gangolli Markov chain [30] | Patefield 1981 [29]; Diaconis & Gangolli 1995 [30] (neither read) | 10.2307/2346669; 10.1007/978-1-4612-0801-3_3 | Contingency tables under independence | Adaptation (needs an integerised T) | Integer counts | Arbitrary rounding; it concentrates near a b^T | scipy `stats.random_table` (SciPy 1.10 or later) | Not recommended. Equivalent in practice to a b^T |
| Sampling variability of regularised plans | "the empirical regularized transport plan itself asymptotically follows a Gaussian law", with bootstrap consistency [31] | Klatt, Tameling & Munk 2020 [31] | 10.1137/19M1278788 | Confidence bands for regularised OT | Adaptation. Bootstrap cells to put error bars on R and on retrieval metrics | Finite metric space; i.i.d. sampling | Addresses sampling noise, not lineage | Custom | Sensitivity (cell bootstrap) |
| Wasserstein two-sample tests | Survey of tests based on the Wasserstein distance [32] | Ramdas, García Trillos & Cuturi 2017 [32] | 10.3390/e19020047 | Homogeneity testing | Not needed | n/a | n/a | n/a | Not recommended (it answers a different question) |

### 2.7 Tree reconstruction uncertainty

| 1 Name | 2 Procedure | 3 Original publication | 4 DOI | 5 Original use | 6 KP-Tracer use | 7 Assumptions | 8 Limits | 9 Implementation | 10 Recommendation |
|---|---|---|---|---|---|---|---|---|---|
| Character bootstrap | Resample characters with replacement and rebuild the tree [33]. Cassiopeia has per-site `sample_bootstrap_character_matrices` and per-intBC `sample_bootstrap_allele_tables`, the latter "because oftentimes there are dependencies between cut-sites on the same intBC" [5] | Felsenstein 1985 [33] (not read); Cassiopeia [5] | 10.1111/j.1558-5646.1985.tb00420.x | Confidence limits on clades | Direct procedure. Rebuild with the solver used in [1] and recompute every metric | Characters are independent (blocks are, in the intBC version) | Few intBCs per clone means low bootstrap power. Re-running Cassiopeia-Hybrid is expensive | Cassiopeia (as named) plus solver | Primary robustness check (intBC-block version) |
| Transfer bootstrap expectation | Branch support measured by a gradual transfer distance instead of presence or absence [34] | Lemoine et al. 2018 [34] | 10.1038/s41586-018-0043-0 | Support on large phylogenies | Adaptation | n/a | Supports clades, not metrics | booster | Sensitivity (only if clade supports are reported) |
| Robinson-Foulds distance | Number of bipartitions found in one tree but not the other [35] | Robinson & Foulds 1981 [35] (not read) | 10.1016/0025-5564(81)90043-2 | Tree comparison | Direct, comparing Cassiopeia and NJ topologies | Comparable leaf sets | Polytomies reduce the bipartition count, so normalisation matters | Cassiopeia `critique.robinson_foulds`; ete3 `robinson_foulds` | Report as a covariate |
| Triplets correct | Share of sampled leaf triplets whose ordering agrees, averaged with equal weight over LCA depths (about 10,000 triplets; depths with at least 20 cells) [3] | Jones 2020 [3] | 10.1186/s13059-020-02000-8 | Accuracy of reconstruction against simulated truth | Adaptation: agreement between NJ and Cassiopeia, not accuracy | One tree treated as the reference | Unresolved triplets in polytomies | Cassiopeia `critique.triplets_correct` | Report as a covariate |
| Alternative reconstruction (NJ vs Cassiopeia) | Yang: expansion proportions from NJ vs Cassiopeia trees, Pearson ρ = 0.87; CNV-tree agreement in 20 of 21 tumours [1]. LineageOT fits NJ on barcode Hamming distances [2]. DREAM benchmark: "structural features of the trees prevented optimal reconstructions" [37] | Saitou & Nei 1987 [36]; [1,2,37] | 10.1093/oxfordjournals.molbev.a040454; 10.1016/j.cels.2021.05.008 | Robustness checks | Direct | n/a | **NJ and Cassiopeia trees use the same characters. They are not independent replicates; agreement shows robustness to the algorithm, not correctness** | KP-Tracer ships both | Primary robustness check, with that caveat stated |
| Bayesian tree posterior | TiDeTree, a Bayesian framework for lineage-recorder trees [38]; integration over posterior trees as in BaTS [14] | Seidel & Stadler 2022 [38] (not read) | 10.1098/rspb.2022.1844 | Tree and population-dynamics inference | Adaptation | Model-based (BEAST 2) | Computationally heavy at thousands of cells | TiDeTree (BEAST 2) | Not recommended for this round |

## 3. Established methods vs our adaptations

**Direct uses (the method as published, on the kind of objects it was built for):**

- Yang's modified (mutation-count, Camin-Sokal) phylogenetic distance on the MetFamily trees, primary vs metastasis [1]. The direct part is the distance. Stratifying it by gate label is an adaptation.
- Evolutionary Coupling, D(M,K) with a 2000-shuffle z-score, between groups of cells on a MetFamily tree [1].
- The nearest-relative concordance statistics: Allelic EffectivePlasticity, CNV nearest-neighbour probability, AlleleMetRate [1,4]. The statistic is direct; the labels are new.
- Fitch-Hartigan parsimony normalised by edges (TreeMetRate, EffectivePlasticity) [1,4,10,11]. The statistic is direct; applying it to gate labels is new.
- Tip-label permutation nulls [1,13,14].
- Cassiopeia weighted Hamming with -log prior weights; indel priors; missing-data conventions [3,5].
- Cassiopeia vs NJ comparison, Robinson-Foulds, and intBC-block character bootstrap [1,5,35].
- The independence coupling a b^T as the uninformative reference [6,26].

**Adaptations (new objects or a new question):**

- Every use of the gate label g (retained or rejected) as a trait on the tree. No published work tests whether OT-rejected cells cluster on a lineage tree.
- Nearest-relative distance m(i) from source cells to the target set, compared between rejected and retained cells by AUROC or Mann-Whitney U. Yang's analysis goes the other way: metastatic cell to primary, by expansion.
- All coupling-vs-lineage evaluation. The published metrics (LineageOT L_D and L_A, moslin's normalised error, CoSpar and LARRY fate-bias correlation) assume ancestor-descendant ground truth across time points. KP-Tracer primary and metastatic cells are sampled at the same time and related through unobserved common ancestors, so no γ* exists.
- The tree-relevance retrieval metrics (MRR, AUPRC, precision@k) on rows or columns of T.
- The tree-optimal normaliser W_tree from Evans & Matsen [23]. The equality with weighted UniFrac is established; using it to normalise a coupling's tree cost is our proposal.
- Clade-level fate-bias agreement and JSD between clade-composition vectors.
- The permuted-identity and expression-stratified nulls (our proposal).

**Not sourced:** normalised MRCA relatedness; the tree-cost ratio R; the expression-stratified null (see section 5).

## 4. Recommended principal metrics and why

All metrics are computed per MetFamily, on every MetFamily with a completed gate and an available tree. Missing families are reported, never filtered. Results are pooled only by stratified summaries.

1. **Rejected vs retained nearest-relative distance to metastasis.** Compute m(i) = min_{k in K} d(i,k) under three relatednesses: mutation-count tree distance, edge-count tree distance (unifurcations collapsed), and the tree-free weighted-Hamming allele distance. Report AUROC (= U/(n1 n2) [42]) of rejection against m(i), with a within-family label-permutation p-value.
   *Why:* it tests claim (a) directly. It uses the authors' own distance [1]. Ranks absorb the ties at polytomies. The allele version does not depend on tree reconstruction.

2. **Normalised tree cost of the coupling.** R = (C(T) - W_tree(a,b)) / (C(a b^T) - W_tree(a,b)).
   - R = 0: T is as lineage-economical as any coupling with those marginals.
   - R = 1: T is no better than independence.
   Compare against the permuted-identity and expression-stratified nulls.
   *Why:* it scores claim (b) with no ground-truth coupling. The normalisation follows moslin [6], and the lower bound is the Evans-Matsen earth mover's distance on the tree [23].

3. **Normalised Fitch-Hartigan parsimony score of the gate label.** Compute it on source leaves (pruned family tree) and report a z-score against label permutations.
   *Why:* it is the established statistic for "is this binary trait clade-structured" [10,11,14]. KP-Tracer's own authors used it for metastasis and plasticity [1,4]. The Hartigan form handles polytomies.

4. **Lineage retrieval of the coupling.** For each target cell, rank sources by T_ij and score with MRR and AUPRC, where relevant = closest relatives by the same three relatednesses as in metric 1. Report each against its permutation baseline.
   *Why:* it gives per-cell, interpretable evidence that the coupling's top mass falls on phylogenetic relatives. This is the analogue of moslin's top-ancestor criterion [6].

5. **Stability across reconstructions.** Recompute metrics 1-4 on:
   - the Cassiopeia tree;
   - the NJ tree;
   - intBC-block bootstrap trees [5].

   Report the sign, and the spread across replicates, for each.
   *Why:* trees are inferred. Alternative reconstructions from the same characters are not independent [1,37]. Agreement shows robustness to the algorithm, not correctness, so the tree-free allele versions in metrics 1 and 4 are the necessary complement.

**Baselines alongside the nulls.** The LineageOT and moslin benchmarks compare against an expression-only counterpart. Accordingly, report metrics 2 and 4 for plain balanced OT and for kNN on the same representation. This separates what the gate adds from what heritable expression already gives.

## 5. Gaps: nothing found

- No published method evaluates whether cells rejected by a partial or unbalanced OT gate lack lineage relatives on the other side. Every use of the gate label is our proposal.
- No published metric evaluates a coupling against a lineage tree when there is no ancestor-descendant ground truth, i.e. same-time primary and metastasis. The tree-cost ratio R and the tree-relevance retrieval metrics are our proposals.
- Normalised MRCA relatedness: no established source found; our proposal (not recommended).
- Expression-stratified permutation null for lineage concordance: no established source found; our proposal.
- No established significance test for an OT coupling against a tree. Klatt et al. [31] cover sampling variability only.
- "TreeMetRate" and "phylogenetic metastatic score" are not in Yang et al. 2022. TreeMetRate is in Quinn et al. 2021 [4].
- No plasticity or fitness score in Yang 2022 measures primary-to-metastasis relatedness. Fitness came from `jungle.infer_fitness` (Neher et al. 2014 method, not read), with branch lengths set to 0 or 1 according to whether mutations separate the nodes [1]. Expansions came from a coalescent test (thresholds: depth at least 1, clade at least 15% of the tree) [1].
- A numeric definition of "MetFamily" or "large enough metastatic population" was not found in [1]. Use the released labels as given.
- The DREAM benchmark's metric definitions [37] were not read (abstract only).

## 6. References

Access levels:

- [read: methods]: the Methods section was read in full or in its relevant parts.
- [read: abstract only]: only the abstract was read.
- [not accessed]: the bibliographic record was verified via Crossref or a resolving URL, but no text was read.

1. Yang D, Jones MG, Naranjo S, et al. Lineage tracing reveals the phylodynamics, plasticity, and paths of tumor evolution. *Cell* 185(11):1905-1923 (2022). https://doi.org/10.1016/j.cell.2022.04.015 (PMC9452598). [read: methods, STAR Methods in full]
2. Forrow A, Schiebinger G. LineageOT is a unified framework for lineage tracing and trajectory inference. *Nat Commun* 12:4940 (2021). https://doi.org/10.1038/s41467-021-25133-1 (PMC8367995). [read: methods]
3. Jones MG, Khodaverdian A, Quinn JJ, et al. Inference of single-cell phylogenies from lineage tracing data using Cassiopeia. *Genome Biol* 21:92 (2020). https://doi.org/10.1186/s13059-020-02000-8 (PMC7155257). [read: methods]
4. Quinn JJ, Jones MG, Okimoto RA, et al. Single-cell lineages reveal the rates, routes, and drivers of metastasis in cancer xenografts. *Science* 371(6532):eabc1944 (2021). https://doi.org/10.1126/science.abc1944 (PMC7983364). [read: abstract only, plus the main-text results. Supplementary methods not accessed; TreeMetRate is defined in the main text]
5. Cassiopeia software, YosefLab. https://github.com/YosefLab/Cassiopeia. Read: `solver/dissimilarity_functions.py`, `solver/solver_utilities.py`, `tools/autocorrelation.py`, `tools/topology.py`, `tools/small_parsimony.py`, `data/utilities.py`, `data/CassiopeiaTree.py`; docs page for `weighted_hamming_distance`. [read: methods (source code)]
6. Lange M, Piran Z, Klein M, Spanjaard B, Klein D, Junker JP, Theis FJ, Nitzan M. Mapping lineage-traced cells across time points with moslin. *Genome Biol* 25:277 (2024). https://doi.org/10.1186/s13059-024-03422-4 (PMC11492637). [read: methods]
7. Wang SW, Herriges MJ, Hurley K, Kotton DN, Klein AM. CoSpar identifies early cell fate biases from single-cell transcriptomic and lineage information. *Nat Biotechnol* 40:1066-1074 (2022). https://doi.org/10.1038/s41587-022-01209-1. [read: abstract only for the journal version; main text of bioRxiv preprint 10.1101/2021.05.06.443026 v1 read]
8. Weinreb C, Rodriguez-Fraticelli A, Camargo FD, Klein AM. Lineage tracing on transcriptional landscapes links state to fate during differentiation. *Science* 367:eaaw3381 (2020). https://doi.org/10.1126/science.aaw3381 (PMC7608074). [read: abstract only, plus the main-text benchmark section]
9. Sokal RR, Rohlf FJ. The comparison of dendrograms by objective methods. *Taxon* 11(2):33-40 (1962). https://doi.org/10.2307/1217208. [not accessed]
10. Fitch WM. Toward defining the course of evolution: minimum change for a specific tree topology. *Syst Zool* 20:406-416 (1971). https://doi.org/10.2307/2412116. [not accessed]
11. Hartigan JA. Minimum mutation fits to a given tree. *Biometrics* 29:53-65 (1973). https://doi.org/10.2307/2529676. [not accessed]
12. Slatkin M, Maddison WP. A cladistic measure of gene flow inferred from the phylogenies of alleles. *Genetics* 123:603-613 (1989). https://doi.org/10.1093/genetics/123.3.603. [read: abstract only]
13. Wang TH, Donaldson YK, Brettle RP, Bell JE, Simmonds P. Identification of shared populations of human immunodeficiency virus type 1 infecting microglia and tissue macrophages outside the central nervous system. *J Virol* 75:11686-11699 (2001). https://doi.org/10.1128/JVI.75.23.11686-11699.2001 (PMC114755). [read: methods]
14. Parker J, Rambaut A, Pybus OG. Correlating viral phenotypes with phylogeny: accounting for phylogenetic uncertainty. *Infect Genet Evol* 8:239-246 (2008). https://doi.org/10.1016/j.meegid.2007.08.001. [read: abstract only]
15. Fritz SA, Purvis A. Selectivity in mammalian extinction risk and threat types: a new measure of phylogenetic signal strength in binary traits. *Conserv Biol* 24(4):1042-1051 (2010). https://doi.org/10.1111/j.1523-1739.2010.01455.x. [read: abstract only, plus caper `phylo.d` documentation]
16. Pagel M. Inferring the historical patterns of biological evolution. *Nature* 401:877-884 (1999). https://doi.org/10.1038/44766. [read: abstract only]
17. Blomberg SP, Garland T, Ives AR. Testing for phylogenetic signal in comparative data: behavioral traits are more labile. *Evolution* 57:717-745 (2003). https://doi.org/10.1111/j.0014-3820.2003.tb00285.x. [read: abstract only]
18. Gittleman JL, Kot M. Adaptation: statistics and a null model for estimating phylogenetic effects. *Syst Zool* 39:227-241 (1990). https://doi.org/10.2307/2992183. [not accessed; start page 227 verified, end page unverified]
19. Webb CO, Ackerly DD, McPeek MA, Donoghue MJ. Phylogenies and community ecology. *Annu Rev Ecol Syst* 33:475-505 (2002). https://doi.org/10.1146/annurev.ecolsys.33.010802.150448. [not accessed]
20. DeTomaso D, Yosef N. Hotspot identifies informative gene modules across modalities of single-cell genomics. *Cell Syst* 12:446-456 (2021). https://doi.org/10.1016/j.cels.2021.04.005. [read: abstract only]
21. Lozupone C, Knight R. UniFrac: a new phylogenetic method for comparing microbial communities. *Appl Environ Microbiol* 71:8228-8235 (2005). https://doi.org/10.1128/AEM.71.12.8228-8235.2005. [read: abstract only]
22. Lozupone CA, Hamady M, Kelley ST, Knight R. Quantitative and qualitative beta diversity measures lead to different insights into factors that structure microbial communities. *Appl Environ Microbiol* 73:1576-1585 (2007). https://doi.org/10.1128/AEM.01996-06 (PMC1828774). [read: methods (equations shown as images; term definitions read)]
23. Evans SN, Matsen FA. The phylogenetic Kantorovich-Rubinstein metric for environmental sequence samples. *J R Stat Soc B* 74:569-592 (2012). https://doi.org/10.1111/j.1467-9868.2011.01018.x (PMC3405733). [read: methods]
24. Lin J. Divergence measures based on the Shannon entropy. *IEEE Trans Inf Theory* 37(1):145-151 (1991). https://doi.org/10.1109/18.61115. [not accessed]
25. Cover TM, Thomas JA. *Elements of Information Theory*, 2nd ed. Wiley (2006). https://doi.org/10.1002/047174882X. [not accessed]
26. Peyré G, Cuturi M. Computational optimal transport. *Found Trends Mach Learn* 11(5-6):355-607 (2019). https://doi.org/10.1561/2200000073 (arXiv 1803.00567). [read: methods (Section 4.1, Proposition 4.1)]
27. Cuturi M. Sinkhorn distances: lightspeed computation of optimal transport. *NeurIPS* 26:2292-2300 (2013). https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport. [read: abstract only]
28. Sinkhorn R. A relationship between arbitrary positive matrices and doubly stochastic matrices. *Ann Math Stat* 35:876-879 (1964). https://doi.org/10.1214/aoms/1177703591. [not accessed]
29. Patefield WM. Algorithm AS 159: an efficient method of generating random R x C tables with given row and column totals. *Appl Stat* 30:91-97 (1981). https://doi.org/10.2307/2346669. [not accessed; SciPy `random_table` documentation read]
30. Diaconis P, Gangolli A. Rectangular arrays with fixed margins. In: *Discrete Probability and Algorithms*, IMA Vol Math Appl 72:15-41 (1995). https://doi.org/10.1007/978-1-4612-0801-3_3. [not accessed]
31. Klatt M, Tameling C, Munk A. Empirical regularized optimal transport: statistical theory and applications. *SIAM J Math Data Sci* 2(2):419-443 (2020). https://doi.org/10.1137/19M1278788. [read: abstract only]
32. Ramdas A, García Trillos N, Cuturi M. On Wasserstein two-sample testing and related families of nonparametric tests. *Entropy* 19(2):47 (2017). https://doi.org/10.3390/e19020047. [read: abstract only]
33. Felsenstein J. Confidence limits on phylogenies: an approach using the bootstrap. *Evolution* 39:783-791 (1985). https://doi.org/10.1111/j.1558-5646.1985.tb00420.x. [not accessed]
34. Lemoine F, Domelevo Entfellner JB, Wilkinson E, et al. Renewing Felsenstein's phylogenetic bootstrap in the era of big data. *Nature* 556:452-456 (2018). https://doi.org/10.1038/s41586-018-0043-0. [read: abstract only]
35. Robinson DF, Foulds LR. Comparison of phylogenetic trees. *Math Biosci* 53:131-147 (1981). https://doi.org/10.1016/0025-5564(81)90043-2. [not accessed]
36. Saitou N, Nei M. The neighbor-joining method: a new method for reconstructing phylogenetic trees. *Mol Biol Evol* 4:406-425 (1987). https://doi.org/10.1093/oxfordjournals.molbev.a040454. [not accessed; DOI resolves to this title, but volume and pages are unverified (Crossref has no metadata for them)]
37. Gong W, Granados AA, Hu J, et al. Benchmarked approaches for reconstruction of in vitro cell lineages and in silico models of C. elegans and M. musculus developmental trees. *Cell Syst* 12(8):810-826 (2021). https://doi.org/10.1016/j.cels.2021.05.008. [read: abstract only]
38. Seidel S, Stadler T. TiDeTree: a Bayesian phylogenetic framework to estimate single-cell trees and population dynamic parameters from genetic lineage tracing data. *Proc R Soc B* 289 (2022). https://doi.org/10.1098/rspb.2022.1844. [not accessed]
39. Sashittal P, Schmidt H, Chan M, Raphael BJ. Startle: a star homoplasy approach for CRISPR-Cas9 lineage tracing. *Cell Syst* 14:1113-1121 (2023). https://doi.org/10.1016/j.cels.2023.11.005. [not accessed]
40. Liu J, Huang Y, Singh R, Vert JP, Noble WS. Jointly embedding multiple single-cell omics measurements. *WABI 2019*, LIPIcs 143:10 (2019). https://doi.org/10.4230/LIPIcs.WABI.2019.10 (preprint 10.1101/644310). [not accessed; record verified at drops.dagstuhl.de]
41. Demetci P, Santorella R, Sandstede B, Noble WS, Singh R. SCOT: single-cell multi-omics alignment with optimal transport. *J Comput Biol* 29:3-18 (2022). https://doi.org/10.1089/cmb.2021.0446. [read: methods of bioRxiv preprint 10.1101/2020.04.28.066787 v2 (evaluation section)]
42. Hanley JA, McNeil BJ. The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology* 143:29-36 (1982). https://doi.org/10.1148/radiology.143.1.7063747. [not accessed]
43. Davis J, Goadrich M. The relationship between precision-recall and ROC curves. *Proc ICML* 233-240 (2006). https://doi.org/10.1145/1143844.1143874. [not accessed]
44. Saito T, Rehmsmeier M. The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10:e0118432 (2015). https://doi.org/10.1371/journal.pone.0118432. [not accessed]
45. Voorhees EM. The TREC-8 question answering track report. *Proc TREC-8* (1999). https://trec.nist.gov/pubs/trec8/papers/qa_report.pdf. [not accessed: the URL resolves but the PDF is a scan without a text layer. Whether this is the first definition of MRR is unverified]
46. Huerta-Cepas J, Serra F, Bork P. ETE 3: reconstruction, analysis, and visualization of phylogenomic data. *Mol Biol Evol* 33:1635-1638 (2016). https://doi.org/10.1093/molbev/msw046. [read: abstract only]
47. Paradis E, Schliep K. ape 5.0: an environment for modern phylogenetics and evolutionary analyses in R. *Bioinformatics* 35:526-528 (2019). https://doi.org/10.1093/bioinformatics/bty633. [not accessed]
48. Schliep KP. phangorn: phylogenetic analysis in R. *Bioinformatics* 27:592-593 (2011). https://doi.org/10.1093/bioinformatics/btq706. [not accessed]
49. Revell LJ. phytools: an R package for phylogenetic comparative biology (and other things). *Methods Ecol Evol* 3:217-223 (2012). https://doi.org/10.1111/j.2041-210X.2011.00169.x. [not accessed]
50. Keck F, Rimet F, Bouchez A, Franc A. phylosignal: an R package to measure, test, and explore the phylogenetic signal. *Ecol Evol* 6:2774-2780 (2016). https://doi.org/10.1002/ece3.2051. [read: abstract only]
51. Flamary R, Courty N, Gramfort A, et al. POT: Python Optimal Transport. *J Mach Learn Res* 22(78):1-8 (2021). https://jmlr.org/papers/v22/20-451.html. [not accessed; landing page verified]
52. scikit-bio, `skbio.diversity.beta.weighted_unifrac` documentation. https://scikit.bio/docs/latest/generated/skbio.diversity.beta.weighted_unifrac.html. [read: documentation]
53. SciPy, `scipy.stats.random_table` documentation (added in SciPy 1.10.0). https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.random_table.html. [read: documentation]
54. KPTracer-release repository. https://github.com/mattjones315/KPTracer-release. [read: README only]
55. LineageOT repository. https://github.com/aforr/LineageOT. [not accessed; URL resolves]
