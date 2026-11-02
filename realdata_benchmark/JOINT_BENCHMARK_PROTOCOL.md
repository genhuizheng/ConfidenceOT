# Joint Splatter + LTL331 PDX preprocessing benchmark: protocol

Written 2026-10-31, before any configuration below other than the two frozen LTL331 arms was fitted with OT, and
before any Rank-K / Rank-HVG / P25 / P50 representation was built. Changes after this commit are added as dated
sections at the end, never edits.

**Already seen when this was written** (so not blind to them): the frozen LTL331 run (`ranknm256_noscale_ds_cos`,
`logcpm`; 7 pairs, M4-E / M4-R), its validity audit (`ltl331_nepc/07_`, `08_`), the DEG / GSEA depth audit, the
P10 downsampling information-loss table (`DOWNSAMPLING_INFORMATION_LOSS.tsv`), the tie-rule finding (production
`ranknm` breaks ties in reverse alphabetical gene order), the KP / Quinn Level 2 / 3 results and the native Splatter
benchmark results (`benchmark_results/native`). Nothing below is chosen by OT accuracy or by a biological result.

**Goal.** Find which preprocessing choices give rejection-aware OT reliable unmatched detection, real-data
robustness and interpretable couplings, and whether the real-data limitations come from Top-K truncation,
downsampling, cost geometry, calibration or their interaction. No configuration is assumed best.

## 1. Arms

| arm | data | truth | where |
|---|---|---|---|
| A | original Splatter benchmark (`benchmark/DESIGN.md`): K = 5 groups; `all_shared`, `population_lost` (one population unmatched, both directions), `populations_disjoint`; levels `L0_matched`, `L1_depth`, `L2_depth_dropout`; original seeds and generated counts | simulator `Group` | TACC (`benchmark_technical`), reused read-only |
| B | PDX-informed Splatter: parameters estimated from LTL331 counts (section 4), same three cases, fixed truths | simulator `Group` | TACC (R container `benchmark/tacc/splatter.def`) |
| C | LTL331 GSE297328, 8 time points, all 7 adjacent pairs; 16 -> 20 weeks is the focus, never the only pair reported | none (authors' coarse_celltype, velocity labels are annotations, not truth) | local pilot, TACC full |
| D | KP-Tracer, Quinn A549: frozen protocol 508609f Level 1 / 2 / 3 | CRISPR lineage (authors' labels, frozen subclones) | TACC; versioned extension only |

Anatomical sites (where a cell was sampled) and CRISPR editing sites (Cas9 target sites / intBCs) are never mixed.

## 2. Configuration factors and their exact definitions

Every configuration is a combination of the factors below; `PREPROCESSING_CONFIG_REGISTRY.tsv` lists each one with
its role (main / sensitivity / historical record) and whether it ran before.

- **Input depth (DS).** `none` = the QC'd original counts. `P10`, `P25`, `P50` = `confidenceot.equalise_depth`
  (exact multivariate hypergeometric subsampling without replacement; cells at or below the target untouched) to the
  pooled quantile of the analysed cells' depth: LTL331 2,817 / 5,023 / 9,360 UMIs. At P50 about half of all cells
  (14-64% per time point) are below the target, so P50 is a partial equalisation; reported as such. Seeds 1, 2, 3
  for every DS level; the frozen P10 files (27_, their own seed) are kept as a fourth P10 draw. Splatter: the same
  quantiles of each condition's pooled depth.
- **Encoding.**
  - `logcpm` (the user's "log1p"): library size 1e4 -> log1p; the non-rank control. Raw-count arms are removed from
    every new grid (user, 2026-10-31); they appear in the registry only as a record of what ran.
  - `ranknm-K` (Rank-K): per cell, the K genes with the highest count (division by the cell total does not change
    the order; no gene-median division), value 1 - r/K for r = 0..K-1, every other gene 0. K = 128, 256, 512, 1024,
    2048. Downstream as the production rank arm: the 2,000 highest-variance encoded genes over both sides, no gene
    scaling, PCA 30.
  - `rankhvg-H` (Rank-HVG): H = 2000, 3000. HVGs = the H genes of highest variance of log1p(1e4 x count / total)
    over both sides stacked (the `logcpm` arm's own rule, so only the encoding differs from the control), computed on
    the counts the configuration reads. Within the HVGs, per cell, only expressed genes (count > 0) are ranked, by
    count, descending; value 1 - (r - 1) / H for rank r = 1..n_expressed; genes with count 0 stay 0, so no zero gene
    receives a nonzero rank and there is no Top-K boundary inside the HVGs (every expressed HVG is kept). No further
    gene selection; no gene scaling; PCA 30.
  - `rank-K` with gene-median division (`rank256` family) stays in the historical grid only.
- **Tie rule** (rank encodings). `production` = numpy stable argsort over the stored row order, exactly as
  `rank_value_encode` behaves (reverse alphabetical for `ranknm`); `expected` = the encoding averaged over random
  tie-breaking (every member of a tie group gets the mean value of the slots the group occupies; a group straddling
  the Top-K cut shares those slots: value x slots / group size); `midrank` (Rank-HVG) = tied genes share their
  average rank. `random` (seeds 1-3) only to measure gene-order sensitivity.
- **Gene scaling.** `noscale` (main for rank and logcpm) and `scale` (sensitivity, as in the historical grid).
- **Cost.** `cos` = PCA rows L2-normalised, squared Euclidean = 2 - 2 cos (main); `sqeuc` = squared Euclidean on PCA
  coordinates (sensitivity, every representative configuration). Median sampled cross-pair scaling as in production.
- **OT.** epsilon 0.1, lambda 1 / 1, tolerance 1e-4, within-side split null, 5 + 5 replicates, grid 5, acceptance
  0.99 (main; the real-data rule), 0.90 for Arm A comparability with the original grid. M4-E main; M4-R reported, and
  a non-converged M4-R never enters an average or a formal statement. Balanced OT (entropic, epsilon 0.1) on the same
  cost, full and on the M4-E retained block (matched coverage). No solver default changes.

## 3. Pre-specified experiment matrix

The Cartesian product is not run. Contrasts:

| id | contrast | fixed | varied | stage |
|---|---|---|---|---|
| R1 | Top-K truncation | DS none, tie expected, noscale, cos | K in 128..2048 | diagnostics all pairs; OT 16 -> 20 local, all pairs TACC |
| R2 | tie handling | DS none and P10, noscale, cos | K in {256, 1024} x tie {production, expected, random1-3} | diagnostics; OT for production vs expected |
| R3 | HVG-restricted ranking | DS none, noscale, cos | rankhvg-2000, rankhvg-3000 (midrank) vs ranknm-1024 / 2048 (expected) vs logcpm_noscale_cos | diagnostics; OT |
| R4 | downsampling | representatives: logcpm_noscale_cos, ranknm256 production (the frozen family), ranknm1024 expected, rankhvg-2000 midrank | DS none, P10 (seeds 1-3 + frozen), P25, P50 (seeds 1-3) | diagnostics; OT none + one seed per DS level on 16 -> 20 local, TACC for the rest |
| R5 | cost geometry | the four representatives at DS none and P10 | cos vs sqeuc | diagnostics; OT |
| R6 | gene scaling | logcpm and ranknm256 production, DS none and P10, cos | scale vs noscale | diagnostics; OT |
| H  | historical grid | the 24 non-raw labels of the original 32-arm grid, as executed | none | diagnostics all pairs; OT 16 -> 20 local |

Representatives were fixed by rule, not by result: the production configuration; the non-rank control; the rank-K
setting nearest the middle of the sweep on a log scale (1024); the smaller Rank-HVG. K and H are never re-chosen
from results.

## 4. Arm B: PDX-informed Splatter

- `splatEstimate` on LTL331 original counts of the analysed cells, one estimate per time point and one pooled.
- Fidelity check before any OT, against the real counts: library-size distribution, detected genes, mean-variance,
  fraction of zeros per gene, dropout curve, and the Top-K tie profile at P10 (share of Top-256 slots decided by ties,
  tie-group size at the cut). Reported in `SPLATTER_PDX_PARAMETER_AUDIT.md`; a parameter set that misses a check is
  reported, not tuned against OT results.
- Fixed truths: the three Arm A cases with K = 5 groups. Factors (one at a time from the estimated centre): library
  size (estimated, x0.5, x0.25), dropout (estimated, + target-side batch dropout as Arm A L2), composition (equal
  groups; one dominant group at 0.6, mirroring 20-week NE dominance), state overlap (DE probability / factor
  location at the estimate and at half). Seeds 1-3. Parameters are frozen before the first OT run.

## 5. Evaluation (every configuration, every arm where defined)

- Information: UMI retention, detected genes, Top-K slots decided by ties, Top-K membership stability across DS
  seeds and against DS none, PCA subspace stability (seed, DS draw), cost correlation across configurations.
- Technical robustness: gate / cost association with pre-equalisation depth (AUC, Spearman), DS-seed gate stability
  (kappa), tie-rule (gene-order) sensitivity, dropout sensitivity (Arm A L2, Arm B dropout).
- Gate: Splatter shared-population false rejection, unmatched detection, disjoint false asserted mass; source /
  target coverage; calibration validity; convergence.
- Coupling: Splatter ground-truth population transport and conditional accuracy, absolute supported mass;
  matched-coverage balanced OT correspondence; row entropy; enrichment over the marginal-independent coupling
  (permutation null preserving abundances).
- PDX: NE / PRAD / Basal / Other retention with intervals, within-type depth-adjusted effects, kNN RNA baseline,
  target-composition and target-size controls (`ltl331_nepc/08_` design), annotation-level transport beyond
  marginals, DS-seed and arm agreement.
- DEG / GSEA only for cases that pass the depth / detection audit (median Wilcoxon score and positive-gene share
  near the null on equalised counts), exploratory, hallmark only; never used to choose a configuration.

## 6. Reporting

All configurations are reported; none is dropped for its result. Every table separates observed result, technical
association and biological inference. No NEPC ancestry, fate or transdifferentiation claim is made from gates or
couplings. Frozen outputs are never overwritten: new work lives in `results/realdata_benchmark_20261031/` (local)
and a versioned root on TACC.

## Dated change, 2026-11-01: scope is RNA-only (user)

- This benchmark studies RNA preprocessing only. Cancelled, no code or TACC job: CRISPR editing-site-only OT,
  RNA + editing-site OT, held-out intBC input / evaluation, CRISPR character cost engineering, multimodal cost fusion.
  None of these had been written.
- Priorities: (1) Splatter shared / unmatched (Arms A, B); (2) LTL331 PDX preprocessing robustness and NEPC
  validity; (3) KP-Tracer / Quinn RNA-only exploratory validation, low priority.
- KP-Tracer / Quinn: costs from scRNA-seq only; CRISPR trees, clone and MetFamily labels are external evaluation
  information only; the existing RNA-only Level 1 / 2 / 3 results are reused and the frozen protocols are untouched.
  CRISPR phylogenetic correspondence is not a ground truth that RNA-based ConfidenceOT has to recover. New compute,
  if any, is a few representative RNA preprocessing sensitivity runs, not a CRISPR-specific benchmark.
- Table 1 arm D and section 5's lineage items read accordingly.

## Dated correction, 2026-11-01: Arm A root

Table 1 names `benchmark_technical` as Arm A's TACC root. That is the superseded thinning benchmark (three cases, no
`populations_disjoint`; not archived, `benchmark/ARCHIVE_MANIFEST.md`). Arm A is `benchmark_native`, the original
benchmark with all four cases (all_shared, population_lost, population_emerged, populations_disjoint). Found from the
first worklist (2,592 rows, no disjoint pairs) before any Splatter fit; that worklist is set aside, not used.

## Dated extension, 2026-11-01: Splatter N = 5,000, cumulative levels L0-L5 (user)

Organised as cumulative levels instead of separate stages; each level inherits every factor of the one before and
adds one; values fixed in `splatter_n5000/20_generate_n5000.R` before any L3-L5 fit:
- L0-L2: the frozen native definitions (matched; target depth x 0.5; + target dropout), regenerated with the same seeds
  and checked against the native generation; their frozen fits are reused when identical.
- L3 = L2 + batch effect (Splatter batch factors, expected squared per-gene log shift 0.5 D, D = 2 de.prob
  (de.facLoc^2 + de.facScale^2)); L4 = L3 + target perturbation of shared Group1 (label kept, 0.5 D); L5 = L4 + target
  proportions 0.40 / 0.25 / 0.15 / 0.12 / 0.08; ablations A4 = L2 + perturbation, A5 = L2 + imbalance.
- L3-L5 and the ablations use L2's seed (paired). N = 5,000 cells per side, Cases 1 / 2 (both directions) / 3,
  replicates 1-3; eight configurations (the four R4 / R5 representatives at DS none and P10 seed 1, cosine, noscale).
- Per row IC-POT (c = 0.5, the literature value) and Vanilla UOT on the same cost besides M4-E / M4-R / balanced OT;
  DEG-A / B / C / D against the simulator's truth (21_ / 23_ / 24_).

## Dated section, 2026-11-01: which configurations are read biologically on LTL331 (fixed before the collected tables are opened)

Written before PDX_GATE_COUPLING_METRICS.tsv / SPLATTER_GATE_COUPLING_METRICS.tsv were downloaded or read. A
configuration is eligible for the PDX biological readout when, with M4-E at acceptance 0.99, all hold:
1. Splatter (Arm A, n_cells 1,000 and 5,000, replicates 1-3, every level): median false rejection of shared cells
   <= 0.10 on each side in all_shared; median recall of the unmatched population >= 0.80 (source side in
   population_lost, target side in population_emerged); median false asserted mass <= 0.05 in populations_disjoint.
2. LTL331: M4-E inference valid in >= 5 of the 7 pairs; median |AUC(pre-equalisation depth, rejected) - 0.5| <= 0.10
   over the 14 pair-sides.
3. Stability, where measured (DS and tie-rule representatives, REPRESENTATION_DIAGNOSTICS / RANK_TIE_SENSITIVITY):
   OT-free kNN-gate kappa >= 0.80 between DS seeds and between tie rules.
If none passes every criterion, that is reported and every configuration's position on the three criteria is shown
(no configuration is picked for its biology). The biological readout is the same for every eligible configuration:
all 7 pairs; per pair and side the retained fraction of each annotated type with intervals, the depth-stratified
and kNN-baseline checks of `ltl331_nepc/07_`, latent time and cell cycle; DEG / GSEA (hallmark only) within each
annotated type, rejected vs retained, on the frozen P10 equalised counts whatever the configuration (so depth cannot
drive the contrast), read only where the depth audit of `LTL331_BIOLOGICAL_VALIDITY.md` section 5 passes; a pathway
is reported as consistent only when its sign agrees in every eligible configuration. Observed result, technical
association and biological inference stay separate; no ancestry or fate claim.
