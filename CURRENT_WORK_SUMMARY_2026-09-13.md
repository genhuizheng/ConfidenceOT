# ConfidenceOT Current Work Summary

**Last updated:** 2026-09-14 (section 5.7 supersedes the gate results in sections 3, 5 and 6)
**Local repository:** `D:\Xia_lab\CellOT`  
**TACC repository:** `/scratch/10119/ghzheng/OT_project/code/ConfidenceOT`  
**TACC result root:** `/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results`

## 1. Overall objective

The current project evaluates whether ConfidenceOT can identify cell states that are compatible or incompatible between paired primary and metastatic single-cell RNA-seq samples.

The central biological question is:

> What distinguishes primary malignant cells whose expression states are compatible with a matched metastatic lesion from primary malignant cells whose states are not represented in that lesion?

This wording is intentionally descriptive. A ConfidenceOT gate does not by itself prove metastatic potential, clonal origin, or lineage. Such claims require orthogonal validation, for example shared CNV clones, somatic mutations, lineage tracing, or an independent cohort.

## 2. Interpretation of ConfidenceOT outputs

For one exact `patient x primary site x metastatic site` pair:

- `retained=True` means that the cell was retained by the fitted M4 gate and is relatively compatible with the opposite side under that particular model, representation, cost, and initialization.
- `rejected=True` means that the cell was excluded by the fitted gate and is relatively poorly explained by the opposite side under that fit.
- The rejection budget is an upper bound on the rejected fraction, not a requirement to reject that fraction.
- The rejection score is a continuous cell-level diagnostic associated with the binary gate; a larger score indicates stronger evidence toward rejection under the fitted solution.
- Cross-pair/cap consensus states summarize repeated occurrences of the same cell: consistently rejected, consistently retained, or discordant across eligible pair/cap runs.

The labels must not be renamed `metastasis-potential` and `non-metastasis-potential` until they pass robustness and biological validation.

As of 2026-09-14 the defensible reading is narrower still. For every gate fitted before that date, `retained` means "a primary cell lying near the core of the metastatic distribution in a sequencing-depth-confounded PCA geometry, under a rejection cost whose null could not distinguish a homogeneous cloud from a real correspondence". Section 5.7 gives the evidence and the changes made in response. Results produced before those changes should not be read as compatibility statements at all.

As of 2026-09-16 the third and fourth bullets above are withdrawn as written, on the evidence in section 5.8. `retained_fraction` is not an estimate of the incompatible fraction under any calibration: it measures primary--metastasis transcriptional similarity at partial rho 0.79 against an independent pseudobulk measure, spans 0.000 to 0.879 across GSE180661 patients, and falls to 0.0087 on GSE271675, where the two sides genuinely diverge. `retained` is also not a property of a cell but of a (cell, metastasis) pair: naming a different lesion moved one patient from 0.420 to 0.032 over the same cells. Within one patient the retained cells are still the ones nearer that metastasis, which is what the rule computes; that is the whole of what the label currently supports.

## 3. Main dataset: GSE180661 HGSOC

### 3.1 Dataset inventory

GSE180661 is a high-grade serous ovarian cancer single-cell dataset containing primary lesions and matched metastatic sites such as infracolic omentum, ascites, bowel, and pelvic peritoneum.

| Scope | Coverage |
|---|---:|
| Original patients | 41 |
| Sample files | 74 |
| Cells | 929,690 |
| Enumerable primary-metastasis combinations | 121 |
| All-cell eligible patients | 33 |
| Author malignant label | `Ovarian.cancer.cell` |
| Malignant-only eligible exact pairs | 94 |
| Malignant-only represented patients | 29 |

The number of patients differs by analysis because the malignant-only workflow requires enough author-annotated malignant cells on both sides of an exact pair. Multi-primary origin ranking is a smaller subset of the malignant analysis.

### 3.2 Figure-oriented workflow

#### Figure 1: cohort and sample inventory

- Sample/site availability across patients.
- Primary and metastatic cell counts.
- Cell-type composition by patient and site.
- Sampling time was intentionally ignored because it was not consistently available.

#### Figure 2A: all-cell OT validation

- Unit: one patient, one primary sample/site, and one metastatic sample/site.
- Use all annotated cell types.
- Summarize source-to-target transported mass as a cell-type transition matrix.
- Main validation question: whether biologically corresponding populations, such as T cells, preferentially map to one another.
- This branch validates transport structure; it is not the main malignant-cell discovery analysis.

#### Figure 2B: malignant-only OT

- Use author-annotated malignant cells only.
- Fit each exact primary-metastasis pair independently.
- Visualize primary and metastatic malignant cells jointly for pairwise mapping, while also providing clearly separated primary and metastatic panels.
- Export cell-level confidence and gate state for each side.

#### Figure 3: primary internal differential expression

- Within every exact pair, split primary malignant cells into source-retained and source-rejected groups.
- Sum raw integer counts across cells to form one pseudobulk per `pair_id x comparison_status`.
- Pseudobulk is a raw-count sum, not an average; library-size differences are handled by PyDESeq2 normalization.
- Combine all eligible pair pseudobulks using the paired design `~ pair_id + comparison_status`.
- Test all genes, including genes used in the OT representation.
- Report gene name, log2 fold change, FDR, mean CPM, patient detection fraction, patient direction consistency, and whether the gene was used by OT.

#### Figure 4: pathway analysis

- Run GSEApy prerank using the PyDESeq2 Wald statistic.
- Show positive and negative normalized enrichment scores in a dot plot.
- Retain pathway direction and FDR in the output table.

#### Figure 5: external clinical validation

- Construct a candidate expression signature from the primary internal DEG/GSEA results.
- Test the signature in TCGA-OV with Kaplan-Meier and Cox models.
- Treat survival analysis as external association rather than proof of metastatic lineage.

## 4. Preprocessing and OT settings

### 4.1 Cancer workflows

The standard representation is:

`raw counts -> library-size normalization to 10,000 -> log1p -> joint HVG selection -> gene scaling -> joint PCA -> pairwise cost -> OT`

Settings varied during sensitivity analyses:

- Main GSE180661 malignant analyses used 2,000 HVGs and 30 PCs.
- Post-QC thresholds were `nCount >= 1000`, `nFeature >= 500`, and mitochondrial percentage `<= 20%`.
- The mitochondrial threshold was inert for GSE225857. That dataset carries no `MT-` prefixed gene symbols, so `pct_counts_mitochondrial` is constant zero and the 20% rule never removed a cell. The claim that all three datasets received identical post-QC filtering does not hold.
- The post-QC workflow retained 176,070 of 202,731 source-cell occurrences (86.85%) and 172,683 of 197,090 target-cell occurrences (87.62%).
- The main post-QC sensitivity analysis used source cap 0.85 and target cap 0.95.
- Null-calibrated rejection costs were used unless a workflow was explicitly labelled as a fixed-cost experiment.
- Every run described in this document used the rotation null and an enforced rejection budget. Both defaults changed on 2026-09-14; see section 5.7. New runs calibrate against a within-side split null, report the budget rather than enforcing it, and may be preceded by the depth-equalisation step. `pair_metrics.csv` and `run.json` now record `calibration_null`, `budget_enforced` and the decomposed calibration-validity flags, so a stored result states which convention produced it.

### 4.2 All-cell pan-cancer run

The completed all-cell run covered:

- 13 datasets
- 86 patients
- 252 exact primary-metastasis pairs
- 504 method rows: 252 M4-E and 252 M4-R
- Source cap 0.85 and target cap 0.95 in the completed run
- Up to 10,000 observed cells per side

All 252 pairs completed. M4-E converged for all pairs. M4-R had valid calibration for only 3 of 252 pairs, with frequent outer-loop cycles, so M4-R was retained primarily as a diagnostic sensitivity result rather than a biological inference engine.

That 3-of-252 figure has since been traced and does not mean the calibration failed. `calibration_valid` was a strict conjunction that any M4-R terminal warning zeroed, and M4-R only cross-checks a rejection cost fitted with M4-E; its gate has no monotone-objective guarantee, so exhausting the outer loop is a property of that variant. On a case where every component was sound -- feasible cost found, M4-E clean, held-out acceptance 0.917 and 0.923 against a 0.90 requirement -- the flag still read False because M4-R cycled on 6 of 10 nulls. The result now reports `feasible_cost_found`, `m4e_calibration_clean`, `m4r_validation_clean` and `m4e_inference_valid` separately, and the biological inference rests on the last of these.

## 5. Completed sensitivity analyses and findings

### 5.1 Source-cap sensitivity

The target cap was fixed at 0.95 while the source cap was varied. Earlier pre-QC comparisons gave:

| Source cap | Source weighted rejection | Target weighted rejection | Exact winner match vs 0.95 |
|---:|---:|---:|---:|
| 0.95 | 0.911 | 0.615 | 1.000 |
| 0.90 | 0.860 | 0.606 | 0.882 |
| 0.85 | 0.828 | 0.670 | 0.794 |

Additional source caps, including 0.75, 0.60, and 0.50, were subsequently included in sensitivity work. These runs are used to examine cap binding, cell-group sizes, gate stability, DEG direction, and biological coherence. A cap must not be selected solely because it produces more metastasis-related genes.

That standard now also governs the choice of calibration null and of depth normalisation introduced in section 5.7. Both were selected against simulation ground truth, where the correct answer is known by construction, and neither may be chosen because it yields a more attractive differential-expression result.

### 5.2 Post-QC DEG observation

At source cap 0.85, the source-rejected primary cells showed strong inflammatory, interferon, chemokine, stress, and invasion-associated signals, including examples such as `CXCL10`, `CXCL11`, `CXCL8`, `CCL20`, `OASL`, `SAA1`, `RSAD2`, `IDO1`, `MMP7`, `ADM`, and `IGFBP3`.

This observation is biologically interesting but contradicts the initial assumption that source-retained cells could automatically be interpreted as the more metastasis-competent population. It therefore triggered algorithmic and data-quality audits rather than a reversal of labels based only on desired biology.

### 5.3 Initialization sensitivity

M4-E was rerun from the stored baseline, the closest feasible deterministic budget-floor start, and five random feasible starts for 94 exact malignant pairs. An all-zero gate is infeasible because the rejection budget is an upper bound; the production default with no supplied gate is all-one, meaning that every cell is initially retained. The stored baseline was reconstructed exactly, but random-start robustness was limited:

| Metric | Source | Target |
|---|---:|---:|
| Median whole-gate agreement | 0.816 | 0.762 |
| Median retained-set Jaccard | 0.276 | 0.389 |
| Consensus-stable cell-occurrence fraction | 0.569 | 0.584 |

All fits converged without detected cycles, but convergence did not imply a unique gate. Thus, a single M4-E retained/rejected partition is not currently initialization-independent.

### 5.4 Other diagnostic analyses

The following independent analyses were implemented or run:

- Fixed rejection cost sensitivity at `c = 0.4, 0.5, 0.6, 0.7`.
- A second OT analysis restricted to cells rejected by the first source-cap-0.85 OT.
- One-pair-per-patient analysis, selecting the exact pair with the largest value of `min(source malignant cells, target malignant cells)` for every patient.
- Pre-QC versus post-QC comparison.
- Pair-level QC and library-complexity checks.
- Source-cap-specific pseudobulk DEG and pathway analyses.

The second-OT result contained many positive DEGs but weak patient-direction consistency for most leading genes. It is therefore exploratory and does not establish that first-round rejected cells form a coherent metastatic subpopulation.

### 5.5 Source-only malignant OT

A newer analysis removed target-side rejection and retained only the primary/source gate. Each dataset and each exact primary-metastasis pair were fitted independently with:

- Source = primary malignant cells.
- Target = metastatic malignant cells.
- Source rejection cap = 0.85.
- Target rejection cap = 0.00, so all metastatic target cells remain available as the reference distribution.
- Production M4-E initialization = all cells retained on both sides.

Completed source-only results were:

| Dataset | Exact pairs | Source weighted rejection rate | Target rejection rate |
|---|---:|---:|---:|
| GSE180661 | 94 | 0.814 | 0.000 |
| GSE181919 | 4 | 0.844 | 0.000 |
| GSE225857 | 5 | 0.652 | 0.000 |

These rates describe fitted compatibility gates, not measured metastatic probabilities. The high rejected fractions also show that the source cap is permissive and that biological interpretation must be supported by independent expression or lineage evidence.

### 5.6 Metastasis-signature UCell validation

An internal expression-based validation was completed for all three source-only datasets. For each dataset independently:

1. Author-labelled malignant cells actually analyzed by OT were selected.
2. Raw integer counts were summed into one primary and one metastatic pseudobulk per patient. Multiple primary sites and multiple metastatic sites were collapsed within their respective patient-side pseudobulk.
3. PyDESeq2 fitted `~ patient_id + side`, with positive log2 fold change defined as metastasis-enriched.
4. Positive genes were ranked by the PyDESeq2 Wald statistic, without excluding OT-representation genes, to form Top-50, Top-100, and Top-200 metastasis signatures.
5. Official pyUCell 0.7.3 scored each primary malignant cell from within-cell expression ranks. No KNN smoothing, subtype stratification, or batch correction was applied.
6. UCell scores were merged with exact-pair source-only M4-E retained/rejected gates. The primary inference unit was the patient median across exact pairs; cell-occurrence plots were retained only as descriptive figures.

Patient-level results were:

| Dataset | Signature | Patients | Retained median | Rejected median | Median retained - rejected | Paired Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| GSE180661 | Top 50 | 29 | 0.06640 | 0.05997 | 0.00627 | 1.86e-08 |
| GSE180661 | Top 100 | 29 | 0.06012 | 0.05578 | 0.00420 | 2.00e-06 |
| GSE180661 | Top 200 | 29 | 0.07317 | 0.06777 | 0.00463 | 6.30e-07 |
| GSE181919 | Top 50 | 4 | 0.02349 | 0.02505 | -0.00986 | 0.375 |
| GSE181919 | Top 100 | 4 | 0.01595 | 0.01943 | -0.00865 | 0.375 |
| GSE181919 | Top 200 | 4 | 0.02080 | 0.02681 | -0.00732 | 0.375 |
| GSE225857 | Top 50 | 5 | 0.00000 | 0.00002 | 0.00000 | 1.000 |
| GSE225857 | Top 100 | 5 | 0.00666 | 0.00626 | 0.00018 | 0.188 |
| GSE225857 | Top 200 | 5 | 0.01466 | 0.01189 | 0.00075 | 0.188 |

GSE180661 therefore shows a statistically consistent but small enrichment of metastasis-derived expression programs in source-retained primary cells. GSE181919 trends in the opposite direction without significance, and GSE225857 shows only weak, non-significant positive differences. The result supports the gate direction modestly in GSE180661 but does not establish a robust cross-dataset separation or a cell-level classifier.

This is internal validation because each dataset supplies both its DEG-derived signature and its gate assessment. It also contains a scale mismatch: the signature is patient-level and site-collapsed, whereas the gate is exact-pair-specific. Leave-one-patient-out and exact-pair-aware analyses remain stronger future tests.

Two readings above have since been corrected.

The GSE180661 result is a depth artefact rather than modest support for the gate direction. Retained and rejected primary cells in that dataset differ in median depth by 1.32x (13,773 against 10,433 counts). UCell ranks genes within each cell, which makes it robust to depth but not invariant to it: a shallow cell detects fewer genes, so a signature's genes reach `maxRank` less often and its score falls. That mechanism produces exactly the observed pattern, a difference of about 0.005 that is nonetheless significant at 1.9e-08, because a systematic artefact is small but perfectly consistent.

The GSE225857 Top-50 medians of 0.00000 in both groups are not a biological negative. Rejected cells there detect a median of 1,689 genes, so most signature genes are simply undetected and the score is zero by construction rather than by absence of the program. That row should not be read as failed replication.

### 5.7 Gate specificity: what the gate was measuring, and what changed

#### 5.7.1 What the stored gates separate

Two diagnostics were written against completed runs and require no refitting:
`cancer_metastasis/25_diagnose_gate_covariates.py` and
`cancer_metastasis/26_diagnose_pairing_quality.py`.

The transport solver itself is faithful. The gate is a clean threshold on
`decision_cost`: its AUC against the retained set is exactly 0.000, meaning
perfect separation, and the gate agrees with the calibrated rule
`decision_cost < rejection_cost` on 99.7% of cells. Two earlier hypotheses were
tested and withdrawn: the rejection-budget floor contributes at most 3.8%
of the retained set on real data, and exactly nothing in GSE225857, and the unbalanced-transport mass term does
not reorder the gate.

`decision_cost` is, however, substantially a sequencing-depth readout.

| Statistic | GSE180661 | GSE181919 | GSE225857 |
|---|---:|---:|---:|
| rho(decision_cost, total_counts) | -0.330 | -0.593 | -0.568 |
| AUC(total_counts to retained) | 0.673 | 0.767 | 0.791 |
| Median depth, retained | 13,773 | 37,447 | 13,002 |
| Median depth, rejected | 10,433 | 18,982 | 4,243 |
| Depth ratio | 1.32x | 1.97x | 3.07x |

The direction is the same in all three datasets: retained cells are the more
deeply sequenced ones. Removing the depth component from the cost changes 25%
to 36% of the retained set, a Jaccard of 0.47 to 0.60 against a chance overlap
of 0.08 to 0.18.

The transport plan is depth-stratified as well, not only the gate. On
GSE180661, a primary cell's reciprocal-dominant metastatic partner has
correlated depth, rho = 0.64; the nearest-neighbour version is 0.59 and holds
within the retained and rejected subsets alike. This bears on the Figure 2A
cell-type transition matrix, because cell types differ systematically in
library size, so that validation needs its own check.

Two mechanisms contribute about equally, which matters because their remedies
differ. Depth occupies a substantial single component
(`max_abs_spearman_pc_depth` = 0.60) and also drives dispersion
(rho(depth, nearest-neighbour distance) = -0.41). Regressing one component out
therefore cannot suffice.

#### 5.7.2 Two defects established against ground truth

`scripts/validate_depth_null_specificity.py` builds data whose correct answer
is a construction rather than a simulator parameter. Its first version was
wrong and its numbers should not be quoted: `simulate_counts` drew a separate
gene-mean vector per side, so the homogeneous arms gave the two sides unrelated
expression profiles and rejecting them was correct. The symptom was a
within-side null median cost 37x below the observed median. The vector is now
shared and every arm was re-measured.

**Defect 1, the rejection rate carried no information.** With one homogeneous
population, uniform depth, and source and target from the same generative
distribution, so that no incompatible cell exists, the method rejected 84.7%,
exactly the rejection budget. The arm containing a genuine 20% incompatible
subpopulation also rejected 84.7%. The rate could not distinguish the two.

With the budget reported rather than enforced, the underlying severity is
worse: at production parameters the same homogeneous population is rejected
entirely, 1.000, because the calibrated rule retains nothing. The 84.7% figure
was the cardinality floor holding the rate at the cap, not the rule agreeing
with it.

**Defect 2, the rejected identity is depth-driven.** With no biological
difference at all, depth spread alone reproduced the real-data signature at
300 cells per side: AUC 0.64 to 0.68 against 0.67 to 0.79 observed, rho -0.35
to -0.62 against -0.33 to -0.59 observed, and a depth ratio of 1.75x against
1.3x to 3.1x observed.

At production parameters, and after Defect 1 was fixed, it is far more severe:
AUC reaches 0.995 and rho -0.959 at a depth spread of sigma 0.9. Section 5.7.4
gives the series. This defect is not addressed by either change made to the
calibration, and only the depth-equalisation step can reach it.

#### 5.7.3 Cause and changes

The rotation null rotates the target cloud about its own centroid, which
preserves every cell's distance to that centroid. Radial position is exactly
what the gate thresholds, so this null cannot separate a homogeneous cloud from
a real correspondence, and it placed the rejection cost below the median
observed cost. Every rejection rate then equalled the budget, which is why the
cap looked arbitrary: any cap produced a rate equal to itself.

Three changes followed, each validated only against simulation ground truth.

1. **Within-side split null**, `confidenceot.within_side_null_costs`, now the
   default. Two halves of one side contain no incompatible cells by
   construction and carry that side's own depth heterogeneity, so requiring
   that they be accepted absorbs both nuisances. The cost is the smallest one
   reaching 0.90 acceptance. Every call site states its semantics explicitly;
   the two `mouse_embryo` entry points stay pinned to the rotation null so
   their published numbers remain reproducible.
2. **The rejection budget is reported, not enforced.** With the cost calibrated
   against a null containing no incompatible cells the cardinality floor has
   nothing to protect, and enforcing it only let an arbitrary parameter set the
   answer. An all-rejected gate is now reachable, which is the verdict that
   nothing here matches rather than a state to clip away. A budget of exactly
   zero stays binding either way, because rejecting at most none is a statement
   of intent; the source-only design depends on this for its target cap of
   0.00.
3. **Depth is equalised outside the algorithm**,
   `cancer_metastasis/27_downsample_counts.py`. It subsamples every cell's
   reads to one shared depth and writes new h5ads with a rewritten manifest, so
   the existing pipeline runs unchanged and the transport, pseudobulk,
   differential expression and UCell stages all see the same counts. Counts
   stay integers, which is what keeps the step decoupled: the pseudobulk stage
   rejects non-integer input, so Pearson residuals could only have lived inside
   the representation code and were dropped for that reason.

A separate correctness fix: the cardinality floor was inflated by floating
point. At the production budget of 0.85, `(1 - 0.85) * 1000` evaluates to
150.00000000000003, so a bare ceiling retained 151 cells instead of 150. The
CUDA path had been compensating with its own expression, which is why the two
devices could disagree by one cell; both now share a rounded helper.

#### 5.7.4 Measured effect of the changes

At production parameters, 1,500 cells per side with 2,000 HVGs and 30 PCs, with
the budget reported rather than enforced. One replicate for the null
comparison, three for the depth series.

**The rejection rate is fixed, and the choice of null is the whole difference.**

| Quantity | Rotation null | Within-side null | Truth |
|---|---:|---:|---:|
| Homogeneous rejection rate | 1.000 | 0.090 | ~0 |
| Homogeneous `sign_rule_retained_fraction` | 0.000 | 0.910 | ~1 |
| Calibrated rejection cost | 0.538 | 0.807 | - |
| Median decision cost | 1.012 | 0.696 | - |
| Perturbed rejection rate | 0.999 | 0.259 | 0.200 |
| Perturbed detection recall | 1.000 | 1.000 | - |
| Perturbed detection precision | 0.200 | 0.773 | - |
| Perturbed detection F1 | 0.334 | 0.872 | - |
| Wall time, two arms | 13m19s | 3m50s | - |

The rotation null rejects every cell of a population that contains nothing to
reject. Its `sign_rule_retained_fraction` of 0.000 says the calibrated rule
wanted to reject all of them, so the 84.7% recorded before was the cardinality
floor holding the rate at the cap rather than the rule agreeing with it.
Removing the floor is what made the severity visible, which is also why the
floor could only be removed after the cost was recalibrated. Its precision of
0.200 equals the perturbed fraction exactly, which is the arithmetic signature
of a gate carrying no information: reject everything and precision necessarily
falls to the base rate while recall is trivially 1.000.

The within-side null retains 0.910 of a homogeneous population against an
acceptance requirement of 0.90, so the calibration lands where it was designed
to. On the perturbed arm it recovers every one of the 300 genuinely
incompatible cells and adds 88 false positives out of 1,200 compatible cells,
and `auc_perturbed_rejected` is 0.997.

**The rate is now insensitive to depth. The identity is not.**

Three replicates across the depth series, medians:

| Depth sigma | Rejection rate | `sign_rule_retained` | Rejection cost | AUC(depth) | rho(cost, depth) | Depth ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.088 | 0.912 | 0.806 | 0.500 | - | 1.00x |
| 0.3 | 0.093 | 0.907 | 0.911 | 0.949 | -0.787 | 1.63x |
| 0.6 | 0.091 | 0.909 | 1.168 | 0.990 | -0.928 | 2.97x |
| 0.9 | 0.091 | 0.909 | 1.411 | 0.995 | -0.959 | 4.71x |

None of these arms contains a biological difference of any kind. Depth spread
alone moves the rejection rate not at all, because the within-side null raises
the rejection cost in step with the within-side spread and so absorbs the
heterogeneity into the threshold. That is the null doing its job.

It does not, and cannot, fix which cells are rejected. An AUC of 0.995 means
the rejected 9% are very nearly the shallowest 9%. The threshold decides how
many cells are rejected; the cost geometry decides which, and only a change of
geometry can affect that. Fixing Defect 1 therefore made Defect 2 maximally
visible rather than reducing it: with the cardinality floor gone, the rejected
set is the extreme tail, and under pure depth heterogeneity the extreme tail is
the shallow tail.

These simulated AUCs must not be compared directly against the 0.67 to 0.79
measured on cancer data. Those gates were fitted under the rotation null with
the budget enforced, a different regime entirely. What the cancer gates look
like after refitting is unknown until they are refitted.

At 300 cells per side with the budget still enforced, the same comparison gave
0.847 against 0.063 on the homogeneous arm and an F1 of 0.382 against 0.775,
and the depth series read 0.64 to 0.68 rather than 0.95 to 0.995. The direction
holds at both scales; the production figures are the ones to quote.

#### 5.7.5 What this does not yet establish

Nothing here has been re-run on cancer data. Every gate, differential
expression result, pathway result and UCell score in sections 3, 5 and 6 was
produced under the rotation null with an enforced cap and without depth
correction. They should be treated as superseded rather than as results
awaiting confirmation.

The cross-patient mismatched control the project still lacks remains the
decisive test: patient A's primary against patient B's metastasis, run through
an unmodified pipeline. Matched pairs must separate from mismatched pairs, and
the keratinisation and SPRR signal that appears in the rejected group of all
three datasets must disappear. Section 5.1's standard applies: the
configuration is fixed on simulation results, and the differential expression
is run once afterwards.

### 5.8 What the retained fraction measures, established on two cancers

Everything in this section was run on 2026-09-15 and 2026-09-16 after the
changes in 5.7. Each acceptance criterion was written into a script docstring
or into `cancer_metastasis/DEG_PRESPECIFICATION_2026-09-15.md` and committed
before the numbers existed.

#### 5.8.1 The cross-patient control the project lacked

`cancer_metastasis/28_build_mismatched_manifest.py` pairs each primary with a
different patient's metastasis by drawing from the pooled lesions of every
other patient, choosing the donor closest in cell count so that patient
identity is the only thing that varies. `29_compare_matched_mismatched.py`
reads the two runs cell by cell.

| Quantity | Matched | Mismatched |
|---|---:|---:|
| `retained_fraction`, median of 94 pairs | 0.351 | 0.000 |
| `rejection_cost` | 0.589 | 0.390 |
| Median `decision_cost`, rejected | 1.172 | 1.017 |

The paired shift is a signed-rank p of 8.1e-17, and `spearman_decision_cost`
between the two runs over the same primary cells is 0.378, so two thirds of the
cost ordering is lost when the partner changes. The gate does use the
metastatic side.

`gate_jaccard` is 0.000 in both arms and its test returns p = 0.585. That is
degenerate rather than informative: the mismatched arm retains nothing, so
there is no set to overlap. The prespecified reading anticipated the opposite
failure, a Jaccard near 1.0, and the retained-fraction shift is the statistic
that carries the result.

Only 26 of 94 mismatched pairs retain any cell at all, and those cells are
near-empty droplets: `auc_predownsample_total_counts` 0.065, 795 detected genes
against 1,357 in the rejected set, 0.92% mitochondrial reads against 12.6%.
The direction is the reverse of the matched arm, where retained cells are the
deeper ones.

#### 5.8.2 The retained fraction is a similarity readout

The mechanism predicts this. The threshold is calibrated by splitting one side
against itself and dividing by the median cross-side distance, and the observed
cost is divided by the same quantity, so

    c ~ within-side heterogeneity / cross-side separation

and `cost < c` reduces, after the scale cancels, to

    d(primary cell, metastasis) < typical within-primary distance

The denominator is a similarity, so the retained fraction should track how
alike the two sides are. `30_validate_similarity_readout.py` tests that against
pseudobulk correlation, which involves no PCA, no transport and no gene
selection by the solver.

| Arm | raw rho | partial rho, cell count held | p |
|---|---:|---:|---:|
| Union, 188 pairs | 0.671 | **0.795** | 4.9e-42 |
| Union, fixed HV basis | 0.757 | **0.826** | 6.7e-48 |
| Matched only, 94 pairs | 0.445 | **0.767** | 2.9e-19 |
| Mismatched only | -0.225 | -0.026 | 0.80 |

The prespecified disqualifier fired on its literal wording: pseudobulk
correlation rises with cell count at rho 0.701, larger than the result itself.
The wording was wrong. Cell count predicts the retained fraction at -0.058
(p = 0.43), so the path through it contributes about -0.10, negative where the
observed relationship is positive. Controlling it raises the matched estimate
from 0.445 to 0.767 rather than lowering it: the confound was suppressing the
relationship. The criterion was replaced with the partial correlation, which is
the correct test, rather than argued away.

Lesion type does not explain it. The 17 ascitic pairs have a median retained
fraction of 0.021 against 0.403 for solid lesions, which is itself coherent
with detached, anchorage-independent ascitic cells. Holding ascites as a
covariate moves the estimate from 0.792 to 0.757, and the 77 solid-lesion pairs
alone give 0.785 (p = 4.6e-17). The gradient is continuous, not a proxy for
ascites.

The relationship holds **within the matched arm alone**, so it is not two
clusters joined by a line.

#### 5.8.3 The prostate dataset inverts the answer

GSE271675 is a 10x Multiome series with matched primary prostate tumours and
lymph-node metastases, re-annotated and integrated by Faming Zhao. His own
paired pseudobulk DESeq2 and Hallmark GSEA report reduced androgen response and
increased EMT and interferon in the nodes, and increased proliferation in the
primaries -- that is, the two sides genuinely diverge. GSE180661's two sides do
not: their pseudobulk correlation is 0.941 and
`metastasis_all_vs_primary_all` yields **zero** genes at FDR < 0.05 with
|log2FC| >= 1.

`34_prepare_prostate_pairs.py` writes one slim H5AD per specimen, drops the
2,923 immune, stromal and `Unknown` cells, names the largest node per patient
so each primary cell is gated once, and emits 24 pairs. Patient3 has 11,406
primary cells and no node cells, so four patients contribute. Depth was
equalised to 1,556 counts, the 10th percentile of a pooled median of 4,905.

| | GSE180661 | GSE271675 |
|---|---:|---:|
| Two sides alike? | pseudobulk rho 0.941, 0 DEGs | AR, EMT, IFN all differ |
| `retained_fraction`, median | 0.351 | **0.0087** |
| `forced_in_share_of_retained` | 0.000 | 0.000 |
| `sign_rule_concordance` | 1.000 | 1.000 |

**On the dataset where there is metastatic divergence to study, the gate
retains under one per cent of primary cells.** The solver is faithful in both:
nothing is forced, and the gate is exactly its calibrated rule.

The residual retained set is the low-content tail, the same signature as the
ovarian mismatched arm:

| Statistic | Retained | Rejected |
|---|---:|---:|
| Median pre-downsampling depth | 1,464 | 4,767 |
| Median detected genes | 872 | 981 |
| Median mitochondrial % | 0.644 | 0.193 |
| `auc_predownsample_total_counts` | 0.189 | - |
| `auc_n_genes_by_counts` | 0.146 | - |
| `auc_pct_counts_mitochondrial` | 0.818 | - |
| `median_downsample_untouched` | 1.000 | 0.000 |

`depth_residual_gate_jaccard` is 0.091 against a chance 0.004, so removing the
depth component replaces almost the whole retained set. The median retained
cell was never subsampled, meaning it already sat below 1,556 counts.

The differential expression is not interpretable. The 20-cell minimum leaves
three patients, six pseudobulk samples and two residual degrees of freedom
under `~patient_id + comparison_status`; two genes reach FDR < 0.05. That is
what a near-empty retained set produces, not a finding.

#### 5.8.4 Statement of the defect, and a fix to evaluate

The criterion asks whether a primary cell sits closer to the metastasis than
two halves of the primary sit to each other. Its denominator is the primary's
own internal heterogeneity, which has no relation to metastatic competence, and
its threshold is normalised by the cross-side separation. The retained fraction
is therefore jointly determined by within-side heterogeneity and by
between-side separation, and is close to inversely proportional to the latter.
A homogeneous primary rejects nearly everything however near its metastasis
lies; a pair of divergent tissues rejects nearly everything because the
denominator dominates. The interesting biological case is the one the rule
cannot answer.

Two things are consequently **not** supported. The retained fraction is not an
estimate of how many primary cells are metastasis-competent: it spans 0.000 to
0.879 across GSE180661 patients and falls to 0.0087 on GSE271675 without any
claim about those tumours' biology changing. And `retained` is not a property of
a cell: naming a different lesion for SPECTRUM-OV-003 moved its retained
fraction from 0.420 to 0.032 over the same 1,332 cells, and 64% of one
patient's primary cells carry different labels against different lesions of
that same patient. Retention is a property of a (cell, metastasis) pair.

What **is** supported is narrower and real. Within one patient, the retained
cells are the ones nearer that metastasis, which is what the rule computes and
is a defensible reading. Across pairs, the retained fraction measures
primary--metastasis transcriptional similarity, at partial rho 0.79 against a
measure the algorithm never sees.

The fix to evaluate, not yet implemented, is to build the null on the source
cells' own distances to the same target rather than on the source's internal
spread: ask where a cell's distance to the metastasis falls among all source
cells' distances to that metastasis. The cross-side separation then appears in
both the observed statistic and the null and cancels, so the gate stops
drifting with overall tissue similarity. The cost is that the rejected fraction
becomes a pure ranking and must be set by something else -- which is honest,
since the present fraction was never the incompatible-cell prevalence it was
read as. Two datasets now bracket the test: a working rule should move
GSE180661 down from 0.351 and GSE271675 up from 0.0087, each towards its own
structure.

A deflationary reading has to be answered separately. If pseudobulk correlation
recovers 0.79 of the retained fraction, the transport's added value needs its
own demonstration: what does the gate predict that
`spearmanr(sum(A), sum(B))` does not?

#### 5.8.5 Cell identity, and two methodological corrections

GSE180661's major cell types come from CellAssign over nine categories --
B.cell, Dendritic.cell, Endothelial.cell, Fibroblast, Myeloid.cell,
Ovarian.cancer.cell, Plasma.cell, T.cell, Mast.cell -- and **none of them is
normal epithelium**. The `Ovarian.cancer.cell` markers are epithelial rather
than malignant: WFDC2, CD24, CLDN3, KRT7/8/17/18/19, EPCAM, WT1, CLDN4, MSLN,
FOLR1, MUC1. GEO records the primary sites as "adnexa (ovary and fallopian
tube)", and the authors' own sub-clustering of these cells contains
`Ciliated.cell.1` and `Ciliated.cell.2`. Ciliated tube epithelium therefore
carries the malignant label, and the `cell_subtype` obs column -- which this
project had not read -- holds the finer assignment.

It does not explain the gate. Within patients, only two subtypes shift
significantly: `Cancer.cell.6`, the hypoxic cluster, at +0.0222 (16/24 patients,
p = 0.019), and `Ciliated.cell.1` at +0.0067 (18/24, p = 0.0087), the most
consistent direction in the table but 0.67 of a percentage point. Ciliated cells
are about 3% of the rejected set. The GSE180661 differential expression is
therefore not a cell-type composition artefact: `Cancer.cell.1` differs by 14.7
percentage points but none of its markers -- DAPL1, SCGB2A1, CRABP1 -- appears
among the 40 genes.

Two corrections worth keeping:

**Pooled cell composition misleads.** Three conclusions drawn from pooled
cross-tabulations on 2026-09-15 did not survive the within-patient paired test:
that ciliated cells were not enriched among rejected cells, that unassigned
cells were enriched 3.0-fold, and that `Cancer.cell.3` was enriched among
retained cells. Only the last is a real Simpson effect -- within patients it is
+0.0139, 15/24, matching the differential expression rather than the pooled
count. The pair or the patient is the unit; pooled cell counts are not.

**`"NA"` in `cell_subtype` is a category, not a missing value.** It is the
authors' label for cells their clustering left unassigned. `pd.read_csv` maps
it to NaN by default, which silently zeroed a test of exactly that category.
Any read of that column needs `keep_default_na=False`.

## 6. Replication datasets

Independent author-labelled malignant-cell workflows were created for:

- GSE181919: author label `Malignant.cells`.
- GSE225857: author tumour clusters `Tu01` through `Tu11`.

Each dataset is analyzed independently and is never pooled with GSE180661 for OT or DEG. The workflow applies the same post-QC thresholds, 2,000 HVGs, 30 PCs, source cap 0.85, target cap 0.95, exact-pair pseudobulk, paired PyDESeq2, and GSEApy.

The later source-only comparison uses target cap 0.00 rather than 0.95 and must be treated as a separate analysis. In the joint malignant UMAP summaries, GSE181919 contained 526 post-QC primary and 284 metastatic malignant cells from four patients; GSE225857 contained 9,585 primary and 13,987 metastatic malignant cells from five patients. GSE181919 had only one broad malignant label, whereas GSE225857 contained 11 author tumour subtypes.

Preliminary inspection suggests strong dataset-specific effects. Normal epithelial or goblet-like annotation can create apparently strong matching unrelated to metastatic lineage, particularly when malignant labels are broad. Replication results must therefore be presented with annotation audits and dataset-specific directions rather than as a single pooled effect.

## 7. TACC execution model

### 7.1 Cluster resources

- Allocation: `MCB26031`.
- `gh`: standard GPU work; avoided for the newest small sensitivity jobs when queue behavior was unfavorable.
- `gh-dev`: short GPU analyses and initialization sensitivity.
- `gg`: CPU analyses, pseudobulk, DEG, GSEA, plotting, dependency installation, and the current Splatter scaling benchmark.

The workflow uses Slurm dependencies so that audit/manifest, OT, pseudobulk, DEG, GSEA, and finalization stages start only after their required upstream stages complete successfully. A `DependencyNeverSatisfied` state means an upstream job failed or was cancelled and the downstream job must be cancelled and resubmitted after fixing the cause.

### 7.2 Main TACC result locations

- GSE180661 malignant OT and biological analysis: under `confidenceot_results`.
- Post-QC source-cap-0.85 OT: `gse180661_sourcecap_0p85_postqc_ot_20260908/source_cap_0p85`.
- Initialization sensitivity: `gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_rerun_20260910`.
- Initialization summary: `gse180661_m4e_initialization_sensitivity_sourcecap_0p85_postqc_summary_rerun_20260910`.
- Pan-cancer all-cell OT: `pancancer_all_cell_sourcecap_0p85_targetcap_0p95_20260902`.
- Splatter scaling benchmark: `/scratch/10119/ghzheng/OT_project/benchmark_scaling`.
- Three-dataset source-only OT: `source_only_GSE180661_GSE181919_GSE225857_20260913`.
- UCell gate validation: `primary_gate_ucell_validation_GSE180661_GSE181919_GSE225857_v3_20260914`.
- Dedicated pyUCell environment: `/scratch/10119/ghzheng/conda_envs/pyucell_073`.
- Gate covariate diagnostics: `gate_covariate_diagnostic_sourceonly_v3_20260914` and `gate_covariate_diagnostic_bidirectional_20260914`.
- Transport pairing diagnostics: `pairing_quality_GSE180661_20260914`.
- Specificity validation against ground truth: `depth_null_prod_probe` and `depth_null_prod_rotation`.
- Depth-equalised counts and their rewritten manifest: `downsampled_GSE180661_20260914`.

## 8. Splatter simulation benchmark

### 8.1 Purpose

The simulation benchmark measures population-level anomaly detection, directional rejection accuracy, robustness to batch and gene perturbation, and single-fit OT runtime as the number of cells increases.

### 8.2 Simulation grid

| Component | Values |
|---|---|
| Cell counts | 1,000; 5,000; 10,000; 20,000 |
| Batch condition | none; mild |
| Target gene perturbation | 0%; 5%; 10%; 20% |
| Perturbation magnitude | two-fold signed up/down perturbation |
| Scenarios | S1 extinction; S2 emergence; S3 source outlier; S4 bifurcation; S5 abundance shift |
| Replicates | 2 |

This gives 80 M4 cases: `4 sizes x 2 batches x 5 scenarios x 2 replicates`. Partial OT is run for the 20 `N=1000` comparator cases.

### 8.3 Simulation preprocessing

Every simulated pair uses:

`raw counts -> library-size normalization to 10,000 -> log1p -> top 500 HVGs -> gene-wise standardization -> joint PCA with 20 PCs -> squared Euclidean cost -> median-positive cost scaling -> OT`

The joint PCA is fitted once on the clean source-plus-target counts and then frozen. Every perturbed dataset is projected through the same HVGs, scaler, and PCA. Thus, perturbation doses and methods are compared in exactly the same representation and cost geometry.

### 8.4 Compared methods

- Traditional balanced OT.
- Vanilla UOT mass-deficit baseline.
- Fixed-cost M4-E and M4-R with balanced and UOT backbones.
- Null-calibrated M4-E and M4-R with balanced and UOT backbones.
- Partial OT comparator.

The current scaling runs use source and target rejection budgets of 0.15, fixed binary rejection cost `c = 0.5`, entropy regularization `epsilon = 0.1`, UOT marginal penalties `lambda_a = lambda_b = 1.0`, two calibration nulls, and two held-out validation nulls.

The large-N jobs use `--skip-soft-gate`. Binary ConfidenceOT M4-E and M4-R are called without explicit initial gates, so both source and target begin from deterministic all-one gates. Escape-score and random initializations exist only in the skipped soft-gate diagnostic path and are not used for the reported binary scaling runs. Consequently, this benchmark evaluates the production all-cell-in initialization but does not itself measure initialization sensitivity.

### 8.5 Benchmark outputs

Planned result figures and tables include:

- Population rejection heatmaps by scenario.
- Directional F1 heatmaps across batch and perturbation settings.
- Single-fit runtime scaling across cell counts.
- Method- and size-level metric tables.
- A final audit JSON and downloadable result bundle.

The reported single-OT fit time excludes normalization, HVG selection, PCA, cost construction, null calibration, file I/O, plotting, and scheduler waiting. Preprocessing and calibration are recorded separately where applicable.

### 8.6 Current simulation status

- The original scaling submission did not run because `splatter` was unavailable.
- The TACC repository-local R library was then installed successfully.
- Installed simulation package: Splatter 1.34.0, together with the required Bioconductor dependencies.
- Installation job `994564` completed successfully on `gg`.
- The benchmark root is `/scratch/10119/ghzheng/OT_project/benchmark_scaling`.
- The first 32-thread submission completed all 20 `N=1000` cases but was stopped during `N=5000` because each GG allocation supplied a full 144-core node while the workflow used only 32 threads.
- The replacement configuration uses 144 requested CPU cores per case, nine independent case workers, and up to nine concurrent GG nodes. All sizes use the same requested resources so runtime comparisons are aligned.
- Single-fit runtime is elapsed wall-clock time measured with `time.perf_counter`; aggregate process CPU time is retained separately.
- The N=1,000, N=5,000, and N=10,000 job arrays have finished. N=20,000 job `994713` remains queued/running separately and is intentionally being allowed to finish.
- The old four-size finalization dependency `994714` is being cancelled so that a non-blocking interim report can be generated from the first three completed sizes. The interim report uses a separate hard-linked result snapshot and does not modify the original benchmark data or stop N=20,000.
- The planned interim location is `/scratch/10119/ghzheng/OT_project/benchmark_scaling_interim_n1000_5000_10000`.

Submission command:

```bash
cd /scratch/10119/ghzheng/OT_project/code/ConfidenceOT

conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot

export CONFIDENCEOT_SCALING_ROOT=/scratch/10119/ghzheng/OT_project/benchmark_scaling
export CONFIDENCEOT_WORKER_COUNT=9
export CONFIDENCEOT_MAX_PARALLEL=9

bash scripts/tacc/submit_splatter_scaling.sh
```

## 9. Current interpretation

1. ConfidenceOT produces a pair-specific partition and a continuous per-cell score, but the rejection *rate* has never estimated the incompatible fraction, under either calibration. Before 2026-09-14 it carried no information at all: simulated data with no incompatible cells and simulated data with a genuine 20% incompatible subpopulation both rejected 84.7%, the rejection budget. The within-side null and the unenforced budget fixed that on simulated data, where the homogeneous arm falls to 0.090 and the perturbed arm recovers every incompatible cell. On real data the repaired rate turns out to measure something else: primary--metastasis transcriptional similarity, at partial rho 0.79 against an independent pseudobulk measure, 0.351 on GSE180661 whose two sides correlate at 0.941 and 0.0087 on GSE271675 whose two sides genuinely diverge. Section 5.8.4 states the defect and the change to evaluate. The simulation did not catch this because both of its sides are drawn from one generative process, so cross-side separation never departs far from within-side spread.
2. The all-cell transition matrix is not yet established as a structural validation. The transport plan pairs cells by sequencing depth at rho = 0.64, and cell types differ systematically in library size, so a diagonal-looking matrix could arise without any biological correspondence. This needs its own check before Figure 2A can carry the validation argument.
3. Malignant-only gates identify expression differences, but the current direction is not equivalent to metastatic potential.
4. The inflammatory and invasive signals in source-rejected cells are real observations about the fitted partition, and the labels were never reversed. The partition itself, however, separates cells by sequencing depth, and the groups differ in depth by 1.3x to 3.1x. A pseudobulk built from shallower cells is relatively enriched for high-abundance transcripts, which is the most likely reading of the keratinisation and SPRR signal that appears on the rejected side of all three datasets.
5. Post-QC filtering does not by itself resolve the biological-direction issue.
6. Rejection specificity splits into two parts, and only one of them is fixed. The rejection *rate* is repaired: on simulated data containing no incompatible cells the rotation null rejected every one of them at production parameters, and the within-side null brings the same case to 0.090 while recovering all 300 genuinely incompatible cells in the positive control. The rejected *identity* is not: with the rate no longer pinned by the cap, depth spread alone determines which cells are rejected almost completely, AUC 0.995 at sigma 0.9 in data with no biological difference. A threshold can absorb depth; only the cost geometry decides the ranking. This now sits ahead of M4-E gate initialization sensitivity as the leading limitation.
7. M4-R frequently exhausts its outer loop, which is expected: its gate has no monotone-objective guarantee. That is not evidence against the rejection cost, which is fitted with M4-E, and the earlier reading of `calibration_valid` conflated the two. M4-R remains a cross-check rather than the basis of the primary claim.
8. Dataset annotation and epithelial-state contamination are major possible confounders.
9. The GSE180661 UCell result is explainable by a sequencing-depth difference and is not evidence for two metastatic-potential states. Retained and rejected primary cells there differ in median depth by 1.32x. UCell ranks within a cell, which is depth-robust but not depth-invariant: shallow cells detect fewer genes, so signature genes reach `maxRank` less often. A tiny effect that is nonetheless overwhelmingly significant is the signature of a systematic artefact, not of a small biological difference.
10. The cross-patient mismatched control has now been run and passed: matched pairs retain 0.351 and mismatched pairs 0.000, signed-rank p = 8.1e-17, and only a third of the cost ordering survives the swap. The partition does use the metastatic side and is patient-specific. That was the gate on any biological claim, and it is cleared. What replaces it as the blocking question is 5.8.4: the retained fraction is close to inversely proportional to how far apart the two sides are, so the method returns almost nothing exactly when there is divergence to study. No biological claim should rest on a rate with that property. Distinguishing patient A from patient B in HGSOC is easy on clonal grounds, so passing the mismatched control is necessary and not sufficient; the control that would bite is same-patient primary against primary, for which 16 of 29 GSE180661 patients have two primary samples.

## 10. Immediate next steps

Ordered so that nothing downstream is run before the thing it depends on is
established. Section 5.7 supersedes the earlier ordering, in which items 1 to 3
came first.

1. Finish the production-parameter specificity run for all five arms and three
   replicates, under both `--calibration-null` values. The homogeneous arms
   must reject near zero and `perturbed_f1` must not collapse. A collapse would
   mean the within-side null is too permissive and the decision statistic needs
   a contrast form rather than a new threshold.
2. Run `27_downsample_counts.py` and read only `downsample_report.json` first.
   `target_depth` and `cells_already_at_or_below_target_fraction` decide
   whether depth correction by subsampling is viable at all; GSE225857's
   rejected cells sit at a median of 4,243 counts, so a global low quantile may
   discard more than is acceptable.
3. Confirm CPU and CUDA agree. The floor is now a shared helper so parity is
   structural, but `tests/test_torch_kernel_matches_numpy_reference_on_cpu`
   needs a run in an environment that has PyTorch.
4. Re-run the Splatter benchmark under the new null into a fresh
   `CONFIDENCEOT_SCALING_ROOT`, keeping the old outputs for comparison. A large
   jump in valid calibrations would itself be evidence that the old null caused
   the 3-of-252 figure.
5. Only then refit the cancer pairs, with the configuration fixed on items 1
   and 2 alone. Write down the configuration, the comparison and a minimum
   effect size before looking at any differential expression, then run once.
6. Run the cross-patient mismatched control described in 5.7.5. This is the
   control the project has never had, and no differential expression result
   should be presented without it.
7. Re-run `25_diagnose_gate_covariates.py` with `--predownsample-depth` and
   `26_diagnose_pairing_quality.py` against the new output. Acceptance:
   `auc_predownsample_total_counts` near 0.5,
   `spearman_decision_cost_predownsample_total_counts` near 0, and
   `depth_residual_gate_jaccard` near 1.
8. Check whether the Figure 2A cell-type transition matrix survives the same
   scrutiny. Cell types differ systematically in library size, and the
   transport plan pairs cells by depth at rho = 0.64, so a diagonal-looking
   matrix could arise without any biology.
9. Use neutral labels (`source-retained`, `source-rejected`, or `compatible`,
   `restricted`) until independent evidence supports a metastatic-potential
   interpretation.
10. Reserve TCGA survival analysis for a stable, prespecified signature that
    survives the robustness checks above.

Carried over from the earlier list and still open, but no longer ahead of the
items above: the ground-truth initialization benchmark comparing all-one with
deterministic and random feasible starts; patient-level UCell effect-size
summaries with leave-one-patient-out or exact-pair-aware signatures; and
patient-aware or one-pair-per-patient quantification of repeated-sample
influence. Each of these measures a property of gates that section 5.7 shows
were not measuring compatibility, so each should be repeated after item 5
rather than interpreted from the existing runs.
