# KP-Tracer tumour level: available representations, comparability and CNV feasibility (2026-10-15)

**Sources (local, verified copies of the TACC outputs):**
- `results/kp_tracer_lineage_validation/export/`: `readout_tumour_level/`, `config/ot_runs/<label>/<pair>/run.json`, `tables/cells.csv.gz`. All 489 files were checked against MD5SUMS.
- The exploratory tables are written by `kp_lineage/tumour_level/01_ranking_and_representations.py` to `results/kp_tracer_lineage_validation/tumour_level_exploratory/`.

**What this did not touch.** The running validation_v3 jobs (audit and coupling pilot) were neither touched nor read.

## 1. Which KP-Tracer runs exist

| Label | Completed pairs | Eligible pairs | Mice with pairs | Gate | Acceptance | Output |
|---|---|---|---|---|---|---|
| ranknm256_noscale_ds_cos | 104 | 104 | 11 | M4-E (M4-R also saved) | 0.99, within-side split null | `kp_tracer_confidenceot/ot/kptracer/ranknm256_noscale_ds_cos` |
| raw | 104 | 104 | 11 | M4-E (M4-R also saved) | 0.99, within-side split null | `kp_tracer_confidenceot/ot/kptracer/raw` |

**Nothing else exists for KP-Tracer.**
- `ot/kptracer/` holds these two labels only.
- The 32-arm preprocessing factorial and the cycling × CNV arms were run on the human datasets (GSE180661, GSE225857, GSE271675), never on KP-Tracer.
- No KP-Tracer run used `representation_source` cnv or rna_cnv; `run.json` records `cnv: null`.

**Same analysis population in both labels.**
- Same 131-row pair manifest, same 104 eligible pairs (at least 20 malignant cells per side), same malignant call (`not is_normal`, the authors' label), same MetFamily labels, same retained-fraction definition.
- Same informative lesions for the readout: 31 in 8 mice in both labels.
- The cell sets agree pair by pair, which was checked: all 208 pair-sides (104 pairs × source and target) hold identical cell IDs in the two labels. The raw arm applies inside `02_run_pair.py` the same QC that `27_downsample_counts.py` applies before equalising (1000 counts, 500 genes, 20% mitochondrial).
- The audit v3 (running) will report the same check as `raw_arm_cell_sets_identical`.

## 2. What each label does (from run.json and the preprocessing code)

| Step | raw | ranknm256_noscale_ds_cos |
|---|---|---|
| Input counts | original integer counts | each file equalised to one pooled depth, the 10th percentile (`27_`, binomial thinning) |
| Cell QC | in 02_run_pair.py: 1000 counts, 500 genes, 20% mito | in 27_, before equalising: the same thresholds |
| Per-cell transform | none ("values used as supplied"); no library-size normalisation | rank encoding: within each cell, the top 256 genes by count ranked 1, 1 − 1/256, ..., others 0 (no gene-median division) |
| HVG | 2,000, top variance of the raw values | 2,000, top variance of the rank encoding |
| Gene scaling | z-score before PCA | none (PCA centres only) |
| PCA | 30, joint over both sides | 30, joint over both sides |
| Cost | squared Euclidean on PCs | cosine (unit-length PCs), squared Euclidean then 2 − 2cos |
| Cost scaling | ÷ median of random source-target pairs | the same |
| Gate | M4-E, ε 0.1, λ 1, acceptance 0.99, within-side split null | the same |

**Consequence.**
- The two labels differ in five settings at once: depth equalisation, per-cell transform, HVG basis, gene scaling and cost geometry.
- A difference between them cannot be attributed to any one step.
- A controlled one-step comparison needs new KP-Tracer runs (section 4).

## 3. CNV feasibility

**Released data.**
- `KPTracer-Data` (README and the 2026-10-14 inspection) has expression, trees, character matrices, metadata, fitness, expansions and plasticity scores.
- **No CNV profile is released.**
- The audit v3 inventory (running) will list every file, to confirm this.

**What the authors did** (Yang et al. 2022, STAR Methods):
- inferCNV on single-cell expression, per tumour;
- reference: normal lung cells identified by MULTI-seq deconvolution;
- cells with outlying CNV counts filtered.

It was used to corroborate subclonal structure. Expression-derived CNV is an inference from the same RNA, not a separate DNA measurement.

**Our pipeline** (human datasets: R inferCNV per patient → `44_export_cnv_cells.R` → `02_run_pair.py --representation-source cnv | rna_cnv`) has never been run on KP-Tracer. Running it would need:
1. A mouse gene-position file (mm10 or mm39) matching the KP-Tracer gene symbols. None is in the repository.
2. Reference cells per mouse: the KP-Tracer `is_normal` cells (about 29,759 in the processed h5ad) are available in every tumour h5ad.
3. inferCNV per mouse, all of that mouse's tumours in one run, so that primary and metastasis sit on one scale as in the human pipeline.
   - It must use expression only: no tree, intBC, lentiBC or MetFamily input. Then the family labels cannot leak into the CNV features.
   - Coverage must be checked: cells removed by inferCNV filtering, and barcodes shared between libraries, would have no profile.
4. Biological informativeness to check first: KP tumours carry relatively few large CNVs, and Yang et al. report CNVs for a subset of tumours. A CNV representation could be uninformative for many pairs.

**Status:** R4 (CNV-only) and R5 (RNA+CNV) are **not evaluated**. No result is shown for them.

## 4. Experiments needed and not yet run (none submitted)

| ID | What | Purpose | Approximate cost |
|---|---|---|---|
| X1 | KP-Tracer one-step RNA arms on the same 104 pairs: e.g. ranknm256_noscale_cos (no equalisation), ranknm256_noscale_ds (squared Euclidean), raw_ds_cos, logcpm_noscale_ds_cos | attribute the raw vs main difference to single steps | 104 pairs per arm; at the original timing (0.5–15 min per pair) about one 6-task array per arm |
| X2 | inferCNV per mouse (11 mice) with the `is_normal` reference, then `44_` export | CNV features | 11 R jobs; the human runs needed fewer threads for large units |
| X3 | cnv and rna_cnv arms on the 104 pairs, with the same gate settings | R4, R5 | as X1, per arm |

**Submission.** Each needs its own plan shown before submission, and must stay within the 40-job cap while validation_v3 runs.
