# ConfidenceOT_20261015_v3.pptx: KP-Tracer section, slide by slide

- **Deck:** `D:\Xia_lab\Progress_report\progress_report7\ConfidenceOT_20261015_v3.pptx`, 31 slides. v2 and the original 20261015 deck are unchanged.
- **Builder:** `cancer_metastasis/tools/build_deck_20261015_v3.py`, which reuses the drawing helpers of `build_deck_20261015_v2.py`.
- **Editability:** every schematic and heatmap is made of editable shapes; the charts are native PowerPoint charts; the explanations are in the notes.

| Slide | Title | Content | Data source |
|---|---|---|---|
| 20 | KP-Tracer: Independent Lineage Evidence | kept from v2: expression → ConfidenceOT; edits → inferred tree → family labels → reference; observed vs inferred | schematic |
| 21 | How CRISPR Lineage Tracing Works | recorder (intBC, 3 cut sites, 10–30 sites per ES clone, Cre-induced Cas9) → growth and metastasis → character matrix (0 unedited, – missing) → Cassiopeia-Hybrid tree; families as an author annotation | schematic; verified against Yang et al. 2022 STAR Methods (PMC9452598) |
| 22 | Which Primary Matches Each Metastasis? | candidate-ranking view of the design: per-pair runs, r(P, M), several positives allowed, d_M as the prespecified summary | schematic, values illustrative |
| 23 | Retained Fractions Reveal Candidate Primary Matches | heatmap of every completed pair, 11 mice: r on one 0–1 scale; outline = same family; hatched = not run (< 20 malignant cells); grey lesions lack both family groups | `export/readout_tumour_level/pairs_ranknm256_noscale_ds_cos.csv` |
| 24 | Tumor-Level Concordance Across Lesions | v2 (retitled): 31 lesion d_M, 27/3/1, median +0.086, mean +0.144, Wilcoxon p = 4.99e-7 | `metastasis_contrast_source_ranknm256_noscale_ds_cos.csv`, `readout_summary.json` |
| 25 | Tumor-Level Consistency Across Mice | v2 (retitled): lesions by mouse with medians; 8/8; sign test 0.0039 (exploratory); raw beside it | the same, plus `_raw` |
| 26 | Candidate-Primary Ranking (Exploratory) | top-1, reciprocal rank, pairwise AUROC for both labels against the exact random expectation; bootstrap over mice | `tumour_level_exploratory/ranking_summary.json` |
| 27 | RNA Preprocessing and Family Discrimination | configuration table (raw vs main), paired mouse medians, retained-fraction and gate-change summary | `representation_mice.tsv`, `representation_pairs.tsv`, `preprocessing_configurations.tsv` |
| 28 | Raw vs Preprocessed RNA: Matched Heatmaps | the 8 informative mice, identical ordering, scale and markers | `pairs_*.csv` |
| 29 | CNV and RNA+CNV: Not Yet Evaluated | status table R1–R5 and what a CNV arm needs; no result | `TUMOUR_LEVEL_REPRESENTATION_AUDIT.md` |
| 30 | What the Tumor-Level Benchmark Supports | three columns: supported / exploratory / not shown | the above |
| 31 | Summary and next steps | kept from v2 | — |

**Exploratory tables** (all in `results/kp_tracer_lineage_validation/tumour_level_exploratory/`):
- `candidates.tsv`, `lesion_ranking.tsv`, `mouse_ranking.tsv`, `ranking_summary.json`;
- `representation_pairs.tsv`, `representation_mice.tsv`;
- `run_manifest.tsv`, `preprocessing_configurations.tsv`;
- `readout_check.json` (d_M re-read and matched to the prespecified readout).

**Checks.**
- The tie-handling ranking metrics and their random expectations were checked by brute force over all orderings.
- The cell IDs of the two labels are identical in all 208 pair-sides.
