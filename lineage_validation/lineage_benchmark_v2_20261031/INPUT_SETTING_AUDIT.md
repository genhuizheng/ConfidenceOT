# Input-setting audit: RNA-only, Site-only, RNA+Site

> **SUPERSEDED IN PART, 2026-10-31.** This file read "site" as the anatomical site (tumour / tissue of origin). The user then stated that "site" means the **CRISPR editing-site character matrix** (per-cell target-site states: unedited, an observed indel allele, or missing). Everything below about anatomical site stays true as an audit of anatomical metadata, and its conclusion applies **only** to anatomical site. It does **not** apply to CRISPR editing sites, which are per-cell and vary within a side. The CRISPR reading is audited in `CRISPR_SITE_FEASIBILITY_AUDIT.md` in this folder; use that file for the Site-only and RNA+Site designs.

Status: 2026-10-31, read-only audit of the code and of the existing designs (sources: `PREVIOUS_EXPERIMENT_AUDIT.md` §3 in this folder, `cancer_metastasis/02_run_pair.py`, `lineage_validation/01_build_inputs.py`, `lineage_validation/level2_pairwise_ot/run_pairwise.py`). No run was made for this file.

## 1. What exists

| Setting | Implemented? | Where | Runs on KP-Tracer / Quinn |
|---|---|---|---|
| RNA-only | yes | `02_run_pair.py` `--representation-source rna`; cost = squared Euclidean on unit-norm PCs (= 2 − 2cos) of `ranknm256_noscale_ds_cos`, divided by the median random source–target pair cost (lines 705–725) | Level 1 (protocol, M4-E / M4-R), Level 2 / 3 (balanced OT on logcpm = "A"; UOT with rejection budget 0 = "B") |
| RNA+CNV | yes | `--representation-source rna_cnv`: RNA PCs and infercnvpy CNV PCs, each block scaled to median pair cost 1, concatenated | as RNA |
| Site-only | **no** | no site, tissue or lesion term in any cost; site is only copied to output tables or used to select cells (`02_run_pair.py:217–219`, `run_pairwise.py:96–142`) | none |
| RNA+Site | **no** | — | none |

The old `rna_cnv` is RNA + CNV. It is not RNA+Site and is not used as such anywhere in this extension.

## 2. Site metadata that exists

- **KP-Tracer:** `Tumor` (T* = a primary lung tumour; other suffixes = a metastasis such as a lymph node or the liver), `SubTumor` (sequenced piece), `is_normal`, `MetFamily`. The prepared files carry `site` only as the category primary / metastasis.
- **Quinn:** `sampleID` = tissue; LL (30kLL) = left lung = the injection site, the only "primary"; the other tissues are metastatic sites. A549 only, no normal cells.

## 3. Whether a site feature can carry cell-level information in each design

A site feature can only change a coupling where the site label **varies within a side**. If every source cell shares one site and every target cell shares one site, the site term adds one constant to every cost entry: the coupling and every per-cell gate decision are unchanged (Site-only becomes a constant cost, i.e. the independent coupling, with all gate scores tied).

| Design | Source sites | Target sites | What a site term would do |
|---|---|---|---|
| Level 1a pairwise (one primary → one metastasis; E1, E2) and the v3 couplings | 1 | 1 | **Constant. No information.** Site-only is undefined as a gate (all cells tied); RNA+Site = RNA. |
| Level 2 / 3 (one primary → pooled metastatic sites) | 1 | 2–8 | Varies only on the target side: c_site(i, j) = d(primary, site_j). It shifts whole target sites up or down. Under unbalanced OT this makes whole metastatic sites more or less likely to be rejected, as a site prior. Within one site every cell is still tied, and Level 2 / 3 truth is within-site or within-family, so it cannot help Level 2 / 3 except by moving mass between sites. |
| Quinn (LL → pooled tissues) | 1 (LL) | 5–6 | Same as above: a target-side site prior. |
| Protocol D_A (pooled primaries → one metastasis) | 2–5 primaries | 1 | Varies on the source side. **Each KP primary is one MetFamily**, so the site label of a primary is almost the lineage label: a site term here encodes the ground truth (**leakage**). |
| Protocol D_B / D_C3 (pooled metastases / both) | several | several | Target-side site = which metastasis; source-side as D_A (leakage on KP). |

## 4. The missing piece is a site distance, and no lineage-free one exists in the data

A site term needs d(site_a, site_b) between a primary site and a metastatic site. The data give only categorical labels. The candidates are:
1. **Categorical 0/1** (0 same site, 1 different): every primary → metastasis entry is 1, so it is constant in every design above, the same as no site.
2. **Anatomical or organ distance** (lung–lymph node closer than lung–liver, for example): no measured coordinates or agreed metric exist for KP-Tracer or Quinn. Any numbers would be ours, and there is no lineage-free way to choose them.
3. **Primary-tumour identity on the source side** (D_A): this is MetFamily in KP-Tracer, which is leakage.

So with the available metadata:
- **Site-only** is either constant (no information), a site prior with an arbitrary distance, or leakage. It cannot be scored at Level 2 / 3, because within a site every cell is tied.
- **RNA+Site** equals RNA-only wherever the site term is constant. It differs from RNA-only only by a whole-site prior on the target side (Level 2 / 3, Quinn), or by leakage (KP D_A).

## 5. Decision needed from the user before any RNA+Site job is written

- **What "site" means for Site-only / RNA+Site:** which column, and which site–site distance. Options: (1) categorical; (2) a fixed anatomical ordering supplied by the user before any result; (3) none, reporting RNA+Site as not identifiable with these data.
- **Which designs it applies to.** On KP D_A the source-side site is the lineage label and must not be used.
- **Weighting.** If a site term is used, its weight relative to the RNA cost has to be fixed in advance without CRISPR labels. The only label-free rule in the project is median pair-cost scaling, which is undefined for a constant block (median = the constant).

Until this is decided, no Site-only or RNA+Site run is proposed (see `PROPOSED_TACC_JOBS.md`).
