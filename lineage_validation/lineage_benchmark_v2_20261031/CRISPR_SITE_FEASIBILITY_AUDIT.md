# CRISPR editing-site feasibility audit (Site = target-site character matrix)

Status: 2026-10-31. Everything in §1–§4 was computed locally in this session from the exported KP-Tracer package; the commands are in §8. Quinn is **not** audited here: its lineage files are on TACC only (§5). No OT was run, no existing file was changed, and nothing was submitted to TACC.

This file replaces the Site-only / RNA+Site conclusion of `INPUT_SETTING_AUDIT.md`, which read "site" as the anatomical site. "Site" here means a CRISPR target site (cut site) of the recorder, whose per-cell state is unedited, an observed indel allele, or missing.

---

## 1. The per-cell character matrix exists for KP-Tracer, locally

Source: `results/kp_tracer_lineage_validation/export/trees/character_matrices/*_character_matrix.txt`, listed in `MANIFEST.tsv` as byte-identical copies of the authors' files (Zenodo 5847462, `KPTracer-Data.tar.gz`). 85 matrices in all; 10 of them are family (`_Fam` / `_All`) matrices that span a primary and its metastases.

Format: tab-separated, index `cellBC` (for example `L9.TTTGTCATCTGTCAAG-1`), columns `r0 … rN`, one column per character.

**State alphabet** (counted over all cells × characters of the 10 family matrices):

| Symbol | Meaning | Note |
|---|---|---|
| `-` | missing / not detected | **not an allele**; see §4 |
| `0` | unedited (uncut) | a real observed state, shared by descent and by never having been cut |
| `2, 3, 4, …` | distinct observed indel alleles | integer ids, matrix-specific; **state `1` never occurs** in any matrix |

Allele-state counts per matrix run from 18 (3457_Apc_T4_Fam) to 292 (3724_NT_All). Per character the number of distinct alleles has a median of 2–37 depending on the family.

## 2. Cell ids align exactly with the OT inputs

`cellBC` is the same identifier as the expression `obs_names` and the `cell_id` of the Level 2 inputs (`KPTracer_meta.csv` index format `L28.CTCTAATAGGAGTTTA-1`).

Checked against `lineage_validation/level2_pairwise_ot/inputs/family_cells.tsv.gz`: **every Level 2 KP cell has a row in its family character matrix — coverage 1.000 in all five KP cases.** The earlier audit also found that all 242,974 source-cell observations of the OT outputs are present in `KPTracer_meta.csv` (`kp_lineage/validation_v2/LINEAGE_DATA_AUDIT.md` §C2).

So no realignment is needed: the character matrix can be indexed by the same cell ids the OT source/target sets already use.

## 3. The independent unit is the intBC, not the character — and it is scarce

Each KP-Tracer integration barcode (intBC) carries three cut sites, so three characters come from one physical integration. They share dropout: if the intBC is not captured, all of its characters are missing together.

**This is directly visible in the data.** Grouping characters by their identical per-cell missingness pattern:

| Family matrix | Characters | Distinct missingness patterns | Group sizes |
|---|---|---|---|
| 3515_Lkb1_T1_Fam | 30 | 10 | 3,3,3,3,3,3,3,3,3,3 |
| 3454_Lkb1_All | 24 | 8 | 3,3,3,3,3,3,3,3 |
| 3508_Apc_All | 29 | 10 | 3,3,3,3,3,3,3,3,3,2 |
| 3457_Apc_T1_Fam | 24 | 9 | 3,3,3,3,3,3,3,2,1 |
| 3457_Apc_T4_Fam | 9 | 6 | 2,2,2,1,1,1 |
| 3724_NT_All | 9 | 6 | 2,2,2,1,1,1 |

The groups are exactly the consecutive triples `r0,r1,r2`, `r3,r4,r5`, … Where a group is smaller than 3, either a cut site was dropped from the matrix, or the family's overall missingness is so low (3724: 2.5%, 3457_T4: 6.6%) that two characters of one intBC differ in a handful of cells and so fall into separate patterns. **For those two families the grouping is consistent with 3 intBCs but is not proven by missingness alone; the allele table would settle it (§5).**

**Consequence for design B (held-out sites).** A split into input and evaluation partitions is only independent if it splits **intBCs**, not characters. The budget is therefore:

| Family | intBCs (approx.) | A 50/50 intBC split gives |
|---|---|---|
| 3515 | 10 | 5 input / 5 evaluation |
| 3508 | 10 | 5 / 5 |
| 3457 T1 | 8–9 | 4 / 4 |
| 3454 | 8 | 4 / 4 |
| **3724** | **3** | **1–2 / 1–2** |
| 3457 T4 | 3 | 1–2 / 1–2 |

**3724 is the problem.** It is the one family that carries essentially all of the existing Level 2 and Level 3 signal (balanced accuracy 0.93–0.99; ΔD_norm 0.32–0.36), and it has the fewest independent sites. A held-out evaluation there would rest on 1–2 intBCs, which is too thin to call a result robust. Families with 8–10 intBCs support design B properly, but those are exactly the families where the existing signal is near chance.

Even within one intBC the three cut sites are not independent: they are cut by the same Cas9 exposure and an indel can span neighbouring sites (resection), so a large deletion removes several at once. Splitting *within* an intBC is not a valid independence claim.

## 4. The character model: what must not be done

Audited against the project's own rules and the data:

1. **Missing is not a state.** `-` means the intBC was not captured, not that the site is different. Treating `-` as an allele would make two cells that both lost the same intBC look related. Dropout is whole-intBC and ranges from 0.6% to 64% per character (median 2–11% per family), and it correlates with capture depth, which is the known confound in this project. Any distance must be computed over the characters that are observed in **both** cells, and normalised by how many those are.
2. **Hamming distance is not evolutionary distance.** Two cells sharing allele `7` at a character is strong evidence of common ancestry only if allele `7` is rare; a common indel arises independently in many cells (homoplasy). The authors' Cassiopeia pipeline uses indel priors for exactly this reason, and the README states that priors ship with each tree. **The priors file is not in the local export** (no `*prior*` file; `MANIFEST.tsv` lists trees and character matrices only), so it has to be fetched from `KPTracer-Data` on TACC before any weighted distance is built.
3. **`0` (unedited) is not neutral.** Two cells both unedited at a site share a state only because neither has been cut yet. Under a Cassiopeia-style model, shared `0` carries no ancestry information beyond shared timing, while a shared non-zero allele does. The distance has to treat `0` differently from an allele; a plain categorical match does not.
4. **No re-inference of trees.** `SUBCLONE_RULES.md` §1 fixes the authors' trees as the truth and forbids re-inferring them. Design B builds a *compatibility metric* from held-out characters directly — it does not build a new tree — so it stays inside that rule. Any variant that re-infers a tree from a character subset would need a dated rule change first.

## 5. Not verified here, and needed before implementation

| Item | Why it matters | Where |
|---|---|---|
| **KP allele tables** (intBC × cut site per cell) | The only way to prove the intBC grouping, in particular for 3724 and 3457 T4 | `KPTracer-Data` on TACC; not in the local export |
| **KP indel priors** | Required for a homoplasy-aware distance (§4.2) | same |
| **Per-metastasis matrices** | The KP README says individual character matrices are not reported for each metastasis. The family matrices do cover metastatic cells (coverage 1.000 over the Level 2 cell sets, which include the metastatic sites), so this looks like a non-issue, but it should be confirmed per site | same |
| **Quinn: allele table, character matrices, trees** | Quinn's files were copied by `quinn2021/01_convert_geo.py` into `converted/lineage/<tag>/{alleleTable,character_matrices,trees}` for every experiment except LM0. Nothing of this is local, so Quinn's character count, missingness, state alphabet, intBC structure and cell-id format are **unknown here** | `/scratch/10119/ghzheng/OT_project/data/quinn2021_GSE161363/converted/lineage/` |
| **Quinn cell-id alignment** | The 10k experiment is known to name cells differently in the matrix and the meta (`<sampleID>.<16nt>-<library>` vs `<16nt>-<library>`), so the lineage files may need the same rebuild the converter does | same |

## 6. The three designs, as they can actually be built

Written to the user's definitions; none has been run.

**C. RNA-only OT.** Unchanged from the existing pipeline: `ranknm256_noscale_ds_cos`, no CRISPR input, CRISPR used only for scoring. This is the clean control and the only design with no leakage question.

**A. Full-barcode-informed OT.** Cost from all observed characters of both cells. Honest status: the evaluation truth (the authors' tree, the frozen L1/L2 subclones) is derived from these same characters, so Level 2 and Level 3 scores are **not independent**. Report A as a diagnostic upper bound, never as validation. Level 1 is less affected but still lineage-informed.

**B. Held-out-editing-sites OT.** Partition **intBCs** (not characters) by a rule fixed before any score is read, e.g. by intBC index order, input = even positions, evaluation = odd; and report every partition's result, not the best one. The OT cost uses input intBCs only; the evaluation metric is built from the evaluation intBCs only (a direct character-level compatibility measure, no new tree). Robustness is reported across all partitions. Feasible for 3454, 3457 T1, 3508, 3515; **marginal for 3724 and 3457 T4** (§3).

**Shared rules for A and B.**
- Distance over characters observed in both cells, normalised by that count; missing excluded; shared `0` and shared allele weighted differently; allele weights from the authors' priors if they are available, otherwise a fixed, documented scheme chosen before any evaluation.
- RNA+site weighting fixed in advance by the project's existing label-free rule (each block scaled to median random source–target pair cost 1, as `rna_cnv` already does in `02_run_pair.py:705–725`), never tuned on CRISPR labels. The block is not constant, so the median scaling is well defined — unlike the anatomical-site case.
- Report per case: input-site count, held-out-site count, missingness, clone coverage, validity flags, and the spread across partitions.

## 7. What I recommend, given the numbers

The design is sound and the data support it for KP-Tracer. The honest limitation to state now, before any compute: the families with enough intBCs for a real held-out test are the families where RNA and CNV already sit at chance, and the one family with a strong effect has about three independent sites. So design B is likely to produce a well-controlled but low-powered answer on 3724 and a well-powered answer on families where there may be nothing to find. That is still worth doing — it is a genuine negative-or-positive test rather than a circular one — but it should not be expected to rescue the Level 2 / Level 3 result.

## 8. Reproducing §1–§3

```
python results/kp_tracer_lineage_validation/export  (read-only)
  trees/character_matrices/*_character_matrix.txt   state counts, per-character missingness
  lineage_validation/level2_pairwise_ot/inputs/family_cells.tsv.gz   coverage join on cell_id
```
The per-case numbers are in `CRISPR_SITE_FEASIBILITY.tsv` in this folder.
