# CRISPR lineage × site ground truth: definitions

Fixed 2026-10-22, before any OT output is read in this analysis. They follow the frozen protocol's ground truth (`../VALIDATION_PROTOCOL.md` §5, FROZEN 508609f) and the frozen subclone rules (`../within_family_subclones/SUBCLONE_RULES.md`, a545f1c), and add the per-site bookkeeping this analysis needs.

**CRISPR labels never enter an OT solver** here or in the Level 1 runs they are compared with, and they are never adjusted after an OT result is seen.

## Labels: two levels per dataset, never mixed

| Dataset | Level | Label of a cell | Unknown when | Source |
|---|---|---|---|---|
| KP-Tracer | **family** | `MetFamily` of the cell's tumour (constant within each of the 77 tumours, checked) | family ends in `_Unk` | authors' metadata (`results/kp_tracer_lineage_validation/export/metadata/kptracer_meta_subset.csv.gz`) |
| KP-Tracer | **subclone** (L1, L2) | the frozen subclone of SUBCLONE_RULES.md, for eligible families only | the cell is not a reliable leaf of its family tree (missing share > 0.5, or no leaf) | `../within_family_subclones/output/cells/` |
| Quinn | **clone** | the authors' `LineageGroup` | no author meta row (such cells have no site either) | converted obs (`…/meta/Quinn2021_obs_<mouse>.csv.gz`) |
| Quinn | **subclone** (L1, L2) | the frozen tree-derived subclone, eligible clones only | as for KP-Tracer | `../within_family_subclones/output/cells/` |

**Subclone labels.** A labelled cell whose subclone label is `ambiguous` (between partitions, SUBCLONE_RULES.md) is **ambiguous**, not unknown.
- Family / clone labels have no ambiguous state.
- KP `_Unk` and a missing Quinn LineageGroup are **unknown**.

**Sites.**
- KP-Tracer: the tumour (`Tumor`). A tumour named `<mouse>_<genotype>_T<n>` is a primary lung tumour; every other tumour is a metastasis (lymph node N, kidney K, liver L, …).
  - Different primaries are independently initiated tumours, so their families are different lineages by design.
- Quinn: the tissue (`sampleID`). `LL` (or `<mouse>LL`, e.g. `30kLL`) is the primary injection site; every other tissue is metastatic.
- Cells: malignant cells only (KP `is_normal` False, 47,549 of 47,551 cells; every Quinn cell is A549).

## Mouse-level matrices (Phase 1; `lineage_site_matrices/`)

For each mouse / experiment and level:
- **counts:** rows are lineages, columns are sites (primaries first, then metastases); entries are cell counts. Unknown is a separate row, never folded into a lineage.
- **fractions:** each site's column divided by the site's cells. The unknown row is included, so it shows annotation coverage.
- **Sharing across the sampled sites.** These describe the mouse, not an OT comparison:
  - `shared_across_sites`: the lineage is seen in ≥ 1 primary and ≥ 1 metastasis;
  - `primary_sites_only`;
  - `metastatic_sites_only`;
  - `unknown`.

## Comparison-level classes (Phase 2: what a given OT run could match)

For one OT comparison with its **actual** analysed source set S⁰ and target set T⁰ (the cells of that run's `cell_confidence.csv`; frozen protocol §3.4, §6). Each class is defined per side and per level:

| Class | Source cell i | Target cell j |
|---|---|---|
| `observed_shared` | label known and ∃ j ∈ T⁰ with the same label | label known and ∃ i ∈ S⁰ with the same label (the protocol's *matchable*) |
| `primary_only_observed` | label known, no j ∈ T⁰ with the label | — |
| `metastasis_only_observed` | — | label known, no i ∈ S⁰ with the label (the protocol's *unmatched_observed*) |
| `ambiguous` | subclone level: label `ambiguous` | same |
| `unknown` | label unknown | same |

- **Only that run's cells count.** A site of the mouse that is not in this comparison never makes a lineage shared.
- **Pooled comparisons are their own case.** In the pooled designs (D_A pooled primaries, D_B pooled metastases, D_C3 both pooled), the counterpart is any cell of the pooled side. These are recorded with the design.
- **Observation is not existence.** Unobserved is never read as absent, so the negatives are read per dataset (protocol §5.2):
  - **KP-Tracer, family level:** a family with no cell on the other side is a different primary's lineage, or a family whose primary was not sampled. It is a by-design negative for matching.
  - **Quinn:** a clone not seen in the LL sample may exist there unsampled. It is **not** a confirmed negative and is always reported separately.
  - **Subclone level (both):** a subclone not observed on the other side may exist there unsampled. Not a confirmed negative.
- **Sensitivity.** "Observed" means ≥ 1 cell. A second reading requires ≥ 5 cells of the lineage on the other side (the protocol's Quinn sensitivity rule); lineages with 1–4 such cells are then reported as low-support.

## Gate semantics (Phase 2; already audited)

As in `../within_family_subclones/audit/AUDIT_REPORT.md` and protocol §6:
- **Retained:** `retained` is True (signed rejection margin < 0).
- **Rejected:** analysed but not retained.
- **Excluded:** removed at step 1 of a two-step strategy; not in that step-2 run.
- **Effective coupling:** P̃ = ρ ⊙ P ⊙ τ. A rejected cell's row or column of the raw UOT plan can carry mass; that mass is never counted as retained matching.
