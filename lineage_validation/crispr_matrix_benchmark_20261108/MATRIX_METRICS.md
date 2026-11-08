# Matrix-based CRISPR validation of the v3 grid (KP-Tracer, Quinn 2021): definitions

Status: **FIXED 2026-11-08, before any matrix-based score was computed from a coupling.** Written after the audit of
the character matrices and allele tables (`00_audit_matrices.py`, `results/crispr_matrix_benchmark_20261108/audit/`)
and before any scoring code was run. A later change is a dated section at the end, marked post hoc; no edit-distance or
grouping parameter is changed after a score is seen.

Scope set by the user (2026-11-08):
- RNA-only OT. Nothing new is fitted: the existing v3 grid is reused (296 configurations, frozen calibration v3,
  M4-E / M4-R couplings, `CRISPR_LINEAGE_BENCHMARK.md` §4).
- The CRISPR character data are the **primary independent evidence**. The Cassiopeia trees are **secondary** evidence.
- CRISPR data never enter a cost, the calibration or a choice of preprocessing.
- The old tree-based protocol and its results are kept as they are.

## 1. Lineage data (from the audit)

**Source of the character states.** The authors' allele tables, not the integer-coded matrices, because the integer
codes carry no indel identity and cannot be compared across characters.
- KP: `KPTracer.alleleTable.FINAL.txt` (Zenodo 5847462). 571,092 rows, 72,328 cells, 81 intBCs, 82 MetFamilies, no
  duplicated cell × intBC.
- Quinn: `GSM49053{34,36,38,40}_alleleTable.<mouse>.txt.gz`. One per experiment, with `LineageGroup`.

**Character.** (intBC, cut site), with cut sites r1, r2, r3.

**State of cell c at a character:**
- **missing:** intBC b is not detected in c. Missing is never a state and never a similarity of 0.
- **uncut:** the cut-site string contains `None`. Both spellings occur: `None` and `[None]`, the latter inside its
  flanks.
- **indel:** any other string, compared as the full string (flanks plus bracket), as Cassiopeia does. Two insertions at
  the same position with different inserted bases are therefore different states.

**Cell IDs.**
- KP: `cellBC` equals the OT cell id.
- Quinn: `cellBC` equals the OT id (5k: `<site>.<bc>` on both sides; 100k and 30k: `<bc>` on both). In 10k the OT id
  is `<site>.<bc>` and `cellBC` is `<bc>`: the prefix is stripped when there is no exact match. A barcode that maps to
  more than one row is an error, not a choice.

**Lineage of a cell** (for scope only, never for a score's input weighting):
- KP: `MetFamily` in the allele table.
- Quinn: `LineageGroup`.

**Audit facts that shape the rules:**
- KP edits are near-saturated: uncut is 0.1-14% of observed states.
- Many KP characters carry one allele in ≥ 90% of a family's cells, so they are clonal and uninformative within the
  family.
- In KP, 31% of allele-table rows repeat the same indel string at adjacent cut sites of one intBC: one resection seen
  at two sites.
- Quinn: uncut is 28-65% of observed states.
- Coverage: 100% of KP and 96-100% of Quinn OT cells carry ≥ 1 informative edit. About 100% of cross-side pairs in a
  lineage have ≥ 1 comparable site, and 50-93% share ≥ 1 informative edit.

## 2. Fixed building blocks

**(a) Indel prior and weight** (Jones et al. 2020, Cassiopeia `compute_empirical_indel_priors`):
- Per dataset (KP; each Quinn experiment), q(s) = number of intBCs in which indel string s occurs at any cut site,
  across all cells of that dataset, divided by the number of intBCs of the dataset.
- Weight w(s) = −log q(s). Rarer indels weigh more, so recurrent indels (homoplasy) count less.

**(b) Informative character.**
- Within a lineage, a character is founder, and uninformative, when one indel string is carried by ≥ 90% of the
  lineage's cells observed there.
- Founder characters are excluded everywhere below.
- The 0.9 threshold is the audit's, fixed before scoring.

**(c) Events within an intBC** (cut-site dependence):
- In one cell, the indel strings of an intBC are reduced to the set of distinct strings.
- A resection that spans r1 and r2 is therefore one event, not two.
- Uncut sites carry no event.

**(d) Cell-pair edit agreement** s_ij, for source i and target j of the same lineage:
- Comparable intBCs: observed in both cells, carrying ≥ 1 non-founder event in at least one of the two cells.
- For every event e (indel string) present in i or j at a comparable intBC:
  - agreement a_e = w(e) if both cells carry e, else 0;
  - maximum m_e = w(e).
- s_ij = Σ_e a_e / Σ_e m_e. Each event is counted once per pair even when both carry it.
  - It is the prior-weighted Jaccard index of the two cells' event sets over comparable intBCs: Cassiopeia's weighted
    Hamming idea, with shared uncut and missing states removed.
  - A conflict (different indels at the same site) and an indel against uncut both lower s.
  - Shared uncut states contribute nothing.
- **Not evaluable:** no comparable intBC. Such a pair is never given s = 0.

**(e) Coupling quantities** (M4-E primary, M4-R reported). They are the retained-block definitions of
`CRISPR_LINEAGE_BENCHMARK.md` §3:
- P is the saved coupling with every rejected source row and rejected target column set to 0.
- Row mass m_i = Σ_j P_ij; for a retained target j, q_ij = P_ij / Σ_i P_ij.
- **Independent reference with the same marginals:** π_i = m_i / Σ_k m_k (the coupling a ⊗ b of the retained block).

## 3. Levels

**L1, family / clone (unchanged in meaning):**
- **KP:** d = same-family − mean other-family source retained fraction, per metastasis. 104 pairs; 31 metastases in
  8 mice. Read from `FIT_METRICS_v3tacc.tsv.gz` (M4-E rows).
- **Quinn** (L2/L3 cases):
  - own-clone mass gain per retained target cell j with a known clone present on the source side:
    g_j = Σ_{i: clone(i)=clone(j)} q_ij − Σ_{i: clone(i)=clone(j)} π_i;
  - per experiment, the mean of g_j;
  - coverage = scored targets / all target cells.

**L2, edit-defined groups** (primary: intBC allele groups):
- **Unit of truth:** within a lineage, one partition per informative intBC b. Each cell observed at b is in the class of
  its intBC allele: the 3-site tuple of strings, uncut included.
  - A class is kept when it has ≥ 20 cells, ≥ 5 on each side, and carries ≥ 1 non-founder event.
  - Cells in other classes at b are "minor" and not scored.
  - A partition needs ≥ 2 kept classes; otherwise it is not evaluable.
- **OT assignment** of a retained target cell j: argmax over kept classes g of E_gj = S̃_gj / π_g (the Level 2 rule
  on these classes), where:
  - S_gj = Σ_{i∈g} q_ij and S̃_gj = S_gj / Σ_{g kept} S_gj;
  - π_g = Σ_{i∈g} m_i / Σ_{i∈kept} m_i.
  - Only classes with π_g ≥ 0.01 and ≥ 5 source cells are candidates.
- **Score:**
  - balanced accuracy (mean recall over the true classes) − chance (1 / candidates), per partition;
  - per lineage, the mean over its partitions;
  - per mouse, the mean over its lineages.
- **Baselines:**
  - **size rule:** predict the candidate with the largest π_g. Reported as BA_OT − BA_size;
  - **source composition:** the independent coupling gives E = 1 for every class, which reduces to the size rule;
    reported as the same number.
- **Coverage:** scored target cells / target cells of the lineage, and the share of target cells in kept classes.
- **Secondary partition:** hierarchical clustering with a categorical distance, never k-means on allele codes.
  - Distance 1 − s_ij on all cells of the lineage (both sides). Pairs that are not evaluable get the lineage's median
    distance only for the clustering, never in a score.
  - Average linkage (UPGMA).
  - Cut at the first split from the root that gives ≥ 2 clusters of ≥ 20 cells with ≥ 5 on each side (the rule of
    `SUBCLONE_RULES.md` §4, applied to this dendrogram).
  - Scored as above, with the clusters as classes.

**L3, cell-level edit similarity** (primary matrix metric):
- For a retained target j, using the evaluable sources E_j of its lineage:
  - S_OT(j) = Σ_{i∈E_j} q_ij s_ij / Σ_{i∈E_j} q_ij;
  - S_ind(j) = Σ_{i∈E_j} π_i s_ij / Σ_{i∈E_j} π_i;
  - S_best(j) = max_{i∈E_j} s_ij.
  - ΔS_j = S_OT − S_ind; ΔS_norm,j = ΔS_j / (S_best − S_ind), undefined when S_best = S_ind.
- **Coverage of j:** c_j = Σ_{i∈E_j} q_ij, the OT mass of j on evaluable same-lineage sources. j is scored when c_j > 0.
- **Aggregation:** cell → lineage (median ΔS_norm; mean ΔS beside it) → mouse (mean over lineages) → dataset (mean over
  mice).
- **Coverage reported beside every number:** scored retained targets / all target cells; retained targets / all target
  cells; mean c_j.

**Tree-secondary** (depends on the authors' inferred trees; labelled so in every table):
- L2-tree: subclone balanced accuracy − chance.
- L3-tree: ΔD_norm.
- Both are defined in `CRISPR_LINEAGE_BENCHMARK.md` §3, on the same retained-block couplings.

## 4. Units, strata and checks

**Units scored:**
- **(i)** The 9 L2/L3 cases: KP 3454 T1, 3457 T1, 3457 T4, 3515 T1, 3724 T1; Quinn 100k, 10k, 30k, 5k.
  - Every lineage present in a case is used, not only the tree-eligible ones.
- **(ii)** KP L1 same-family pairs (primary → metastasis of its own MetFamily) with their v3 couplings: L3 and L2 per
  pair. This stratum adds mice without tree-defined subclones.

**Strata:** KP and Quinn are always separate. Mouse is the unit of aggregation; a Quinn experiment counts as one mouse.
No result pools KP with Quinn.

**3724:**
- every KP summary is also given without 3724;
- every 3724 lineage is reported on its own.

**Sequencing depth:** per lineage and configuration, the Spearman correlations of ΔS_j and of the gate score with each
cell's pre-equalisation total counts. Reported beside the scores, never used to correct them.

**Configurations:** all 296, every unit, M4-E primary and M4-R beside. Flags are columns, never filters
(`calibration_m4e_inference_valid`, convergence).

**Reading:** the rule of `CRISPR_LINEAGE_BENCHMARK.md` §4, applied per level and dataset:
- best set by mouse bootstrap (10,000 resamples, seed 20261108);
- leave-one-mouse-out, with 3724 named;
- the Splatter v3 rank beside each configuration.
- No composite score and no PASS / FAIL.

**Literature basis:**
- indel priors and weighted Hamming distance: Jones et al. 2020 (Cassiopeia);
- event-level counting within an intBC (resections spanning cut sites): Yang et al. 2022 and Quinn et al. 2021 allele
  tables;
- independent coupling as the reference: moslin / LineageOT practice;
- UPGMA on a categorical distance: classical phylogenetic clustering.

## 5. Not done

- No new OT fit, no CRISPR-informed cost, no tree re-inference.
- No k-means on allele codes.
- No parameter change after scores are seen.
