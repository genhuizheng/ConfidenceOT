# Level 2 results: Standard OT vs Our OT on CRISPR subclones, three strategies

Run 2026-10-22 on TACC. Design as fixed in `PLAN.md`; no parameter was changed after any result was seen.
- **Commits.** Code `c52bfed`; the KP-Tracer site-label fix `cac2ee9` came before any KP coupling.
- **Jobs.** RNA-only 1091504/1091505 (Quinn) and the KP retry. CNV-only 1091646/1091647, RNA+CNV 1091648/1091661, scoring 1091662.
- **Download.** `level2_full_small_20261022_204810.tar.gz`, md5 `382946ac0343f3d255b88b85e69e4c71` (checked), unpacked in `data/` (not committed).
- **Rescore.** Rescoring locally from the raw run outputs gives tables identical to the TACC scores.
- **Committed tables.** `results/`.

## Runs

- **All 54 runs are ok:** 9 cases × 3 strategies × A / B, from 26 families. No failure or missing run (`results/run_status.tsv`).
- **Every run converged** (`results/diagnostics.tsv`).
- **A (Balanced OT):** row and column marginals exact (relative deviation < 1e-4).
- **B (our UOT, budget 0):**
  - no cell gated (retained 1.0 / 1.0);
  - total mass 0.64–0.73;
  - largest row-marginal deviation 0.37–0.47 for RNA-only, 0.54–1.00 for CNV-only and RNA+CNV. Under the CNV representations the KL relaxation lets some primary cells send almost no mass (soft down-weighting, not rejection).

## Evaluability

- **Evaluable family-levels**, identical in every strategy and method, because it depends on the labels only:
  - L1: 17 of 26 families;
  - L2: 12 of 21 family-levels.
- **Not evaluable (kept, listed in `results/summary.tsv`):**
  - one true subclone among the scored cells: 3457 T1 (L1, L2), 3457 T4 (L1), several Quinn clones;
  - fewer than 2 candidate subclones: e.g. 3454 T1 L1 and Quinn 5k LG11/12/13 L1.
- **Caveats on the evaluable set:**
  - In 14 of 29 evaluable family-levels the smallest true subclone has fewer than 20 scored metastatic cells, so balanced accuracy is driven by a few cells. Example: 3454 L2, where 1 of the 531 cells is L1.1.2.
  - 5.2% of scored cells belong to a subclone that is not a candidate (π < 0.01 or < 5 primary cells), so no rule can call them correctly.

## Subclone identification (enrichment argmax; `results/comparison.tsv`, `results/comparison_overview.tsv`)

**Most families sit at chance under both workflows.** Per strategy, level and method, the median balanced accuracy minus chance is −0.024 to +0.029. Families above chance:

| Strategy | L1: A | L1: B | L2: A | L2: B |
|---|---|---|---|---|
| RNA-only | 10/17 | 8/17 | 6/12 | 5/12 |
| CNV-only | 6/17 | 8/17 | 5/12 | 5/12 |
| RNA+CNV | 8/17 | 8/17 | 8/12 | 7/12 |

**Where there is signal, both workflows find it.**
- **KP 3724 T1** (S1 seeded by L1.1, L1 / L2 / L3 by L1.2):
  - per-site recall 0.90–1.00 for RNA-only and RNA+CNV, both methods;
  - CNV-only L3 recall: A 0.65, B 0.80;
  - balanced accuracy at L2: RNA A 0.993 / B 0.968; CNV 0.927 / 0.947; RNA+CNV 0.983 / 0.977.
- **KP 3515 T1, L1:**
  - site L1 has recall 0.85–0.93;
  - the N1 cells of L1.2 (12 cells) have recall 0.67–0.92;
  - the other sites (all L1.1) are near 0.5.
- **Quinn 100k LG4, L1** (52 cells): CNV-only 0.856 / 0.856 (A / B); RNA-only 0.506 / 0.653.

**A vs B: small and not consistent in direction.**
- **RNA-only.**
  - A is higher in 13 of 17 families at L1 and 9 of 12 at L2. The median B − A is −0.010 (L1) and −0.006 (L2).
  - Largest differences: 3515 L1 −0.094 and L2 −0.131; Quinn 100k LG4 L1 +0.147, 5k LG15 L1 +0.149, 30k LG2 L2 +0.146 (22 cells).
- **CNV-only.**
  - The representation is the same for A and B, so only the solver differs.
  - B is higher in 11 of 17 at L1 (median +0.002); at L2, 4 B higher vs 6 A higher.
- **RNA+CNV.** At L1, 6 B higher vs 11 A higher (median −0.004); at L2, 5 vs 6.

## Rare primary subclones (π < 0.10; `results/per_subclone.tsv`)

- **Counts.** 107–109 rare subclones have scored metastatic cells. The count differs slightly between A and B because π is each method's own row-mass share.
- **Recall.** Median recall is 0 for every strategy and method; mean recall is 0.11–0.13. Only 5 (A) or 6 (B) rare subclones have recall > 0.5.
- **Not recoverable by construction.** 55 of the rare subclones are not candidates, so the enrichment rule cannot assign them.

## Enrichment relative to each method's actual marginals

- **E_own** = mean S̃ / π of the true subclone, per true subclone with scored cells:
  - medians 1.07–1.20;
  - 69–80% of true subclones have E_own > 1, in every strategy and method.
- **RNA-only:** B's E_own is lower than A's (L1 1.074 vs 1.119; L2 1.112 vs 1.197).
- **CNV-only and RNA+CNV:** close (e.g. RNA+CNV L1 1.134 vs 1.124).
- **Cell-level mean log₂ enrichment** (`mean_own_log2E`) is below 0 for both methods, because it is a log mean, and less negative for B in most families: RNA+CNV L1 14 of 17, L2 10 of 12.

## Coverage

- **Scored share.** Median share of the family's metastatic cells that are scored: 0.81 (L1), 0.41 (L2). The same for A and B.
- **Mass from the cell's own family / clone:**
  - KP-Tracer: 1.0, because the primary holds only its family;
  - Quinn: about 0.3 at L1, because the LL side holds every clone.
- **RNA-only:** B's own-clone share is a little lower than A's (0.306 vs 0.326; A higher in 16 of 17).

## Two-step strategies and CNV

- CNV→RNA and RNA→CNV are not run: without rejection they equal RNA-only and CNV-only exactly (`strategy_check/README.md`).
- The CNV representation is the existing infercnvpy export through `02_run_pair.py:cnv_coordinates`. Its "− 1" changes neither coordinates nor cost (audit on the 16 real units, same file).

## Summary

- **Overall.** With one primary → pooled metastases and no rejection, CRISPR-defined subclones are identified above chance only where metastatic sites are seeded by different primary branches. That is KP 3724, the L1 site and the N1 L1.2 cells of KP 3515, and Quinn 100k LG4 under CNV. Elsewhere both workflows are at chance.
- **A vs B.** Standard OT and Our OT differ little:
  - RNA-only leans to A;
  - CNV-only, where only the solver differs, is near equal;
  - RNA+CNV is mixed.
- **Rare primary subclones** are almost never recovered by either workflow.

These are descriptive results with no test statistic (PLAN.md). Level 3 (cell-level lineage correspondence) can reuse the saved couplings on TACC.
