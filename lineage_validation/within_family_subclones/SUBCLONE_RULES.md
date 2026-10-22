# Within-family subclones: rules fixed before any OT output is read

Status: **fixed 2026-10-22, before the subclone definitions were compared with any gate or coupling.** This is an exploratory biological application, separate from the pre-specified T1/T2 of `../VALIDATION_PROTOCOL.md`; that protocol is not changed. A later change to these rules is a dated section at the end, marked post hoc.

**Seen before writing these rules:**
- the figures of `../local_lineage_visualization/` (3508, 3515, Quinn 10k LG1/LG2, 5k LG3), which show family-level OT results;
- the audit counts of trees and leaves.

No subclone-level OT quantity had been computed.

## 1. Data

- **Trees.** The authors' trees only, never re-inferred.
  - KP-Tracer: the family's Cassiopeia tree, `<family>_Fam` if it exists, else `<mouse>_<genotype>_All` restricted to the family's leaves (the induced subtree: nodes with no family leaf below them are dropped, unary nodes left as they are).
  - Quinn: the clone's `*_tree_hybrid_priors.alleleThresh.processed.txt`.
- **Character matrix.** The matrix deposited with that tree (KP-Tracer `<stem>_character_matrix.txt`, Quinn `*_character_matrix.alleleThresh.txt`). States as written: `0` is unedited, `-` / `-1` / empty is missing.
- **Cells.** Tree leaves mapped to scRNA cells:
  - KP-Tracer: leaf name = cell id, the cell in `KPTracer_meta`, `is_normal` false;
  - Quinn: the frozen pipeline's leaf → cell map (`labels/tree_leaves.tsv.gz`; in 10k `<sampleID>.<leaf>`).
  - Primary = the family's own primary tumour (KP-Tracer) or LL (Quinn); every other site is metastatic.
- **Author annotations.**
  - KP-Tracer `SubTumor` is the physical piece of a tumour that was sequenced (T1_1 … T1_5), a spatial sampling label, not a lineage subclone. It is reported beside the subclones and never used to define them.
  - Quinn deposits no subclone annotation.

## 2. Reliable cells

A mapped cell is **reliable** if its missing share in the matrix is ≤ 0.5. Other mapped cells are `ambiguous` (their tree position rests on too few characters).

## 3. CRISPR-supported branch

A branch is a node B of the tree; its cells are the reliable cells below it. B is **supported** against its parent context P (the cells of the node where the split is made) if at least one marker edit exists, i.e. a character k and a state s ≠ 0 with:
- ≥ 50% of B's cells non-missing at k;
- ≥ 80% of B's non-missing cells at k carrying s;
- ≤ 20% of the non-missing cells of P \ B at k carrying s.

Every marker edit of every subclone is saved (character, state, frequency inside, frequency outside).

## 4. Subclones (one deterministic partition per family)

- **Major child.** A child holding ≥ 20 reliable cells.
- **Level 1.**
  - Start at the family root. While the current node has exactly one major child, descend into it; the cells of its other children become `ambiguous` (minor branch).
  - At the first node with ≥ 2 major children (the split node), every supported major child is a level-1 subclone. Unsupported major children and minor children become `ambiguous`.
  - With no such node, the family has no subclone structure.
- **Level 2.** The same procedure inside each level-1 subclone, with that subclone as root and parent context. Level 2 is reported as a finer, secondary view.
- **Names.** `L1.<k>`, ordered by reliable-cell count; each keeps its tree node id (preorder index in the authors' tree file), its cell ids, site composition and marker edits.
- **Shared subclone.** A subclone with ≥ 5 reliable primary and ≥ 5 reliable metastatic cells.
- Subclone boundaries depend only on the tree, the matrix and the cell annotation, never on an OT result.

## 5. Eligibility for OT classification

A family or clone is classified when:
- it has ≥ 2 level-1 subclones;
- it has ≥ 20 reliable primary and ≥ 20 reliable metastatic cells;
- an existing run has the family's primary as source, with saved couplings:
  - KP-Tracer: `D_B` with that primary (`<mouse>__D_B__<primary>__metastases`); if the mouse has no D_B, every `D_A` run of the family's metastases;
  - Quinn: `D_B` (LL → pooled tissues).

Families that fail are listed with the reason.

## 6. Classification of a metastatic cell j (all five strategies, M4-E; M4-R reported)

- **Matching weights.** q_ij as in the protocol §6: the saved coupling over retained source cells, column-normalised, with rejected cells weighing 0. S_gj = Σ_{i∈g} q_ij over source groups g:
  - the family's level-1 subclones;
  - `family unlabelled` (family primary cells with no reliable subclone label or not on the tree);
  - `other` (source cells of other clones / other primaries).
- **Family weight.** w_F(j) = Σ over the family's groups.
- **Within-family weight.** S̃_gj = S_gj / w_F(j).
- **Outcome** (in this order):
  1. `no CRISPR label`: j has no reliable level-1 subclone label;
  2. `unmatched_observed`: j's subclone has no analysed primary source cell in the run (no counterpart; not an error);
  3. `counterpart rejected`: j's subclone has source cells but the gate retained none of them (reported apart, not as a plain error);
  4. `abstain`: j was rejected, or has no retained-source mass, or w_F(j) = 0 (matching score 0);
  5. `ambiguous`: the top two S̃ differ by < 0.05, or the top group is `family unlabelled`;
  6. `correct` / `incorrect`: the top subclone equals / differs from j's subclone.
- **Reported per family and strategy:**
  - counts of every outcome;
  - accuracy = correct / (correct + incorrect);
  - coverage = (correct + incorrect) / (cells with a label and a counterpart);
  - the confusion matrix;
  - mean S̃ per (true subclone × source subclone);
  - two reference predictions:
    - **size**: always the subclone with the most retained source cells;
    - **chance**: the expected accuracy of drawing a source subclone in proportion to its retained source cells.
- **Wording.** Results are stated as "OT puts the metastatic cell's mass on the primary subclone that shares its CRISPR branch", never as "came from" or a direction.
