"""Audit of the CRISPR character matrices behind the lineage benchmark (KP-Tracer families, Quinn 2021 clones):
cell ids, characters, missing / uncut / indel states, allele frequencies, the intBC (3 cut sites) grouping, and how many
OT cells and cross-side cell pairs have comparable edits. Read only; no OT output is read.

    python lineage_validation/crispr_matrix_benchmark_20261108/00_audit_matrices.py --out OUT

Inputs (local exports of the authors' files):
  KP     results/kp_tracer_lineage_validation/export/trees/character_matrices/<stem>_character_matrix.txt
         (Zenodo 5847462); cellBC = OT cell id
  Quinn  lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export/lineage/Quinn2021/
         <mouse>/character_matrices/character_matrices/m<mouse>_lg<k>_character_matrix.alleleThresh.txt;
         cellBC = tree leaf, mapped to the OT cell id by labels/tree_leaves.tsv.gz
  cells  the OT cells of the 9 L2/L3 cases (level2_pairwise_ot/data/full_20261022_204810/full/<case>/cells.tsv.gz: side,
         cell, site), the cells the v3 grid fitted for these units
States: '-' / '-1' / empty = missing (no read, never a state); 0 = uncut; any other integer = an indel allele. The codes
are per character and per matrix (allele 2 at r0 is unrelated to allele 2 at r1).
Writes OUT/matrices.tsv (one row per matrix), OUT/characters.tsv (one row per character), OUT/intbc_missingness.tsv
(consecutive triples), OUT/case_coverage.tsv (per case and side), OUT/pair_comparability.tsv (sampled cross-side
pairs per case and lineage) and OUT/AUDIT.md.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KP_MAT = REPO / "results" / "kp_tracer_lineage_validation" / "export" / "trees" / "character_matrices"
QX = REPO / "lineage_validation" / "local_lineage_visualization" / "data" / "local_lineage_visualization_export"
L2 = REPO / "lineage_validation" / "level2_pairwise_ot"
RUNS = L2 / "data" / "full_20261022_204810" / "full"
MISSING = {"-", "-1", "", "nan", "NaN"}
FOUNDER = 0.9      # an allele carried by >= 90% of a character's observed cells is shared by (almost) everyone
SEED = 20261108


def read_matrix(path: Path) -> pd.DataFrame:
    """Integer matrix: -1 missing, 0 uncut, >= 1 allele code (codes as written)."""
    raw = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).set_index("cellBC")
    if raw.index.duplicated().any():
        raise SystemExit(f"{path.name}: duplicated cellBC")
    v = raw.apply(lambda c: c.map(lambda x: -1 if x.strip() in MISSING else int(x)))
    return v.astype(np.int32)


def char_table(m: pd.DataFrame, name: str) -> pd.DataFrame:
    X = m.to_numpy(); rows = []
    for k, col in enumerate(m.columns):
        x = X[:, k]; obs = x >= 0; ed = x >= 1
        vals, cnt = np.unique(x[ed], return_counts=True)
        top = cnt.max() / obs.sum() if ed.any() and obs.any() else 0.0
        rows.append({"matrix": name, "character": col, "index": k, "intbc_guess": k // 3, "cells": len(x),
                     "missing": float((~obs).mean()), "uncut": float((x == 0).mean()), "edited": float(ed.mean()),
                     "uncut_of_observed": float((x[obs] == 0).mean()) if obs.any() else np.nan,
                     "alleles": int(len(vals)), "top_allele_share_of_observed": float(top),
                     "founder": bool(top >= FOUNDER), "state_1_present": bool((x == 1).any())})
    return pd.DataFrame(rows)


def triples(m: pd.DataFrame, name: str) -> pd.DataFrame:
    """Consecutive triples (r0-r2, r3-r5, ...): share of cells whose missingness is identical across the triple."""
    miss = m.to_numpy() < 0; rows = []
    for g in range(0, m.shape[1], 3):
        b = miss[:, g:g + 3]
        rows.append({"matrix": name, "triple": g // 3, "characters": b.shape[1],
                     "identical_missingness": float((b.all(1) | (~b).all(1)).mean()), "any_missing": float(b.any(1).mean())})
    return pd.DataFrame(rows)


def comparable(a: np.ndarray, b: np.ndarray, informative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per pair (rows of a vs rows of b): characters observed in both where at least one cell carries an informative
    allele; and characters where both carry the same informative allele."""
    both = (a >= 0) & (b >= 0)
    inf_a = (a >= 1) & informative[None, :]
    inf_b = (b >= 1) & informative[None, :]
    comp = both & (inf_a | inf_b)
    shared = both & inf_a & inf_b & (a == b)
    return comp.sum(1), shared.sum(1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=20000, help="cross-side pairs sampled per case and lineage")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    mats = {}
    for f in sorted(KP_MAT.glob("*_character_matrix.txt")):
        mats[("KPTracer", f.name.replace("_character_matrix.txt", ""))] = f
    for f in sorted(glob.glob(str(QX / "lineage" / "Quinn2021" / "*" / "character_matrices" / "character_matrices" / "*_character_matrix.alleleThresh.txt"))):
        f = Path(f); mats[("Quinn2021", f.name.replace("_character_matrix.alleleThresh.txt", ""))] = f

    mrows, crows, trows, loaded = [], [], [], {}
    for (ds, name), f in mats.items():
        m = read_matrix(f); loaded[(ds, name)] = m
        ct = char_table(m, name); crows.append(ct); trows.append(triples(m, name))
        X = m.to_numpy(); obs = X >= 0; ed = X >= 1
        inf = ~ct.founder.to_numpy()
        mrows.append({"dataset": ds, "matrix": name, "cells": m.shape[0], "characters": m.shape[1],
                      "missing": float((~obs).mean()), "uncut": float((X == 0).mean()), "edited": float(ed.mean()),
                      "uncut_of_observed": float((X[obs] == 0).mean()),
                      "founder_characters": int(ct.founder.sum()), "characters_never_edited": int((ct.edited == 0).sum()),
                      "alleles_total": int(ct.alleles.sum()), "alleles_per_character_median": float(ct.alleles.median()),
                      "state_1_present": bool(ct.state_1_present.any()),
                      "cells_no_observed": float((obs.sum(1) == 0).mean()),
                      "cells_no_edit": float((ed.sum(1) == 0).mean()),
                      "cells_no_informative_edit": float(((ed & inf[None, :]).sum(1) == 0).mean()),
                      "observed_per_cell_median": float(np.median(obs.sum(1))),
                      "informative_edits_per_cell_median": float(np.median((ed & inf[None, :]).sum(1))),
                      "multiple_of_3": m.shape[1] % 3 == 0})
    M = pd.DataFrame(mrows); C = pd.concat(crows); T = pd.concat(trows)
    M.to_csv(a.out / "matrices.tsv", sep="\t", index=False)
    C.to_csv(a.out / "characters.tsv", sep="\t", index=False)
    T.to_csv(a.out / "intbc_missingness.tsv", sep="\t", index=False)

    # OT cells of the 9 L2/L3 cases -> matrix rows
    leaves = pd.read_csv(QX / "labels" / "tree_leaves.tsv.gz", sep="\t", dtype=str)
    feas = pd.read_csv(REPO / "lineage_validation" / "within_family_subclones" / "output" / "feasibility.tsv", sep="\t", dtype=str)
    cases = pd.read_csv(L2 / "inputs" / "cases.tsv", sep="\t", dtype=str)
    cov, comp_rows = [], []
    for case in cases.itertuples():
        cells = pd.read_csv(RUNS / case.case_id / "cells.tsv.gz", sep="\t", dtype=str)
        if case.dataset == "KPTracer":
            fam = case.families.replace(f"KPTracer_{case.mouse}_", "")
            stem = feas[(feas.dataset == "KPTracer") & (feas.family_or_clone == fam)].tree.iloc[0].replace("_tree.nwk", "")
            lin = {"": (stem, cells.cell.to_numpy())}
            groups = {stem: cells.assign(row=cells.cell)}
        else:  # OT id <site>.<barcode>; a clone matrix lists barcodes; every barcode must sit in at most one matrix
            own = {}
            for (ds, stem), m in loaded.items():
                if ds == "Quinn2021" and stem.startswith(f"m{case.mouse}_"):
                    for bc in m.index:
                        if bc in own:
                            raise SystemExit(f"{bc} in {own[bc]} and {stem}")
                        own[bc] = stem
            # ids differ by experiment: 5k <site>.<bc> on both, 100k / 30k <bc> on both, 10k <site>.<bc> vs <bc>
            bc = cells.cell.where(cells.cell.isin(own.keys()), cells.cell.str.split(".", n=1).str[-1])
            cells = cells.assign(row=bc, lineage=bc.map(own))
            groups = {stem: g for stem, g in cells.dropna(subset=["lineage"]).groupby("lineage")}
        for side in ("source", "target"):
            n = int((cells.side == side).sum())
            in_mat, with_edit, with_inf = 0, 0, 0
            for stem, g in groups.items():
                key = (case.dataset, stem)
                if key not in loaded:
                    continue
                m = loaded[key]; ct = C[C.matrix == stem]; inf = ~ct.founder.to_numpy()
                r = g[(g.side == side) & g.row.isin(m.index)].row
                X = m.loc[r].to_numpy()
                in_mat += len(r); with_edit += int(((X >= 1).sum(1) > 0).sum()); with_inf += int((((X >= 1) & inf[None, :]).sum(1) > 0).sum())
            cov.append({"case_id": case.case_id, "dataset": case.dataset, "side": side, "ot_cells": n,
                        "in_a_matrix": in_mat, "with_any_edit": with_edit, "with_informative_edit": with_inf,
                        "share_in_matrix": in_mat / n if n else np.nan, "share_informative": with_inf / n if n else np.nan})
        for stem, g in groups.items():
            key = (case.dataset, stem)
            if key not in loaded:
                comp_rows.append({"case_id": case.case_id, "lineage": stem, "status": "no matrix"}); continue
            m = loaded[key]; ct = C[C.matrix == stem]; inf = ~ct.founder.to_numpy()
            s = g[(g.side == "source") & g.row.isin(m.index)].row.to_numpy()
            t = g[(g.side == "target") & g.row.isin(m.index)].row.to_numpy()
            if len(s) == 0 or len(t) == 0:
                comp_rows.append({"case_id": case.case_id, "lineage": stem, "status": "one side absent",
                                  "source_cells": len(s), "target_cells": len(t)}); continue
            k = a.pairs
            A = m.loc[rng.choice(s, k)].to_numpy(); B = m.loc[rng.choice(t, k)].to_numpy()
            cp, sh = comparable(A, B, inf)
            comp_rows.append({"case_id": case.case_id, "lineage": stem, "status": "sampled", "source_cells": len(s),
                              "target_cells": len(t), "pairs": k, "share_comparable": float((cp > 0).mean()),
                              "median_comparable_characters": float(np.median(cp)),
                              "share_sharing_informative_edit": float((sh > 0).mean()),
                              "mean_shared_informative_edits": float(sh.mean())})
    pd.DataFrame(cov).to_csv(a.out / "case_coverage.tsv", sep="\t", index=False)
    pd.DataFrame(comp_rows).to_csv(a.out / "pair_comparability.tsv", sep="\t", index=False)
    print(M.groupby("dataset")[["cells", "characters", "missing", "uncut", "edited", "founder_characters"]].describe().T.round(3).to_string())
    print(pd.DataFrame(cov).to_string())


if __name__ == "__main__":
    main()
