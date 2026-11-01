"""Lineage-label alignment audit of the existing KP-Tracer / Quinn evaluation (read-only, local copies).

    python realdata_benchmark/06_lineage_label_audit.py --lineage-root lineage_validation --out OUT_DIR

Two different "site" vocabularies exist and are kept apart here:
  anatomical site      the tissue a cell was sampled from (KP-Tracer tumour / metastasis names such as 3454_Lkb1_N1;
                       Quinn sampleID tissues such as LL, the injection site). Columns: site, site_kind.
  CRISPR editing site  a Cas9 target site / intBC of the lineage recorder (allele tables on TACC, not local). None of
                       the tables read here hold editing-site data; the lineage labels L1 / L2 and the families are
                       derived from them by the authors' trees.
For every Level 2 case, the cells the Level 2 / 3 runs used (data/full_*/full/<case>/cells.tsv.gz) are matched by cell
ID to the Level 2 family table (inputs/family_cells.tsv.gz), the lineage-site cell annotation
(lineage_site_application/lineage_site_matrices/cells_annotation.tsv.gz) and the Level 3 ground truth
(level3_cell_lineage/results/ground_truth/targets.tsv.gz): anatomical site and L1 / L2 must agree wherever two tables
hold the cell. Writes OUT/LINEAGE_LABEL_AUDIT.tsv (per case and table pair: cells compared, missing, disagreements by
field) and prints the totals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def norm(v):
    return "" if pd.isna(v) else str(v)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lineage-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "LINEAGE_LABEL_AUDIT.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    L = a.lineage_root
    fam = pd.read_csv(L / "level2_pairwise_ot" / "inputs" / "family_cells.tsv.gz", sep="\t")
    ann = pd.read_csv(L / "lineage_site_application" / "lineage_site_matrices" / "cells_annotation.tsv.gz", sep="\t")
    gt = pd.read_csv(L / "level3_cell_lineage" / "results" / "ground_truth" / "targets.tsv.gz", sep="\t")
    cases = pd.read_csv(L / "level2_pairwise_ot" / "inputs" / "cases.tsv", sep="\t")
    runs = sorted((L / "level2_pairwise_ot" / "data").glob("full_*/full"))
    if not runs:
        sys.exit("no Level 2 run directory")
    rows = []
    for _, case in cases.iterrows():
        f = runs[-1] / case.case_id / "cells.tsv.gz"
        if not f.is_file():
            rows.append({"case_id": case.case_id, "table": "level2 cells.tsv", "status": "missing"}); continue
        cells = pd.read_csv(f, sep="\t").drop_duplicates("cell").set_index("cell")
        ds, mouse = case.dataset, str(case.mouse)
        refs = {
            "family_cells": fam[(fam.dataset == ds) & (fam.mouse.astype(str) == mouse)].drop_duplicates("cell_id").set_index("cell_id"),
            "cells_annotation": ann[(ann.dataset == ds) & (ann.mouse.astype(str) == mouse)].drop_duplicates("cell").set_index("cell"),
            "level3_targets": gt[gt.case_id == case.case_id].drop_duplicates("cell").set_index("cell"),
        }
        for name, ref in refs.items():
            shared = cells.index.intersection(ref.index)
            r = {"case_id": case.case_id, "dataset": ds, "mouse": mouse, "table": name, "status": "compared",
                 "level2_cells": len(cells), "table_cells": len(ref), "shared": len(shared),
                 "level2_cells_absent_from_table": int(len(cells) - len(shared))}
            r["site_disagree"] = int(sum(norm(cells.at[c, "site"]) != norm(ref.at[c, "site"]) for c in shared)) if "site" in ref else None
            rows.append(r)
        # lineage labels between the three label tables, on their shared cells
        for x, y in (("family_cells", "cells_annotation"), ("family_cells", "level3_targets"), ("cells_annotation", "level3_targets")):
            A, B = refs[x], refs[y]; shared = A.index.intersection(B.index)
            r = {"case_id": case.case_id, "dataset": ds, "mouse": mouse, "table": f"{x} vs {y}", "status": "compared", "shared": len(shared)}
            for col in ("site", "L1", "L2", "reliable"):
                if col in A and col in B:
                    va = [norm(A.at[c, col]) for c in shared]; vb = [norm(B.at[c, col]) for c in shared]
                    r[f"{col}_disagree"] = int(sum(x != y for x, y in zip(va, vb)))
                    if col in ("L1", "L2"):
                        # a conflict is two different subclone labels; empty vs 'ambiguous' / a label of a family
                        # not eligible in one table is an encoding difference (unreliable leaf, eligibility), not a conflict
                        labelled = [(x, y) for x, y in zip(va, vb) if x not in ("", "ambiguous") and y not in ("", "ambiguous")]
                        r[f"{col}_both_labelled"] = len(labelled)
                        r[f"{col}_conflict_between_labels"] = int(sum(x != y for x, y in labelled))
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(target, sep="\t", index=False)
    cols = [c for c in out if c.endswith("_disagree")]
    print(out.groupby("table")[["shared"] + cols].sum(min_count=1).to_string())
    print(f"-> {target}")


if __name__ == "__main__":
    main()
