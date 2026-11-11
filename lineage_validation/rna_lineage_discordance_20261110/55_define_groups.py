"""DEG / GSEA groups (PLAN_v2.md section 5) from the l5best comparison, defined before any expression is read.

    python lineage_validation/rna_lineage_discordance_20261110/55_define_groups.py --root results/rna_lineage_discordance_20261110

Within one unit x lineage x site x side: "divergent" = the top 10% of div, "convergent" = the top 10% of conv,
"concordant" = div and conv both within the 40-60% band (40_summarize.py flags). A contrast (divergent vs
concordant, convergent vs concordant) is kept when both groups have >= 20 cells.
Cells are used once: KP target (metastasis) cells of every unit; KP source (primary) cells only from the L2/L3 cases;
Quinn both sides of the cases. Writes ROOT/deg_groups.tsv (contrast_id, unit_id, cell, group) and
ROOT/deg_contrasts.tsv (one row per contrast with its sizes, lineage, site, mouse and resolution flag).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MIN = 20


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    C = pd.read_csv(a.root / "summary" / "cells_all.tsv.gz", sep="\t", dtype={"mouse": str, "cell": str}, low_memory=False)
    C = C[C.tag == "l5best"]
    keep = (C.side == "target") | (C.stratum == "case")
    C = C[keep]
    lin = pd.read_csv(a.root / "summary" / "discordance_lineages.tsv", sep="\t", dtype={"mouse": str})
    res = lin[lin.tag == "l5best"].set_index(["unit_id", "lineage"]).resolved
    rows, groups = [], []
    for (u, l, s, sd), g in C.groupby(["unit_id", "lineage", "site", "side"]):
        conc = g[g.concordant]
        for kind in ("divergent", "convergent"):
            case = g[g[kind]]
            if len(case) < MIN or len(conc) < MIN:
                continue
            cid = f"{u}__{l}__{s}__{sd}__{kind}".replace(" ", "_").replace("/", "_")
            rows.append({"contrast_id": cid, "unit_id": u, "dataset": g.dataset.iloc[0], "mouse": g.mouse.iloc[0], "lineage": l,
                         "site": s, "side": sd, "kind": kind, "case_n": len(case), "reference_n": len(conc),
                         "resolved": res.get((u, l), None)})
            groups.append(pd.concat([case.assign(group="case"), conc.assign(group="reference")])[["unit_id", "cell", "group"]].assign(contrast_id=cid))
    R = pd.DataFrame(rows)
    R.to_csv(a.root / "deg_contrasts.tsv", sep="\t", index=False)
    pd.concat(groups)[["contrast_id", "unit_id", "cell", "group"]].to_csv(a.root / "deg_groups.tsv", sep="\t", index=False)
    print(R.groupby(["dataset", "kind"]).agg(contrasts=("contrast_id", "size"), mice=("mouse", "nunique"), case_cells=("case_n", "sum")).to_string())


if __name__ == "__main__":
    main()
