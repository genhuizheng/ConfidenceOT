"""Pilot pair table and CRISPR group table for Level 2 (PLAN.md), written from the frozen subclone definitions.

    python lineage_validation/level2_pairwise_ot/make_inputs.py

Reads ../within_family_subclones/output (feasibility.tsv, subclones.tsv, cells/<family>.csv.gz, made by
define_subclones.py under SUBCLONE_RULES.md). Writes inputs/:
  pilot_pairs.tsv     index, pair_id, dataset, mouse, source_site, target_site, families (';')
  subclones.tsv       name, level, subclone of every family named in pilot_pairs.tsv
  family_cells.tsv.gz name, dataset, mouse, cell_id, site, reliable, L1, L2 (every family cell, source and
                      target sites)
The OT run never reads these labels before the coupling exists: run_pairwise.py uses them only to sum the
finished coupling by source subclone.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFS = HERE.parent / "within_family_subclones" / "output"
# Fixed in PLAN.md before any Level 2 coupling (CRISPR composition only).
PILOT = [("KPTracer", "3457", "3457_Apc_T1", "3457_Apc_K1"),
         ("KPTracer", "3515", "3515_Lkb1_T1", "3515_Lkb1_L2"),
         ("KPTracer", "3515", "3515_Lkb1_T1", "3515_Lkb1_K2"),
         ("KPTracer", "3724", "3724_NT_T1", "3724_NT_S1"),
         ("KPTracer", "3724", "3724_NT_T1", "3724_NT_L2"),
         ("Quinn2021", "10k", "LL", "M2")]


def main() -> None:
    feas = pd.read_csv(DEFS / "feasibility.tsv", sep="\t", dtype=str)
    feas = feas[feas["subclone_structure_eligible"].str.lower().eq("true")]
    subs = pd.read_csv(DEFS / "subclones.tsv", sep="\t", dtype=str)
    rows, names = [], []
    for index, (dataset, mouse, src, tgt) in enumerate(PILOT):
        mine = feas[feas["dataset"].eq(dataset) & feas["mouse"].eq(mouse)]
        # KP-Tracer: the family is the primary tumour itself; Quinn: every eligible clone of the mouse.
        mine = mine[mine["family_or_clone"].eq(src)] if dataset == "KPTracer" else mine
        fams = [f"{dataset}_{mouse}_{f}" for f in mine["family_or_clone"]]
        if not fams:
            raise ValueError(f"{dataset} {mouse} {src}: no eligible family")
        rows.append({"index": index, "pair_id": f"{dataset}_{mouse}__{src}__{tgt}", "dataset": dataset, "mouse": mouse,
                     "source_site": src, "target_site": tgt, "families": ";".join(fams)})
        names += fams
    names = sorted(set(names))
    out = HERE / "inputs"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "pilot_pairs.tsv", sep="\t", index=False)
    subs[subs["name"].isin(names)][["name", "level", "subclone"]].to_csv(out / "subclones.tsv", sep="\t", index=False)
    cells = []
    for name in names:
        table = pd.read_csv(DEFS / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
        if table["cell_id"].duplicated().any():
            raise ValueError(f"{name}: duplicated cell ids")
        dataset, mouse = name.split("_")[:2]
        cells.append(table.assign(name=name, dataset=dataset, mouse=mouse)[["name", "dataset", "mouse", "cell_id", "site", "reliable", "L1", "L2"]])
    pd.concat(cells).to_csv(out / "family_cells.tsv.gz", sep="\t", index=False, compression="gzip")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
