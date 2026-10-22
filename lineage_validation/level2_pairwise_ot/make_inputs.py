"""Pilot table and CRISPR group table for Level 2 (PLAN.md), written from the frozen subclone definitions.

    python lineage_validation/level2_pairwise_ot/make_inputs.py

Reads ../within_family_subclones/output (feasibility.tsv, subclones.tsv, cells/<family>.csv.gz, made by
define_subclones.py under SUBCLONE_RULES.md). Writes inputs/:
  pilot_pairs.tsv     index, pair_id, dataset, mouse, source_site, target_sites (';', or ALL), families (';')
  subclones.tsv       name, level, subclone of every family named in pilot_pairs.tsv
  family_cells.tsv.gz name, dataset, mouse, cell_id, site, reliable, L1, L2 (every family cell, source and
                      target sites)
Target (PLAN.md, 2026-10-22 decision): one plan from the family's primary to all its metastatic sites.
KP-Tracer: every metastatic site with >= 20 reliable family cells carrying a level-1 subclone label.
Quinn: the lineage-validation metastases side (every tissue with >= 20 cells, ALL), since every tissue
holds cells of every clone. The rule reads CRISPR composition only.
The OT run never reads these labels before the coupling exists: run_pairwise.py uses them only to sum the
finished coupling by source subclone.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DEFS = HERE.parent / "within_family_subclones" / "output"
MIN_SITE_CELLS = 20
# Fixed in PLAN.md before any Level 2 coupling (CRISPR composition only).
PILOT = [("KPTracer", "3457", "3457_Apc_T1"), ("KPTracer", "3515", "3515_Lkb1_T1"), ("KPTracer", "3724", "3724_NT_T1"),
         ("Quinn2021", "10k", "LL")]


def main() -> None:
    feas = pd.read_csv(DEFS / "feasibility.tsv", sep="\t", dtype=str)
    feas = feas[feas["subclone_structure_eligible"].str.lower().eq("true")]
    subs = pd.read_csv(DEFS / "subclones.tsv", sep="\t", dtype=str)
    rows, names = [], []
    for index, (dataset, mouse, src) in enumerate(PILOT):
        mine = feas[feas["dataset"].eq(dataset) & feas["mouse"].eq(mouse)]
        # KP-Tracer: the family is the primary tumour itself; Quinn: every eligible clone of the mouse.
        mine = mine[mine["family_or_clone"].eq(src)] if dataset == "KPTracer" else mine
        fams = [f"{dataset}_{mouse}_{f}" for f in mine["family_or_clone"]]
        if not fams:
            raise ValueError(f"{dataset} {mouse} {src}: no eligible family")
        if dataset == "KPTracer":
            (fam,) = fams
            cells = pd.read_csv(DEFS / "cells" / f"{fam}.csv.gz", dtype=str, keep_default_na=False)
            l1 = set(subs.loc[subs["name"].eq(fam) & subs["level"].eq("1"), "subclone"])
            met = cells[cells["kind"].eq("metastasis") & cells["reliable"].eq("True") & cells["L1"].isin(l1)]
            counts = met["site"].value_counts()
            sites = sorted(counts.index[counts >= MIN_SITE_CELLS])
            left_out = {s: int(c) for s, c in counts.items() if c < MIN_SITE_CELLS}
            target, pair_id = ";".join(sites), f"{dataset}_{mouse}__{src}__family_metastases"
        else:
            target, pair_id, left_out = "ALL", f"{dataset}_{mouse}__{src}__metastases", {}
        rows.append({"index": index, "pair_id": pair_id, "dataset": dataset, "mouse": mouse, "source_site": src,
                     "target_sites": target, "families": ";".join(fams), "sites_left_out": json.dumps(left_out)})
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
