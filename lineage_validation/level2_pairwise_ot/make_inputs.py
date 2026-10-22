"""Case table, run manifest and CRISPR group table for the full Level 2 run (PLAN.md), from the frozen subclone
definitions.

    python lineage_validation/level2_pairwise_ot/make_inputs.py

Reads ../within_family_subclones/output (feasibility.tsv, subclones.tsv, cells/<family>.csv.gz, made by
define_subclones.py under SUBCLONE_RULES.md). Every family / clone with subclone_structure_eligible = True
is included; none is dropped for being hard to evaluate (the scorer marks those not_evaluable). Writes inputs/:
  cases.tsv           index, case_id, dataset, mouse, source_site, target_sites (';', or ALL), families (';'),
                      sites_left_out (json), estimated_entries, resource (standard / large)
  subclones.tsv       name, level, subclone of every family in cases.tsv
  family_cells.tsv.gz name, dataset, mouse, cell_id, site, reliable, L1, L2 (every family cell, both sides)
  run_manifest.tsv    one row per case x method (A_standard_OT, B_our_OT; methods.py): the dry-run manifest
A case is one plan from a primary to its pooled metastases:
  KP-Tracer: one case per eligible family; source = the family's primary tumour; target = every metastatic
             site with >= 20 reliable family cells carrying a level-1 subclone label.
  Quinn:     one case per mouse with an eligible clone; source = the LL side, target = the lineage-validation
             metastases side (ALL: every tissue with >= 20 cells); families = the mouse's eligible clones.
The rule reads CRISPR composition only. resource = large when (primary family cells) x (metastatic family
cells) > 2e6; the product is a lower bound on the plan size (unlabelled cells are not counted), used only to
set the SLURM time limit.
The OT run never reads these labels before the coupling exists: run_pairwise.py uses them only to sum the
finished coupling by source subclone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from methods import EPSILON, METHODS  # noqa: E402

DEFS = HERE.parent / "within_family_subclones" / "output"
MIN_SITE_CELLS = 20
LARGE_ENTRIES = 2e6


def main() -> None:
    feas = pd.read_csv(DEFS / "feasibility.tsv", sep="\t", dtype=str)
    feas = feas[feas["subclone_structure_eligible"].str.lower().eq("true")]
    subs = pd.read_csv(DEFS / "subclones.tsv", sep="\t", dtype=str)
    l1 = subs[subs["level"].eq("1")].groupby("name")["subclone"].apply(set)
    read_cells = lambda name: pd.read_csv(DEFS / "cells" / f"{name}.csv.gz", dtype=str, keep_default_na=False)
    cases = []
    for _, row in feas[feas["dataset"].eq("KPTracer")].iterrows():
        fam = f"KPTracer_{row['mouse']}_{row['family_or_clone']}"
        cells = read_cells(fam)
        lab = cells[cells["reliable"].eq("True") & cells["L1"].isin(l1[fam])]
        met = lab[lab["kind"].eq("metastasis")]["site"].value_counts()
        sites = sorted(met.index[met >= MIN_SITE_CELLS])
        cases.append({"dataset": "KPTracer", "mouse": row["mouse"], "source_site": row["family_or_clone"],
                      "target_sites": ";".join(sites), "families": fam,
                      "sites_left_out": json.dumps({s: int(c) for s, c in met.items() if c < MIN_SITE_CELLS}),
                      "estimated_entries": int(lab["kind"].eq("primary").sum()) * int(met[sites].sum()),
                      "case_id": f"KPTracer_{row['mouse']}__{row['family_or_clone']}__family_metastases"})
    for mouse, rows in feas[feas["dataset"].eq("Quinn2021")].groupby("mouse", sort=False):
        fams = [f"Quinn2021_{mouse}_{c}" for c in rows["family_or_clone"]]
        sites = json.loads(rows["sites"].iloc[0])
        ll = [s for s in sites if s.replace(mouse, "").replace("30k", "") == "LL" or s == "LL"]
        if len(ll) != 1:
            raise ValueError(f"Quinn {mouse}: LL site {ll}")
        n_src = n_tgt = 0
        for fam in fams:
            lab = read_cells(fam)
            lab = lab[lab["reliable"].eq("True") & lab["L1"].isin(l1[fam])]
            n_src += int(lab["site"].eq(ll[0]).sum())
            n_tgt += int((~lab["site"].eq(ll[0])).sum())
        cases.append({"dataset": "Quinn2021", "mouse": mouse, "source_site": ll[0], "target_sites": "ALL",
                      "families": ";".join(fams), "sites_left_out": "{}", "estimated_entries": n_src * n_tgt,
                      "case_id": f"Quinn2021_{mouse}__LL__metastases"})
    cases = pd.DataFrame(cases)
    cases.insert(0, "index", range(len(cases)))
    cases["resource"] = ["large" if e > LARGE_ENTRIES else "standard" for e in cases["estimated_entries"]]
    cases = cases[["index", "case_id", "dataset", "mouse", "source_site", "target_sites", "families", "sites_left_out",
                   "estimated_entries", "resource"]]
    names = sorted({f for fs in cases["families"] for f in fs.split(";")})
    if len(names) != len(feas):
        raise ValueError(f"{len(names)} families in the cases, {len(feas)} eligible")
    out = HERE / "inputs"
    out.mkdir(exist_ok=True)
    cases.to_csv(out / "cases.tsv", sep="\t", index=False, lineterminator="\n")
    subs[subs["name"].isin(names)][["name", "level", "subclone"]].to_csv(out / "subclones.tsv", sep="\t", index=False, lineterminator="\n")
    cells = []
    for name in names:
        table = read_cells(name)
        if table["cell_id"].duplicated().any():
            raise ValueError(f"{name}: duplicated cell ids")
        dataset, mouse = name.split("_")[:2]
        cells.append(table.assign(name=name, dataset=dataset, mouse=mouse)[["name", "dataset", "mouse", "cell_id", "site", "reliable", "L1", "L2"]])
    cells = pd.concat(cells)
    cells.to_csv(out / "family_cells.tsv.gz", sep="\t", index=False, compression="gzip")
    manifest = []
    for _, r in cases.iterrows():
        fam = cells[cells["name"].isin(r["families"].split(";")) & cells["reliable"].eq("True")]
        fam = fam[[x in l1[n] for n, x in zip(fam["name"], fam["L1"])]]
        by_site = {s: int(c) for s, c in fam["site"].value_counts().sort_index().items()}
        for method, m in METHODS.items():
            manifest.append({"index": r["index"], "case_id": r["case_id"], "method": method, **m, "epsilon": EPSILON,
                             "source_site": r["source_site"], "target_sites": r["target_sites"], "families": r["families"],
                             "resource": r["resource"], "level1_labelled_family_cells_by_site": json.dumps(by_site),
                             "output": f"<root>/level2_pairwise_ot/full/{r['case_id']}/{method}"})
    pd.DataFrame(manifest).to_csv(out / "run_manifest.tsv", sep="\t", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(cases.drop(columns=["families"]).to_string(index=False))
    print(f"{len(cases)} cases, {len(names)} families, {len(manifest)} OT runs")


if __name__ == "__main__":
    main()
