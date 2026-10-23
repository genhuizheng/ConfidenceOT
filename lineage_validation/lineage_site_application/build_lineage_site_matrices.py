"""Phase 1: CRISPR lineage x site ground truth for every KP-Tracer and Quinn mouse (GROUND_TRUTH_DEFINITIONS.md).

    python lineage_validation/lineage_site_application/build_lineage_site_matrices.py --out lineage_site_matrices

Reads only the authors' annotations and the frozen subclone tables; no OT output. Writes OUT/:
  cells_annotation.tsv.gz             every malignant cell: dataset, mouse, cell, site, site_kind, lineage (family /
                                      clone) and its status, family_or_clone of its subclone table, L1, L2 and the
                                      subclone status (labelled / ambiguous / unknown: unreliable / no tree leaf /
                                      not eligible)
  <dataset>_<mouse>/lineage_counts.tsv     lineage x site counts (primaries first), unknown as its own row
  <dataset>_<mouse>/lineage_fractions.tsv  each site's column divided by the site's cells
  <dataset>_<mouse>/lineage_sharing.tsv    per lineage: cells in primaries / metastases, sites, sharing across the
                                      sampled sites (shared_across_sites / primary_sites_only / metastatic_sites_only)
  <dataset>_<mouse>/subclone_<family>_L<k>_counts.tsv / _fractions.tsv   the same at the subclone level
  coverage.tsv                        per mouse and family / clone: cells, unknown lineage, tree leaves, reliable
                                      leaves, subclone-labelled, ambiguous
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LV = HERE.parent
REPO = LV.parent
KP_META = REPO / "results" / "kp_tracer_lineage_validation" / "export" / "metadata" / "kptracer_meta_subset.csv.gz"
QUINN_META = LV / "local_lineage_visualization" / "data" / "local_lineage_visualization_export" / "meta"
DEFS = LV / "within_family_subclones" / "output"
QUINN_MICE = ["5k", "10k", "30k", "100k"]


def kp_site_kind(tumor: str) -> str:
    return "primary" if re.search(r"_T\d+$", tumor) else "metastasis"


def quinn_site_kind(site: str, mouse: str) -> str:
    return "primary" if site.replace(mouse, "") == "LL" else "metastasis"


def load_cells() -> pd.DataFrame:
    m = pd.read_csv(KP_META, low_memory=False, usecols=["cell_id", "Tumor", "Mouse", "MetFamily", "is_normal"])
    m = m[m["Tumor"].notna() & ~m["is_normal"].astype(bool)]
    kp = pd.DataFrame({"dataset": "KPTracer", "mouse": m["Tumor"].str.split("_").str[0], "cell": m["cell_id"].astype(str),
                       "site": m["Tumor"].astype(str), "lineage": m["MetFamily"].astype(str)})
    kp["site_kind"] = kp["site"].map(kp_site_kind)
    kp["lineage_status"] = np.where(kp["lineage"].str.endswith("_Unk"), "unknown", "known")
    rows = [kp]
    for mouse in QUINN_MICE:
        q = pd.read_csv(QUINN_META / f"Quinn2021_obs_{mouse}.csv.gz", low_memory=False, usecols=["cell_id", "LineageGroup", "sampleID"])
        q = q[q["sampleID"].notna()]
        lab = q["LineageGroup"]
        d = pd.DataFrame({"dataset": "Quinn2021", "mouse": mouse, "cell": q["cell_id"].astype(str), "site": q["sampleID"].astype(str),
                          "lineage": np.where(lab.notna(), "LG" + lab.astype("Int64").astype(str), "unknown")})
        d["site_kind"] = [quinn_site_kind(s, mouse) for s in d["site"]]
        d["lineage_status"] = np.where(lab.notna(), "known", "unknown")
        rows.append(d)
    cells = pd.concat(rows, ignore_index=True)
    if cells.duplicated(["dataset", "mouse", "cell"]).any():
        raise ValueError("a cell id repeats within a mouse")
    return cells


def add_subclones(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    feas = pd.read_csv(DEFS / "feasibility.tsv", sep="\t", dtype=str)
    cells["family_or_clone"], cells["L1"], cells["L2"], cells["subclone_status"] = "", "", "", "not eligible"
    cov = []
    for _, f in feas.iterrows():
        name = f"{f['dataset']}_{f['mouse']}_{f['family_or_clone']}"
        path = DEFS / "cells" / f"{name}.csv.gz"
        eligible = str(f["subclone_structure_eligible"]).lower() == "true"
        lineage = f["family_or_clone"] if f["dataset"] == "KPTracer" else f["family_or_clone"]
        sel = cells["dataset"].eq(f["dataset"]) & cells["mouse"].eq(str(f["mouse"])) & cells["lineage"].eq(lineage)
        n_lin = int(sel.sum())
        row = {"dataset": f["dataset"], "mouse": f["mouse"], "family_or_clone": f["family_or_clone"], "eligible": eligible,
               "lineage_cells": n_lin, "tree": f.get("tree", ""), "reason": f.get("reason", "")}
        if not path.exists():
            cov.append({**row, "tree_leaves": 0, "reliable_leaves": 0, "labelled_L1": 0, "ambiguous_L1": 0})
            continue
        t = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("cell_id")
        idx = cells.index[sel]
        in_tree = cells.loc[idx, "cell"].isin(t.index)
        cells.loc[idx, "family_or_clone"] = name
        cells.loc[idx[~in_tree.to_numpy()], "subclone_status"] = "unknown: no tree leaf"
        hit = idx[in_tree.to_numpy()]
        tt = t.loc[cells.loc[hit, "cell"]]
        rel = tt["reliable"].eq("True").to_numpy()
        cells.loc[hit, "L1"] = np.where(rel, tt["L1"].to_numpy(), "")
        cells.loc[hit, "L2"] = np.where(rel, tt["L2"].to_numpy(), "")
        status = np.where(~rel, "unknown: unreliable leaf (missing > 0.5)",
                          np.where(tt["L1"].eq("ambiguous").to_numpy(), "ambiguous",
                                   np.where(tt["L1"].str.startswith("L").to_numpy(), "labelled", "unknown: no subclone")))
        if not eligible:
            status = np.where(status == "labelled", "labelled (family not eligible)", status)
        cells.loc[hit, "subclone_status"] = status
        cov.append({**row, "tree_leaves": int(in_tree.sum()), "reliable_leaves": int(rel.sum()),
                    "labelled_L1": int((status == "labelled").sum()), "ambiguous_L1": int((status == "ambiguous").sum())})
    return cells, pd.DataFrame(cov)


def matrices(d: pd.DataFrame, row: str, out: Path, stem: str):
    sites = d.drop_duplicates("site")[["site", "site_kind"]]
    order = sites.sort_values(["site_kind", "site"], key=lambda s: s.map({"primary": 0, "metastasis": 1}) if s.name == "site_kind" else s)["site"].tolist()
    counts = pd.crosstab(d[row], d["site"]).reindex(columns=order, fill_value=0)
    unk = [r for r in counts.index if str(r).startswith("unknown") or r in ("ambiguous",)]
    counts = counts.loc[[r for r in counts.index if r not in unk] + unk]
    kind = sites.set_index("site")["site_kind"].reindex(order)
    header = pd.DataFrame([kind.to_numpy()], columns=order, index=["site_kind"])
    pd.concat([header, counts.astype(object)]).to_csv(out / f"{stem}_counts.tsv", sep="\t")
    frac = counts / counts.sum(0).replace(0, np.nan)
    pd.concat([header, frac.round(6).astype(object)]).to_csv(out / f"{stem}_fractions.tsv", sep="\t")
    return counts, kind


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cells = load_cells()
    cells, cov = add_subclones(cells)
    args.out.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out / "cells_annotation.tsv.gz", sep="\t", index=False, compression="gzip")
    mouse_cov = []
    for (ds, mouse), d in cells.groupby(["dataset", "mouse"], sort=False):
        o = args.out / f"{ds}_{mouse}"
        o.mkdir(exist_ok=True)
        lab = d["lineage"].where(d["lineage_status"].eq("known"), "unknown")
        counts, kind = matrices(d.assign(lin=lab), "lin", o, "lineage")
        prim, met = kind.index[kind.eq("primary")], kind.index[kind.eq("metastasis")]
        share = pd.DataFrame({"cells_primary": counts[prim].sum(1), "cells_metastatic": counts[met].sum(1),
                              "primary_sites": (counts[prim] > 0).sum(1), "metastatic_sites": (counts[met] > 0).sum(1),
                              "sites": [";".join(counts.columns[counts.loc[r] > 0]) for r in counts.index]})
        share["sharing"] = np.where(share.index == "unknown", "unknown",
                                    np.where((share.cells_primary > 0) & (share.cells_metastatic > 0), "shared_across_sites",
                                             np.where(share.cells_primary > 0, "primary_sites_only", "metastatic_sites_only")))
        share.index.name = "lineage"
        share.to_csv(o / "lineage_sharing.tsv", sep="\t")
        for fam, g in d[d["family_or_clone"].ne("")].groupby("family_or_clone"):
            for level in ("L1", "L2"):
                lab2 = np.where(g["subclone_status"].isin(["labelled", "labelled (family not eligible)"]) & g[level].str.startswith("L"),
                                g[level], np.where(g["subclone_status"].eq("ambiguous"), "ambiguous", g["subclone_status"].str.replace("labelled.*", "unknown: no subclone at this level", regex=True)))
                if not (pd.Series(lab2).str.startswith("L")).any():
                    continue
                matrices(g.assign(sub=lab2), "sub", o, f"subclone_{fam.split('_', 2)[2]}_{level}")
        n = len(d)
        mouse_cov.append({"dataset": ds, "mouse": mouse, "cells": n, "sites": d["site"].nunique(),
                          "primary_sites": d.loc[d.site_kind.eq("primary"), "site"].nunique(),
                          "metastatic_sites": d.loc[d.site_kind.eq("metastasis"), "site"].nunique(),
                          "lineages_known": int(share.index.drop("unknown", errors="ignore").size),
                          "unknown_lineage_share": float((d["lineage_status"] == "unknown").mean()),
                          "lineages_shared_across_sites": int((share["sharing"] == "shared_across_sites").sum()),
                          "lineages_primary_sites_only": int((share["sharing"] == "primary_sites_only").sum()),
                          "lineages_metastatic_sites_only": int((share["sharing"] == "metastatic_sites_only").sum()),
                          "cells_in_subclone_tables": int(d["family_or_clone"].ne("").sum()),
                          "cells_subclone_labelled": int(d["subclone_status"].eq("labelled").sum())})
    pd.DataFrame(mouse_cov).to_csv(args.out / "coverage_mice.tsv", sep="\t", index=False)
    cov.to_csv(args.out / "coverage_families.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(mouse_cov).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
