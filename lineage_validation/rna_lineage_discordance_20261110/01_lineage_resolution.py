"""Step 1 of PLAN.md: lineages per mouse, their CRISPR resolution, and the all-pairs edit agreement s_ij (no RNA).

    python lineage_validation/rna_lineage_discordance_20261110/01_lineage_resolution.py \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --bundles results/crispr_matrix_benchmark_20261108/bundles \
        --out results/rna_lineage_discordance_20261110/lineages

A lineage = one KP MetFamily / Quinn LineageGroup within one mouse (experiment); its cells = the union of the OT cells
of every bundle unit of that mouse (both sides). The allele-table states, priors and s_ij are those of
crispr_matrix_benchmark_20261108/matrix_core.py (MATRIX_METRICS.md), here over all pairs of the lineage.
Writes OUT/resolution.tsv (every lineage, resolved or why not), OUT/<mouse>__<lineage>.npz for resolved lineages
(cells, site, informative events per cell, s and evaluable as condensed upper triangles) and OUT/cells.tsv.gz.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MX = REPO / "lineage_validation" / "crispr_matrix_benchmark_20261108"
sys.path.insert(0, str(MX))
import matrix_core as mc  # noqa: E402

MIN_CELLS, MIN_EVENTS, MIN_DISTINCT, MIN_IQR = 50, 3, 0.1, 0.05


def bundles_module():
    spec = importlib.util.spec_from_file_location("bb", MX / "01_build_bundles.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--bundles", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    bb = bundles_module()
    tabs = bb.load_tables(a.inputs)
    units = pd.read_csv(a.inputs / "lineage_preprocessing_factorial_20261103" / "UNITS.tsv", sep="\t", dtype=str).set_index("unit_id")
    # cells per mouse from the bundles: id, side role, site, mapped allele-table id, lineage
    rows = []
    for f in sorted(glob.glob(str(a.bundles / "*.pkl"))):
        b = pickle.load(open(f, "rb")); u = units.loc[b["unit_id"]]
        key = "KPTracer" if b["dataset"] == "KPTracer" else f"Quinn2021:{b['mouse']}"
        t = tabs[key][0]
        ids = pd.Index(t.drop_duplicates("cell").cell)
        for side in ("source", "target"):
            cid = pd.Series(b[f"{side}_ids"])
            mapped = bb.map_ids(cid, ids)
            site = b["target_site"] if side == "target" else np.repeat(u.source if b["dataset"] == "KPTracer" else "LL", len(cid))
            rows.append(pd.DataFrame({"dataset": b["dataset"], "mouse": b["mouse"], "cell": cid, "table_cell": mapped,
                                      "site": site, "lineage": b[f"{side}_lineage"]}))
    cells = pd.concat(rows).drop_duplicates(["dataset", "mouse", "cell"]).reset_index(drop=True)
    cells.to_csv(a.out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    res = []
    for (ds, mouse, lin), g in cells.dropna(subset=["lineage"]).groupby(["dataset", "mouse", "lineage"]):
        key = "KPTracer" if ds == "KPTracer" else f"Quinn2021:{mouse}"
        t, prior = tabs[key]
        rec = {"dataset": ds, "mouse": mouse, "lineage": lin, "cells": len(g), "sites": g.site.nunique()}
        if len(g) < MIN_CELLS:
            res.append({**rec, "resolved": False, "reason": f"< {MIN_CELLS} cells"}); continue
        L = mc.Lineage(t[t.lineage == lin], g.table_cell.to_numpy(), np.repeat("source", len(g)), prior)
        inf = np.asarray(L.incidence.sum(1)).ravel()
        sets = pd.Series([tuple(r.indices) for r in L.incidence.tocsr()]).nunique()
        rows_ = np.arange(len(g))
        s, ok = L.agreement(rows_, rows_)
        iu = np.triu_indices(len(g), 1)
        sv, okv = s[iu], ok[iu]
        e = sv[okv]
        rec.update(informative_events=len(L.events), founder_characters=L.founder_characters,
                   median_events_per_cell=float(np.median(inf)), distinct_sets_share=sets / len(g),
                   pairs_evaluable=float(okv.mean()), pairs_strictly_between=float(((e > 0) & (e < 1)).mean()) if e.size else np.nan,
                   s_iqr=float(np.subtract(*np.percentile(e, [75, 25]))) if e.size else np.nan, s_median=float(np.median(e)) if e.size else np.nan)
        why = [w for w, bad in ((f"median events per cell < {MIN_EVENTS}", rec["median_events_per_cell"] < MIN_EVENTS),
                                (f"distinct event sets / cells < {MIN_DISTINCT}", rec["distinct_sets_share"] < MIN_DISTINCT),
                                (f"IQR of s < {MIN_IQR}", not rec["s_iqr"] >= MIN_IQR)) if bad]
        rec.update(resolved=not why, reason="; ".join(why))
        res.append(rec)
        if not why:
            np.savez_compressed(a.out / f"{ds}__{mouse}__{lin}.npz", cell=g.cell.to_numpy(dtype=str), site=g.site.to_numpy(dtype=str),
                                events_per_cell=inf.astype(np.int32), s=sv.astype(np.float32), ok=okv)
        print(f"{ds} {mouse} {lin}: {len(g)} cells, median events {rec['median_events_per_cell']:.0f}, "
              f"IQR {rec['s_iqr']:.3f}, {'resolved' if not why else why}", flush=True)
    R = pd.DataFrame(res)
    R.to_csv(a.out / "resolution.tsv", sep="\t", index=False)
    print(R.groupby(["dataset", "resolved"]).agg(lineages=("lineage", "size"), cells=("cells", "sum")))


if __name__ == "__main__":
    main()
