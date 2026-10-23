"""Tests of the Phase 1 ground truth (local).

    python lineage_validation/lineage_site_application/tests/test_ground_truth.py

Checks: the site-kind rules; the built matrices against the authors' annotations (column sums equal the site's
cells, unknown kept as its own row); the frozen Level 1 labels agree cell by cell where they exist locally; the
sharing classes follow the counts; no OT output is read by the builder.
Prints PASS / FAIL; exit 1 on any failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(APP))
import build_lineage_site_matrices as b  # noqa: E402

RESULTS = []
OUT = APP / "lineage_site_matrices"


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    check("KP site kind: T<n> primary, everything else metastasis",
          [b.kp_site_kind(x) for x in ["3515_Lkb1_T1", "3515_Lkb1_T12", "3515_Lkb1_N1", "3454_Lkb1_Rib", "3465_Lkb1_DN1"]]
          == ["primary", "primary", "metastasis", "metastasis", "metastasis"])
    check("Quinn site kind: LL / 30kLL primary, other tissues metastatic",
          [b.quinn_site_kind(s, m) for s, m in [("LL", "5k"), ("30kLL", "30k"), ("30kM1", "30k"), ("Liv", "5k")]]
          == ["primary", "primary", "metastasis", "metastasis"])
    src = Path(b.__file__).read_text(encoding="utf-8")
    check("the builder reads no OT output (no gates, couplings or cell_confidence)",
          not any(k in src for k in ("cell_confidence", "coupling", "level2_pairwise_ot", "gates/")))
    cells = pd.read_csv(OUT / "cells_annotation.tsv.gz", sep="\t", dtype=str)
    ok, detail = True, []
    for (ds, m), d in cells.groupby(["dataset", "mouse"]):
        c = pd.read_csv(OUT / f"{ds}_{m}" / "lineage_counts.tsv", sep="\t", index_col=0)
        c = c.drop(index="site_kind").astype(int)
        per_site = d["site"].value_counts()
        same = (c.sum(0).sort_index() == per_site.reindex(c.columns).sort_index()).all()
        unk = (d["lineage_status"] == "unknown").sum()
        ok &= same and (unk == 0 or c.loc["unknown"].sum() == unk)
        if not same:
            detail.append(f"{ds} {m}")
    check("every mouse: column sums equal the site's cells, unknown cells only in the unknown row", ok, ";".join(detail))
    s = pd.read_csv(OUT / "KPTracer_3515" / "lineage_sharing.tsv", sep="\t", index_col=0)
    check("3515: T1 family shared across sites, T2 and T3 primary-only",
          s.loc["3515_Lkb1_T1", "sharing"] == "shared_across_sites" and set(s.loc[["3515_Lkb1_T2", "3515_Lkb1_T3"], "sharing"]) == {"primary_sites_only"})
    s60 = pd.read_csv(OUT / "KPTracer_3460" / "lineage_sharing.tsv", sep="\t", index_col=0)
    check("3460: the N1 family without a sampled primary is metastatic-sites-only (protocol 5.3)",
          (s60["sharing"] == "metastatic_sites_only").sum() == 1)
    lab = pd.read_csv(APP.parent / "local_lineage_visualization" / "data" / "local_lineage_visualization_export" / "labels" / "cells.tsv.gz",
                      sep="\t", dtype=str, keep_default_na=False)
    agree = 0
    for (ds, m), g in lab.groupby(["dataset", "mouse"]):
        mine = cells[(cells.dataset == ds) & (cells.mouse == m)].set_index("cell")
        j = g.set_index("cell").join(mine[["lineage", "site"]], rsuffix="_new", how="inner")
        new = j["lineage_new"] if ds == "KPTracer" else j["lineage_new"].str.replace("LG", "")
        agree += int(len(j) == len(g) and (j["lineage"] == new).all() and (j["site"] == j["site_new"]).all())
    check("frozen Level 1 labels (6 mice local) agree cell by cell on lineage and site", agree == 6, f"{agree}/6")
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
