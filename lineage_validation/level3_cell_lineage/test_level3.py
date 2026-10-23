"""Tests of Level 3 on a synthetic tree and planted couplings (local; no real data).

    python lineage_validation/level3_cell_lineage/test_level3.py

Checks: tree distances against brute force; every status (evaluable, ambiguous, not_informative, three
not_evaluable reasons); an oracle coupling (all mass on the tree-nearest relatives) gives delta_D_norm 1 and
top-1 in the nearest set; the independent coupling built from non-uniform row masses gives delta_D 0 and log2
enrichment 0 (the baseline uses the actual marginals, not uniform); mass on the farthest relatives gives
delta_D_norm < 0; mass on non-comparable sources lowers c_ot and leaves the conditional quantities unchanged;
score_level3.py end to end (a missing coupling is reported); mouse means weight families equally.
Prints PASS / FAIL; exit 1 on any failure.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l3_core as c  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


# Family F tree: clade X = ((p0,p1,m0),(p2,m1)), clade Y = ((p3,p4),(p5,m2)), unary wrapper around Y, m3 unreliable,
# star family G: (q0,q1,q2,q3,q4,m9) so every source is equidistant from m9; family H: 3 sources only.
NWK_F = "((((p0,p1,m0),(p2,m1)),(((p3,p4),(p5,m2)))),(m3,p6,p7));"
NWK_G = "(q0,q1,q2,q3,q4,m9);"
NWK_H = "((h0,h1),(h2,mh));"


def brute(ft, a, b):
    anc = set()
    v = a
    while v >= 0:
        anc.add(v); v = ft.parent[v]
    w = b
    while w not in anc:
        w = ft.parent[w]
    return ft.depth[a] + ft.depth[b] - 2 * ft.depth[w]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "trees").mkdir(); (tmp / "defs" / "cells").mkdir(parents=True)
        for name, nwk in (("F_tree.nwk", NWK_F), ("G_tree.nwk", NWK_G), ("H_tree.nwk", NWK_H)):
            (tmp / "trees" / name).write_text(nwk)
        fams = {"T_M1_F": ("F_tree.nwk", ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"], ["m0", "m1", "m2", "m3"]),
                "T_M1_G": ("G_tree.nwk", ["q0", "q1", "q2", "q3", "q4"], ["m9"]),
                "T_M2_H": ("H_tree.nwk", ["h0", "h1", "h2"], ["mh"])}
        rows = []
        for fam, (tree, src, tgt) in fams.items():
            pd.DataFrame([{"leaf": x, "cell_id": x, "site": "T" if x in src else "M", "kind": "primary" if x in src else "metastasis",
                           "reliable": "False" if x == "m3" else "True", "missing_share": 0.6 if x == "m3" else 0.1, "L1": "", "L2": ""}
                          for x in src + tgt]).to_csv(tmp / "defs" / "cells" / f"{fam}.csv.gz", index=False)
            rows.append({"dataset": "T", "mouse": fam.split("_")[1], "family_or_clone": fam.split("_")[2], "tree": tree})
        pd.DataFrame(rows).to_csv(tmp / "defs" / "feasibility.tsv", sep="\t", index=False)
        c.DEFS = tmp / "defs"
        trees = c.tree_files([tmp / "trees"])
        ft = c.FamilyTree(trees["F_tree.nwk"])
        src_nodes = np.array([ft.leaf_of[x] for x in ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7"]])
        ft.prepare(src_nodes)
        ok = all(int(ft.distances(ft.leaf_of[t])[k]) == brute(ft, ft.leaf_of[t], s) for t in ["m0", "m1", "m2"]
                 for k, s in enumerate(src_nodes))
        check("tree distances equal brute force (unary node contracted)", ok, str(ft.distances(ft.leaf_of["m2"])))
        # one case: sources = p0..p7 + q0..q4 + x0 (no leaf), targets = m0..m3, m9, z (no leaf)
        src = ["p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "q0", "q1", "q2", "q3", "q4", "x0"]
        tgt = ["m0", "m1", "m2", "m3", "m9", "z"]
        cells = pd.DataFrame({"side": ["source"] * len(src) + ["target"] * len(tgt), "cell": src + tgt,
                              "site": ["T"] * len(src) + ["M"] * len(tgt)})
        case = pd.Series({"case_id": "T_M1__case", "families": "T_M1_F;T_M1_G", "index": "0"})
        gt, allt, n_src = c.ground_truth(case, cells, trees)
        st = allt.set_index("cell")["status"]
        check("statuses: m0, m1, m2 evaluable; m3 not_evaluable (unreliable); m9 not_informative (star); z not_evaluable (no leaf)",
              list(st[["m0", "m1", "m2", "m3", "m9", "z"]]) == ["evaluable", "evaluable", "evaluable", "not_evaluable", "not_informative", "not_evaluable"],
              str(st.to_dict()))
        caseH = pd.Series({"case_id": "T_M2__case", "families": "T_M2_H", "index": "1"})
        cellsH = pd.DataFrame({"side": ["source"] * 3 + ["target"], "cell": ["h0", "h1", "h2", "mh"], "site": ["T"] * 3 + ["M"]})
        _, allH, _ = c.ground_truth(caseH, cellsH, trees)
        check("fewer than 5 comparable primaries -> not_evaluable", allH["reason"].iloc[0].startswith("fewer than 5"))
        # ambiguous: target whose nearest set is >= 50% of the comparable primaries
        nwk_amb = "((a0,a1,a2,mA),(a3,a4));"
        (tmp / "trees" / "A_tree.nwk").write_text(nwk_amb)
        pd.DataFrame([{"leaf": x, "cell_id": x, "site": "T", "kind": "primary", "reliable": "True", "missing_share": 0.0, "L1": "", "L2": ""}
                      for x in ["a0", "a1", "a2", "a3", "a4", "mA"]]).to_csv(tmp / "defs" / "cells" / "T_M3_A.csv.gz", index=False)
        pd.concat([pd.read_csv(tmp / "defs" / "feasibility.tsv", sep="\t"),
                   pd.DataFrame([{"dataset": "T", "mouse": "M3", "family_or_clone": "A", "tree": "A_tree.nwk"}])]).to_csv(
            tmp / "defs" / "feasibility.tsv", sep="\t", index=False)
        trees = c.tree_files([tmp / "trees"])
        _, allA, _ = c.ground_truth(pd.Series({"case_id": "T_M3__case", "families": "T_M3_A", "index": "2"}),
                                    pd.DataFrame({"side": ["source"] * 5 + ["target"], "cell": ["a0", "a1", "a2", "a3", "a4", "mA"],
                                                  "site": ["T"] * 5 + ["M"]}), trees)
        check("nearest set holding 3 of 5 primaries -> ambiguous", allA["status"].iloc[0] == "ambiguous", allA["reason"].iloc[0])
        # matching quantities
        ftF, s_rows, ftab = gt["T_M1_F"]
        m0 = ftab[ftab["cell"].eq("m0")].iloc[0]
        d = ftF.distances(int(m0["target_node"])).astype(float)
        rng = np.random.default_rng(0)
        mrow = rng.uniform(0.2, 2.0, n_src)  # non-uniform source marginal, as UOT
        near = s_rows[d == d.min()]
        q_or = np.zeros(n_src); q_or[near] = 1 / len(near)
        r = c.cell_metrics(q_or, mrow, s_rows, d)
        check("oracle coupling: delta_D_norm = 1, top-1 in the nearest set, enrichment > 0",
              np.isclose(r["delta_D_norm"], 1) and r["top1_in_nearest"] and r["log2_close_enrichment"] > 0, f"{r['delta_D_norm']:.3f}")
        q_ind = mrow / mrow.sum()
        r = c.cell_metrics(q_ind, mrow, s_rows, d)
        check("independent coupling with non-uniform marginals: delta_D 0, log2 enrichment 0 (actual marginals used)",
              abs(r["delta_D"]) < 1e-12 and abs(r["log2_close_enrichment"]) < 1e-12, f"{r['delta_D']:.2e}")
        r_uniform_wrong = c.cell_metrics(q_ind, np.ones(n_src), s_rows, d)
        check("... and a uniform baseline would have reported a spurious effect", abs(r_uniform_wrong["delta_D"]) > 1e-3,
              f"{r_uniform_wrong['delta_D']:.3f}")
        far = s_rows[d == d.max()]
        q_far = np.zeros(n_src); q_far[far] = 1 / len(far)
        r = c.cell_metrics(q_far, mrow, s_rows, d)
        check("mass on the farthest relatives: delta_D_norm < 0", r["delta_D_norm"] < 0, f"{r['delta_D_norm']:.3f}")
        q_mix = 0.5 * q_or.copy(); q_mix[src.index("x0")] += 0.5
        r = c.cell_metrics(q_mix, mrow, s_rows, d)
        check("half the mass on a non-comparable source: c_ot 0.5, conditional delta_D_norm still 1",
              np.isclose(r["c_ot"], 0.5) and np.isclose(r["delta_D_norm"], 1), f"c_ot {r['c_ot']:.2f}")
        # end to end: score_level3 on planted couplings (BOT-like oracle for rna/A, independent for rna/B), cnv missing
        runs = tmp / "runs"
        for folder in ("full", "full_cnv", "full_rna_cnv"):
            (runs / folder / "T_M1__case").mkdir(parents=True)
            cells.to_csv(runs / folder / "T_M1__case" / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
        m_ = np.full(n_src, 1 / n_src)
        P_or = np.zeros((n_src, len(tgt)))
        for j, t in enumerate(tgt):
            row = allt[allt["cell"].eq(t)].iloc[0]
            if row["status"] == "evaluable":
                fam_t = gt[row["family"]]
                dd = fam_t[0].distances(int(row["target_node"]))
                P_or[fam_t[1][dd == dd.min()], j] = 1.0
            else:
                P_or[:, j] = 1.0
        P_or /= P_or.sum()
        P_ind = np.outer(mrow / mrow.sum(), np.full(len(tgt), 1 / len(tgt)))
        for folder in ("full", "full_rna_cnv"):
            for method, P in (("A_standard_OT", P_or), ("B_our_OT", P_ind)):
                (runs / folder / "T_M1__case" / method).mkdir()
                np.savez_compressed(runs / folder / "T_M1__case" / method / "coupling.npz", coupling=P.astype(np.float32))
        pd.DataFrame([{"index": "0", "case_id": "T_M1__case", "dataset": "T", "mouse": "M1", "families": "T_M1_F;T_M1_G"}]).to_csv(
            tmp / "cases.tsv", sep="\t", index=False)
        import score_level3 as sl
        sl.c.tree_files = lambda roots=None: trees
        sys.argv = ["score_level3.py", "--runs", str(runs), "--out", str(tmp / "scores"), "--case-table", str(tmp / "cases.tsv")]
        sl.main()
        rs = pd.read_csv(tmp / "scores" / "run_status.tsv", sep="\t")
        check("run_status: 4 ok, cnv missing for both methods with the absent file named",
              (rs["status"] == "ok").sum() == 4 and set(rs.loc[rs["strategy"].eq("cnv"), "status"]) == {"missing"}
              and rs.loc[rs["status"].eq("missing"), "file"].str.endswith("coupling.npz").all())
        sc = pd.read_csv(tmp / "scores" / "cells" / "T_M1__case.tsv.gz", sep="\t")
        e = sc[sc["status"].eq("evaluable")]
        check("end to end: planted oracle (A) delta_D_norm 1 for every evaluable cell, planted independent (B) 0",
              np.allclose(e.loc[e["method"].eq("A_standard_OT"), "delta_D_norm"], 1, atol=1e-5)
              and np.allclose(e.loc[e["method"].eq("B_our_OT"), "delta_D_norm"], 0, atol=1e-5))
        check("not_informative cell carried but its delta_D_norm undefined", sc[sc["status"].eq("not_informative")]["delta_D_norm"].isna().all())
        # summaries: a mouse with two families weights them equally
        import summarize_level3 as sm
        gt_dir = tmp / "gt"; gt_dir.mkdir()
        pd.DataFrame([{"family": f, "flag_high_missingness": False, "flag_low_tree_resolution": False, "n_evaluable": 1,
                       "metastatic_cells_in_family": 1, "sources": 1, "sources_comparable": 1} for f in ("T_M1_F", "T_M1_G")]).to_csv(
            gt_dir / "coverage.tsv", sep="\t", index=False)
        sys.argv = ["summarize_level3.py", "--scores", str(tmp / "scores"), "--ground-truth", str(gt_dir), "--out", str(tmp / "summary")]
        sm.main()
        fs = pd.read_csv(tmp / "summary" / "family_summary.tsv", sep="\t")
        ms = pd.read_csv(tmp / "summary" / "mouse_summary.tsv", sep="\t")
        a = fs[(fs.strategy == "rna") & (fs.method == "A_standard_OT")]
        check("family summary: A oracle delta_D_norm 1 with a bootstrap interval; mouse mean is the mean over families",
              np.isclose(a["delta_D_norm"].iloc[0], 1, atol=1e-5)
              and np.isclose(ms[(ms.strategy == "rna") & (ms.method == "A_standard_OT")]["delta_D_norm"].iloc[0], a["delta_D_norm"].mean()))
        ab = pd.read_csv(tmp / "summary" / "ab_paired.tsv", sep="\t")
        check("A vs B paired on the same cells: B - A = -1 for the planted pair",
              np.isclose(ab.loc[ab.strategy.eq("rna"), "delta_D_norm__B_minus_A"].iloc[0], -1, atol=1e-5))
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
