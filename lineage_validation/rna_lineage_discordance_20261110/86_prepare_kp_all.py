"""PLAN_v2.md extension E1 / E2, locally on the 85_export_kp_all.py export: audit of every KP cell, the eligible cell
table for 70_state_lineage.py, and the harmonization checks of the "(Met)" labels with the fixed merge rule.

    python lineage_validation/rna_lineage_discordance_20261110/86_prepare_kp_all.py --root results/rna_lineage_discordance_20261110 \
        --inputs results/crispr_matrix_benchmark_20261108/inputs --export results/rna_lineage_discordance_20261110/kp_all_extract/kp_all_20261111

Eligible (E2): is_normal False, a Cluster-Name, a tumour (primary / metastasis by the README rule), a MetFamily, and an
allele-table record (FINAL, else unfiltered; 01_build_bundles.load_tables / map_ids). Writes ROOT/kp_all_audit/:
  audit_mouse.tsv       per mouse: cells at each criterion, ID overlap, unfiltered share, family agreement, depth
  audit_ot_overlap.tsv  OT-unit cells (state_lineage/cells.tsv.gz) inside the eligible set, and eligible cells in them
  kp_all_cells.tsv.gz   eligible cells: cell (allele-table id), mouse, site, state, depth, detected_genes, allele_source, in_ot
  harmonize_*.tsv       E1 checks for every "X (Met)" whose parent "X" exists: marker overlap (top 50), Spearman of the
                        label means over the union of top-50 genes, cosine distance of PCA-30 centroids (pooled and per
                        mouse where both labels have >= 20 cells), and the label composition by mouse and side
  state_map.tsv         original -> harmonized: a candidate merges only if its parent is its closest label on both the
                        Spearman and the centroid distance, pooled (fixed before computing); every other label maps to itself
"""

from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOP, MIN_PER_MOUSE = 50, 20


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--export", type=Path, required=True)
    a = p.parse_args()
    out = a.root / "kp_all_audit"; out.mkdir(exist_ok=True)
    cot = load("cot", HERE / "10_crispr_ot.py"); bb = cot.bundles_module()
    t = bb.load_tables(a.inputs)["KPTracer"][0]
    tc = t.drop_duplicates("cell").set_index("cell")
    o = pd.read_csv(a.export / "obs_all.tsv.gz", sep="\t", dtype=str)
    o["total_counts"] = o.total_counts.astype(float); o["detected_genes"] = o.detected_genes.astype(float)
    o["normal"] = o.is_normal.str.lower().isin(["true", "1", "1.0"])
    o["labelled"] = ~o.cluster_name.isin(["nan", "", "None"]) & o.cluster_name.notna()
    o["tumour"] = o.tumour_kind.isin(["primary", "metastasis"])
    o["has_family"] = ~o.met_family.isin(["nan", "", "None"]) & o.met_family.notna()
    o["table_cell"] = bb.map_ids(o.cell, pd.Index(tc.index))
    o["in_allele_table"] = o.table_cell.notna()
    o["allele_source"] = np.where(o.in_allele_table, np.where(o.table_cell.map(tc.from_unfiltered).fillna(False).astype(bool), "unfiltered", "FINAL"), "none")
    o["table_family"] = o.table_cell.map(tc.lineage)
    o["eligible"] = ~o.normal & o.labelled & o.tumour & o.has_family & o.in_allele_table
    # audit per mouse
    rows = []
    for m, g in o.groupby("mouse"):
        e = g[g.eligible]
        rows.append({"mouse": m, "cells": len(g), "normal": int(g.normal.sum()), "malignant": int((~g.normal).sum()),
                     "malignant_labelled": int((~g.normal & g.labelled).sum()), "malignant_labelled_tumour": int((~g.normal & g.labelled & g.tumour).sum()),
                     "in_allele_table": int(g.in_allele_table.sum()), "normal_in_allele_table": int((g.normal & g.in_allele_table).sum()),
                     "eligible": len(e), "eligible_unfiltered_share": float((e.allele_source == "unfiltered").mean()) if len(e) else np.nan,
                     "family_agrees_share": float((e.table_family == e.met_family).mean()) if len(e) else np.nan,
                     "sites": e.tumor.nunique(), "metastatic_cells": int((e.tumour_kind == "metastasis").sum()), "states": e.cluster_name.nunique(),
                     "median_depth": float(e.total_counts.median()) if len(e) else np.nan,
                     "median_detected": float(e.detected_genes.median()) if len(e) else np.nan})
    A = pd.DataFrame(rows); A.to_csv(out / "audit_mouse.tsv", sep="\t", index=False)
    print(A.to_string(index=False))
    tot = A[["cells", "normal", "malignant_labelled_tumour", "in_allele_table", "normal_in_allele_table", "eligible"]].sum()
    print(tot.to_string())
    e = o[o.eligible].copy()
    ot = pd.read_csv(a.root / "state_lineage" / "cells.tsv.gz", sep="\t", dtype={"cell": str, "mouse": str})
    e["in_ot"] = e.table_cell.isin(set(ot.cell))
    ov = pd.DataFrame([{"ot_cells": len(ot), "ot_in_eligible": int(ot.cell.isin(set(e.table_cell)).sum()),
                        "eligible": len(e), "eligible_in_ot": int(e.in_ot.sum())}])
    st_ot = ot.set_index("cell").state; agree = (e[e.in_ot].table_cell.map(st_ot) == e[e.in_ot].cluster_name).mean()
    ov["state_label_agrees_on_overlap"] = float(agree)
    ov.to_csv(out / "audit_ot_overlap.tsv", sep="\t", index=False); print(ov.to_string(index=False))
    cells = pd.DataFrame({"cell": e.table_cell, "mouse": e.mouse, "site": e.tumor, "state": e.cluster_name, "depth": e.total_counts,
                          "detected_genes": e.detected_genes, "allele_source": e.allele_source, "in_ot": e.in_ot,
                          "tumour_kind": e.tumour_kind})
    cells.to_csv(out / "kp_all_cells.tsv.gz", sep="\t", index=False, compression="gzip")
    # consistency of the README rule with 70_'s primary regex on the eligible sites
    rx = cells.site.map(lambda s: bool(re.search(r"_T\d+$", s)))
    print(f"README primary vs 70_ regex agree on {float((rx == (cells.tumour_kind == 'primary')).mean()):.4f} of eligible cells")
    # E1 harmonization checks
    labels = sorted(o[o.labelled & ~o.normal].cluster_name.unique())
    cand = {x: x[:-len(" (Met)")] for x in labels if x.endswith(" (Met)") and x[:-len(" (Met)")] in labels}
    print("candidates:", cand)
    mk = pd.read_csv(a.export / "markers.tsv.gz", sep="\t")
    top = {k: set(g.sort_values("rank").names.head(TOP)) for k, g in mk.groupby("label")}
    lm = pd.read_csv(a.export / "label_means.tsv.gz", sep="\t", index_col=0)
    sp = lm.T.rank().corr()                                   # Spearman between label means over the union genes
    z = np.load(a.export / "pca_logcpm30.npz"); X = z["X"].astype(np.float64); lab = z["label"].astype(str); cid = z["cell"].astype(str)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    cen = pd.DataFrame(Xn).groupby(lab).mean()
    cn = cen.to_numpy() / np.linalg.norm(cen.to_numpy(), axis=1, keepdims=True)
    cosd = pd.DataFrame(1 - cn @ cn.T, index=cen.index, columns=cen.index)
    mouse_of = o.set_index("cell").mouse.reindex(cid).to_numpy(); side_of = o.set_index("cell").tumour_kind.reindex(cid).to_numpy()
    hrows, prow, crows = [], [], []
    for c, par in cand.items():
        others = [x for x in labels if x != c]
        ov_ = pd.Series({x: len(top[c] & top[x]) for x in others}).sort_values(ascending=False)
        spr = sp.loc[c, others].sort_values(ascending=False); dd = cosd.loc[c, others].sort_values()
        merge = spr.index[0] == par and dd.index[0] == par
        hrows.append({"candidate": c, "parent": par, "marker_overlap_parent": int(ov_[par]), "marker_overlap_best_other": ov_.drop(par).index[0],
                      "marker_overlap_best_other_n": int(ov_.drop(par).iloc[0]), "spearman_parent": spr[par], "spearman_closest": spr.index[0],
                      "spearman_closest_other": spr.drop(par).index[0], "spearman_closest_other_value": spr.drop(par).iloc[0],
                      "cosdist_parent": dd[par], "cosdist_closest": dd.index[0], "cosdist_closest_other": dd.drop(par).index[0],
                      "cosdist_closest_other_value": dd.drop(par).iloc[0], "merge": bool(merge)})
        for x in others:
            prow.append({"candidate": c, "label": x, "marker_overlap": int(ov_[x]), "spearman": spr[x], "cosdist": dd[x]})
        for m in sorted(set(mouse_of[(lab == c)]) & set(mouse_of[(lab == par)])):
            mc_, mp_ = (lab == c) & (mouse_of == m), (lab == par) & (mouse_of == m)
            if mc_.sum() < MIN_PER_MOUSE or mp_.sum() < MIN_PER_MOUSE:
                continue
            others_m = [x for x in labels if x != c and ((lab == x) & (mouse_of == m)).sum() >= MIN_PER_MOUSE]
            cm = {x: Xn[(lab == x) & (mouse_of == m)].mean(0) for x in [c] + others_m}
            cm = {k: v / np.linalg.norm(v) for k, v in cm.items()}
            dm = pd.Series({x: 1 - float(cm[c] @ cm[x]) for x in others_m}).sort_values()
            crows.append({"candidate": c, "mouse": m, "candidate_cells": int(mc_.sum()), "parent_cells": int(mp_.sum()), "cosdist_parent": dm[par],
                          "closest": dm.index[0], "closest_other": dm.drop(par).index[0] if len(dm) > 1 else None,
                          "closest_other_value": dm.drop(par).iloc[0] if len(dm) > 1 else np.nan})
    H = pd.DataFrame(hrows); H.to_csv(out / "harmonize_decision.tsv", sep="\t", index=False)
    pd.DataFrame(prow).to_csv(out / "harmonize_pooled_all_labels.tsv", sep="\t", index=False)
    pd.DataFrame(crows).to_csv(out / "harmonize_per_mouse.tsv", sep="\t", index=False)
    comp = cells.groupby(["state", "mouse", "tumour_kind"]).size().rename("cells").reset_index()
    comp[comp.state.isin(list(cand) + list(cand.values()))].to_csv(out / "harmonize_composition.tsv", sep="\t", index=False)
    mp = {x: x for x in labels}
    for r in hrows:
        if r["merge"]:
            mp[r["candidate"]] = r["parent"]
    pd.DataFrame({"original": list(mp), "harmonized": list(mp.values())}).to_csv(out / "state_map.tsv", sep="\t", index=False)
    print(H.round(3).T.to_string()); print(pd.DataFrame(crows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
