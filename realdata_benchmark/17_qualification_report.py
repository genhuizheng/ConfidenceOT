"""Qualification report: every LTL331 configuration against the registered eligibility criteria (protocol section of
2026-11-01, db5c2e7), why each one fails, and what the depth criterion does and does not show. Descriptive only: no
threshold is changed and CONFIG_ELIGIBILITY.tsv (14_) is read, not rewritten.

    python realdata_benchmark/17_qualification_report.py --tables TACC_TABLES --diag DIAG_DIR --run-root LTL331_RUN_ROOT \
        --converted CONVERTED --out OUT

Writes into OUT (refuses if it exists):
  QUALIFICATION_TABLE.tsv     one row per configuration: Splatter criterion-1 values, fit status (exit / convergence /
                              cycles, all M4-E fits), calibration-valid pairs and the valid-pair pattern, the depth
                              criterion with its signed median and the same AUC within annotated type, AUC of detected
                              genes, criterion 3 under three readings, the historical / registered twin and whether its
                              gates are identical, the failure classes
  DEPTH_DECOMPOSITION.tsv     per configuration, pair and side: rejected fraction, AUC of pre-equalisation depth (all
                              cells; within each coarse type, weighted by cells; types with >= 10 rejected and >= 10
                              retained), AUC of detected genes on the raw counts (nFeature_RNA) and on the frozen P10
                              equalised counts
  SPLATTER_TECHNICAL.tsv      per configuration and technical level: Splatter criterion-1 medians and the AUC of
                              total counts / detected genes among all_shared rejections
  ELIGIBLE_VS_RANK256.tsv     per pair and side, the eligible configuration against the original ranknm256_noscale_ds_cos
                              and the configurations that differ from it in one factor: rejected fraction, gate kappa
                              against the original, depth / nFeature AUC, NE minus non-NE retained, coupling metrics
  NFEATURE_P10.tsv            detected genes per cell on the frozen P10 equalised counts, per sample, against K
Criterion 3 readings: as_run (14_ as executed: a ranknm256 tie kappa assigned by encoding stem and DS); exact_min (a
configuration is measured only when its exact representation, production ties, was in a tie table; min over pairs,
sides and non-production rules); exact_median (same, median over pairs and rules, min over the two sides). Expected-tie
configurations are tie-invariant by construction and listed as such.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import rb_common as rb

ORIGINAL = "label:ranknm256_noscale_ds_cos"
ELIGIBLE = "ranknm1024-expected|P10sfrozen|noscale|sqeuc"
STEPS = [ELIGIBLE, ORIGINAL, "ranknm256-production|P10sfrozen|noscale|sqeuc", "ranknm1024-expected|P10sfrozen|noscale|cos",
         "ranknm1024-production|P10sfrozen|noscale|cos", "ranknm256-expected|P10sfrozen|noscale|cos", "label:rank256_noscale_ds_cos"]
# exact representation (production ties) measured in a tie table: config id -> tie-table config name
TIE_EXACT = {"label:ranknm256_noscale_ds_cos": "ranknm256_noscale_ds_cos", "ranknm256-production|P10sfrozen|noscale|cos": "ranknm256_noscale_ds_cos",
             "label:ranknm256_noscale_cos": "ranknm256_noscale_cos", "ranknm256-production|none|noscale|cos": "ranknm256_noscale_cos",
             "label:rank256_noscale_ds_cos": "rank256_noscale_ds_cos", "label:rank256_noscale_cos": "rank256_noscale_cos",
             "label:ranknm256_noscale_ds": "ranknm256_noscale_ds", "ranknm256-production|P10sfrozen|noscale|sqeuc": "ranknm256_noscale_ds",
             "ranknm1024-production|P10sfrozen|noscale|cos": None}


def auc(x, rejected):
    x = np.asarray(x, float); r = np.asarray(rejected, bool); ok = ~np.isnan(x); x, r = x[ok], r[ok]
    n1, n0 = r.sum(), (~r).sum()
    if n1 == 0 or n0 == 0:
        return np.nan
    rk = rankdata(x)
    return float((rk[r].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("tables", "diag", "run_root", "converted", "out"):
        p.add_argument("--" + k.replace("_", "-"), type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    elig = pd.read_csv(a.diag / "analysis" / "CONFIG_ELIGIBILITY.tsv", sep="\t").set_index("config_id")
    pdx = pd.read_csv(a.tables / "PDX_GATE_COUPLING_METRICS.tsv", sep="\t")
    m4e = pdx[pdx.method == "M4-E"]
    spl = pd.read_csv(a.tables / "SPLATTER_GATE_COUPLING_METRICS.tsv", sep="\t")
    spl = spl[(spl.method == "M4-E") & np.isclose(spl.acceptance, 0.99)]

    # cells
    obs = pd.concat([ad.read_h5ad(f, backed="r").obs[["coarse_celltype", "nFeature_RNA"]] for f in sorted((a.converted / "h5ad").glob("*.h5ad"))])
    obs["coarse_celltype"] = obs.coarse_celltype.astype(str)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    nf, nfs = [], []
    for f in sorted((a.run_root / "inputs" / "ds" / "h5ad").glob("*.h5ad")):
        x = ad.read_h5ad(f); d = np.asarray((x.X > 0).sum(1)).ravel()
        nf.append(pd.Series(d, index=x.obs_names))
        nfs.append({"sample": f.stem.replace("LTL331__", ""), "cells": len(d), "total_counts_median": float(np.median(np.asarray(x.X.sum(1)).ravel())),
                    "detected_median": float(np.median(d)), "detected_p10": float(np.quantile(d, 0.1)), "detected_p90": float(np.quantile(d, 0.9)),
                    **{f"fraction_below_{k}": float(np.mean(d < k)) for k in (128, 256, 512, 1024, 2048)}})
    pd.DataFrame(nfs).to_csv(a.out / "NFEATURE_P10.tsv", sep="\t", index=False)
    nf = pd.concat(nf)
    g = pd.read_csv(a.tables / "PDX_CELL_GATES.tsv.gz", sep="\t")
    g = g[g.method == "M4-E"].join(obs, on="observation_id")
    g["depth"] = g.observation_id.map(depth); g["nfeature_p10"] = g.observation_id.map(nf); g["rejected"] = g.rejected.astype(bool)
    g["pair"] = g.pair_id.str.replace("LTL331__", "", regex=False)

    # depth decomposition
    dd = []
    for (cid, pair, side), u in g.groupby(["config_id", "pair", "side"]):
        within, wn = [], []
        for t, w in u.groupby("coarse_celltype"):
            if w.rejected.sum() >= 10 and (~w.rejected).sum() >= 10:
                within.append(auc(w.depth, w.rejected)); wn.append(len(w))
        dd.append({"config_id": cid, "pair": pair, "side": side, "cells": len(u), "rejected_fraction": u.rejected.mean(),
                   "auc_depth": auc(u.depth, u.rejected), "auc_depth_within_type": float(np.average(within, weights=wn)) if within else np.nan,
                   "types_in_within": len(within), "auc_nfeature_raw": auc(u.nFeature_RNA, u.rejected), "auc_nfeature_p10": auc(u.nfeature_p10, u.rejected),
                   "ne_minus_nonne_retained": (~u.rejected[u.coarse_celltype == "NE"]).mean() - (~u.rejected[u.coarse_celltype != "NE"]).mean()
                   if (u.coarse_celltype == "NE").sum() >= 20 else np.nan})
    dd = pd.DataFrame(dd); dd.to_csv(a.out / "DEPTH_DECOMPOSITION.tsv", sep="\t", index=False)

    # Splatter by technical level
    rows = []
    for (cid, lvl), s in spl.groupby(["config_id", "technical_level"]):
        sh = s[s.biological_case == "all_shared"]
        rows.append({"config_id": cid, "technical_level": lvl, "rows": len(s),
                     "shared_false_rejection_source": sh.source_false_rejection_rate.median(), "shared_false_rejection_target": sh.target_false_rejection_rate.median(),
                     "recall_lost_source": s[s.biological_case == "population_lost"].source_recall.median(),
                     "recall_emerged_target": s[s.biological_case == "population_emerged"].target_recall.median(),
                     "disjoint_false_asserted_mass": s[s.biological_case == "populations_disjoint"].false_asserted_mass.median(),
                     "shared_auc_total_counts_source": sh.source_auc_total_counts.median(), "shared_auc_detected_genes_source": sh.source_auc_detected_genes.median(),
                     "shared_auc_total_counts_target": sh.target_auc_total_counts.median(), "shared_auc_detected_genes_target": sh.target_auc_detected_genes.median()})
    pd.DataFrame(rows).to_csv(a.out / "SPLATTER_TECHNICAL.tsv", sep="\t", index=False)

    # criterion 3 readings
    ties = []
    for f in sorted(a.diag.glob("RANK_TIE_SENSITIVITY*.tsv")):
        t = pd.read_csv(f, sep="\t"); ties.append(t[t.tie_rule != "production"])
    ties = pd.concat(ties) if ties else pd.DataFrame(columns=["config"])
    kmin, kmed = {}, {}
    for cfg, t in ties.groupby("config"):
        s_, t_ = t.source_knn_gate_kappa_vs_production, t.target_knn_gate_kappa_vs_production
        kmin[cfg] = float(min(s_.min(), t_.min()))
        kmed[cfg] = float(min(t.groupby("pair").source_knn_gate_kappa_vs_production.median().median(),
                              t.groupby("pair").target_knn_gate_kappa_vs_production.median().median()))

    # twins
    ids = set(m4e.config_id)
    twin_same = {}
    for cid in ids:
        tw = elig.loc[cid, "splatter_twin"] if cid in elig.index else None
        if cid.startswith("label:") and isinstance(tw, str) and tw in ids:
            x = g[g.config_id == cid].set_index(["pair", "side", "observation_id"]).rejected
            y = g[g.config_id == tw].set_index(["pair", "side", "observation_id"]).rejected
            j = x.index.intersection(y.index)
            same = bool((x.loc[j] == y.loc[j]).all()) and len(j) == len(x) == len(y)
            twin_same[cid], twin_same[tw] = f"{tw} identical={same}", f"{cid} identical={same}"

    out = []
    for cid, m in m4e.groupby("config_id"):
        e = elig.loc[cid]; d = dd[dd.config_id == cid]
        r = {"config_id": cid}
        for k in ("splatter_twin", "splatter_shared_false_rejection_source", "splatter_shared_false_rejection_target", "splatter_recall_lost_source",
                  "splatter_recall_emerged_target", "splatter_disjoint_false_asserted_mass", "criterion1_pass"):
            r[k] = e[k]
        r.update(fits=len(m), fits_exit0=int((m.exit == 0).sum()), fits_converged=int((m.inner_converged & m.outer_converged).sum()),
                 fits_cycled=int(m.cycle_detected.sum()), calibration_valid_pairs=int(m.m4e_inference_valid.sum()),
                 valid_pattern="".join("V" if v else "-" for v in m.sort_values("pair_index").m4e_inference_valid),
                 median_abs_depth_auc_dev=e["ltl331_median_abs_depth_auc_dev"], median_signed_depth_auc=float(d.auc_depth.median()),
                 median_abs_within_type_depth_auc_dev=float((d.auc_depth_within_type - 0.5).abs().median()),
                 median_abs_nfeature_p10_auc_dev=float((d.auc_nfeature_p10 - 0.5).abs().median()),
                 median_rejected_fraction=float(d.rejected_fraction.median()), criterion2_pass=e["criterion2_pass"],
                 c3_as_run_kappa=e["tie_kappa_min"], c3_as_run_pass=e["criterion3_pass"])
        tie_free = "expected" in cid.split("|")[0] or "midrank" in cid.split("|")[0]
        key = TIE_EXACT.get(cid)
        if tie_free:
            r.update(c3_exact="tie-invariant by construction", c3_exact_min=np.nan, c3_exact_median=np.nan, c3_exact_min_pass=True, c3_exact_median_pass=True)
        elif key in kmin:
            r.update(c3_exact=key, c3_exact_min=kmin[key], c3_exact_median=kmed[key], c3_exact_min_pass=kmin[key] >= 0.80, c3_exact_median_pass=kmed[key] >= 0.80)
        else:
            r.update(c3_exact="not measured" if cid.split("|")[0].startswith(("ranknm", "label:rank")) else "no ties to measure (logcpm)",
                     c3_exact_min=np.nan, c3_exact_median=np.nan, c3_exact_min_pass=True, c3_exact_median_pass=True)
        r["twin"] = twin_same.get(cid, "")
        cls = []
        if r["fits_exit0"] < r["fits"] or r["fits_converged"] < r["fits"] or r["fits_cycled"] > 0:
            cls.append("solver")
        if str(e.get("criterion1", "")).startswith("not evaluated"):
            cls.append("splatter_not_evaluated")
        elif not e["criterion1_pass"]:
            parts = [n for n, bad in (("shared_false_rejection", max(e.splatter_shared_false_rejection_source, e.splatter_shared_false_rejection_target) > 0.10),
                                       ("recall", min(e.splatter_recall_lost_source, e.splatter_recall_emerged_target) < 0.80),
                                       ("disjoint_mass", e.splatter_disjoint_false_asserted_mass > 0.05)) if bad]
            cls.append("splatter_performance:" + "+".join(parts))
        if r["calibration_valid_pairs"] < 5:
            cls.append("calibration_invalid")
        if r["median_abs_depth_auc_dev"] > 0.10:
            cls.append("depth_association")
        if not r["c3_as_run_pass"]:
            cls.append("tie_stability(as_run)")
        r["failure_classes"] = ";".join(cls) if cls else "none"
        r["eligible_as_run"] = bool(e["eligible"])
        r["eligible_exact_min"] = bool(e["criterion1_pass"] and e["criterion2_pass"] and r["c3_exact_min_pass"])
        r["eligible_exact_median"] = bool(e["criterion1_pass"] and e["criterion2_pass"] and r["c3_exact_median_pass"])
        out.append(r)
    out = pd.DataFrame(out).sort_values(["eligible_as_run", "criterion1_pass", "criterion2_pass", "median_abs_depth_auc_dev"], ascending=[False, False, False, True])
    out.to_csv(a.out / "QUALIFICATION_TABLE.tsv", sep="\t", index=False)

    # eligible against the original and one-factor neighbours
    ref = g[g.config_id == ORIGINAL].set_index(["pair", "side", "observation_id"]).rejected
    cm = ["coupling_row_entropy_median", "transported_mass", "retained_block_mass_fraction", "full_vs_balanced_type_corr",
          "enrich_PRAD_to_PRAD", "enrich_NE_to_NE", "target_types_covered", "m4e_inference_valid"]
    cmp_rows = []
    for cid in STEPS:
        for (pair, side), u in g[g.config_id == cid].groupby(["pair", "side"]):
            x = u.set_index(["pair", "side", "observation_id"]).rejected; j = x.index.intersection(ref.index)
            row = dd[(dd.config_id == cid) & (dd.pair == pair) & (dd.side == side)].iloc[0].to_dict()
            row["gate_kappa_vs_original"] = rb.kappa(x.loc[j], ref.loc[j]) if len(j) else np.nan
            row["cells_shared_with_original"] = len(j)
            row.update(m4e[(m4e.config_id == cid) & (m4e.pair == pair)][cm].iloc[0].to_dict())
            cmp_rows.append(row)
    pd.DataFrame(cmp_rows).to_csv(a.out / "ELIGIBLE_VS_RANK256.tsv", sep="\t", index=False)
    print(out[["config_id", "criterion1_pass", "calibration_valid_pairs", "median_abs_depth_auc_dev", "median_abs_within_type_depth_auc_dev",
               "c3_exact", "failure_classes", "eligible_as_run", "eligible_exact_min", "eligible_exact_median"]].to_string(index=False))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
