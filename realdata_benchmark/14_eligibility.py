"""CONFIG_ELIGIBILITY.tsv: every LTL331 configuration against the criteria registered in JOINT_BENCHMARK_PROTOCOL.md
("which configurations are read biologically on LTL331", fixed before these tables were read).

    python realdata_benchmark/14_eligibility.py --tables TACC_TABLES_DIR --diag DIAG_DIR --out OUT_DIR

Criterion 1, Splatter (SPLATTER_GATE_COUPLING_METRICS.tsv; M4-E, acceptance 0.99, n_cells 1,000 and 5,000, replicates
1-3, every level): median shared-cell false rejection <= 0.10 on each side in all_shared; median recall >= 0.80
(source in population_lost, target in population_emerged); median false asserted mass <= 0.05 in populations_disjoint.
A historical LTL331 label is judged by the registered configuration it is identical to (same production label path:
e.g. logcpm_noscale_cos = logcpm|none|noscale|cos); a label with no such twin has no 0.99 Splatter run and is "not
evaluated", which does not pass.
Criterion 2, LTL331 (PDX_GATE_COUPLING_METRICS.tsv, M4-E): inference valid in >= 5 of 7 pairs; median
|AUC(pre-equalisation depth, rejected) - 0.5| <= 0.10 over the 14 pair-sides.
Criterion 3, stability where measured: OT-free kNN-gate kappa >= 0.80 between DS seeds (DOWNSAMPLING_ROBUSTNESS.tsv
of 09_, when present) and between tie rules (RANK_TIE_SENSITIVITY.tsv: production vs expected / random /
alphabetical, ranknm256 configurations); "not measured" passes.
Every configuration is listed with every number; the eligible set is not chosen by any biological result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TWIN = {}
for stem, enc in (("logcpm", "logcpm"), ("ranknm256", "ranknm256-production")):
    for scale in ("", "_noscale"):
        for ds in ("", "_ds"):
            for cos in ("", "_cos"):
                TWIN[f"label:{stem}{scale}{ds}{cos}"] = f"{enc}|{'P10sfrozen' if ds else 'none'}|{'noscale' if scale else 'scale'}|{'cos' if cos else 'sqeuc'}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", type=Path, required=True)
    p.add_argument("--diag", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "CONFIG_ELIGIBILITY.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    s = pd.read_csv(a.tables / "SPLATTER_GATE_COUPLING_METRICS.tsv", sep="\t")
    s = s[(s.method == "M4-E") & np.isclose(s.acceptance, 0.99)]
    pdx = pd.read_csv(a.tables / "PDX_GATE_COUPLING_METRICS.tsv", sep="\t")
    pdx = pdx[pdx.method == "M4-E"]
    spl = {}
    for cid, g in s.groupby("config_id"):
        sh, lo, em, dj = (g[g.biological_case == c] for c in ("all_shared", "population_lost", "population_emerged", "populations_disjoint"))
        spl[cid] = {"splatter_pairs": len(g),
                    "splatter_shared_false_rejection_source": sh.source_false_rejection_rate.median(),
                    "splatter_shared_false_rejection_target": sh.target_false_rejection_rate.median(),
                    "splatter_recall_lost_source": lo.source_recall.median(),
                    "splatter_recall_emerged_target": em.target_recall.median(),
                    "splatter_disjoint_false_asserted_mass": dj.false_asserted_mass.median()}
    tie = None
    f = a.diag / "RANK_TIE_SENSITIVITY.tsv"
    if f.exists():
        t = pd.read_csv(f, sep="\t"); t = t[t.tie_rule != "production"]
        tie = t.groupby("config")[["source_knn_gate_kappa_vs_production", "target_knn_gate_kappa_vs_production"]].min().min(axis=1)
    dsr = None
    f = a.diag / "DOWNSAMPLING_ROBUSTNESS.tsv"
    if f.exists():
        d = pd.read_csv(f, sep="\t"); kc = [c for c in d if c.endswith("_gate_kappa") and "_s" in c and "vs_ds_none" not in c]
        if kc:
            d["min_seed_kappa"] = d[kc].min(axis=1); dsr = d.groupby("representative").min_seed_kappa.min()
    rows = []
    for cid, g in pdx.groupby("config_id"):
        twin = TWIN.get(cid, cid if not cid.startswith("label:") else None)
        r = {"config_id": cid, "splatter_twin": twin or "none"}
        sp = spl.get(twin) if twin else None
        if sp is None:
            r.update(criterion1="not evaluated (no 0.99 Splatter run)", criterion1_pass=False)
        else:
            r.update(sp)
            r["criterion1_pass"] = bool(sp["splatter_shared_false_rejection_source"] <= 0.10 and sp["splatter_shared_false_rejection_target"] <= 0.10
                                        and sp["splatter_recall_lost_source"] >= 0.80 and sp["splatter_recall_emerged_target"] >= 0.80
                                        and sp["splatter_disjoint_false_asserted_mass"] <= 0.05)
        aucs = np.r_[g.source_auc_depth_rejected.to_numpy(float), g.target_auc_depth_rejected.to_numpy(float)]
        r.update(ltl331_valid_pairs=int(g.m4e_inference_valid.astype(bool).sum()), ltl331_pairs=len(g),
                 ltl331_median_abs_depth_auc_dev=float(np.nanmedian(np.abs(aucs - 0.5))),
                 ltl331_median_source_rejected=float(g.source_rejected.median()), ltl331_median_target_rejected=float(g.target_rejected.median()))
        r["criterion2_pass"] = bool(r["ltl331_valid_pairs"] >= 5 and r["ltl331_median_abs_depth_auc_dev"] <= 0.10)
        enc = (twin or cid).split("|")[0]
        k_tie = None
        if tie is not None:
            for cfg, v in tie.items():
                if (("ranknm256" in cfg and enc.startswith("ranknm256")) and (("_ds_" in cfg) == ("P10" in (twin or cid) or "_ds" in cid))):
                    k_tie = float(v)
        k_ds = float(dsr.get(enc)) if dsr is not None and enc in dsr.index else None
        r.update(tie_kappa_min=k_tie, ds_seed_kappa_min=k_ds)
        r["criterion3_pass"] = all(k is None or k >= 0.80 for k in (k_tie, k_ds))
        r["eligible"] = bool(r["criterion1_pass"] and r["criterion2_pass"] and r["criterion3_pass"])
        rows.append(r)
    out = pd.DataFrame(rows).sort_values(["eligible", "criterion1_pass", "criterion2_pass"], ascending=False)
    a.out.mkdir(parents=True, exist_ok=True); out.to_csv(target, sep="\t", index=False)
    print(out[["config_id", "criterion1_pass", "criterion2_pass", "criterion3_pass", "eligible"]].to_string(index=False))
    print(f"eligible: {int(out.eligible.sum())} of {len(out)} -> {target}")


if __name__ == "__main__":
    main()
