"""L1 family-level readout on the v3 grid, every configuration (MATRIX_METRICS.md section 3, CRISPR_LINEAGE_BENCHMARK.md
section 3): d = same-family - mean other-family source retained fraction per metastasis, from FIT_METRICS.

    python lineage_validation/crispr_matrix_benchmark_20261108/20_l1_family.py --inputs INPUTS --out OUT

Entry: a metastasis with >= 1 completed same-family and >= 1 completed other-family primary in its mouse (31 in 8
mice). Per configuration and method: per-metastasis d (source side; target side beside), per-mouse mean, the mean of
mouse means (primary), mice with mean d > 0, median d over metastases, and the prespecified one-sided Wilcoxon over
metastases (descriptive; all d, and d != 0 as the 2026-10-14 script did). Writes OUT/l1_metastasis.tsv.gz and
OUT/l1_family.tsv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    units = pd.read_csv(a.inputs / "lineage_preprocessing_factorial_20261103" / "UNITS.tsv", sep="\t", dtype=str)
    units = units[units.level == "L1"]
    fm = pd.read_csv(a.inputs / "v3_grid_20261104" / "kp" / "FIT_METRICS_v3tacc.tsv.gz", sep="\t",
                     usecols=["unit_id", "config_id", "method", "source_final_rejection_rate", "target_final_rejection_rate",
                              "calibration_m4e_inference_valid", "outer_converged"])
    fm = fm[fm.unit_id.isin(units.unit_id)].merge(units[["unit_id", "mouse", "source", "target", "same_met_family"]], on="unit_id")
    fm["src_ret"] = 1 - fm.source_final_rejection_rate; fm["tgt_ret"] = 1 - fm.target_final_rejection_rate
    rows = []
    for (cid, method, mouse, met), g in fm.groupby(["config_id", "method", "mouse", "target"]):
        same, other = g[g.same_met_family == "True"], g[g.same_met_family == "False"]
        if len(same) == 0 or len(other) == 0:
            continue
        rows.append({"config_id": cid, "method": method, "mouse": mouse, "metastasis": met,
                     "d": same.src_ret.mean() - other.src_ret.mean(), "d_target": same.tgt_ret.mean() - other.tgt_ret.mean(),
                     "same_retained": same.src_ret.mean(), "other_retained": other.src_ret.mean(),
                     "calibration_valid_share": float(g.calibration_m4e_inference_valid.astype(bool).mean())})
    M = pd.DataFrame(rows)
    M.to_csv(a.out / "l1_metastasis.tsv.gz", sep="\t", index=False, compression="gzip")
    summ = []
    for (cid, method), g in M.groupby(["config_id", "method"]):
        mm = g.groupby("mouse")[["d", "d_target"]].mean()
        nz = g.d[g.d != 0]
        summ.append({"config_id": cid, "method": method, "metastases": len(g), "mice": len(mm),
                     "mean_mouse_d": mm.d.mean(), "mice_d_pos": int((mm.d > 0).sum()), "median_d": g.d.median(),
                     "mean_mouse_d_target": mm.d_target.mean(), "mice_d_target_pos": int((mm.d_target > 0).sum()),
                     "wilcoxon_p_all": float(wilcoxon(g.d, alternative="greater", zero_method="zsplit").pvalue) if g.d.abs().sum() > 0 else np.nan,
                     "wilcoxon_p_nonzero": float(wilcoxon(nz, alternative="greater").pvalue) if len(nz) > 0 else np.nan,
                     "mean_mouse_d_without_3724": g[g.mouse != "3724"].groupby("mouse").d.mean().mean(),
                     "calibration_valid_share": g.calibration_valid_share.mean()})
    S = pd.DataFrame(summ)
    S.to_csv(a.out / "l1_family.tsv", sep="\t", index=False)
    print(f"{S.config_id.nunique()} configurations x {S.method.nunique()} methods; metastases per configuration {sorted(S.metastases.unique())}, mice {sorted(S.mice.unique())}")


if __name__ == "__main__":
    main()
