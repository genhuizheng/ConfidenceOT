"""The prespecified KP-Tracer readout (READOUT_PRESPECIFICATION_2026-10-14.md).

    python kp_lineage/02_kptracer_readout.py PREPARED_DIR OT_BLOCK_DIR OUT_DIR \\
        [--label ranknm256_noscale_ds_cos --label raw]

PREPARED_DIR holds pair_manifest_all.csv from 01_prepare_kptracer_pairs.py;
OT_BLOCK_DIR is <root>/ot/kptracer, with one directory per preprocessing
label beneath it, as tacc_preprocessing_factorial.slurm writes them.

For each label, and nothing else, as prespecified:

* **Per pair** the source retained fraction is 1 - source_final_rejection_rate
  of the M4-E row of pair_metrics.csv, and the target retained fraction
  likewise. Every pair of the manifest is listed: ineligible ones with their
  reason, eligible ones without a SUCCESS marker as missing. Nothing is
  filtered or substituted.
* **Per metastasis** that has at least one completed same-family and one
  completed other-family primary in its mouse: d = mean same-family retained
  fraction - mean other-family retained fraction (there is one same-family
  primary per metastasis in this data; the mean is the rule if there were
  more).
* **Primary test**: one-sided Wilcoxon signed-rank of d > 0 across those
  metastases. Supported if median d > 0 and p < 0.05.
* **Alongside**: per mouse, how many mice have median d > 0; the same
  contrast on the target retained fraction; and the full per-pair table.

The ds arm is primary; the raw arm is reported beside it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

METHOD = "M4-E"


def pair_metrics(label_dir: Path, pair_id: str) -> tuple[str, dict]:
    candidates = sorted((label_dir / pair_id).glob("scope_malignant/*/SUCCESS"))
    if not candidates:
        return "missing", {}
    if len(candidates) > 1:
        raise RuntimeError(f"{pair_id}: several completed outputs {candidates}; refusing to choose")
    table = pd.read_csv(candidates[0].parent / "pair_metrics.csv")
    rows = table[table["method"].astype(str).eq(METHOD)]
    if len(rows) != 1:
        raise RuntimeError(f"{pair_id}: {len(rows)} {METHOD} rows in pair_metrics.csv")
    row = rows.iloc[0]
    return "completed", {
        "source_retained_fraction": 1.0 - float(row["source_final_rejection_rate"]),
        "target_retained_fraction": 1.0 - float(row["target_final_rejection_rate"]),
        "source_analyzed_n": int(row["source_analyzed_n"]),
        "target_analyzed_n": int(row["target_analyzed_n"]),
        "output": str(candidates[0].parent),
    }


def contrast(pairs: pd.DataFrame, value: str) -> pd.DataFrame:
    done = pairs[pairs["status"].eq("completed")]
    rows = []
    for (mouse, metastasis), group in done.groupby(["patient_id", "target_sample"], sort=True):
        same = group[group["same_met_family"]]
        other = group[~group["same_met_family"]]
        rows.append({
            "patient_id": mouse, "metastasis": metastasis,
            "same_family_primaries": ";".join(same["source_sample"]),
            "other_family_primaries": ";".join(other["source_sample"]),
            "same_family_retained": same[value].mean() if len(same) else np.nan,
            "other_family_retained_mean": other[value].mean() if len(other) else np.nan,
            "enters_test": bool(len(same) and len(other)),
        })
    table = pd.DataFrame(rows)
    table["d"] = table["same_family_retained"] - table["other_family_retained_mean"]
    return table


def test(table: pd.DataFrame) -> dict:
    entering = table[table["enters_test"]]
    d = entering["d"].to_numpy(dtype=float)
    if len(d) == 0:
        return {"metastases": 0}
    nonzero = d[d != 0]
    result = (stats.wilcoxon(nonzero, alternative="greater") if len(nonzero) else None)
    mice = entering.groupby("patient_id")["d"].median()
    p = float(result.pvalue) if result is not None else float("nan")
    return {
        "metastases": int(len(d)),
        "mice": int(entering["patient_id"].nunique()),
        "median_d": float(np.median(d)),
        "mean_d": float(np.mean(d)),
        "positive_d": int((d > 0).sum()), "negative_d": int((d < 0).sum()), "zero_d": int((d == 0).sum()),
        "wilcoxon_statistic": float(result.statistic) if result is not None else float("nan"),
        "wilcoxon_p_one_sided": p,
        "supported": bool(np.median(d) > 0 and p < 0.05),
        "mice_with_median_d_above_0": int((mice > 0).sum()),
        "per_mouse_median_d": {str(k): float(v) for k, v in mice.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("ot_block", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--label", action="append", dest="labels", default=None)
    args = parser.parse_args()
    labels = args.labels or ["ranknm256_noscale_ds_cos", "raw"]
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.prepared / "pair_manifest_all.csv")
    manifest["same_met_family"] = manifest["same_met_family"].fillna(False).astype(bool)
    manifest["eligible"] = manifest["eligible"].astype(bool)
    summary = {"method": METHOD, "labels": {}}
    for label in labels:
        label_dir = args.ot_block / label
        records = []
        for row in manifest.to_dict("records"):
            record = {key: row.get(key) for key in (
                "pair_id", "patient_id", "source_sample", "target_sample", "source_malignant_n",
                "target_malignant_n", "same_met_family", "eligible", "skip_reason")}
            if not row["eligible"] or not isinstance(row.get("source_sample"), str) or not row["source_sample"]:
                record["status"] = "ineligible"
            else:
                status, values = pair_metrics(label_dir, str(row["pair_id"]))
                record["status"] = status
                record.update(values)
            records.append(record)
        pairs = pd.DataFrame(records)
        pairs.to_csv(args.out / f"pairs_{label}.csv", index=False)
        source = contrast(pairs, "source_retained_fraction")
        target = contrast(pairs, "target_retained_fraction")
        source.to_csv(args.out / f"metastasis_contrast_source_{label}.csv", index=False)
        target.to_csv(args.out / f"metastasis_contrast_target_{label}.csv", index=False)
        summary["labels"][label] = {
            "pairs": {k: int(v) for k, v in pairs["status"].value_counts().items()},
            "missing_pairs": pairs.loc[pairs["status"].eq("missing"), "pair_id"].tolist(),
            "primary_source": test(source),
            "alongside_target": test(target),
        }
    summary["primary_label"] = labels[0]
    (args.out / "readout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
