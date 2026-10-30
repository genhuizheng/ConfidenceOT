"""Pair manifest of the LTL331 time course: each time point to the next (adjacent pairs).

    python ltl331_nepc/02_build_manifest.py --converted CONVERTED_DIR --out manifest_raw.csv

Reads CONVERTED_DIR/samples.tsv (01_convert_geo.py) and orders the time points by weeks after castration (preCX = 0;
the relapsed LTL331R points last at 20 and 22 weeks). Writes one row per adjacent pair in the format 27_ and
02_run_pair.py read (pair_id, dataset_id, patient_id, design, source_sample, target_sample, source_h5ad,
target_h5ad, source_h5ads_json, target_h5ads_json, eligible, skip_reason), plus source_weeks, target_weeks and
delta_weeks. Every pair is eligible; a time point with fewer than 20 cells would be marked, not dropped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

MIN_CELLS = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    s = pd.read_csv(args.converted / "samples.tsv", sep="\t").sort_values(["time_weeks", "relapse"]).reset_index(drop=True)
    rows = []
    for a, b in zip(s.index[:-1], s.index[1:]):
        src, tgt = s.loc[a], s.loc[b]
        small = [x["sample_id"] for x in (src, tgt) if x["cells"] < MIN_CELLS]
        rows.append({"pair_id": f"LTL331__{src['sample_id']}__{tgt['sample_id']}", "dataset_id": "GSE297328", "patient_id": "LTL331",
                     "design": "adjacent_time_points", "source_sample": src["sample_id"], "target_sample": tgt["sample_id"],
                     "source_h5ad": src["h5ad"], "target_h5ad": tgt["h5ad"],
                     "source_h5ads_json": json.dumps([src["h5ad"]]), "target_h5ads_json": json.dumps([tgt["h5ad"]]),
                     "source_weeks": int(src["time_weeks"]), "target_weeks": int(tgt["time_weeks"]),
                     "delta_weeks": int(tgt["time_weeks"] - src["time_weeks"]),
                     "eligible": not small, "skip_reason": f"fewer than {MIN_CELLS} cells: {small}" if small else ""})
    m = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(args.out, index=False)
    print(m[["pair_id", "source_weeks", "target_weeks", "eligible"]].to_string(index=False))


if __name__ == "__main__":
    main()
