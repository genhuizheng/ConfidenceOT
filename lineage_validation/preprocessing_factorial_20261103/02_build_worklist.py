"""Registry and worklist of the lineage preprocessing factorial (PLAN.md sections 3, 7). Light; runs anywhere the units
table is.

    python lineage_validation/preprocessing_factorial_20261103/02_build_worklist.py --root ROOT

Writes ROOT/FACTORIAL_CONFIG_REGISTRY.tsv (the 592 grid configurations + the reproduction label), ROOT/WORKLIST.tsv
(every unit x grid configuration, plus the reproduction label on every L1 unit) and ROOT/PILOT_WORKLIST.tsv (a subset
of the same rows, same output directories, so the full run skips what the pilot finished). Refuses if any exists.
Row order: configuration-major, units largest first within a configuration, so the workers running at one time hold
units of mixed size (memory) and the largest of each wave start first.
Pilot (fixed here, before any result): every L23 unit x 4 configurations (logcpm none cos, ranknm256 production P10
cos, rank_value1024 expected none sqeuc, rankhvg2000 midrank P50 scale cos), 24 L1 units spread over the size range x
3 configurations (the first three of that list), and the reproduction label on 6 of those L1 units.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "realdata_benchmark"))
import factorial_common as fc  # noqa: E402

REPRODUCTION = [{"config_id": "label:ranknm256_noscale_ds_cos|pca30", "label": "ranknm256_noscale_ds_cos", "n_pcs": 30}]
PILOT_CONFIGS = ["logcpm|none|noscale|cos|pca50", "ranknm256-production|P10s1|noscale|cos|pca50",
                 "rank_value1024-expected|none|noscale|sqeuc|pca50", "rankhvg2000-midrank|P50s1|scale|cos|pca50"]


def safe(s: str) -> str:
    return s.replace("|", "__").replace(":", "_")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    outs = [a.root / n for n in ("FACTORIAL_CONFIG_REGISTRY.tsv", "WORKLIST.tsv", "PILOT_WORKLIST.tsv")]
    for f in outs:
        if f.exists():
            sys.exit(f"{f} exists; not overwritten")
    reg = fc.registry(REPRODUCTION)
    missing = [c for c in PILOT_CONFIGS if c not in set(reg.config_id)]
    if missing:
        sys.exit(f"pilot configurations not in the grid: {missing}")
    units = pd.read_csv(a.root / "UNITS.tsv", sep="\t", dtype={"mouse": str}).sort_values("size", ascending=False)
    rows = []
    for k, c in enumerate(reg.itertuples()):
        us = units[units.level == "L1"] if c.role == "reproduction" else units
        for u in us.itertuples():
            rows.append({"config_order": k, "config_id": c.config_id, "unit_id": u.unit_id, "level": u.level, "dataset": u.dataset,
                         "mouse": u.mouse, "size": u.size, "out": str(a.root / "rows" / safe(u.unit_id) / safe(c.config_id))})
    w = pd.DataFrame(rows)
    pilot_units_l1 = units[units.level == "L1"].iloc[np.unique(np.linspace(0, (units.level == "L1").sum() - 1, 24).round().astype(int))].unit_id
    pick = (w.level.eq("L23") & w.config_id.isin(PILOT_CONFIGS)) | \
           (w.unit_id.isin(pilot_units_l1) & w.config_id.isin(PILOT_CONFIGS[:3])) | \
           (w.unit_id.isin(pilot_units_l1.iloc[::4]) & w.config_id.eq(REPRODUCTION[0]["config_id"]))
    reg.to_csv(outs[0], sep="\t", index=False)
    w.to_csv(outs[1], sep="\t", index=False)
    w[pick].to_csv(outs[2], sep="\t", index=False)
    print(reg.groupby(["role", "encoding"]).size().to_string())
    print(f"{len(reg)} configurations; worklist {len(w)} rows ({w.level.value_counts().to_dict()}); pilot {int(pick.sum())} rows")


if __name__ == "__main__":
    main()
