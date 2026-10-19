"""Numbers only: the arm heatmap's columns, directional F1 and false asserted mass, per arm.

    python benchmark/58_summarise_arms.py SCORE_DIR --arms LABEL [LABEL ...]

SCORE_DIR holds the per-condition tables 40_score.py wrote as <arm>__<condition>.csv
(benchmark/tacc/score_arms.slurm). For each arm they are pooled into pooled_<arm>.csv,
and for each method (M4-E, M4-R):

* the arm heatmap's readouts (45_plot_arm_heatmap.READOUTS, imported, not copied):
  each is the fraction of a named set of cells the gate rejected, with its ground
  truth (0 = should be kept, 1 = should be rejected), as the mean over the pairs of
  that case at each technical level;
* directional F1 as 55_plot_method_comparison.py defines it: the F1 of rejection on
  the side holding the cells to reject (source for population_lost and
  populations_disjoint, target for population_emerged), 2PR/(P+R) per pair, 0 when
  nothing was rejected; mean and SD over those pairs;
* false asserted mass by case (cross-population mass between cells kept on both
  sides; ideal 0 everywhere).

Writes summary_long.tsv and summary_wide.tsv in SCORE_DIR and prints the wide table.
No figure.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("arm_heatmap", HERE / "45_plot_arm_heatmap.py")
arm_heatmap = importlib.util.module_from_spec(_spec)
sys.modules["arm_heatmap"] = arm_heatmap
_spec.loader.exec_module(arm_heatmap)
READOUTS, LEVELS = arm_heatmap.READOUTS, arm_heatmap.LEVELS
DIRECTIONAL = {"population_lost": "source", "population_emerged": "target", "populations_disjoint": "source"}
EXPECTED_PAIRS = 180


def directional_f1(frame: pd.DataFrame) -> pd.Series:
    values = []
    for record in frame.itertuples(index=False):
        side = DIRECTIONAL.get(record.biological_case)
        if side is None:
            values.append(np.nan)
            continue
        recall = getattr(record, f"{side}_recall")
        precision = getattr(record, f"{side}_precision")
        if not np.isfinite(recall) or recall == 0 or not np.isfinite(precision):
            values.append(0.0)
        else:
            values.append(2 * precision * recall / (precision + recall))
    return pd.Series(values, index=frame.index)


def summarise(pooled: pd.DataFrame, arm: str) -> list[dict]:
    rows = []
    for method, block in pooled.groupby("method"):
        pairs = block["pair_id"].nunique()
        base = {"arm": arm, "method": method}
        rows.append({**base, "quantity": "pairs scored", "level": "all", "value": pairs})
        for group, title, case, column, truth in READOUTS:
            for level, short in LEVELS:
                part = block[(block["biological_case"] == case) & (block["technical_level"] == level)]
                rows.append({**base, "quantity": f"{group} | {title.replace(chr(10), ' ')} | ground truth {truth}",
                             "level": short, "value": float(part[column].mean()), "pairs": len(part)})
        directional = block[block["biological_case"].isin(DIRECTIONAL)].copy()
        directional["f1"] = directional_f1(directional)
        rows.append({**base, "quantity": "directional F1, mean", "level": "all",
                     "value": float(directional["f1"].mean()), "pairs": len(directional)})
        rows.append({**base, "quantity": "directional F1, SD", "level": "all",
                     "value": float(directional["f1"].std()), "pairs": len(directional)})
        for case, part in directional.groupby("biological_case"):
            rows.append({**base, "quantity": f"directional F1, {case}", "level": "all",
                         "value": float(part["f1"].mean()), "pairs": len(part)})
        for case, part in block.groupby("biological_case"):
            if "false_asserted_mass" in part:
                rows.append({**base, "quantity": f"false asserted mass, {case}", "level": "all",
                             "value": float(part["false_asserted_mass"].mean()), "pairs": len(part)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("score_dir", type=Path)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    rows = []
    for arm in args.arms:
        parts = sorted(args.score_dir.glob(f"{arm}__*.csv"))
        frames = [pd.read_csv(path) for path in parts if path.stat().st_size > 0]
        if not frames:
            print(f"{arm}: no scored tables", file=sys.stderr)
            continue
        pooled = pd.concat(frames, ignore_index=True)
        pooled.to_csv(args.score_dir / f"pooled_{arm}.csv", index=False)
        missing = EXPECTED_PAIRS - pooled["pair_id"].nunique()
        if missing:
            print(f"{arm}: {missing} of {EXPECTED_PAIRS} pairs have no scored run", file=sys.stderr)
        rows += summarise(pooled, arm)
    long = pd.DataFrame(rows)
    long.to_csv(args.score_dir / "summary_long.tsv", sep="\t", index=False)
    wide = long.pivot_table(index=["method", "quantity", "level"], columns="arm", values="value", sort=False)
    wide = wide[[arm for arm in args.arms if arm in wide.columns]]
    wide.to_csv(args.score_dir / "summary_wide.tsv", sep="\t")
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 90):
        print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
