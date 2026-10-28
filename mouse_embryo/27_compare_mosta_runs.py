"""Compare two MOSTA runs pair by pair: a rerun against the run it repeats.

Made for the rerun of the finished ranknm256_noscale_ds_cos run that adds the
saved couplings (23_ from 2026-10-28): same representations, same code path,
so every output the old run wrote should come back unchanged.  Nothing is
decided here; every difference is counted and reported.

For every pair of manifest/pairs.csv, both runs' pairs/<id>/SUCCESS are
looked for; a pair complete in only one run is listed.  For a pair complete in
both, every output the old run wrote is compared:

* the gates (m4e/m4r retained and raw retained of bins_source / bins_target):
  bins differing, per method and side;
* every other column of bins_*.csv.gz, annotation_fractions.csv,
  annotation_transitions.csv.gz and pair_metrics.csv (except its timing
  columns), every array of transport_*.npz, every field of calibration.json
  (except its timing) and the analysis fields of run.json: values differing,
  and the largest absolute and relative difference.

Writes <new root>/comparison/<old run name>/: values.csv (pair x file x
column), gates.csv, summary.json and REPORT.md.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import Layout, read_pairs, write_json  # noqa: E402

TABLES = ("bins_source.csv.gz", "bins_target.csv.gz", "annotation_fractions.csv",
          "annotation_transitions.csv.gz", "pair_metrics.csv")
ARRAYS = ("transport_source.npz", "transport_target.npz")
TIMING = {"fit_seconds", "wall_seconds", "summary_seconds_shared", "pair_seconds_shared", "calibration_seconds"}
RUN_FIELDS = ("source_bins", "target_bins", "preprocessing_label", "representation_seed", "solver_seed_rule",
              "cost_scale", "cost_max", "rejection_cost", "epsilon", "lambda", "tolerance",
              "rejection_budget_unenforced", "within_side_acceptance_minimum", "calibration_bins_per_side",
              "block_rows", "summary_block_rows", "source_categories", "target_categories", "rank_cut_short_bins")
GATE_COLUMNS = ("m4e_retained", "m4e_raw_retained", "m4r_retained", "m4r_raw_retained")


def compare_values(old, new) -> dict:
    """Counts and largest differences of two equally shaped value collections."""
    old = np.asarray(old)
    new = np.asarray(new)
    if old.shape != new.shape:
        return {"n_values": int(max(old.size, new.size)), "n_different": -1, "max_abs": np.nan, "max_rel": np.nan,
                "note": f"shape {old.shape} vs {new.shape}"}
    if old.dtype.kind in "biuf" and new.dtype.kind in "biuf":
        a = old.astype(np.float64)
        b = new.astype(np.float64)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        finite = np.isfinite(a) & np.isfinite(b)
        difference = np.abs(a - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = np.where(np.abs(a) > 0, difference / np.abs(a), np.where(difference > 0, np.inf, 0.0))
        return {"n_values": int(a.size), "n_different": int((~same).sum()),
                "max_abs": float(difference[finite].max(initial=0.0)),
                "max_rel": float(relative[finite].max(initial=0.0)), "note": ""}
    same = old.astype(str) == new.astype(str)
    return {"n_values": int(old.size), "n_different": int((~same).sum()), "max_abs": np.nan, "max_rel": np.nan,
            "note": "text"}


def flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def compare_pair(task: tuple[str, str, str]) -> tuple[list[dict], list[dict]]:
    old_root, new_root, pair_id = task
    old_dir = Path(old_root) / "pairs" / pair_id
    new_dir = Path(new_root) / "pairs" / pair_id
    values, gates = [], []

    def record(file, column, result):
        values.append({"pair_id": pair_id, "file": file, "column": column, **result})

    for name in TABLES:
        old = pd.read_csv(old_dir / name)
        new = pd.read_csv(new_dir / name)
        missing = sorted(set(old.columns) - set(new.columns))
        if missing:
            record(name, ",".join(missing), {"n_values": len(old), "n_different": -1, "max_abs": np.nan,
                                             "max_rel": np.nan, "note": "columns missing in the new run"})
        if len(old) != len(new):
            record(name, "(rows)", {"n_values": len(old), "n_different": -1, "max_abs": np.nan, "max_rel": np.nan,
                                    "note": f"{len(old)} vs {len(new)} rows"})
            continue
        for column in old.columns:
            if column in TIMING or column not in new.columns:
                continue
            result = compare_values(old[column].to_numpy(), new[column].to_numpy())
            record(name, column, result)
            if name.startswith("bins_") and column in GATE_COLUMNS:
                gates.append({"pair_id": pair_id, "side": name[5:11], "column": column, "n_bins": len(old),
                              "n_different": result["n_different"]})
    for name in ARRAYS:
        with np.load(old_dir / name) as old, np.load(new_dir / name) as new:
            for key in old.files:
                if key not in new.files:
                    record(name, key, {"n_values": int(old[key].size), "n_different": -1, "max_abs": np.nan,
                                       "max_rel": np.nan, "note": "missing in the new run"})
                    continue
                record(name, key, compare_values(old[key], new[key]))
    for name, keep in (("calibration.json", None), ("run.json", RUN_FIELDS)):
        old = flatten(json.loads((old_dir / name).read_text(encoding="utf-8")))
        new = flatten(json.loads((new_dir / name).read_text(encoding="utf-8")))
        for key, value in old.items():
            top = key.split(".")[0]
            if (keep is not None and top not in keep) or key.split(".")[-1] in TIMING:
                continue
            if key not in new:
                record(name, key, {"n_values": 1, "n_different": -1, "max_abs": np.nan, "max_rel": np.nan,
                                   "note": "missing in the new run"})
                continue
            if isinstance(value, list) or isinstance(new[key], list):
                try:
                    numbers = (np.asarray(value, dtype=np.float64), np.asarray(new[key], dtype=np.float64))
                except (TypeError, ValueError):
                    numbers = None
                if numbers is not None:
                    record(name, key, compare_values(*numbers))
                else:
                    same = json.dumps(value) == json.dumps(new[key])
                    record(name, key, {"n_values": 1, "n_different": int(not same), "max_abs": np.nan,
                                       "max_rel": np.nan, "note": "list"})
            elif value is None or new[key] is None or isinstance(value, (str, bool)):
                record(name, key, {"n_values": 1, "n_different": int(value != new[key]), "max_abs": np.nan,
                                   "max_rel": np.nan, "note": "scalar"})
            else:
                record(name, key, compare_values(np.array([value], dtype=np.float64), np.array([new[key]], dtype=np.float64)))
    return values, gates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old_root", type=Path)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    started = time.perf_counter()
    old, new = Layout(args.old_root), Layout(args.new_root)
    pairs = read_pairs(new)
    complete = {name: {pid for pid in pairs["pair_id"] if (layout.pair(pid) / "SUCCESS").is_file()}
                for name, layout in (("old", old), ("new", new))}
    both = sorted(complete["old"] & complete["new"])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(compare_pair, [(str(old.root), str(new.root), pid) for pid in both], chunksize=4))
    values = pd.DataFrame([row for chunk, _ in results for row in chunk])
    gates = pd.DataFrame([row for _, chunk in results for row in chunk])
    output = new.root / "comparison" / old.root.name
    output.mkdir(parents=True, exist_ok=True)
    values.to_csv(output / "values.csv", index=False)
    gates.to_csv(output / "gates.csv", index=False)
    differing = values[values["n_different"] != 0]
    by_file = values.groupby(["file", "column"]).agg(
        pairs=("pair_id", "nunique"), values=("n_values", "sum"),
        pairs_differing=("n_different", lambda column: int((column != 0).sum())),
        values_differing=("n_different", lambda column: int(column.clip(lower=0).sum())),
        max_abs=("max_abs", "max"), max_rel=("max_rel", "max")).reset_index()
    by_file.to_csv(output / "by_column.csv", index=False)
    summary = {
        "old_root": str(old.root), "new_root": str(new.root),
        "pairs_expected": int(len(pairs)), "complete_old": len(complete["old"]), "complete_new": len(complete["new"]),
        "compared": len(both),
        "only_old": sorted(complete["old"] - complete["new"]), "only_new": sorted(complete["new"] - complete["old"]),
        "pairs_identical_everywhere": int(len(set(both) - set(differing["pair_id"]))),
        "pairs_with_any_difference": int(differing["pair_id"].nunique()),
        "pairs_with_a_gate_difference": int(gates.loc[gates["n_different"] != 0, "pair_id"].nunique()) if len(gates) else 0,
        "gate_bins_differing": {column: int(group["n_different"].clip(lower=0).sum())
                                for column, group in gates.groupby("column")} if len(gates) else {},
        "largest_relative_difference_by_file": values.groupby("file")["max_rel"].max().fillna(0).to_dict(),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    lines = [f"# {new.root.name} against {old.root.name}", "",
             f"Pairs expected {summary['pairs_expected']}, complete in old {summary['complete_old']}, "
             f"in new {summary['complete_new']}, compared {summary['compared']}.  ",
             f"Identical in every compared value: **{summary['pairs_identical_everywhere']}** pairs; "
             f"with any difference: {summary['pairs_with_any_difference']}; "
             f"with a gate difference: {summary['pairs_with_a_gate_difference']}.", "",
             "| file | column | pairs | values | pairs differing | values differing | max abs | max rel |",
             "|---|---|---|---|---|---|---|---|"]
    for _, row in by_file.sort_values(["values_differing", "file"], ascending=[False, True]).iterrows():
        lines.append(f"| {row['file']} | {row['column']} | {row['pairs']} | {row['values']} | {row['pairs_differing']} | "
                     f"{row['values_differing']} | {row['max_abs']:.3g} | {row['max_rel']:.3g} |")
    if summary["only_old"] or summary["only_new"]:
        lines += ["", f"Complete only in old: {summary['only_old']}", f"Complete only in new: {summary['only_new']}"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:5]))


if __name__ == "__main__":
    main()
