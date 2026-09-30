"""Do the CPU and the GPU give the same answer?

``api.py`` names the CPU implementation the numerical reference, so the GPU
is allowed to be faster and is not allowed to be different. This separates
the two kinds of difference, because they mean opposite things.

**The geometry must be identical.** The cost matrix is built by the same
NumPy code on both paths -- ``squared_euclidean`` and ``median_pair_scale``
have no device -- so ``cost_scale``, ``cost_median`` and ``cost_max`` come
back bit for bit or something moved that should not have. A tolerance here
would hide exactly the failure worth catching.

**The solve may differ within tolerance.** Sinkhorn is an iterative fixed
point run to a threshold, and a GPU reduces in a different order, so the
decision costs agree to roughly the tolerance the solve was run at rather
than to the last bit. What matters is whether the *gate* moved: a cell whose
decision cost sits within rounding of the rejection cost can land either way,
and a handful of such cells is the expected disagreement. Many of them is
not, and the report gives the count and the margin of every one so the
difference can be read rather than accepted.

    python benchmark/60_compare_devices.py CPU_ROOT CUDA_ROOT --out report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# The solve runs to --tolerance 1e-4, so agreement to 1e-5 relative on a
# derived quantity is the solve reproducing itself, not the GPU being exact.
SOLVE_TOLERANCE = 1e-5
# The geometry has no device and no iteration, so it has no excuse.
GEOMETRY_TOLERANCE = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu_root", type=Path)
    parser.add_argument("cuda_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--solve-tolerance", type=float, default=SOLVE_TOLERANCE)
    return parser.parse_args()


def only_run(root: Path) -> Path:
    found = sorted(root.glob("*/scope_*/budget_*/run.json"))
    if len(found) != 1:
        raise SystemExit(
            f"{root}: expected exactly one completed run, found {len(found)}")
    return found[0].parent


def main() -> None:
    args = parse_args()
    cpu, cuda = only_run(args.cpu_root), only_run(args.cuda_root)
    report: dict = {"cpu_run": str(cpu), "cuda_run": str(cuda),
                    "solve_tolerance": args.solve_tolerance}

    left = json.loads((cpu / "run.json").read_text(encoding="utf-8"))
    right = json.loads((cuda / "run.json").read_text(encoding="utf-8"))

    geometry = {}
    identical = True
    for key in ("cost_scale", "cost_median", "cost_max"):
        a, b = left.get(key), right.get(key)
        same = a is not None and b is not None and float(a) == float(b)
        identical &= bool(same)
        geometry[key] = {"cpu": a, "cuda": b, "identical": bool(same)}
    report["geometry"] = geometry
    report["geometry_identical"] = bool(identical)

    for key in ("rejection_cost", "calibration_null", "preprocessing_label",
                "calibration_valid_for_m4r"):
        report.setdefault("calibration", {})[key] = {
            "cpu": left.get(key), "cuda": right.get(key)}

    gates = {}
    for name, directory in (("cpu", cpu), ("cuda", cuda)):
        table = pd.read_csv(directory / "cell_confidence.csv")
        gates[name] = table.set_index(
            table["method"].astype(str) + "|" + table["side"].astype(str)
            + "|" + table["observation_id"].astype(str))
    shared = gates["cpu"].index.intersection(gates["cuda"].index)
    if len(shared) != len(gates["cpu"]) or len(shared) != len(gates["cuda"]):
        report["cells_only_on_one_side"] = int(
            len(gates["cpu"]) + len(gates["cuda"]) - 2 * len(shared))
    a = gates["cpu"].loc[shared]
    b = gates["cuda"].loc[shared]

    retained_a = a["retained"].to_numpy(dtype=bool)
    retained_b = b["retained"].to_numpy(dtype=bool)
    moved = retained_a != retained_b
    cost_a = pd.to_numeric(a["decision_cost"], errors="coerce").to_numpy()
    cost_b = pd.to_numeric(b["decision_cost"], errors="coerce").to_numpy()
    rejection = float(left.get("rejection_cost", float("nan")))
    finite = np.isfinite(cost_a) & np.isfinite(cost_b)
    relative = np.abs(cost_a - cost_b) / np.maximum(np.abs(cost_a), 1e-12)

    report["gate"] = {
        "cells": int(len(shared)),
        "retained_cpu": int(retained_a.sum()),
        "retained_cuda": int(retained_b.sum()),
        "cells_that_changed_side": int(moved.sum()),
        "fraction_changed": float(moved.mean()) if moved.size else float("nan"),
        # How close to the threshold the cells that moved were. A cell that
        # flipped while sitting far from the rejection cost is not rounding.
        "changed_cell_margins": sorted(
            float(abs(value - rejection))
            for value in cost_a[moved][:20]) if moved.any() else [],
    }
    report["decision_cost"] = {
        "max_relative_difference": float(np.nanmax(relative[finite]))
        if finite.any() else float("nan"),
        "median_relative_difference": float(np.nanmedian(relative[finite]))
        if finite.any() else float("nan"),
        "within_tolerance": bool(finite.any() and
                                 np.nanmax(relative[finite]) <= args.solve_tolerance),
    }

    verdict = []
    if not report["geometry_identical"]:
        verdict.append("the cost geometry is NOT identical: the two runs are "
                       "not on the same problem and nothing else compares")
    if not report["decision_cost"]["within_tolerance"]:
        verdict.append(
            f"decision costs differ by up to "
            f"{report['decision_cost']['max_relative_difference']:.2e}, "
            f"above the {args.solve_tolerance:.0e} the solve was run to")
    if report["gate"]["cells_that_changed_side"]:
        verdict.append(
            f"{report['gate']['cells_that_changed_side']} of "
            f"{report['gate']['cells']} cells changed side")
    report["verdict"] = verdict or ["the two devices agree"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("geometry_identical", "gate", "decision_cost",
                       "verdict")}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
