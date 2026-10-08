"""Dense against blockwise, on the MOSTA pairs the dense solver can hold.

The full run uses the blockwise solver for every pair, so it has to reproduce
the dense one first.  This runs before any pair too large for the dense
solver.

**Test pairs**, fixed by rule rather than chosen:
* every adjacent pair of E9.5->E10.5 and E10.5->E11.5;
* every same-stage pair of E9.5 and E10.5.

These are the stages whose pairs the dense solver holds comfortably, at most
about 19k x 31k bins.

**The fits.**  Each test pair is calibrated once.  The calibration is shared
between the two solvers, because it runs on dense 1,000 x 1,000 nulls either
way.  M4-E, M4-R and balanced OT are then fitted five ways:

* ``dense32``: ``ConfidenceOT.fit`` on ``squared_euclidean(source, target) /
  scale``, the production dense path;
* ``block32``: blockwise, float32, the default block size, i.e. what the run
  uses;
* ``block32s``: blockwise, float32, 512-row blocks, so the column reductions
  really are accumulated over many blocks;
* ``dense64`` and ``block64s`` (512-row blocks): the same in float64 on a
  float64 cost, which removes rounding and leaves only the algorithm.

**Criteria**, fixed before the run:

* A. float64, all three methods: final and raw gates identical; outer and
  inner iteration counts identical; convergence and cycle flags identical;
  decision costs within 1e-8 relative.  On every test pair whose float64
  dense reference fits in memory; a pair where it does not is recorded as
  "not runnable" and keeps its float32 comparison.
* B. float32, M4-E and balanced OT, the methods the analysis uses:
  * gates identical, except bins whose dense |relative margin| is at most
    1e-4, i.e. on the decision boundary at float32 resolution;
  * decision costs within 1e-4 relative where both are defined;
  * convergence flags identical;
  * objective within 1e-4 relative;
  * per-bin partner-annotation distributions within 1e-3.
* M4-R in float32 is reported with the same statistics but does not decide
  the outcome.  Its gate path has no monotone guarantee, so one boundary flip
  can send two runs to different points of a cycle.  Criterion A is what shows
  the implementation is the same.

Writes equivalence/report.csv (one row per pair, method and configuration),
equivalence/summary.json, and equivalence/EQUIVALENCE_PASS or
EQUIVALENCE_FAIL.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import Layout, read_pairs, write_json  # noqa: E402

TEST_GROUPS = ("E9.5->E10.5", "E10.5->E11.5", "E9.5", "E10.5")
BOUNDARY = 1e-4
SMALL_BLOCK = 512


def runner_module():
    path = Path(__file__).resolve().parent / "23_run_mosta_pair.py"
    spec = importlib.util.spec_from_file_location("mosta_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dense_fit(model, cost: np.ndarray, device: str):
    """The production dense path on a GPU; off one, the same torch solver on the CPU.

    ``ConfidenceOT(device='cpu')`` would run the NumPy reference instead, a
    different implementation, so the CPU branch calls the torch solver
    directly with the arguments ``fit`` would pass.
    """
    if device == "cuda":
        return model.fit(cost)
    from confidenceot.cuda import fit_cuda

    return fit_cuda(np.asarray(cost, dtype=np.float64), dtype=model.cuda_dtype, _torch_device="cpu",
                    **model._solver_kwargs())


def free_device(device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def distribution_from_dense(coupling, source_gate, target_gate, support, target_codes, n_groups):
    supported = coupling if support == "all" else coupling * source_gate[:, None] * target_gate[None, :]
    groups = np.zeros((coupling.shape[0], n_groups))
    for code in range(n_groups):
        groups[:, code] = supported[:, target_codes == code].sum(axis=1)
    mass = supported.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass[:, None] > 0, groups / mass[:, None], np.nan)


def compare(dense, block) -> dict:
    out = {}
    for side in ("source", "target"):
        dense_gate = getattr(dense, f"{side}_gate")
        block_gate = getattr(block, f"{side}_gate")
        differ = dense_gate != block_gate
        raw_differ = getattr(dense, f"{side}_raw_gate") != getattr(block, f"{side}_raw_gate")
        dense_conf = getattr(dense, f"{side}_confidence")
        block_conf = getattr(block, f"{side}_confidence")
        margin = np.abs(dense_conf.relative_rejection_margin)
        defined = (np.abs(dense_conf.decision_cost) > 0) & (np.abs(block_conf.decision_cost) > 0)
        relative = np.abs(block_conf.decision_cost - dense_conf.decision_cost) / np.maximum(
            np.abs(dense_conf.decision_cost), 1e-30)
        out.update({
            f"{side}_gate_differences": int(differ.sum()),
            f"{side}_raw_gate_differences": int(raw_differ.sum()),
            f"{side}_differing_max_abs_relative_margin": float(margin[differ | raw_differ].max()) if (differ | raw_differ).any() else 0.0,
            f"{side}_decision_cost_max_relative_difference": float(relative[defined].max()) if defined.any() else 0.0,
            f"{side}_rejected_fraction_dense": float(np.mean(~dense_gate)),
            f"{side}_rejected_fraction_block": float(np.mean(~block_gate)),
        })
    out.update({
        "outer_dense": dense.n_outer_iterations, "outer_block": block.n_outer_iterations,
        "inner_dense": dense.total_inner_iterations, "inner_block": block.total_inner_iterations,
        "inner_converged_dense": dense.inner_converged, "inner_converged_block": block.inner_converged,
        "outer_converged_dense": dense.outer_converged, "outer_converged_block": block.outer_converged,
        "cycle_dense": dense.cycle_detected, "cycle_block": block.cycle_detected,
        "objective_relative_difference": abs(block.objective - dense.objective) / max(abs(dense.objective), 1e-30),
        "seconds_dense": dense.fit_seconds, "seconds_block": block.fit_seconds,
    })
    return out


def criterion_a(row: dict) -> list[str]:
    problems = []
    for side in ("source", "target"):
        if row[f"{side}_gate_differences"] or row[f"{side}_raw_gate_differences"]:
            problems.append(f"{side} gates differ")
        if row[f"{side}_decision_cost_max_relative_difference"] > 1e-8:
            problems.append(f"{side} decision cost differs by {row[f'{side}_decision_cost_max_relative_difference']:.2e}")
    for key in ("outer", "inner", "inner_converged", "outer_converged", "cycle"):
        if row[f"{key}_dense"] != row[f"{key}_block"]:
            problems.append(f"{key} differs ({row[f'{key}_dense']} vs {row[f'{key}_block']})")
    return problems


def criterion_b(row: dict) -> list[str]:
    problems = []
    for side in ("source", "target"):
        if (row[f"{side}_gate_differences"] or row[f"{side}_raw_gate_differences"]) and \
                row[f"{side}_differing_max_abs_relative_margin"] > BOUNDARY:
            problems.append(f"{side} gate differs off the boundary "
                            f"(|relative margin| {row[f'{side}_differing_max_abs_relative_margin']:.2e})")
        if row[f"{side}_decision_cost_max_relative_difference"] > 1e-4:
            problems.append(f"{side} decision cost differs by {row[f'{side}_decision_cost_max_relative_difference']:.2e}")
    for key in ("inner_converged", "outer_converged"):
        if row[f"{key}_dense"] != row[f"{key}_block"]:
            problems.append(f"{key} differs")
    if row["objective_relative_difference"] > 1e-4:
        problems.append(f"objective differs by {row['objective_relative_difference']:.2e}")
    if row.get("partner_distribution_max_abs_difference", 0.0) > 1e-3:
        problems.append(f"partner distribution differs by {row['partner_distribution_max_abs_difference']:.2e}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--small-block", type=int, default=SMALL_BLOCK)
    parser.add_argument("--groups", nargs="+", default=list(TEST_GROUPS))
    args = parser.parse_args()
    from confidenceot.blockwise import CoordinateCost, transport_reductions
    from confidenceot.preprocessing import squared_euclidean
    import json

    runner = runner_module()
    layout = Layout(args.root)
    pairs = read_pairs(layout)
    tests = pairs[pairs["group"].isin(args.groups)].sort_values("index")
    rows = []
    started = time.perf_counter()
    for _, pair in tests.iterrows():
        pair_id = pair["pair_id"]
        rep = layout.representation(pair_id)
        if not (rep / "REPRESENTATION_SUCCESS").is_file():
            rows.append({"pair_id": pair_id, "method": "all", "configuration": "all",
                         "outcome": "missing representation"})
            continue
        stored = np.load(rep / "representation.npz")
        source, target = stored["source"], stored["target"]
        target_bins = runner.side_table(layout, pair["target_sample"], len(target))
        source_bins = runner.side_table(layout, pair["source_sample"], len(source))
        categories = sorted(set(target_bins["annotation"]))
        target_codes = pd.Categorical(target_bins["annotation"], categories=categories).codes
        source_codes = pd.Categorical(source_bins["annotation"], categories=sorted(set(source_bins["annotation"]))).codes
        scale, calibration, _ = runner.calibrate(source, target, int(pair["index"]), args.device)
        cost32 = CoordinateCost(source, target, scale)
        dense_cost32 = squared_euclidean(source, target) / scale
        source64, target64 = source.astype(np.float64), target.astype(np.float64)
        cost64 = CoordinateCost(source64, target64, scale)
        dense_cost64 = squared_euclidean(source64, target64) / scale
        for method, model in runner.models(float(calibration.rejection_cost), args.device).items():
            model64 = copy.copy(model)
            model64.cuda_dtype = "float64"
            support = "all" if method == "balanced" else "active"
            dense32 = dense_fit(model, dense_cost32, args.device)
            reference32 = distribution_from_dense(dense32.coupling, dense32.source_gate, dense32.target_gate,
                                                  support, target_codes, len(categories))
            # Keep the dense readouts, release the dense coupling.
            dense32 = dataclasses.replace(dense32, coupling=None)
            try:
                dense64 = dense_fit(model64, dense_cost64, args.device)
                reference64 = distribution_from_dense(dense64.coupling, dense64.source_gate, dense64.target_gate,
                                                      support, target_codes, len(categories))
                dense64 = dataclasses.replace(dense64, coupling=None)
            except (RuntimeError, MemoryError) as error:
                if "out of memory" not in str(error).lower() and not isinstance(error, MemoryError):
                    raise
                # The float64 dense reference does not fit this pair. Recorded, and
                # criterion A rests on the pairs where it does; float32 is still compared.
                dense64 = None
                free_device(args.device)
                rows.append({"pair_id": pair_id, "kind": pair["kind"], "group": pair["group"], "method": method,
                             "configuration": "block64s", "criterion": "not runnable",
                             "source_bins": len(source), "target_bins": len(target),
                             "outcome": "dense float64 reference exceeds memory; not compared",
                             "problems": str(error).splitlines()[0][:200]})
            configurations = [
                ("block32", lambda: model.fit_blockwise(cost32), dense32, reference32, cost32, "B"),
                ("block32s", lambda: model.fit_blockwise(cost32, block_rows=args.small_block), dense32, None, cost32, "B"),
            ]
            if dense64 is not None:
                configurations.append(
                    ("block64s", lambda: model64.fit_blockwise(cost64, block_rows=args.small_block), dense64, reference64, cost64, "A"))
            for configuration, fitter, dense, reference, cost, rule in configurations:
                block = fitter()
                row = {"pair_id": pair_id, "kind": pair["kind"], "group": pair["group"], "method": method,
                       "configuration": configuration, "criterion": rule if (rule == "A" or method != "M4-R") else "reported",
                       "source_bins": len(source), "target_bins": len(target),
                       "rejection_cost": float(calibration.rejection_cost), "block_rows": block.block_rows,
                       **compare(dense, block)}
                if reference is not None:
                    summary = transport_reductions(
                        cost, block, source_groups=source_codes, n_source_groups=int(source_codes.max()) + 1,
                        target_groups=target_codes, n_target_groups=len(categories),
                        source_points=source_bins[["x", "y"]].to_numpy(), target_points=target_bins[["x", "y"]].to_numpy(),
                        support=support,
                    )
                    mass = summary["source_mass"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        mine = np.where(mass[:, None] > 0, summary["source_partner_groups"] / mass[:, None], np.nan)
                    both = np.isfinite(mine) & np.isfinite(reference)
                    row["partner_distribution_max_abs_difference"] = float(np.max(np.abs(mine[both] - reference[both]))) if both.any() else 0.0
                    row["partner_distribution_defined_mismatch"] = int((np.isfinite(mine) != np.isfinite(reference)).sum())
                problems = criterion_a(row) if row["criterion"] == "A" else criterion_b(row)
                row["problems"] = "; ".join(problems)
                row["outcome"] = ("pass" if not problems else "fail") if row["criterion"] != "reported" else (
                    "reported: " + ("identical within criterion B" if not problems else "differs"))
                rows.append(row)
                print(f"{pair_id} {method} {configuration}: {row['outcome']} {row['problems']}", flush=True)
    report = pd.DataFrame(rows)
    layout.equivalence.mkdir(parents=True, exist_ok=True)
    report.to_csv(layout.equivalence / "report.csv", index=False)
    criterion = report["criterion"] if "criterion" in report else pd.Series("", index=report.index)
    decisive = report[criterion.isin(["A", "B"])]
    missing = report[report["outcome"].eq("missing representation")]
    failed = decisive[decisive["outcome"].eq("fail")]
    passed = len(failed) == 0 and len(missing) == 0 and len(decisive) > 0
    summary = {
        "test_groups": list(args.groups), "test_pairs": int(tests.shape[0]),
        "pairs_without_representation": missing["pair_id"].tolist(),
        "decisive_comparisons": int(len(decisive)), "failed_comparisons": int(len(failed)),
        "failures": failed[["pair_id", "method", "configuration", "problems"]].to_dict("records"),
        "m4r_float32_reported": report.loc[criterion.eq("reported"), "outcome"].value_counts().to_dict(),
        "float64_reference_not_runnable": report.loc[criterion.eq("not runnable"), ["pair_id", "method"]].to_dict("records"),
        "criteria": {"A": "float64: gates, iterations, flags identical; decision cost within 1e-8",
                     "B": f"float32 M4-E and balanced: gates identical except |relative margin| <= {BOUNDARY}; "
                          "decision cost within 1e-4; flags identical; objective within 1e-4; "
                          "partner distributions within 1e-3"},
        "outcome": "PASS" if passed else "FAIL",
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(layout.equivalence / "summary.json", summary)
    for name in ("EQUIVALENCE_PASS", "EQUIVALENCE_FAIL"):
        (layout.equivalence / name).unlink(missing_ok=True)
    (layout.equivalence / ("EQUIVALENCE_PASS" if passed else "EQUIVALENCE_FAIL")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
