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
way.  M4-E, M4-R and balanced OT are then fitted against two dense baselines:

* ``production``: ``ConfidenceOT.fit`` on ``squared_euclidean(source,
  target) / scale`` built from the float32 coordinates, i.e. the production
  dense path of the device;
* ``exact``: the same on a cost built in float64.

On a GPU the production dense path is the torch solver in float32.  The
blockwise fits there are:
* ``block32``: float32, the default block size;
* ``block32s``: float32, 512-row blocks;
* ``block64s``: float64, 512-row blocks, against ``exact``.

On a CPU the production dense path is the NumPy reference, float64, the path
the cancer runs on gg take.  The blockwise fits there are float64,
SOLVER_DTYPE['cpu']:
* ``block64``: the default block size, against ``production``;
* ``block64s``: 512-row blocks, against ``exact`` and against
  ``exact_torch``, the torch dense solver on the CPU in float64.

Small blocks make the column reductions really accumulate over many blocks.
The ``exact`` comparisons remove rounding and leave only the algorithm.

**Criteria**, fixed before the run:

* A. the ``exact`` comparisons, all three methods: final and raw gates identical; outer
  iteration counts identical; convergence and cycle flags identical; decision
  costs within 1e-8 relative; and, between the two torch solvers, inner
  iteration counts identical.  The NumPy reference counts its inner
  iterations differently: the torch solvers repeat their last solve as the
  final consistency solve and count it, with the same result.  On every test pair whose float64
  dense reference fits in memory; a pair where it does not is recorded as
  "not runnable" and keeps its float32 comparison.
* B. the ``production`` comparisons, M4-E and balanced OT, the methods the analysis uses:
  * gates identical, except bins whose dense |relative margin| is at most
    1e-4, i.e. on the decision boundary at float32 resolution;
  * decision costs within 1e-4 relative where both are defined;
  * convergence flags identical;
  * objective within 1e-4 relative;
  * per-bin partner-annotation distributions within 1e-3.
* M4-R under B is reported with the same statistics but does not decide
  the outcome.  Its gate path has no monotone guarantee, so one boundary flip
  can send two runs to different points of a cycle.  Criterion A is what shows
  the implementation is the same.

Writes equivalence/report.csv (one row per pair, method and configuration),
equivalence/summary.json, and equivalence/EQUIVALENCE_PASS or
EQUIVALENCE_FAIL.

**Where it runs.**  ``--device`` picks the device and with it the production
dense path, as above.  On either device the calibration is the production one
for that device, and it supplies the rejection cost both sides share.
``--shard``/``--n-shards`` split the test pairs over several jobs, and
``--merge`` combines their reports into one decision.
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
        "dense_backend": dense.backend, "block_backend": block.backend,
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
    keys = ["outer", "inner_converged", "outer_converged", "cycle"]
    # The torch solvers repeat their last Sinkhorn solve from the same start as
    # the final consistency solve, and count it; the NumPy reference does not.
    # The results are the same (they agree to 1e-15), the counters are not, so
    # inner counts are compared only between the two torch solvers.
    if row.get("dense_backend") != "numpy":
        keys.insert(1, "inner")
    for key in keys:
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


def compare_pairs(layout: Layout, tests: pd.DataFrame, *, device: str, small_block: int) -> list[dict]:
    """Every configuration of every method on every test pair, one row each."""
    from confidenceot.blockwise import CoordinateCost, transport_reductions
    from confidenceot.preprocessing import squared_euclidean

    runner = runner_module()
    rows = []
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
        scale, calibration, _ = runner.calibrate(source, target, int(pair["index"]), device)
        cost32 = CoordinateCost(source, target, scale)
        dense_cost32 = squared_euclidean(source, target) / scale
        source64, target64 = source.astype(np.float64), target.astype(np.float64)
        cost64 = CoordinateCost(source64, target64, scale)
        dense_cost64 = squared_euclidean(source64, target64) / scale
        for method, model in runner.models(float(calibration.rejection_cost), device).items():
            support = "all" if method == "balanced" else "active"
            if device == "cuda":
                # The production GPU path in SOLVER_DTYPE['cuda'], and the same solver
                # in float64 on a float64 cost.
                exact_model = copy.copy(model)
                exact_model.cuda_dtype = "float64"
                baselines = {
                    "production": lambda: model.fit(dense_cost32),
                    "exact": lambda: exact_model.fit(dense_cost64),
                }
                bits = model.cuda_dtype[-2:]
                plan = (
                    (f"block{bits}", lambda: model.fit_blockwise(cost32), ("production",), cost32, "B"),
                    (f"block{bits}s", lambda: model.fit_blockwise(cost32, block_rows=small_block), ("production",), cost32, "B"),
                    ("block64s", lambda: exact_model.fit_blockwise(cost64, block_rows=small_block), ("exact",), cost64, "A"),
                )
            else:
                # The production CPU path is the NumPy reference (ConfidenceOT.fit with
                # device='cpu', float64), on the cost built as production builds it.
                # The torch dense solver on the CPU is the blockwise solver's twin,
                # which counts its iterations the same way.
                from confidenceot.cuda import fit_cuda

                baselines = {
                    "production": lambda: model.fit(dense_cost32),
                    "exact": lambda: model.fit(dense_cost64),
                    "exact_torch": lambda: fit_cuda(np.asarray(dense_cost64, dtype=np.float64), dtype="float64",
                                                    _torch_device="cpu", **model._solver_kwargs()),
                }
                plan = (
                    ("block64", lambda: model.fit_blockwise(cost32), ("production",), cost32, "B"),
                    ("block64s", lambda: model.fit_blockwise(cost64, block_rows=small_block), ("exact", "exact_torch"), cost64, "A"),
                )
            dense_fits, references = {}, {}
            for name, run_dense in baselines.items():
                try:
                    fitted = run_dense()
                except (RuntimeError, MemoryError) as error:
                    if "out of memory" not in str(error).lower() and not isinstance(error, MemoryError):
                        raise
                    # A dense reference that does not fit is recorded; the other
                    # configurations of this pair are still compared.
                    free_device(device)
                    rows.append({"pair_id": pair_id, "kind": pair["kind"], "group": pair["group"], "method": method,
                                 "configuration": name, "criterion": "not runnable", "device": device,
                                 "source_bins": len(source), "target_bins": len(target),
                                 "outcome": f"dense {name} reference exceeds memory; not compared",
                                 "problems": str(error).splitlines()[0][:200]})
                    continue
                references[name] = distribution_from_dense(fitted.coupling, fitted.source_gate, fitted.target_gate,
                                                           support, target_codes, len(categories))
                # Keep the dense readouts, release the dense coupling.
                dense_fits[name] = dataclasses.replace(fitted, coupling=None)
                free_device(device)
            for configuration, fitter, wanted, cost, rule in plan:
                usable = [name for name in wanted if name in dense_fits]
                if not usable:
                    continue
                block = fitter()
                summary = transport_reductions(
                    cost, block, source_groups=source_codes, n_source_groups=int(source_codes.max()) + 1,
                    target_groups=target_codes, n_target_groups=len(categories),
                    source_points=source_bins[["x", "y"]].to_numpy(), target_points=target_bins[["x", "y"]].to_numpy(),
                    support=support,
                )
                mass = summary["source_mass"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    mine = np.where(mass[:, None] > 0, summary["source_partner_groups"] / mass[:, None], np.nan)
                for baseline in usable:
                    dense, reference = dense_fits[baseline], references[baseline]
                    row = {"pair_id": pair_id, "kind": pair["kind"], "group": pair["group"], "method": method,
                           "configuration": f"{configuration} vs {baseline}",
                           "criterion": rule if (rule == "A" or method != "M4-R") else "reported",
                           "device": device, "source_bins": len(source), "target_bins": len(target),
                           "rejection_cost": float(calibration.rejection_cost), "block_rows": block.block_rows,
                           **compare(dense, block)}
                    both = np.isfinite(mine) & np.isfinite(reference)
                    row["partner_distribution_max_abs_difference"] = float(np.max(np.abs(mine[both] - reference[both]))) if both.any() else 0.0
                    row["partner_distribution_defined_mismatch"] = int((np.isfinite(mine) != np.isfinite(reference)).sum())
                    problems = criterion_a(row) if row["criterion"] == "A" else criterion_b(row)
                    row["problems"] = "; ".join(problems)
                    row["outcome"] = ("pass" if not problems else "fail") if row["criterion"] != "reported" else (
                        "reported: " + ("identical within criterion B" if not problems else "differs"))
                    rows.append(row)
                    print(f"{pair_id} {method} {row['configuration']}: {row['outcome']} {row['problems']}", flush=True)
                free_device(device)
    return rows


def summarise(output: Path, report: pd.DataFrame, tests: pd.DataFrame, groups: list[str], seconds: float) -> bool:
    """Write the report, the summary and the PASS or FAIL marker; True on PASS."""
    import json

    output.mkdir(parents=True, exist_ok=True)
    covered = set(report["pair_id"]) if len(report) else set()
    absent = [pair_id for pair_id in tests["pair_id"] if pair_id not in covered]
    if absent:
        report = pd.concat([report, pd.DataFrame({"pair_id": absent, "method": "all", "configuration": "all",
                                                  "outcome": "no comparison recorded"})], ignore_index=True)
    report.to_csv(output / "report.csv", index=False)
    criterion = report["criterion"] if "criterion" in report else pd.Series("", index=report.index)
    decisive = report[criterion.isin(["A", "B"])]
    missing = report[report["outcome"].isin(["missing representation", "no comparison recorded"])]
    failed = decisive[decisive["outcome"].eq("fail")]
    passed = len(failed) == 0 and len(missing) == 0 and len(decisive) > 0
    summary = {
        "test_groups": list(groups), "test_pairs": int(tests.shape[0]),
        "devices": sorted(report["device"].dropna().unique().tolist()) if "device" in report else [],
        "pairs_without_comparison": missing["pair_id"].tolist(),
        "decisive_comparisons": int(len(decisive)), "failed_comparisons": int(len(failed)),
        "failures": failed[["pair_id", "method", "configuration", "problems"]].to_dict("records"),
        "m4r_float32_reported": report.loc[criterion.eq("reported"), "outcome"].value_counts().to_dict(),
        "float64_reference_not_runnable": report.loc[criterion.eq("not runnable"), ["pair_id", "method"]].to_dict("records"),
        "criteria": {"A": "exact (float64 cost): gates, iterations, flags identical; decision cost within 1e-8",
                     "B": f"production, M4-E and balanced: gates identical except |relative margin| <= {BOUNDARY}; "
                          "decision cost within 1e-4; flags identical; objective within 1e-4; "
                          "partner distributions within 1e-3"},
        "outcome": "PASS" if passed else "FAIL",
        "wall_seconds": seconds,
    }
    write_json(output / "summary.json", summary)
    for name in ("EQUIVALENCE_PASS", "EQUIVALENCE_FAIL"):
        (output / name).unlink(missing_ok=True)
    (output / ("EQUIVALENCE_PASS" if passed else "EQUIVALENCE_FAIL")).write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--small-block", type=int, default=SMALL_BLOCK)
    parser.add_argument("--groups", nargs="+", default=list(TEST_GROUPS))
    parser.add_argument("--output", type=Path, default=None,
                        help="Directory for the report and the PASS/FAIL marker; default <root>/equivalence")
    parser.add_argument("--shard", type=int, default=None,
                        help="Run only this shard of the test pairs and write <output>/shards/report_<shard>.csv")
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--merge", action="store_true",
                        help="Combine <output>/shards/*.csv into the report, summary and marker")
    args = parser.parse_args()
    if (args.shard is None) != (args.n_shards is None):
        parser.error("--shard and --n-shards go together")
    if args.merge and args.shard is not None:
        parser.error("--merge combines shards; it runs none itself")
    layout = Layout(args.root)
    output = args.output or layout.equivalence
    pairs = read_pairs(layout)
    tests = pairs[pairs["group"].isin(args.groups)].sort_values("index")
    started = time.perf_counter()
    if args.merge:
        shards = sorted((output / "shards").glob("report_*.csv"))
        report = pd.concat([pd.read_csv(path) for path in shards], ignore_index=True) if shards else pd.DataFrame(
            columns=["pair_id", "method", "configuration", "outcome"])
        print(f"merging {len(shards)} shard reports")
        sys.exit(0 if summarise(output, report, tests, args.groups, time.perf_counter() - started) else 1)
    if args.shard is not None:
        # Largest first, dealt round-robin, so every shard gets a similar load.
        sizes = pd.read_csv(layout.equalisation / "per_slice.csv").set_index("sample")["bins_analysed"]
        order = tests.assign(elements=tests["source_sample"].map(sizes) * tests["target_sample"].map(sizes))
        order = order.sort_values(["elements", "index"], ascending=[False, True]).reset_index(drop=True)
        mine = order[order.index % args.n_shards == args.shard]
        print(f"shard {args.shard} of {args.n_shards}: {len(mine)} of {len(tests)} test pairs", flush=True)
        rows = compare_pairs(layout, mine, device=args.device, small_block=args.small_block)
        (output / "shards").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output / "shards" / f"report_{args.shard:02d}.csv", index=False)
        print(f"shard {args.shard} done in {time.perf_counter() - started:.0f}s")
        return
    rows = compare_pairs(layout, tests, device=args.device, small_block=args.small_block)
    sys.exit(0 if summarise(output, pd.DataFrame(rows), tests, args.groups, time.perf_counter() - started) else 1)


if __name__ == "__main__":
    main()
