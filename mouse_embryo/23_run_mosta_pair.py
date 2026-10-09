"""Calibrate and fit ConfidenceOT and balanced OT on every MOSTA pair, blockwise.

One pair, start to finish.  This is the production protocol of
cancer_metastasis/02_run_pair.py with nothing added or removed, except that
every bin is used, with no per-side cap:

1. ``rng = default_rng(seed + index * 104729)``.  From it, ``scale`` is the
   median over 1,000,000 sampled cross-side pairs.
2. Calibration as 02_ does it:
   * ``min(2000, N, M)`` label-blind bins per side, drawn from the same rng;
   * within-side split nulls, 5 + 5 replicates per side, seeds seed + index
     and seed + index + 7;
   * ``calibrate_confidence_cost`` with the UOT backbone, epsilon 0.1,
     lambda 1, acceptance 0.99, budget 0.95 unenforced, tolerance 1e-4, grid 5.

   The nulls are 1,000 x 1,000 and run through the dense solver, as in
   production.
3. M4-E (the gate), M4-R (diagnostic) and balanced OT, all three on every bin
   of both slices.  Balanced OT uses epsilon 0.1 and budget 0, so nothing is
   rejected: it is entropic balanced OT on the same cost.  The fits go through
   ``ConfidenceOT.fit_blockwise``: the dense torch solver carried out on row
   blocks of a cost rebuilt from the coordinates (confidenceot/blockwise.py),
   with the same parameters and the same stopping rule.
4. Summaries of the M4-E coupling on its retained x retained support and of
   the balanced coupling everywhere, taken before the potentials are dropped.

``--device`` picks the device and with it that device's production path.  The
run uses one device for every pair.
* cuda, the default and the production run: the torch CUDA solver in float64
  (SOLVER_DTYPE).
* cpu, for validation and reference comparisons: calibration through the
  NumPy reference, as the cancer runs on gg do, and the torch solvers in
  float64, the reference's precision.

Nothing is judged here.  Calibration flags, M4-R convergence and cycles are
written down, and that is all that is done with them.  A pair either completes
and is published whole, or fails with its traceback recorded.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import (  # noqa: E402
    CALIBRATION_GRID_SIZE, CALIBRATION_MAX_BINS, EPSILON, LAMBDA,
    NULL_CALIBRATION_REPLICATES, NULL_VALIDATION_REPLICATES, PREPROCESSING_LABEL,
    REJECTION_BUDGET, SEED, SOLVER_DTYPE, TOLERANCE, WITHIN_SIDE_ACCEPTANCE, Layout, claim,
    file_inventory, ordered_categories, publish, read_pairs, staging_directory,
    write_json,
)

SUCCESS = "SUCCESS"
FLOAT_FORMAT = "%.7g"


def side_table(layout: Layout, sample: str, expected_rows: int) -> pd.DataFrame:
    """The analysed bins of one slice, in equalised-row order."""
    table = pd.read_csv(layout.slice_bins(sample))
    table = table[table["equalised_row"] >= 0].sort_values("equalised_row").reset_index(drop=True)
    if len(table) != expected_rows or not np.array_equal(table["equalised_row"].to_numpy(), np.arange(expected_rows)):
        raise RuntimeError(f"{sample}: slice table does not match the representation ({len(table)} vs {expected_rows})")
    return table


def calibrate(source: np.ndarray, target: np.ndarray, index: int, device: str):
    from confidenceot import calibrate_confidence_cost, median_pair_scale, within_side_null_costs

    rng = np.random.default_rng(SEED + index * 104729)
    scale = median_pair_scale(source, target, rng=rng)
    limit = min(CALIBRATION_MAX_BINS, len(source), len(target))
    source_index = np.sort(rng.choice(len(source), limit, replace=False))
    target_index = np.sort(rng.choice(len(target), limit, replace=False))
    total = NULL_CALIBRATION_REPLICATES + NULL_VALIDATION_REPLICATES
    source_nulls = within_side_null_costs(source[source_index], observed_scale=scale,
                                          seed=SEED + index, n_replicates=total)
    target_nulls = within_side_null_costs(target[target_index], observed_scale=scale,
                                          seed=SEED + index + 7, n_replicates=total)
    split = NULL_CALIBRATION_REPLICATES
    started = time.perf_counter()
    calibration = calibrate_confidence_cost(
        source_nulls[:split] + target_nulls[:split], source_nulls[split:] + target_nulls[split:],
        backbone="uot", epsilon=EPSILON, lambda_a=LAMBDA, lambda_b=LAMBDA,
        null_semantics="within_side_split", within_side_acceptance_minimum=WITHIN_SIDE_ACCEPTANCE,
        source_rejection_budget=REJECTION_BUDGET, target_rejection_budget=REJECTION_BUDGET,
        tolerance=TOLERANCE, grid_size=CALIBRATION_GRID_SIZE, device=device,
        workers=1, fallback_to_cpu=False,
    )
    return scale, calibration, {
        "calibration_bins_per_side": int(limit),
        "calibration_source_rows": source_index.tolist(),
        "calibration_target_rows": target_index.tolist(),
        "calibration_seconds": time.perf_counter() - started,
    }


def models(rejection_cost: float, device: str) -> dict:
    from confidenceot import ConfidenceOT

    # cuda_dtype is the precision of the torch solvers, dense and blockwise, on
    # either device: float64 on both (SOLVER_DTYPE).
    common = dict(rejection_cost=rejection_cost, epsilon=EPSILON, tolerance=TOLERANCE, device=device,
                  cuda_dtype=SOLVER_DTYPE[device])
    return {
        "M4-E": ConfidenceOT(backbone="uot", variant="exact", lambda_a=LAMBDA, lambda_b=LAMBDA,
                             source_rejection_budget=REJECTION_BUDGET,
                             target_rejection_budget=REJECTION_BUDGET, **common),
        "M4-R": ConfidenceOT(backbone="uot", variant="reversible", lambda_a=LAMBDA, lambda_b=LAMBDA,
                             source_rejection_budget=REJECTION_BUDGET,
                             target_rejection_budget=REJECTION_BUDGET, **common),
        # Budget 0 resolves to rejection bounds (0, 0): every bin is kept, so
        # this is entropic balanced OT on the same cost, as the earlier MOSTA
        # run's traditional comparator was.
        "balanced": ConfidenceOT(backbone="balanced", variant="exact",
                                 source_rejection_budget=0.0, target_rejection_budget=0.0, **common),
    }


def centroid_and_spread(points_sum: np.ndarray, square_sum: np.ndarray, mass: np.ndarray):
    with np.errstate(divide="ignore", invalid="ignore"):
        centroid = np.where(mass[:, None] > 0, points_sum / mass[:, None], np.nan)
        variance = np.where(mass > 0, square_sum / mass - np.sum(centroid ** 2, axis=1), np.nan)
    return centroid, np.sqrt(np.clip(variance, 0.0, None))


def normalised(groups: np.ndarray, mass: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(mass[:, None] > 0, groups / mass[:, None], np.nan).astype(np.float32)


def annotation_rows(side: str, labels: np.ndarray, partner_labels: np.ndarray, weights: np.ndarray,
                    gates: dict, co_distribution: np.ndarray, co_mass: np.ndarray,
                    bot_distribution: np.ndarray, partner_categories: list[str]) -> list[dict]:
    """Per annotation of one side: hard fractions and the nominal-mass decomposition.

    ``co_same + co_other + co_retained_without_support + m4e_rejected`` sum to
    the annotation's nominal mass share (1.0) on every side of every pair.
    """
    partner_counts = pd.Series(partner_labels).value_counts()
    lookup = {name: position for position, name in enumerate(partner_categories)}
    rows = []
    for annotation in ordered_categories(labels):
        members = labels == annotation
        nominal = float(weights[members].sum())
        same_column = lookup.get(annotation)
        row = {"side": side, "annotation": annotation, "n_bins": int(members.sum()),
               "partner_present": bool(partner_counts.get(annotation, 0) > 0),
               "partner_bins": int(partner_counts.get(annotation, 0))}
        for method in ("M4-E", "M4-R"):
            retained = gates[method]["retained"][members]
            raw = gates[method]["raw_retained"][members]
            prefix = method.lower().replace("-", "")
            row[f"{prefix}_retained"] = int(retained.sum())
            row[f"{prefix}_rejected"] = int((~retained).sum())
            row[f"{prefix}_rejected_fraction"] = float((~retained).mean())
            row[f"{prefix}_raw_rejected_fraction"] = float((~raw).mean())
        retained = gates["M4-E"]["retained"]
        supported = members & retained & (co_mass > 0)
        same = 0.0 if same_column is None else float(np.sum(weights[supported] * co_distribution[supported, same_column]))
        total = float(np.sum(weights[supported] * np.nansum(co_distribution[supported], axis=1)))
        row["co_same_share"] = same / nominal
        row["co_other_share"] = (total - same) / nominal
        row["co_retained_without_support_share"] = float(weights[members & retained & ~(co_mass > 0)].sum()) / nominal
        for name, rows_mask in (("bot", members), ("bot_co_rejected", members & ~retained)):
            mass_share = float(weights[rows_mask].sum())
            if mass_share > 0:
                same = 0.0 if same_column is None else float(np.sum(weights[rows_mask] * bot_distribution[rows_mask, same_column]))
                row[f"{name}_same_share"] = same / mass_share
                row[f"{name}_other_share"] = 1.0 - same / mass_share
            else:
                row[f"{name}_same_share"] = np.nan
                row[f"{name}_other_share"] = np.nan
        rows.append(row)
    return rows


def transition_rows(direction: str, method: str, own_categories: list[str], partner_categories: list[str],
                    own_codes: np.ndarray, weights: np.ndarray, distribution: np.ndarray, rows_mask: np.ndarray,
                    raw_mass: np.ndarray | None) -> list[dict]:
    """Nominal mass (and raw coupling mass where defined) from each annotation to each partner annotation."""
    usable = rows_mask & np.all(np.isfinite(distribution), axis=1)
    out = []
    for a, own in enumerate(own_categories):
        members = usable & (own_codes == a)
        own_nominal = float(weights[own_codes == a].sum())
        if own_nominal == 0:
            continue
        flow = weights[members] @ distribution[members].astype(np.float64)
        for b, partner in enumerate(partner_categories):
            out.append({
                "direction": direction, "method": method, "annotation": own, "partner_annotation": partner,
                "nominal_mass": float(flow[b]), "annotation_nominal_mass": own_nominal,
                "share_of_annotation": float(flow[b]) / own_nominal,
                "raw_coupling_mass": float(raw_mass[a, b]) if raw_mass is not None else np.nan,
            })
    return out


def run_pair(layout: Layout, row: pd.Series, *, device: str, block_rows: int | None) -> Path:
    from confidenceot.blockwise import CoordinateCost, resolve_block_rows, transport_reductions
    import json

    started = time.perf_counter()
    pair_id = str(row["pair_id"])
    index = int(row["index"])
    representation_dir = layout.representation(pair_id)
    if not (representation_dir / "REPRESENTATION_SUCCESS").is_file():
        raise FileNotFoundError(f"no completed representation for {pair_id}")
    stored = np.load(representation_dir / "representation.npz")
    source, target = stored["source"], stored["target"]
    representation = json.loads((representation_dir / "representation.json").read_text(encoding="utf-8"))
    if representation["preprocessing"]["label"] != PREPROCESSING_LABEL:
        raise SystemExit(f"{pair_id}: representation is {representation['preprocessing']['label']}")
    source_bins = side_table(layout, row["source_sample"], len(source))
    target_bins = side_table(layout, row["target_sample"], len(target))
    n, m = len(source), len(target)

    scale, calibration, calibration_extra = calibrate(source, target, index, device)
    rejection_cost = float(calibration.rejection_cost)
    cost = CoordinateCost(source, target, scale)
    rows_per_block = resolve_block_rows(block_rows, n, m)
    summary_rows = max(1, rows_per_block // 2)

    fits = {}
    metrics = []
    for method, model in models(rejection_cost, device).items():
        fit_started = time.perf_counter()
        fits[method] = model.fit_blockwise(cost, block_rows=rows_per_block)
        fit = fits[method]
        metrics.append({
            "pair_id": pair_id, "kind": row["kind"], "group": row["group"], "method": method,
            "backbone": fit.backbone, "variant": fit.variant,
            "source_sample": row["source_sample"], "target_sample": row["target_sample"],
            "source_bins": n, "target_bins": m, "rejection_cost": rejection_cost, "cost_scale": scale,
            "source_raw_rejection_rate": float(np.mean(~fit.source_raw_gate)),
            "target_raw_rejection_rate": float(np.mean(~fit.target_raw_gate)),
            "source_final_rejection_rate": float(np.mean(~fit.source_gate)),
            "target_final_rejection_rate": float(np.mean(~fit.target_gate)),
            "source_rejection_bounds": str(fit.source_rejection_bounds),
            "target_rejection_bounds": str(fit.target_rejection_bounds),
            "source_budget_override_rate": float(np.mean(fit.source_confidence.budget_overridden)),
            "target_budget_override_rate": float(np.mean(fit.target_confidence.budget_overridden)),
            "inner_converged": fit.inner_converged, "outer_converged": fit.outer_converged,
            "cycle_detected": fit.cycle_detected, "cycle_length": fit.cycle_length,
            "outer_iterations": fit.n_outer_iterations, "total_inner_iterations": fit.total_inner_iterations,
            "objective": fit.objective, "fit_seconds": fit.fit_seconds,
            "wall_seconds": time.perf_counter() - fit_started,
            "block_rows": fit.block_rows, "dtype": fit.dtype, "device": fit.device, "backend": fit.backend,
            "calibration_feasible_cost_found": bool(calibration.feasible_cost_found),
            "calibration_m4e_clean": bool(calibration.m4e_calibration_clean),
            "calibration_m4r_clean": bool(calibration.m4r_validation_clean),
            "calibration_m4e_inference_valid": bool(calibration.m4e_inference_valid),
            "calibration_valid_for_m4r": bool(calibration.calibration_valid),
        })
        print(f"{pair_id} {method}: rejected {metrics[-1]['source_final_rejection_rate']:.3f} / "
              f"{metrics[-1]['target_final_rejection_rate']:.3f}, outer {fit.n_outer_iterations}, "
              f"inner {fit.total_inner_iterations}, {fit.fit_seconds:.0f}s", flush=True)

    source_categories = ordered_categories(source_bins["annotation"])
    target_categories = ordered_categories(target_bins["annotation"])
    source_codes = pd.Categorical(source_bins["annotation"], categories=source_categories).codes.astype(np.int64)
    target_codes = pd.Categorical(target_bins["annotation"], categories=target_categories).codes.astype(np.int64)
    source_points = source_bins[["x", "y"]].to_numpy(dtype=np.float64)
    target_points = target_bins[["x", "y"]].to_numpy(dtype=np.float64)
    m4e = fits["M4-E"]
    summary_started = time.perf_counter()
    reduce = dict(source_groups=source_codes, n_source_groups=len(source_categories),
                  target_groups=target_codes, n_target_groups=len(target_categories),
                  source_points=source_points, target_points=target_points, block_rows=summary_rows)
    co = transport_reductions(cost, m4e, support="active",
                              source_strata={"retained": m4e.source_gate},
                              target_strata={"retained": m4e.target_gate}, **reduce)
    bot = transport_reductions(cost, fits["balanced"], support="all",
                               source_strata={"all": np.ones(n, dtype=bool), "co_rejected": ~m4e.source_gate},
                               target_strata={"all": np.ones(m, dtype=bool), "co_rejected": ~m4e.target_gate},
                               **reduce)
    summary_seconds = time.perf_counter() - summary_started

    staging = staging_directory(layout, pair_id, "ot")
    a = fits["M4-E"].source_weights
    b = fits["M4-E"].target_weights
    sides = {}
    for side, bins, n_side, partner_categories, partner_labels, weights in (
        ("source", source_bins, n, target_categories, target_bins["annotation"].to_numpy(dtype=str), a),
        ("target", target_bins, m, source_categories, source_bins["annotation"].to_numpy(dtype=str), b),
    ):
        table = pd.DataFrame({"bin_row": np.arange(n_side)})
        gates = {}
        for method in ("M4-E", "M4-R"):
            fit = fits[method]
            confidence = getattr(fit, f"{side}_confidence")
            retained = getattr(fit, f"{side}_gate")
            raw = getattr(fit, f"{side}_raw_gate")
            prefix = method.lower().replace("-", "")
            table[f"{prefix}_retained"] = retained.astype(np.int8)
            table[f"{prefix}_raw_retained"] = raw.astype(np.int8)
            table[f"{prefix}_decision_cost"] = confidence.decision_cost
            table[f"{prefix}_margin"] = confidence.signed_rejection_margin
            table[f"{prefix}_coefficient"] = confidence.gate_coefficient
            table[f"{prefix}_budget_overridden"] = confidence.budget_overridden.astype(np.int8)
            gates[method] = {"retained": retained, "raw_retained": raw}
        for prefix, summary in (("co", co), ("bot", bot)):
            mass = summary[f"{side}_mass"]
            centroid, spread = centroid_and_spread(summary[f"{side}_partner_points"],
                                                   summary[f"{side}_partner_square"], mass)
            table[f"{prefix}_partner_mass"] = mass
            table[f"{prefix}_total_mass"] = summary[f"{side}_total_mass"]
            table[f"{prefix}_partner_x"] = centroid[:, 0]
            table[f"{prefix}_partner_y"] = centroid[:, 1]
            table[f"{prefix}_partner_spread"] = spread
        table.to_csv(staging / f"bins_{side}.csv.gz", index=False, compression="gzip", float_format=FLOAT_FORMAT)
        co_distribution = normalised(co[f"{side}_partner_groups"], co[f"{side}_mass"])
        bot_distribution = normalised(bot[f"{side}_partner_groups"], bot[f"{side}_mass"])
        arrays = {
            "partner_annotations": np.asarray(partner_categories, dtype=str),
            "co_partner_distribution": co_distribution,
            "bot_partner_distribution": bot_distribution,
        }
        if side == "source":
            arrays.update(co_pullback=co["pull_retained"].astype(np.float32),
                          bot_pullback=bot["pull_all"].astype(np.float32),
                          bot_pullback_co_rejected=bot["pull_co_rejected"].astype(np.float32))
        else:
            arrays.update(co_pushforward=co["push_retained"].astype(np.float32),
                          bot_pushforward=bot["push_all"].astype(np.float32),
                          bot_pushforward_co_rejected=bot["push_co_rejected"].astype(np.float32))
        np.savez_compressed(staging / f"transport_{side}.npz", **arrays)
        sides[side] = annotation_rows(
            side, bins["annotation"].to_numpy(dtype=str), partner_labels, weights, gates,
            co_distribution, co[f"{side}_mass"], bot_distribution, partner_categories,
        )
    pd.DataFrame(sides["source"] + sides["target"]).to_csv(
        staging / "annotation_fractions.csv", index=False, float_format="%.9g")

    transitions = []
    for method, summary, source_mask, raw in (
        ("co", co, m4e.source_gate, co["group_mass"]),
        ("bot", bot, np.ones(n, dtype=bool), bot["group_mass"]),
        ("bot_co_rejected", bot, ~m4e.source_gate, None),
    ):
        distribution = normalised(summary["source_partner_groups"], summary["source_mass"])
        transitions += transition_rows("forward", method, source_categories, target_categories, source_codes,
                                       a, distribution, source_mask, raw)
    for method, summary, target_mask in (
        ("co", co, m4e.target_gate), ("bot", bot, np.ones(m, dtype=bool)),
        ("bot_co_rejected", bot, ~m4e.target_gate),
    ):
        distribution = normalised(summary["target_partner_groups"], summary["target_mass"])
        transitions += transition_rows("backward", method, target_categories, source_categories, target_codes,
                                       b, distribution, target_mask, None)
    pd.DataFrame(transitions).to_csv(staging / "annotation_transitions.csv.gz", index=False,
                                     compression="gzip", float_format="%.9g")

    for record in metrics:
        record["summary_seconds_shared"] = summary_seconds
        record["pair_seconds_shared"] = time.perf_counter() - started
    pd.DataFrame(metrics).to_csv(staging / "pair_metrics.csv", index=False)
    write_json(staging / "calibration.json", {**calibration.__dict__, **calibration_extra})
    gpu = None
    if device == "cuda":
        import torch

        gpu = {"name": torch.cuda.get_device_name(torch.cuda.current_device()),
               "peak_allocated_bytes": int(torch.cuda.max_memory_allocated())}
    write_json(staging / "run.json", {
        "pair_id": pair_id, "index": index, "kind": row["kind"], "group": row["group"],
        "source_sample": row["source_sample"], "target_sample": row["target_sample"],
        "source_bins": n, "target_bins": m,
        "preprocessing_label": PREPROCESSING_LABEL, "representation_seed": representation["seed"],
        "solver_seed_rule": f"{SEED} + index * 104729", "cost_scale": scale,
        "cost_max": float(co["cost_max"]), "rejection_cost": rejection_cost,
        "epsilon": EPSILON, "lambda": LAMBDA, "tolerance": TOLERANCE,
        "rejection_budget_unenforced": REJECTION_BUDGET,
        "within_side_acceptance_minimum": WITHIN_SIDE_ACCEPTANCE,
        "calibration_bins_per_side": calibration_extra["calibration_bins_per_side"],
        "observed_inference_uses_every_bin": True,
        "block_rows": rows_per_block, "summary_block_rows": summary_rows,
        "source_categories": source_categories, "target_categories": target_categories,
        "rank_cut_short_bins": {
            "source": representation["source_bins_encoded_under_rank_top_n"],
            "target": representation["target_bins_encoded_under_rank_top_n"],
        },
        "code_commit": os.environ.get("CONFIDENCEOT_COMMIT", "unknown"),
        "gpu": gpu, "summary_seconds": summary_seconds,
        "pipeline_seconds": time.perf_counter() - started,
    })
    write_json(staging / SUCCESS, {"pair_id": pair_id, "files": file_inventory(staging)})
    del fits, co, bot
    gc.collect()
    return staging


def worker(layout: Layout, *, device: str, block_rows: int | None, stop_after_hours: float,
           only: str | None, largest: bool = False) -> int:
    pairs = read_pairs(layout)
    sizes = pd.read_csv(layout.equalisation / "per_slice.csv").set_index("sample")["bins_analysed"]
    pairs["elements"] = pairs["source_sample"].map(sizes) * pairs["target_sample"].map(sizes)
    pairs = pairs.sort_values(["elements", "index"], ascending=[False, True])
    if only is not None:
        pairs = pairs[pairs["pair_id"] == only]
        if pairs.empty:
            raise SystemExit(f"{only} is not in {layout.pairs_csv}")
    if largest:
        # The scale probe: the one pair with the most cost entries, run as a
        # normal pair, so its time and peak memory are measured before the
        # workers start. Its output is published like any other.
        pairs = pairs.head(1)
    started = time.time()
    failures = 0
    for _, row in pairs.iterrows():
        pair_id = str(row["pair_id"])
        if (layout.pair(pair_id) / SUCCESS).is_file():
            continue
        if (time.time() - started) / 3600.0 > stop_after_hours:
            print(f"stopping: {stop_after_hours} h elapsed, remaining pairs are left for the next job", flush=True)
            break
        if not (layout.representation(pair_id) / "REPRESENTATION_SUCCESS").is_file():
            continue
        if not claim(layout.claims / "ot", pair_id):
            continue
        print(f"=== {pair_id} ({row['kind']}, {int(row['elements'])} elements) "
              f"{time.strftime('%Y-%m-%dT%H:%M:%S')}", flush=True)
        try:
            staging = run_pair(layout, row, device=device, block_rows=block_rows)
            published = publish(staging, layout.pair(pair_id))
            print(f"=== {pair_id} {'published' if published else 'already present, copy discarded'}", flush=True)
        except Exception:  # recorded for the audit; the worker moves on
            failures += 1
            message = traceback.format_exc()
            layout.pairs.mkdir(parents=True, exist_ok=True)
            (layout.pairs / f"{pair_id}.FAILED").write_text(message, encoding="utf-8")
            print(f"=== {pair_id} FAILED\n{message}", flush=True)
        finally:
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            gc.collect()
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--block-rows", type=int, default=None,
                        help="Rows per cost block; default fills 2**28 entries")
    parser.add_argument("--stop-claiming-after-hours", type=float, default=40.0)
    parser.add_argument("--pair-id", default=None, help="Run only this pair")
    parser.add_argument("--largest", action="store_true",
                        help="Run only the pair with the most cost entries (the scale probe)")
    args = parser.parse_args()
    failures = worker(Layout(args.root), device=args.device, block_rows=args.block_rows,
                      stop_after_hours=args.stop_claiming_after_hours, only=args.pair_id,
                      largest=args.largest)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
