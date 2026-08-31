"""Calibrate and run ConfidenceOT for one exact primary-to-metastasis pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from confidenceot import ConfidenceOT, calibrate_confidence_cost, rotation_null_costs
from common import json_ready, load_exact_side, prepare_joint_representation


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = np.sum(left * left, axis=1)[:, None] + np.sum(right * right, axis=1)[None, :] - 2 * left @ right.T
    return np.maximum(value, 0.0)


def labels(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", data.n_obs)


def confidence_frame(side: str, data, sample: str, result) -> pd.DataFrame:
    value = result.source_confidence if side == "source" else result.target_confidence
    gate = result.source_gate if side == "source" else result.target_gate
    raw_gate = result.source_raw_gate if side == "source" else result.target_raw_gate
    frame = pd.DataFrame({
        "side": side, "observation_id": data.obs_names.astype(str), "sample_id": sample,
        "annotation": labels(data), "retained": gate, "rejected": ~gate,
        "raw_retained": raw_gate, "raw_rejected": ~raw_gate,
        "decision_cost": value.decision_cost, "rejection_cost": value.rejection_cost,
        "signed_rejection_margin": value.signed_rejection_margin,
        "relative_rejection_margin": value.relative_rejection_margin,
        "normalized_rejection_score": value.normalized_rejection_score(),
        "budget_overridden": value.budget_overridden,
        "cost_kind": value.cost_kind,
    })
    for column in ("malignant", "sample_type", "site", "tissue", "lesion"):
        if column in data.obs:
            frame[column] = data.obs[column].astype(str).to_numpy()
    return frame


def transition_table(coupling, source_labels, target_labels, source_gate, target_gate):
    source_categories = sorted(np.unique(source_labels))
    target_categories = sorted(np.unique(target_labels))
    source_hot = np.column_stack([(source_labels == value) & source_gate for value in source_categories]).astype(float)
    target_hot = np.column_stack([(target_labels == value) & target_gate for value in target_categories]).astype(float)
    grouped = source_hot.T @ coupling @ target_hot
    total = grouped.sum(axis=1, keepdims=True)
    conditional = np.divide(grouped, total, out=np.zeros_like(grouped), where=total > 0)
    return pd.DataFrame([
        {"source_annotation": source, "target_annotation": target,
         "transported_mass": grouped[i, j], "source_conditional_probability": conditional[i, j]}
        for i, source in enumerate(source_categories) for j, target in enumerate(target_categories)
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True, help="Zero-based row in the eligible manifest")
    parser.add_argument("--rejection-budget", type=float, default=0.95, help="Safety cap, not a biological target")
    parser.add_argument("--analysis-scope", choices=("all", "malignant"), default="all")
    parser.add_argument("--include-annotation", action="append", default=[],
                        help="Cell-type value retained in malignant scope; may be repeated")
    parser.add_argument("--minimum-scope-cells", type=int, default=20)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--calibration-max-cells", type=int, default=2000)
    parser.add_argument("--max-observed-cells-per-side", type=int, default=10000,
                        help="Deterministic label-blind cap per exact sample; 0 uses every cell")
    parser.add_argument("--null-calibration-replicates", type=int, default=5)
    parser.add_argument("--null-validation-replicates", type=int, default=5)
    parser.add_argument("--calibration-grid-size", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--lambda-a", type=float, default=1.0)
    parser.add_argument("--lambda-b", type=float, default=1.0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    parser.add_argument("--save-coupling", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.rejection_budget < 1:
        raise ValueError("rejection-budget must be in [0,1)")
    manifest = pd.read_csv(args.manifest_csv)
    row = manifest.iloc[args.index]
    if "eligible" in row and not bool(row["eligible"]):
        raise ValueError(f"Ineligible pair: {row.get('skip_reason', '')}")
    pair_id = str(row["pair_id"])
    output = args.output_root / pair_id / f"scope_{args.analysis_scope}" / f"budget_{args.rejection_budget:.2f}"
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    def paths_for(side: str) -> list[str]:
        column = f"{side}_h5ads_json"
        if column in row and pd.notna(row[column]):
            return [str(value) for value in json.loads(str(row[column]))]
        return [str(row[f"{side}_h5ad"])]

    source = load_exact_side(paths_for("source"), str(row["source_sample"]))
    target = load_exact_side(paths_for("target"), str(row["target_sample"]))
    source_exact_n, target_exact_n = source.n_obs, target.n_obs
    if args.analysis_scope == "malignant":
        if not args.include_annotation:
            raise ValueError("malignant scope requires at least one --include-annotation value")
        source_label = labels(source)
        target_label = labels(target)
        source = source[np.isin(source_label, args.include_annotation)].copy()
        target = target[np.isin(target_label, args.include_annotation)].copy()
    source_scope_n, target_scope_n = source.n_obs, target.n_obs
    if source_scope_n < args.minimum_scope_cells or target_scope_n < args.minimum_scope_cells:
        raise ValueError(
            f"scope={args.analysis_scope} is not evaluable: source={source_scope_n}, "
            f"target={target_scope_n}, minimum={args.minimum_scope_cells}"
        )
    sample_rng = np.random.default_rng(args.seed + args.index * 65537)
    if args.max_observed_cells_per_side > 0 and source.n_obs > args.max_observed_cells_per_side:
        keep = np.sort(sample_rng.choice(source.n_obs, args.max_observed_cells_per_side, replace=False))
        source = source[keep].copy()
    if args.max_observed_cells_per_side > 0 and target.n_obs > args.max_observed_cells_per_side:
        keep = np.sort(sample_rng.choice(target.n_obs, args.max_observed_cells_per_side, replace=False))
        target = target[keep].copy()
    source_pca, target_pca, hvg, preprocessing = prepare_joint_representation(
        source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs, seed=args.seed + args.index
    )
    rng = np.random.default_rng(args.seed + args.index * 104729)
    pairs = min(1_000_000, max(len(source_pca) * len(target_pca), 1))
    sampled = np.sum((source_pca[rng.integers(len(source_pca), size=pairs)] - target_pca[rng.integers(len(target_pca), size=pairs)]) ** 2, axis=1)
    scale = float(np.median(sampled[sampled > 0]))
    cost = squared_euclidean(source_pca, target_pca) / scale
    source_index = np.sort(rng.choice(len(source_pca), min(args.calibration_max_cells, len(source_pca)), replace=False))
    target_index = np.sort(rng.choice(len(target_pca), min(args.calibration_max_cells, len(target_pca)), replace=False))
    total_nulls = args.null_calibration_replicates + args.null_validation_replicates
    source_nulls, target_nulls = rotation_null_costs(
        source_pca[source_index], target_pca[target_index], observed_scale=scale,
        seed=args.seed + args.index, n_replicates=total_nulls,
    )
    split = args.null_calibration_replicates
    calibration_started = time.perf_counter()
    calibration = calibrate_confidence_cost(
        source_nulls[:split] + target_nulls[:split], source_nulls[split:] + target_nulls[split:],
        backbone="uot", epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        source_rejection_budget=args.rejection_budget, target_rejection_budget=args.rejection_budget,
        tolerance=args.tolerance, grid_size=args.calibration_grid_size, device=args.device,
        workers=args.workers, fallback_to_cpu=False,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    model = ConfidenceOT(
        backbone="uot", variant="reversible", rejection_cost=calibration.rejection_cost,
        epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
        source_rejection_budget=args.rejection_budget, target_rejection_budget=args.rejection_budget,
        tolerance=args.tolerance, device=args.device,
    )
    fit_started = time.perf_counter()
    result = model.fit(cost)
    fit_seconds = time.perf_counter() - fit_started
    cells = pd.concat([
        confidence_frame("source", source, str(row["source_sample"]), result),
        confidence_frame("target", target, str(row["target_sample"]), result),
    ], ignore_index=True)
    cells.to_csv(output / "cell_confidence.csv", index=False)
    populations = cells.groupby(["side", "annotation"], dropna=False).agg(
        n=("rejected", "size"), raw_rejection_rate=("raw_rejected", "mean"),
        final_rejection_rate=("rejected", "mean"), budget_override_rate=("budget_overridden", "mean"),
        mean_confidence_score=("normalized_rejection_score", "mean")
    ).reset_index()
    populations.to_csv(output / "population_rejection.csv", index=False)
    transition_table(result.coupling, labels(source), labels(target), result.source_gate, result.target_gate).to_csv(
        output / "population_transitions.csv", index=False
    )
    metrics = {
        "pair_id": pair_id, "dataset_id": row["dataset_id"], "patient_id": row["patient_id"],
        "source_sample": row["source_sample"], "target_sample": row["target_sample"],
        "source_exact_n": source_exact_n, "target_exact_n": target_exact_n,
        "analysis_scope": args.analysis_scope,
        "included_annotations": "|".join(args.include_annotation),
        "source_scope_n": source_scope_n, "target_scope_n": target_scope_n,
        "source_analyzed_n": source.n_obs, "target_analyzed_n": target.n_obs,
        "rejection_budget_cap": args.rejection_budget, "rejection_cost": calibration.rejection_cost,
        "calibration_valid": calibration.calibration_valid,
        "source_raw_rejection_rate": float(np.mean(~result.source_raw_gate)),
        "target_raw_rejection_rate": float(np.mean(~result.target_raw_gate)),
        "source_final_rejection_rate": float(np.mean(~result.source_gate)),
        "target_final_rejection_rate": float(np.mean(~result.target_gate)),
        "source_budget_override_rate": float(np.mean(result.source_confidence.budget_overridden)),
        "target_budget_override_rate": float(np.mean(result.target_confidence.budget_overridden)),
        "transported_mass": float(result.coupling.sum()), "calibration_seconds": calibration_seconds,
        "fit_seconds": fit_seconds, "pipeline_seconds": time.perf_counter() - started,
        "inner_converged": result.inner_converged, "outer_converged": result.outer_converged,
        "cycle_detected": result.cycle_detected, "device": result.device, "backend": result.backend,
    }
    pd.DataFrame([metrics]).to_csv(output / "pair_metrics.csv", index=False)
    (output / "calibration.json").write_text(json.dumps(json_ready(calibration), indent=2), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({
        **metrics, "hvg_n": len(hvg), "hvg": hvg, "preprocessing": preprocessing,
    }, indent=2), encoding="utf-8")
    if args.save_coupling:
        np.savez_compressed(output / "coupling.npz", coupling=result.coupling)
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
