"""Calibrate and run ConfidenceOT for one exact primary-to-metastasis pair."""

from __future__ import annotations

import argparse
import gc
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


def budget_tag(source_budget: float, target_budget: float) -> str:
    if np.isclose(source_budget, target_budget, rtol=0.0, atol=5e-12):
        return f"budget_{source_budget:.2f}"
    return f"budget_source_{source_budget:.2f}_target_{target_budget:.2f}"


def labels(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", data.n_obs)


def confidence_frame(method: str, side: str, data, sample: str, result) -> pd.DataFrame:
    value = result.source_confidence if side == "source" else result.target_confidence
    gate = result.source_gate if side == "source" else result.target_gate
    raw_gate = result.source_raw_gate if side == "source" else result.target_raw_gate
    frame = pd.DataFrame({
        "method": method, "side": side,
        "observation_id": data.obs_names.astype(str), "sample_id": sample,
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


def transition_table(method, coupling, source_labels, target_labels, source_gate, target_gate):
    source_categories = sorted(np.unique(source_labels))
    target_categories = sorted(np.unique(target_labels))
    source_hot = np.column_stack([(source_labels == value) & source_gate for value in source_categories]).astype(float)
    target_hot = np.column_stack([(target_labels == value) & target_gate for value in target_categories]).astype(float)
    grouped = source_hot.T @ coupling @ target_hot
    total = grouped.sum(axis=1, keepdims=True)
    conditional = np.divide(grouped, total, out=np.zeros_like(grouped), where=total > 0)
    return pd.DataFrame([
        {"method": method, "source_annotation": source, "target_annotation": target,
         "transported_mass": grouped[i, j], "source_conditional_probability": conditional[i, j]}
        for i, source in enumerate(source_categories) for j, target in enumerate(target_categories)
    ])


def reciprocal_dominant_pairs(
    method: str,
    coupling: np.ndarray,
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    source_gate: np.ndarray,
    target_gate: np.ndarray,
) -> pd.DataFrame:
    """Extract a non-forced one-to-one visualization layer from a soft coupling."""
    coupling = np.asarray(coupling, dtype=float)
    source_gate = np.asarray(source_gate, dtype=bool)
    target_gate = np.asarray(target_gate, dtype=bool)
    if coupling.shape != (len(source_ids), len(target_ids)):
        raise ValueError("Coupling and observation identifiers have inconsistent shapes")
    accepted = coupling.copy()
    accepted[~source_gate, :] = 0.0
    accepted[:, ~target_gate] = 0.0
    row_total = accepted.sum(axis=1)
    column_total = accepted.sum(axis=0)
    if not np.any(row_total > 0) or not np.any(column_total > 0):
        return pd.DataFrame(columns=[
            "method", "source_observation_id", "target_observation_id",
            "transport_mass", "source_conditional_weight",
            "target_conditional_weight", "reciprocal_dominant",
        ])
    row_partner = np.argmax(accepted, axis=1)
    column_partner = np.argmax(accepted, axis=0)
    rows = []
    for source_index, target_index in enumerate(row_partner):
        mass = float(accepted[source_index, target_index])
        if mass <= 0 or column_partner[target_index] != source_index:
            continue
        rows.append({
            "method": method,
            "source_observation_id": str(source_ids[source_index]),
            "target_observation_id": str(target_ids[target_index]),
            "transport_mass": mass,
            "source_conditional_weight": mass / float(row_total[source_index]),
            "target_conditional_weight": mass / float(column_total[target_index]),
            "reciprocal_dominant": True,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True, help="Zero-based row in the eligible manifest")
    parser.add_argument(
        "--rejection-budget", type=float, default=None,
        help="Legacy shared safety cap; side-specific options override it",
    )
    parser.add_argument("--source-rejection-budget", type=float, default=None)
    parser.add_argument("--target-rejection-budget", type=float, default=None)
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
    parser.add_argument(
        "--save-pairing-edges", action="store_true",
        help="Save reciprocal dominant cell pairs and joint PCA coordinates",
    )
    parser.add_argument("--skip-completed", action="store_true",
                        help="Return immediately when the pair output already contains SUCCESS")
    args = parser.parse_args()
    shared_budget = 0.95 if args.rejection_budget is None else args.rejection_budget
    source_budget = (
        shared_budget if args.source_rejection_budget is None
        else args.source_rejection_budget
    )
    target_budget = (
        shared_budget if args.target_rejection_budget is None
        else args.target_rejection_budget
    )
    for name, value in (
        ("source-rejection-budget", source_budget),
        ("target-rejection-budget", target_budget),
    ):
        if not 0 <= value < 1:
            raise ValueError(f"{name} must be in [0,1)")
    manifest = pd.read_csv(args.manifest_csv)
    row = manifest.iloc[args.index]
    if "eligible" in row and not bool(row["eligible"]):
        raise ValueError(f"Ineligible pair: {row.get('skip_reason', '')}")
    pair_id = str(row["pair_id"])
    output = (
        args.output_root / pair_id / f"scope_{args.analysis_scope}"
        / budget_tag(source_budget, target_budget)
    )
    if args.skip_completed and (output / "SUCCESS").is_file():
        print(f"SKIP completed index={args.index} pair_id={pair_id} output={output}")
        return
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
        source_rejection_budget=source_budget, target_rejection_budget=target_budget,
        tolerance=args.tolerance, grid_size=args.calibration_grid_size, device=args.device,
        workers=args.workers, fallback_to_cpu=False,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    source_labels, target_labels = labels(source), labels(target)
    cell_tables = []
    transition_tables = []
    pairing_tables = []
    metric_rows = []
    for method, variant in (("M4-E", "exact"), ("M4-R", "reversible")):
        model = ConfidenceOT(
            backbone="uot", variant=variant, rejection_cost=calibration.rejection_cost,
            epsilon=args.epsilon, lambda_a=args.lambda_a, lambda_b=args.lambda_b,
            source_rejection_budget=source_budget,
            target_rejection_budget=target_budget,
            tolerance=args.tolerance, device=args.device,
        )
        fit_started = time.perf_counter()
        result = model.fit(cost)
        fit_seconds = time.perf_counter() - fit_started
        cell_tables.extend([
            confidence_frame(method, "source", source, str(row["source_sample"]), result),
            confidence_frame(method, "target", target, str(row["target_sample"]), result),
        ])
        transition_tables.append(transition_table(
            method, result.coupling, source_labels, target_labels,
            result.source_gate, result.target_gate,
        ))
        if args.save_pairing_edges:
            pairing_tables.append(reciprocal_dominant_pairs(
                method,
                result.coupling,
                source.obs_names.astype(str).to_numpy(),
                target.obs_names.astype(str).to_numpy(),
                result.source_gate,
                result.target_gate,
            ))
        metric_rows.append({
            "pair_id": pair_id, "dataset_id": row["dataset_id"],
            "patient_id": row["patient_id"], "source_sample": row["source_sample"],
            "target_sample": row["target_sample"], "method": method, "variant": variant,
            "source_exact_n": source_exact_n, "target_exact_n": target_exact_n,
            "analysis_scope": args.analysis_scope,
            "included_annotations": "|".join(args.include_annotation),
            "source_scope_n": source_scope_n, "target_scope_n": target_scope_n,
            "source_analyzed_n": source.n_obs, "target_analyzed_n": target.n_obs,
            "rejection_budget_cap": (
                source_budget if np.isclose(source_budget, target_budget) else np.nan
            ),
            "source_rejection_budget_cap": source_budget,
            "target_rejection_budget_cap": target_budget,
            "rejection_cost": calibration.rejection_cost,
            "calibration_valid_for_m4r": calibration.calibration_valid,
            "source_raw_rejection_rate": float(np.mean(~result.source_raw_gate)),
            "target_raw_rejection_rate": float(np.mean(~result.target_raw_gate)),
            "source_final_rejection_rate": float(np.mean(~result.source_gate)),
            "target_final_rejection_rate": float(np.mean(~result.target_gate)),
            "source_budget_override_rate": float(np.mean(result.source_confidence.budget_overridden)),
            "target_budget_override_rate": float(np.mean(result.target_confidence.budget_overridden)),
            "transported_mass": float(result.coupling.sum()),
            "calibration_seconds_shared": calibration_seconds,
            "fit_seconds": fit_seconds,
            "inner_converged": result.inner_converged,
            "outer_converged": result.outer_converged,
            "cycle_detected": result.cycle_detected,
            "cycle_length": result.cycle_length,
            "outer_iterations": result.n_outer_iterations,
            "total_inner_iterations": result.total_inner_iterations,
            "device": result.device, "backend": result.backend,
        })
        if args.save_coupling:
            np.savez_compressed(output / f"coupling_{method.lower().replace('-', '')}.npz",
                                coupling=result.coupling)
        del result, model
        gc.collect()

    cells = pd.concat(cell_tables, ignore_index=True)
    cells.to_csv(output / "cell_confidence.csv", index=False)
    populations = cells.groupby(["method", "side", "annotation"], dropna=False).agg(
        n=("rejected", "size"), raw_rejection_rate=("raw_rejected", "mean"),
        final_rejection_rate=("rejected", "mean"), budget_override_rate=("budget_overridden", "mean"),
        mean_confidence_score=("normalized_rejection_score", "mean")
    ).reset_index()
    populations.to_csv(output / "population_rejection.csv", index=False)
    pd.concat(transition_tables, ignore_index=True).to_csv(
        output / "population_transitions.csv", index=False)
    if args.save_pairing_edges:
        pd.concat(pairing_tables, ignore_index=True).to_csv(
            output / "reciprocal_cell_pairs.csv.gz", index=False,
            compression="gzip",
        )
        pca_columns = [f"PC{index + 1}" for index in range(source_pca.shape[1])]
        coordinates = pd.concat([
            pd.DataFrame(source_pca, columns=pca_columns).assign(
                side="primary", observation_id=source.obs_names.astype(str).to_numpy(),
                sample_id=str(row["source_sample"]), annotation=source_labels,
            ),
            pd.DataFrame(target_pca, columns=pca_columns).assign(
                side="metastasis", observation_id=target.obs_names.astype(str).to_numpy(),
                sample_id=str(row["target_sample"]), annotation=target_labels,
            ),
        ], ignore_index=True)
        coordinates.to_csv(
            output / "joint_pca_coordinates.csv.gz", index=False, compression="gzip"
        )
    pipeline_seconds = time.perf_counter() - started
    for metrics in metric_rows:
        metrics["pipeline_seconds_shared"] = pipeline_seconds
    pd.DataFrame(metric_rows).to_csv(output / "pair_metrics.csv", index=False)
    (output / "calibration.json").write_text(json.dumps(json_ready(calibration), indent=2), encoding="utf-8")
    (output / "run.json").write_text(json.dumps({
        "pair_id": pair_id, "dataset_id": row["dataset_id"],
        "patient_id": row["patient_id"], "analysis_scope": args.analysis_scope,
        "source_rejection_budget_cap": source_budget,
        "target_rejection_budget_cap": target_budget,
        "rejection_cost": calibration.rejection_cost,
        "calibration_valid_for_m4r": calibration.calibration_valid,
        "pipeline_seconds": pipeline_seconds, "methods": metric_rows,
        "hvg_n": len(hvg), "hvg": hvg, "preprocessing": preprocessing,
    }, indent=2), encoding="utf-8")
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    print(pd.DataFrame(metric_rows).to_string(index=False))


if __name__ == "__main__":
    main()
