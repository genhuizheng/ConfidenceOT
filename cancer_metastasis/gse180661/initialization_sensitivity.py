"""Audit M4-E sensitivity to feasible source/target gate initialization."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from confidenceot import ConfidenceOT
from cancer_metastasis.common import cell_qc_table, load_exact_side, prepare_joint_representation


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def labels(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    raise KeyError("H5AD has no cell-type annotation column")


def squared_euclidean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    value = (
        np.sum(left * left, axis=1)[:, None]
        + np.sum(right * right, axis=1)[None, :]
        - 2 * left @ right.T
    )
    return np.maximum(value, 0.0)


def completed_pair_directory(root: Path, pair_id: str) -> Path:
    candidates = sorted(
        path.parent for path in (root / pair_id / "scope_malignant").glob("*/SUCCESS")
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one completed baseline result for {pair_id}; found {len(candidates)}"
        )
    return candidates[0]


def feasible_random_gate(n: int, budget: float, fraction: float, rng) -> np.ndarray:
    minimum = max(1, min(n, int(math.ceil((1.0 - budget) * n - 1e-12))))
    accepted = max(minimum, min(n, int(round(fraction * n))))
    gate = np.zeros(n, dtype=bool)
    gate[rng.choice(n, size=accepted, replace=False)] = True
    return gate


def projected_zero_gate(n: int, budget: float) -> np.ndarray:
    """Closest deterministic feasible gate to an all-zero requested start."""
    minimum = max(1, min(n, int(math.ceil((1.0 - budget) * n - 1e-12))))
    gate = np.zeros(n, dtype=bool)
    gate[:minimum] = True
    return gate


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("baseline_ot_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--random-starts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--max-observed-cells-per-side", type=int, default=10000)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    row = manifest.iloc[args.index]
    pair_id = str(row["pair_id"])
    output = args.output_root / "pairs" / f"{args.index:03d}_{safe_name(pair_id)}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "SUCCESS").is_file():
        print(f"SKIP completed pair={pair_id}")
        return

    baseline_dir = completed_pair_directory(args.baseline_ot_root, pair_id)
    run = json.loads((baseline_dir / "run.json").read_text(encoding="utf-8"))
    baseline_cells = pd.read_csv(baseline_dir / "cell_confidence.csv")
    baseline_cells = baseline_cells[baseline_cells["method"].eq("M4-E")].copy()

    source = load_exact_side(paths_for(row, "source"), str(row["source_sample"]))
    target = load_exact_side(paths_for(row, "target"), str(row["target_sample"]))
    source = source[labels(source) == "Ovarian.cancer.cell"].copy()
    target = target[labels(target) == "Ovarian.cancer.cell"].copy()

    qc = run.get("cell_qc", {})
    if bool(qc.get("applied", False)):
        source_qc = cell_qc_table(
            source,
            minimum_total_counts=int(qc["minimum_total_counts"]),
            minimum_detected_genes=int(qc["minimum_detected_genes"]),
            maximum_mitochondrial_percent=float(qc["maximum_mitochondrial_percent"]),
        )
        target_qc = cell_qc_table(
            target,
            minimum_total_counts=int(qc["minimum_total_counts"]),
            minimum_detected_genes=int(qc["minimum_detected_genes"]),
            maximum_mitochondrial_percent=float(qc["maximum_mitochondrial_percent"]),
        )
        source = source[source_qc["qc_pass"].to_numpy()].copy()
        target = target[target_qc["qc_pass"].to_numpy()].copy()

    sample_rng = np.random.default_rng(args.seed + args.index * 65537)
    maximum = args.max_observed_cells_per_side
    if maximum > 0 and source.n_obs > maximum:
        source = source[np.sort(sample_rng.choice(source.n_obs, maximum, replace=False))].copy()
    if maximum > 0 and target.n_obs > maximum:
        target = target[np.sort(sample_rng.choice(target.n_obs, maximum, replace=False))].copy()

    source_pca, target_pca, hvg, preprocessing = prepare_joint_representation(
        source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs,
        seed=args.seed + args.index,
    )
    rng = np.random.default_rng(args.seed + args.index * 104729)
    sampled_n = min(1_000_000, max(len(source_pca) * len(target_pca), 1))
    sampled = np.sum(
        (
            source_pca[rng.integers(len(source_pca), size=sampled_n)]
            - target_pca[rng.integers(len(target_pca), size=sampled_n)]
        ) ** 2,
        axis=1,
    )
    scale = float(np.median(sampled[sampled > 0]))
    cost = squared_euclidean(source_pca, target_pca) / scale

    source_budget = float(run["source_rejection_budget_cap"])
    target_budget = float(run["target_rejection_budget_cap"])
    rejection_cost = float(run["rejection_cost"])
    model = ConfidenceOT(
        backbone="uot",
        variant="exact",
        rejection_cost=rejection_cost,
        epsilon=0.1,
        lambda_a=1.0,
        lambda_b=1.0,
        source_rejection_budget=source_budget,
        target_rejection_budget=target_budget,
        tolerance=1e-4,
        device=args.device,
        warn_on_terminal=False,
    )

    starts: list[tuple[str, int, np.ndarray | None, np.ndarray | None]] = [
        ("all_one", 0, None, None),
        (
            "all_zero_projected_to_budget_floor",
            0,
            projected_zero_gate(source.n_obs, source_budget),
            projected_zero_gate(target.n_obs, target_budget),
        ),
    ]
    for replicate in range(args.random_starts):
        start_rng = np.random.default_rng(
            args.seed + args.index * 1_000_003 + replicate * 101 + 17
        )
        starts.append((
            "random_50_percent",
            replicate,
            feasible_random_gate(source.n_obs, source_budget, 0.5, start_rng),
            feasible_random_gate(target.n_obs, target_budget, 0.5, start_rng),
        ))

    fitted = []
    rows = []
    default_result = None
    for strategy, replicate, source_initial, target_initial in starts:
        started = time.perf_counter()
        result = model.fit(
            cost,
            initial_source_gate=source_initial,
            initial_target_gate=target_initial,
        )
        elapsed = time.perf_counter() - started
        if default_result is None:
            default_result = result
        source_score = result.source_confidence.normalized_rejection_score()
        target_score = result.target_confidence.normalized_rejection_score()
        rows.append({
            "pair_id": pair_id,
            "patient_id": str(row["patient_id"]),
            "source_sample": str(row["source_sample"]),
            "target_sample": str(row["target_sample"]),
            "strategy": strategy,
            "replicate": replicate,
            "source_initial_retained_fraction": 1.0 if source_initial is None else float(source_initial.mean()),
            "target_initial_retained_fraction": 1.0 if target_initial is None else float(target_initial.mean()),
            "source_final_retained_fraction": float(result.source_gate.mean()),
            "target_final_retained_fraction": float(result.target_gate.mean()),
            "source_agreement_with_default": float(np.mean(result.source_gate == default_result.source_gate)),
            "target_agreement_with_default": float(np.mean(result.target_gate == default_result.target_gate)),
            "source_retained_jaccard_with_default": jaccard(result.source_gate, default_result.source_gate),
            "target_retained_jaccard_with_default": jaccard(result.target_gate, default_result.target_gate),
            "source_score_correlation_with_default": correlation(source_score, default_result.source_confidence.normalized_rejection_score()),
            "target_score_correlation_with_default": correlation(target_score, default_result.target_confidence.normalized_rejection_score()),
            "objective": float(result.objective),
            "objective_delta_from_default": float(result.objective - default_result.objective),
            "inner_converged": bool(result.inner_converged),
            "outer_converged": bool(result.outer_converged),
            "cycle_detected": bool(result.cycle_detected),
            "outer_iterations": int(result.n_outer_iterations),
            "fit_seconds": elapsed,
        })
        fitted.append((strategy, replicate, result.source_gate.copy(), result.target_gate.copy()))
        del result
        gc.collect()

    assert default_result is not None
    stored_source = baseline_cells[baseline_cells["side"].eq("source")].set_index("observation_id")
    stored_target = baseline_cells[baseline_cells["side"].eq("target")].set_index("observation_id")
    source_ids = source.obs_names.astype(str).to_numpy()
    target_ids = target.obs_names.astype(str).to_numpy()
    if set(source_ids) != set(stored_source.index) or set(target_ids) != set(stored_target.index):
        raise RuntimeError("Reconstructed analyzed cell identities differ from the baseline run")
    source_stored_gate = stored_source.loc[source_ids, "retained"].astype(bool).to_numpy()
    target_stored_gate = stored_target.loc[target_ids, "retained"].astype(bool).to_numpy()

    nondefault = [item for item in fitted if item[0] != "all_one"]
    source_matrix = np.vstack([item[2] for item in nondefault])
    target_matrix = np.vstack([item[3] for item in nondefault])
    source_gate_runs = []
    target_gate_runs = []
    for strategy, replicate, source_gate, target_gate in fitted:
        source_gate_runs.append(pd.DataFrame({
            "pair_id": pair_id,
            "observation_id": source_ids,
            "strategy": strategy,
            "replicate": replicate,
            "retained": source_gate,
        }))
        target_gate_runs.append(pd.DataFrame({
            "pair_id": pair_id,
            "observation_id": target_ids,
            "strategy": strategy,
            "replicate": replicate,
            "retained": target_gate,
        }))
    pd.concat(source_gate_runs, ignore_index=True).to_csv(
        output / "source_initialization_gates.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.concat(target_gate_runs, ignore_index=True).to_csv(
        output / "target_initialization_gates.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame({
        "pair_id": pair_id,
        "observation_id": source_ids,
        "default_retained": default_result.source_gate,
        "stored_baseline_retained": source_stored_gate,
        "retained_fraction_across_random_starts": source_matrix.mean(axis=0),
        "consensus_stable": np.all(source_matrix == source_matrix[0], axis=0),
    }).to_csv(output / "source_cell_initialization_stability.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "pair_id": pair_id,
        "observation_id": target_ids,
        "default_retained": default_result.target_gate,
        "stored_baseline_retained": target_stored_gate,
        "retained_fraction_across_random_starts": target_matrix.mean(axis=0),
        "consensus_stable": np.all(target_matrix == target_matrix[0], axis=0),
    }).to_csv(output / "target_cell_initialization_stability.csv.gz", index=False, compression="gzip")

    run_table = pd.DataFrame(rows)
    run_table["source_default_matches_stored_baseline"] = float(
        np.mean(default_result.source_gate == source_stored_gate)
    )
    run_table["target_default_matches_stored_baseline"] = float(
        np.mean(default_result.target_gate == target_stored_gate)
    )
    run_table.to_csv(output / "initialization_runs.csv", index=False)
    (output / "run.json").write_text(json.dumps({
        "pair_id": pair_id,
        "random_starts_per_strategy": args.random_starts,
        "requested_strategies": ["all_zero", "all_one", "random"],
        "executed_strategies": [
            "all_one",
            "all_zero_projected_to_budget_floor",
            "random_50_percent",
        ],
        "all_zero_status": (
            "infeasible: an empty gate violates the rejection-budget coverage floor "
            "and provides no active support for OT; the closest feasible projected "
            "zero gate was run instead"
        ),
        "source_rejection_budget": source_budget,
        "target_rejection_budget": target_budget,
        "rejection_cost": rejection_cost,
        "hvg_n": len(hvg),
        "preprocessing": preprocessing,
    }, indent=2), encoding="utf-8")
    (output / "SUCCESS").write_text("complete\n", encoding="utf-8")
    print(run_table.to_string(index=False))


if __name__ == "__main__":
    main()
