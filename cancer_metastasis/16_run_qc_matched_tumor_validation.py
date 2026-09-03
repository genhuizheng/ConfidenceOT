"""Validate tumor identity and repeat rejected-versus-retained DEG after QC matching."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors

from common import expression_matrix, load_exact_side


QC_COLUMNS = [
    "total_counts", "n_genes_by_counts",
    "pct_counts_mitochondrial", "pct_counts_ribosomal",
]
MARKER_SETS = {
    "epithelial_ovarian": ["EPCAM", "KRT7", "KRT8", "KRT18", "KRT19", "MUC1", "MSLN", "PAX8", "WFDC2"],
    "immune": ["PTPRC", "LST1", "TYROBP", "FCER1G", "CD3D", "CD3E", "CD74", "HLA-DRA"],
}


def load_numbered_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


DEG_PREP = load_numbered_module("11_run_robust_target_deg.py", "deg_prep")
PYDESEQ2 = load_numbered_module("13_run_paired_pydeseq2.py", "paired_pydeseq2")


def standardized_mean_difference(values: np.ndarray, labels: np.ndarray) -> float:
    left = values[labels == "robust_rejected"]
    right = values[labels == "robust_retained"]
    variance = (np.var(left, ddof=1) + np.var(right, ddof=1)) / 2
    return float((np.mean(left) - np.mean(right)) / np.sqrt(variance)) if variance > 0 else 0.0


def propensity_match(qc: pd.DataFrame, seed: int, caliper_scale: float) -> tuple[np.ndarray, np.ndarray, dict]:
    labels = qc["confidence_status"].astype(str).to_numpy()
    features = qc[QC_COLUMNS].to_numpy(dtype=float)
    center = np.nanmedian(features, axis=0)
    scale = np.nanpercentile(features, 75, axis=0) - np.nanpercentile(features, 25, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    features = np.nan_to_num((features - center) / scale)
    binary = (labels == "robust_rejected").astype(int)
    model = LogisticRegression(max_iter=2_000, class_weight="balanced", random_state=seed)
    logits = model.fit(features, binary).decision_function(features)
    rejected = np.flatnonzero(binary == 1)
    retained = np.flatnonzero(binary == 0)
    small, large = (rejected, retained) if len(rejected) <= len(retained) else (retained, rejected)
    neighbors = NearestNeighbors(n_neighbors=min(100, len(large))).fit(logits[large, None])
    distances, candidates = neighbors.kneighbors(logits[small, None])
    caliper = max(caliper_scale * float(np.std(logits, ddof=1)), 1e-8)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(small))
    used: set[int] = set()
    pairs = []
    for position in order:
        for distance, candidate_position in zip(distances[position], candidates[position]):
            candidate = int(large[candidate_position])
            if candidate not in used and distance <= caliper:
                used.add(candidate)
                pairs.append((int(small[position]), candidate))
                break
    selected = np.asarray([value for pair in pairs for value in pair], dtype=int)
    selected_rejected = selected[labels[selected] == "robust_rejected"]
    selected_retained = selected[labels[selected] == "robust_retained"]
    summary = {"seed": seed, "caliper": caliper, "matched_pair_n": len(pairs)}
    return selected_rejected, selected_retained, summary


def marker_summaries(matrix: sparse.csr_matrix, genes: np.ndarray, qc: pd.DataFrame, group: dict) -> list[dict]:
    library = np.asarray(matrix.sum(axis=1)).ravel()
    normalized = matrix.multiply((10_000 / np.maximum(library, 1))[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    lookup = {gene.upper(): index for index, gene in enumerate(genes)}
    rows = []
    for marker_set, markers in MARKER_SETS.items():
        indices = [lookup[gene] for gene in markers if gene in lookup]
        if not indices:
            continue
        scores = np.asarray(normalized[:, indices].mean(axis=1)).ravel()
        for status in ("robust_rejected", "robust_retained"):
            mask = qc["confidence_status"].astype(str).eq(status).to_numpy()
            rows.append({**group, "marker_set": marker_set, "feature": marker_set,
                         "feature_type": "module_mean", "confidence_status": status,
                         "cell_n": int(mask.sum()), "mean_score": float(np.mean(scores[mask])),
                         "median_score": float(np.median(scores[mask])),
                         "detection_fraction": np.nan,
                         "available_markers": "|".join([genes[index] for index in indices])})
            for index in indices:
                values = normalized[:, index].toarray().ravel()
                rows.append({**group, "marker_set": marker_set, "feature": genes[index],
                             "feature_type": "single_gene", "confidence_status": status,
                             "cell_n": int(mask.sum()), "mean_score": float(np.mean(values[mask])),
                             "median_score": float(np.median(values[mask])),
                             "detection_fraction": float(np.mean(values[mask] > 0)),
                             "available_markers": genes[index]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("group_root", type=Path)
    parser.add_argument("original_pydeseq2_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--caliper-scale", type=float, default=0.2)
    parser.add_argument("--minimum-matched-pairs-per-group", type=int, default=10)
    parser.add_argument("--minimum-cells-per-patient-status", type=int, default=20)
    parser.add_argument("--n-cpus", type=int, default=16)
    parser.add_argument("--maximum-groups", type=int)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest_csv)
    count_tables, metadata_rows, balance_rows, marker_rows, ot_tables = [], [], [], [], []

    ready = sorted(args.group_root.glob("groups/*/PSEUDOBULK_READY"))
    if not ready:
        raise RuntimeError(f"No groups found under {args.group_root}")
    if args.maximum_groups is not None:
        ready = ready[:args.maximum_groups]
    audit_root = args.output_root / "matching_audit"
    audit_root.mkdir(exist_ok=True)
    for group_index, marker in enumerate(ready):
        directory = marker.parent
        diagnostics = json.loads((directory / "diagnostics.json").read_text())
        classification = pd.read_csv(directory / "target_cell_robust_classification.csv.gz")
        qc = pd.read_csv(directory / "cell_qc_metrics.csv.gz")
        qc = qc[qc["confidence_status"].isin(["robust_rejected", "robust_retained"])].copy()
        group = {key: diagnostics[key] for key in ("group_id", "patient_id", "target_sample")}
        row = manifest[manifest["pair_id"].astype(str).eq(str(diagnostics["baseline_pair_id"]))]
        if len(row) != 1:
            raise RuntimeError(f"Manifest did not resolve {diagnostics['baseline_pair_id']}")
        paths = json.loads(str(row.iloc[0]["target_h5ads_json"]))
        data = load_exact_side(paths, str(group["target_sample"]))
        annotations = None
        for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
            if column in data.obs:
                annotations = data.obs[column].astype(str).to_numpy()
                break
        if annotations is None:
            raise KeyError("Target H5AD has no cell-type annotation")
        data = data[annotations == "Ovarian.cancer.cell"].copy()
        lookup = {str(value): index for index, value in enumerate(data.obs_names)}
        ordered_ids = qc["observation_id"].astype(str).tolist()
        missing = [value for value in ordered_ids if value not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} QC cells were absent from target tumor cells")
        raw = sparse.csr_matrix(expression_matrix(data)[[lookup[value] for value in ordered_ids]], dtype=float)
        raw, genes, used_for_ot = DEG_PREP.aggregate_gene_symbols(
            raw, data, set()
        )
        if raw.data.size and (raw.data.min() < 0 or not np.allclose(raw.data, np.round(raw.data))):
            raise ValueError("QC-matched pseudobulk requires raw integer counts")
        old_genes = pd.read_csv(directory / "pseudobulk_gene_metadata.csv.gz")
        old_used = old_genes.set_index("gene")["used_for_ot"].astype(bool)
        used_for_ot = pd.Index(genes).map(old_used).fillna(False).to_numpy(bool)
        ot_tables.append(pd.DataFrame({"gene": genes, "used_for_ot": used_for_ot}))
        marker_rows.extend(marker_summaries(raw, genes, qc, group))
        selected_audit = []
        labels = qc["confidence_status"].astype(str).to_numpy()
        for column in QC_COLUMNS:
            balance_rows.append({**group, "replicate": -1, "metric": column,
                                 "standardized_mean_difference": standardized_mean_difference(qc[column].to_numpy(float), labels)})
        for replicate in range(args.replicates):
            rejected, retained, match_summary = propensity_match(
                qc, args.seed + replicate + group_index * 10_000, args.caliper_scale
            )
            if len(rejected) < args.minimum_matched_pairs_per_group:
                continue
            selected = np.concatenate([rejected, retained])
            selected_labels = labels[selected]
            for column in QC_COLUMNS:
                balance_rows.append({**group, "replicate": replicate, "metric": column,
                                     "standardized_mean_difference": standardized_mean_difference(qc.iloc[selected][column].to_numpy(float), selected_labels)})
            rows = np.vstack([
                np.asarray(raw[rejected].sum(axis=0)).ravel(),
                np.asarray(raw[retained].sum(axis=0)).ravel(),
            ]).astype(np.int64)
            sample_ids = [f"{group['group_id']}__rep{replicate}__robust_rejected",
                          f"{group['group_id']}__rep{replicate}__robust_retained"]
            count_tables.append(pd.DataFrame(rows, index=sample_ids, columns=genes))
            for sample_id, status in zip(sample_ids, ("robust_rejected", "robust_retained")):
                metadata_rows.append({"sample_id": sample_id, **group, "replicate": replicate,
                                      "confidence_status": status, "cell_n": len(rejected), **match_summary})
            for index in selected:
                selected_audit.append({**group, "replicate": replicate,
                                       "observation_id": ordered_ids[index],
                                       "confidence_status": labels[index]})
        pd.DataFrame(selected_audit).to_csv(
            audit_root / f"{directory.name}_matched_cells.csv.gz", index=False, compression="gzip"
        )
        print(f"Prepared {group['group_id']}", flush=True)

    if not count_tables:
        raise RuntimeError("No QC-matched groups passed the minimum matched-pair requirement")
    all_genes = sorted(set().union(*(set(table.columns) for table in count_tables)))
    counts = pd.concat([table.reindex(columns=all_genes, fill_value=0) for table in count_tables])
    metadata = pd.DataFrame(metadata_rows).set_index("sample_id").loc[counts.index]
    ot = pd.concat(ot_tables).groupby("gene", as_index=False).agg(used_for_ot_anywhere=("used_for_ot", "max"))
    balance = pd.DataFrame(balance_rows)
    balance["phase"] = np.where(balance["replicate"].lt(0), "before", "after")
    balance["absolute_standardized_mean_difference"] = balance["standardized_mean_difference"].abs()
    balance.to_csv(args.output_root / "qc_balance.csv", index=False)
    balance.groupby(["phase", "metric"], as_index=False).agg(
        comparison_n=("absolute_standardized_mean_difference", "size"),
        median_absolute_smd=("absolute_standardized_mean_difference", "median"),
        fraction_absolute_smd_below_0p1=(
            "absolute_standardized_mean_difference", lambda x: float(np.mean(x < 0.1))
        ),
    ).to_csv(args.output_root / "qc_balance_summary.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(args.output_root / "tumor_identity_marker_scores.csv", index=False)
    replicate_reports, results = [], []
    for replicate in sorted(metadata["replicate"].unique()):
        meta = metadata[metadata["replicate"].eq(replicate)].copy()
        subcounts = counts.loc[meta.index]
        subcounts = subcounts.groupby([meta["patient_id"], meta["confidence_status"]]).sum()
        cells = meta.groupby(["patient_id", "confidence_status"])["cell_n"].sum()
        eligible = cells.ge(args.minimum_cells_per_patient_status).groupby(level=0).all()
        complete = cells.groupby(level=0).size().eq(2)
        patients = eligible.index[eligible & complete]
        subcounts = subcounts.loc[subcounts.index.get_level_values(0).isin(patients)]
        meta = cells.loc[cells.index.get_level_values(0).isin(patients)].reset_index(name="cell_n")
        subcounts.index = [f"{p}__{s}" for p, s in subcounts.index]
        meta.index = [f"{p}__{s}" for p, s in zip(meta.patient_id, meta.confidence_status)]
        meta = meta.loc[subcounts.index]
        meta["comparison_status"] = meta["confidence_status"].map({
            "robust_rejected": "case", "robust_retained": "reference"
        })
        keep = subcounts.sum(axis=0).ge(10)
        result = PYDESEQ2.run_pydeseq2(subcounts.loc[:, keep].astype(np.int64), meta, n_cpus=args.n_cpus)
        result = result.merge(ot, on="gene", how="left")
        result["used_for_ot_anywhere"] = result["used_for_ot_anywhere"].fillna(False)
        result["replicate"] = replicate
        destination = args.output_root / f"replicate_{replicate}"
        destination.mkdir(exist_ok=True)
        result.to_csv(destination / "pydeseq2_all_genes.csv", index=False)
        non_ot = result[~result.used_for_ot_anywhere].copy()
        non_ot.to_csv(destination / "pydeseq2_non_ot_genes.csv", index=False)
        results.append(non_ot)
        replicate_reports.append({"replicate": int(replicate), "patient_n": len(patients),
                                  "gene_n": len(result), "non_ot_gene_n": len(non_ot)})

    long = pd.concat(results, ignore_index=True)
    consensus = long.groupby("gene", as_index=False).agg(
        replicate_n=("replicate", "nunique"), median_log2_fold_change=("log2_fold_change", "median"),
        direction_consistency=("log2_fold_change", lambda x: max(np.mean(x > 0), np.mean(x < 0))),
    )
    for threshold in (0.5, 1.0):
        selected = long[long.fdr.lt(0.05) & long.log2_fold_change.abs().ge(threshold)]
        frequency = selected.groupby("gene")["replicate"].nunique()
        consensus[f"fdr_005_abs_log2fc_{PYDESEQ2.threshold_tag(threshold)}_replicate_n"] = consensus.gene.map(frequency).fillna(0).astype(int)
    original = pd.read_csv(args.original_pydeseq2_root / "contrasts" / "rejected_vs_retained" / "pydeseq2_non_ot_gene_validation.csv")
    original = original[["gene", "log2_fold_change", "fdr"]].rename(columns={
        "log2_fold_change": "original_log2_fold_change", "fdr": "original_fdr"
    })
    consensus = consensus.merge(original, on="gene", how="left")
    consensus["direction_matches_original"] = np.sign(consensus.median_log2_fold_change) == np.sign(consensus.original_log2_fold_change)
    required_replicates = int(np.ceil(args.replicates / 2))
    for threshold in (0.5, 1.0):
        tag = PYDESEQ2.threshold_tag(threshold)
        consensus[f"qc_stable_fdr_005_abs_log2fc_{tag}"] = (
            consensus[f"fdr_005_abs_log2fc_{tag}_replicate_n"].ge(required_replicates)
            & consensus["direction_consistency"].ge(0.8)
            & consensus["direction_matches_original"]
        )
    consensus.to_csv(args.output_root / "qc_matched_deg_consensus.csv", index=False)

    marker = pd.DataFrame(marker_rows)
    patient_marker = marker.groupby(
        ["patient_id", "confidence_status", "marker_set", "feature", "feature_type"]
    )["mean_score"].median().unstack("confidence_status")
    marker_tests = []
    for keys, values in patient_marker.groupby(level=["marker_set", "feature", "feature_type"]):
        marker_set, feature, feature_type = keys
        values = values.droplevel(["marker_set", "feature", "feature_type"]).dropna()
        try:
            _, pvalue = wilcoxon(values.robust_rejected, values.robust_retained)
        except ValueError:
            pvalue = np.nan
        marker_tests.append({"marker_set": marker_set, "feature": feature,
                             "feature_type": feature_type, "patient_n": len(values),
                             "rejected_median": float(values.robust_rejected.median()),
                             "retained_median": float(values.robust_retained.median()),
                             "paired_wilcoxon_p": float(pvalue)})
    pd.DataFrame(marker_tests).to_csv(args.output_root / "tumor_identity_patient_tests.csv", index=False)
    report = {"group_n": len(ready), "matching_replicates": args.replicates,
              "matching_method": "logistic propensity score; greedy 1:1 without replacement; 0.2-SD caliper",
              "qc_covariates": QC_COLUMNS, "replicate_reports": replicate_reports,
              "primary_claim_limit": "Sensitivity analysis; QC covariates may include biological signal."}
    (args.output_root / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
