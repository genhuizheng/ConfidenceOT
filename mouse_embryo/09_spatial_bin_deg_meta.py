"""Within-section spatial-bin DEG for ConfidenceOT biological discovery.

For each predefined developmental candidate, ConfidenceOT-rejected bins of
the candidate annotation are compared with all ConfidenceOT-retained bins on
the same side of the same MOSTA section. The primary test is Scanpy's
Wilcoxon rank-sum implementation on library-normalized log1p expression.

Genes used to build the OT representation are excluded from DEG, preventing
the same features from both selecting and validating rejected bins. Every
section is tested independently. The across-section table is a descriptive
consensus ranking, not a biological-replicate meta-analysis.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


PAIR_PATTERN = re.compile(r"^(E\d+\.\d+)_(.+)_to_(E\d+\.\d+)_(.+)$")


@dataclass(frozen=True)
class Candidate:
    name: str
    transition: str
    side: str
    annotation: str


DEFAULT_CANDIDATES = (
    Candidate("heart_emergent_state", "E9.5 → E10.5", "target", "Heart"),
    Candidate("lung_primordium_disappearance", "E9.5 → E10.5", "source", "Lung primordium"),
    Candidate("dermomyotome_emergent_state", "E10.5 → E11.5", "target", "Dermomyotome"),
    Candidate("spinal_cord_disappearance", "E10.5 → E11.5", "source", "Spinal cord"),
    Candidate("mesenchyme_disappearance", "E10.5 → E11.5", "source", "Mesenchyme"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--consensus-threshold", type=float, default=0.5)
    parser.add_argument("--minimum-rejected-bins", type=int, default=20)
    parser.add_argument("--minimum-retained-bins", type=int, default=20)
    parser.add_argument("--minimum-gene-total", type=int, default=10)
    parser.add_argument("--target-sum", type=float, default=10_000.0)
    parser.add_argument(
        "--candidates", nargs="+", choices=[candidate.name for candidate in DEFAULT_CANDIDATES],
        help="Run only the named candidates; the default runs all candidates.",
    )
    return parser.parse_args()


def pair_metadata(pair_id: str) -> dict[str, str]:
    match = PAIR_PATTERN.match(pair_id)
    if match is None:
        raise ValueError(f"Unrecognized pair id: {pair_id}")
    source_stage, source_sample, target_stage, target_sample = match.groups()
    return {
        "source_stage": source_stage,
        "source_sample": source_sample,
        "target_stage": target_stage,
        "target_sample": target_sample,
        "transition": f"{source_stage} → {target_stage}",
    }


def load_valid_results(run_root: Path) -> tuple[pd.DataFrame, dict[str, set[str]], pd.DataFrame]:
    cells: list[pd.DataFrame] = []
    transport_genes: dict[str, set[str]] = {}
    certificates: list[dict] = []
    for success in sorted(run_root.glob("pairs/*/analysis/SUCCESS")):
        pair_root = success.parents[1]
        pair_id = pair_root.name
        info = pair_metadata(pair_id)
        certificate = json.loads((pair_root / "analysis" / "calibration.json").read_text())
        valid = bool(certificate["calibration_valid"])
        certificates.append({"pair_id": pair_id, **info, "calibration_valid": valid})
        if not valid:
            continue
        table = pd.read_csv(pair_root / "analysis" / "cell_confidence.csv")
        table.insert(0, "pair_id", pair_id)
        for key, value in info.items():
            table[key] = value
        cells.append(table)
        with np.load(pair_root / "preparation" / "prepared_pair.npz", allow_pickle=False) as prepared:
            transport_genes.setdefault(info["transition"], set()).update(
                prepared["hvg_genes"].astype(str)
            )
    if not cells:
        raise RuntimeError("No calibration-valid pairs were found.")
    return pd.concat(cells, ignore_index=True), transport_genes, pd.DataFrame(certificates)


def consensus_for_candidate(cells: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    sample_column = f"{candidate.side}_sample"
    selected = cells[
        cells.transition.eq(candidate.transition) & cells.side.eq(candidate.side)
    ]
    result = selected.groupby(
        [sample_column, "observation_id", "annotation", "spatial_x", "spatial_y"],
        as_index=False,
    ).agg(
        rejection_frequency=("rejected", "mean"),
        partner_pair_n=("pair_id", "nunique"),
    )
    return result.rename(columns={sample_column: "sample"})


def normalize_log_counts(counts: sparse.csr_matrix, target_sum: float) -> sparse.csr_matrix:
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("A selected MOSTA bin has zero library size.")
    result = counts.multiply((target_sum / library)[:, None]).tocsr()
    result.data = np.log1p(result.data)
    return result


def scanpy_wilcoxon(
    expression: sparse.csr_matrix,
    genes: np.ndarray,
    rejected_n: int,
) -> pd.DataFrame:
    try:
        import scanpy as sc
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "A compatible Scanpy installation is required for DEG. Install "
            "scanpy>=1.10,<2 with anndata>=0.10,<0.13 in the active environment."
        ) from error
    labels = np.where(np.arange(expression.shape[0]) < rejected_n, "rejected", "retained")
    analysis = ad.AnnData(
        X=expression,
        obs=pd.DataFrame(
            {"confidence_status": pd.Categorical(labels)},
            index=[f"bin_{index}" for index in range(len(labels))],
        ),
        var=pd.DataFrame(index=pd.Index(genes, name="gene")),
    )
    sc.tl.rank_genes_groups(
        analysis,
        groupby="confidence_status",
        groups=["rejected"],
        reference="retained",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        tie_correct=True,
        pts=True,
        n_genes=analysis.n_vars,
        use_raw=False,
    )
    result = sc.get.rank_genes_groups_df(analysis, group="rejected").rename(columns={
        "names": "gene",
        "scores": "wilcoxon_score",
        "logfoldchanges": "log2_fold_change",
        "pvals": "p_value",
        "pvals_adj": "fdr",
        "pct_nz_group": "rejected_expression_fraction",
        "pct_nz_reference": "retained_expression_fraction",
    })
    rejected_mean = np.asarray(expression[:rejected_n].mean(axis=0)).ravel()
    retained_mean = np.asarray(expression[rejected_n:].mean(axis=0)).ravel()
    mean_lookup = pd.DataFrame({
        "gene": genes,
        "rejected_mean_log_expression": rejected_mean,
        "retained_mean_log_expression": retained_mean,
        "mean_log_expression_difference": rejected_mean - retained_mean,
    })
    return result.merge(mean_lookup, on="gene", how="left", validate="one_to_one")


def analyze_section(
    path: Path,
    section: pd.DataFrame,
    candidate: Candidate,
    excluded_genes: set[str],
    threshold: float,
    minimum_rejected: int,
    minimum_retained: int,
    minimum_gene_total: int,
    target_sum: float,
) -> tuple[pd.DataFrame | None, dict]:
    rejected = section.rejection_frequency.ge(threshold)
    case_rows = section[rejected & section.annotation.eq(candidate.annotation)].copy()
    control_rows = section[~rejected].copy()
    diagnostics = {
        "sample": section["sample"].iloc[0],
        "candidate_annotation": candidate.annotation,
        "candidate_rejected_bin_n": len(case_rows),
        "all_retained_background_bin_n": len(control_rows),
        "other_rejected_bins_excluded_n": int(rejected.sum() - len(case_rows)),
    }
    if len(case_rows) < minimum_rejected or len(control_rows) < minimum_retained:
        diagnostics["status"] = "insufficient_bins"
        return None, diagnostics

    dataset = ad.read_h5ad(path, backed="r")
    try:
        lookup = {str(value): index for index, value in enumerate(dataset.obs_names)}
        ordered_ids = pd.concat([case_rows, control_rows], ignore_index=True).observation_id.astype(str)
        missing = [value for value in ordered_ids if value not in lookup]
        if missing:
            raise KeyError(f"{len(missing)} ConfidenceOT observation IDs were absent from {path.name}.")
        rows = np.asarray([lookup[value] for value in ordered_ids], dtype=np.int64)
        raw = sparse.csr_matrix(dataset.layers["count"][rows], dtype=np.float64)
        genes = np.asarray(dataset.var_names.astype(str), dtype=str)
    finally:
        dataset.file.close()

    gene_total = np.asarray(raw.sum(axis=0)).ravel()
    keep = (~np.isin(genes, list(excluded_genes))) & (gene_total >= minimum_gene_total)
    raw, genes = raw[:, keep], genes[keep]
    if raw.shape[1] == 0:
        raise RuntimeError(f"No genes remained for {path.name} after independent-gene filtering.")
    expression = normalize_log_counts(raw, target_sum)
    result = scanpy_wilcoxon(expression, genes, len(case_rows))
    result["sample"] = diagnostics["sample"]
    result["comparison"] = "candidate_rejected_vs_all_retained_bins"
    result["candidate_annotation"] = candidate.annotation
    result["rejected_bin_n"] = len(case_rows)
    result["retained_background_bin_n"] = len(control_rows)
    result["excluded_transport_gene_n"] = len(excluded_genes)
    diagnostics.update({"status": "complete", "tested_gene_n": len(result)})
    return result.sort_values("wilcoxon_score", ascending=False), diagnostics


def descriptive_consensus(section_tables: list[pd.DataFrame]) -> pd.DataFrame:
    columns = [
        "wilcoxon_score", "log2_fold_change", "fdr",
        "rejected_expression_fraction", "retained_expression_fraction",
        "mean_log_expression_difference",
    ]
    long = pd.concat(
        [table[["gene", "sample", *columns]] for table in section_tables],
        ignore_index=True,
    )
    rows: list[dict] = []
    for gene, values in long.groupby("gene", sort=False):
        score = values.wilcoxon_score.to_numpy(dtype=float)
        fold_change = values.log2_fold_change.to_numpy(dtype=float)
        median_score = float(np.nanmedian(score))
        sign = np.sign(median_score)
        consistency = float(np.mean(np.sign(score) == sign)) if sign != 0 else 0.0
        rows.append({
            "gene": gene,
            "median_wilcoxon_score": median_score,
            "median_log2_fold_change": float(np.nanmedian(fold_change)),
            "median_mean_log_expression_difference": float(np.nanmedian(values.mean_log_expression_difference)),
            "mean_rejected_expression_fraction": float(np.nanmean(values.rejected_expression_fraction)),
            "mean_retained_expression_fraction": float(np.nanmean(values.retained_expression_fraction)),
            "best_section_fdr": float(np.nanmin(values.fdr)),
            "section_fdr_005_n": int(np.sum(values.fdr < 0.05)),
            "section_direction_consistency": consistency,
            "section_n": int(values["sample"].nunique()),
            "consensus_rank_score": median_score * consistency,
            "inference_scope": "descriptive_across_section_consensus",
        })
    return pd.DataFrame(rows).sort_values("consensus_rank_score", ascending=False)


def main() -> None:
    args = parse_args()
    if not 0 <= args.consensus_threshold <= 1:
        raise ValueError("--consensus-threshold must be between 0 and 1.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, transport_genes, certificates = load_valid_results(args.run_root)
    certificates.to_csv(args.output_dir / "certificate_scope.csv", index=False)
    candidates = [
        candidate for candidate in DEFAULT_CANDIDATES
        if args.candidates is None or candidate.name in args.candidates
    ]
    summary: list[dict] = []
    for candidate in candidates:
        root = args.output_dir / candidate.name
        root.mkdir(exist_ok=True)
        consensus = consensus_for_candidate(cells, candidate)
        consensus.to_csv(root / "cell_consensus.csv.gz", index=False, compression="gzip")
        source_stage, target_stage = candidate.transition.split(" → ")
        stage = source_stage if candidate.side == "source" else target_stage
        section_tables: list[pd.DataFrame] = []
        section_diagnostics: list[dict] = []
        for sample in sorted(consensus["sample"].unique()):
            table, diagnostics = analyze_section(
                args.data_root / f"{stage}_{sample}.MOSTA.h5ad",
                consensus[consensus["sample"].eq(sample)],
                candidate,
                transport_genes[candidate.transition],
                args.consensus_threshold,
                args.minimum_rejected_bins,
                args.minimum_retained_bins,
                args.minimum_gene_total,
                args.target_sum,
            )
            section_diagnostics.append(diagnostics)
            if table is None:
                continue
            table.to_csv(root / f"section_{sample}_bin_deg.csv.gz", index=False, compression="gzip")
            section_tables.append(table)
        pd.DataFrame(section_diagnostics).to_csv(root / "section_diagnostics.csv", index=False)
        if not section_tables:
            summary.append({"candidate": candidate.name, "status": "no_eligible_sections"})
            continue

        consensus_deg = descriptive_consensus(section_tables)
        consensus_deg.to_csv(root / "consensus_deg.csv", index=False)
        consensus_deg[["gene", "consensus_rank_score"]].to_csv(
            root / "gsea_rank.rnk", sep="\t", index=False, header=False
        )
        design = {
            **asdict(candidate),
            "primary_comparison": "candidate-annotation rejected bins versus all retained bins",
            "test_unit": "individual spatial bin within each section",
            "test": "scanpy Wilcoxon rank-sum with tie correction and BH FDR",
            "normalization": f"library-size normalize to {args.target_sum:g}, then log1p",
            "cross_section_result": "descriptive consensus only; no pooled p-value or FDR",
            "consensus_threshold": args.consensus_threshold,
            "excluded_transport_gene_n": len(transport_genes[candidate.transition]),
            "circularity_control": "genes used in OT PCA/HVG representation excluded from DEG",
        }
        (root / "analysis_design.json").write_text(json.dumps(design, indent=2), encoding="utf-8")
        summary.append({
            "candidate": candidate.name,
            "annotation": candidate.annotation,
            "transition": candidate.transition,
            "side": candidate.side,
            "status": "complete",
            "section_n": len(section_tables),
            "tested_gene_n": len(consensus_deg),
            "primary_comparison": design["primary_comparison"],
        })
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(args.output_dir / "spatial_bin_deg_summary.csv", index=False)
    print(f"Spatial-bin DEG outputs: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
