"""Run replicate-wise GSEA for the QC-matched rejected-versus-retained validation."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_gsea_module():
    path = Path(__file__).with_name("15_run_pydeseq2_gseapy.py")
    spec = importlib.util.spec_from_file_location("paired_gsea", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GSEA = load_gsea_module()


def pathway_consensus(
    tables: list[pd.DataFrame],
    replicate_count: int,
    minimum_fraction: float,
) -> pd.DataFrame:
    long = pd.concat(tables, ignore_index=True)
    rows = []
    required = int(np.ceil(replicate_count * minimum_fraction))
    for pathway, group in long.groupby("pathway", sort=True):
        nes = group["NES"].to_numpy(float)
        significant = group["fdr"].lt(0.05).to_numpy()
        positive_n = int(np.sum(significant & (nes > 0)))
        negative_n = int(np.sum(significant & (nes < 0)))
        dominant_positive = positive_n >= negative_n
        same_direction_n = positive_n if dominant_positive else negative_n
        rows.append({
            "pathway": pathway,
            "collection": group["collection"].iloc[0],
            "replicate_n": int(group["replicate"].nunique()),
            "median_nes": float(np.median(nes)),
            "minimum_fdr": float(group["fdr"].min()),
            "median_fdr": float(group["fdr"].median()),
            "rejected_enriched_fdr_005_replicate_n": positive_n,
            "retained_enriched_fdr_005_replicate_n": negative_n,
            "dominant_direction": (
                "robust_rejected_enriched" if dominant_positive
                else "robust_retained_enriched"
            ),
            "significant_same_direction_replicate_n": same_direction_n,
            "stable": same_direction_n >= required,
        })
    return pd.DataFrame(rows).sort_values(
        ["stable", "significant_same_direction_replicate_n", "median_fdr"],
        ascending=[False, False, True],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qc_matched_root", type=Path)
    parser.add_argument("human_pathways_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--minimum-size", type=int, default=10)
    parser.add_argument("--maximum-size", type=int, default=500)
    parser.add_argument("--minimum-stable-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()

    if not 0 < args.minimum_stable_fraction <= 1:
        raise ValueError("--minimum-stable-fraction must be in (0, 1]")
    args.output_root.mkdir(parents=True, exist_ok=True)
    pathways = GSEA.read_gmt(args.human_pathways_gmt)
    replicate_directories = sorted(args.qc_matched_root.glob("replicate_*"))
    if not replicate_directories:
        raise RuntimeError(f"No replicate directories found under {args.qc_matched_root}")

    analyses = []
    for scope, filename in (
        ("all_gene_discovery", "pydeseq2_all_genes.csv"),
        ("non_ot_gene_validation", "pydeseq2_non_ot_genes.csv"),
    ):
        destination = args.output_root / scope
        destination.mkdir(exist_ok=True)
        tables = []
        for run_index, directory in enumerate(replicate_directories):
            replicate = int(directory.name.rsplit("_", 1)[1])
            table = pd.read_csv(directory / filename)
            rank = table[["gene", "wald_statistic"]].dropna().sort_values(
                "wald_statistic", ascending=False
            )
            rank_path = destination / f"replicate_{replicate}_wald.rnk"
            rank.to_csv(rank_path, sep="\t", index=False, header=False)
            GSEA.run_one(
                f"replicate_{replicate}", rank_path, pathways, destination,
                permutations=args.permutations, threads=args.threads,
                minimum_size=args.minimum_size, maximum_size=args.maximum_size,
                seed=args.seed + run_index,
                case_label="robust_rejected", reference_label="robust_retained",
            )
            result = pd.read_csv(destination / f"replicate_{replicate}_gsea_results.csv")
            result["replicate"] = replicate
            tables.append(result)

        long = pd.concat(tables, ignore_index=True)
        long.to_csv(destination / "replicate_gsea_results_long.csv.gz", index=False)
        consensus = pathway_consensus(
            tables, len(replicate_directories), args.minimum_stable_fraction
        )
        consensus.to_csv(destination / "qc_matched_gsea_consensus.csv", index=False)
        stable = consensus[consensus["stable"]]
        analyses.append({
            "analysis": scope,
            "replicate_n": len(replicate_directories),
            "tested_pathway_n": len(consensus),
            "stable_pathway_n": len(stable),
            "stable_rejected_enriched_n": int(
                stable["dominant_direction"].eq("robust_rejected_enriched").sum()
            ),
            "stable_retained_enriched_n": int(
                stable["dominant_direction"].eq("robust_retained_enriched").sum()
            ),
        })

    report = {
        "engine": "GSEApy prerank",
        "ranking_metric": "QC-matched paired PyDESeq2 Wald statistic",
        "replicate_n": len(replicate_directories),
        "minimum_stable_fraction": args.minimum_stable_fraction,
        "stable_definition": "FDR < 0.05 with the same NES direction in at least the requested fraction of matching replicates",
        "positive_direction": "robust_rejected_enriched",
        "negative_direction": "robust_retained_enriched",
        "analyses": analyses,
        "interpretation_limit": "Supplementary QC-matched pathway sensitivity analysis; not an independent discovery analysis.",
    }
    pd.DataFrame(analyses).to_csv(args.output_root / "qc_matched_gsea_summary.csv", index=False)
    (args.output_root / "qc_matched_gsea_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
