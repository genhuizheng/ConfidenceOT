"""Run Python preranked GSEA for rejected-bin DEG consensus rankings."""

from __future__ import annotations

import argparse
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd


def standardized_results(table: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Term": "pathway",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "pval",
        "FDR q-val": "padj",
        "FWER p-val": "fwer",
        "Lead_genes": "leadingEdge",
    }
    result = table.rename(columns=rename).copy()
    required = ["pathway", "ES", "NES", "pval", "padj"]
    missing = [column for column in required if column not in result]
    if missing:
        raise RuntimeError(f"GSEApy result lacked expected columns: {missing}")
    return result.sort_values(["padj", "NES"], ascending=[True, False])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deg_root", type=Path)
    parser.add_argument("mouse_pathways_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--permutations", type=int, default=1_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-size", type=int, default=10)
    parser.add_argument("--maximum-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for candidate_root in sorted(path for path in args.deg_root.iterdir() if path.is_dir()):
        rank_path = candidate_root / "gsea_rank.rnk"
        if not rank_path.exists():
            continue
        rank = pd.read_csv(rank_path, sep="\t", header=None, names=["gene", "score"])
        rank = rank[np.isfinite(rank.score)].drop_duplicates("gene").sort_values("score", ascending=False)
        result = gp.prerank(
            rnk=rank,
            gene_sets=str(args.mouse_pathways_gmt),
            min_size=args.minimum_size,
            max_size=args.maximum_size,
            permutation_num=args.permutations,
            threads=args.threads,
            seed=args.seed,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
        table = standardized_results(result.res2d)
        destination = args.output_root / candidate_root.name
        destination.mkdir(exist_ok=True)
        table.to_csv(destination / "fgsea_results.csv", index=False)
        summaries.append({
            "candidate": candidate_root.name,
            "tested_pathway_n": len(table),
            "pathway_fdr_005_n": int(np.sum(table.padj < 0.05)),
            "engine": "GSEApy prerank",
            "permutations": args.permutations,
        })
        print(f"Completed {candidate_root.name}", flush=True)

    if not summaries:
        raise RuntimeError("No candidate rank files were found.")
    pd.DataFrame(summaries).to_csv(args.output_root / "gsea_summary.csv", index=False)


if __name__ == "__main__":
    main()
