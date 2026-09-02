"""Run Human MSigDB preranked GSEA from paired PyDESeq2 Wald statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd


def collection_name(term: str) -> str:
    upper = term.upper()
    if upper.startswith("HALLMARK_"):
        return "Hallmark"
    if upper.startswith(("GOBP_", "GO_BP_")):
        return "GO Biological Process"
    if upper.startswith("REACTOME_"):
        return "Reactome"
    if upper.startswith(("WP_", "WIKIPATHWAYS_")):
        return "WikiPathways"
    return "Other"


def read_gmt(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number}: {path}")
            result[fields[0]] = list(dict.fromkeys(fields[2:]))
    if not result:
        raise RuntimeError(f"No pathways read from {path}")
    return result


def stable_rank(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep="\t", header=None, names=["gene", "score"])
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    table = table.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates("gene")
    table = table.sort_values(["score", "gene"], ascending=[False, True]).reset_index(drop=True)
    if table.empty:
        raise RuntimeError(f"Empty ranking: {path}")
    tie_order = table.groupby("score", sort=False).cumcount().to_numpy(float)
    table["score"] = table.score.to_numpy(float) - tie_order * 1e-12
    return table


def run_one(label: str, rank_path: Path, pathways: dict[str, list[str]], output: Path,
            *, permutations: int, threads: int, minimum_size: int,
            maximum_size: int, seed: int) -> dict:
    rank = stable_rank(rank_path)
    ranked = set(rank.gene.astype(str))
    overlap_rows = []
    for term, genes in pathways.items():
        overlap_n = len(ranked.intersection(genes))
        overlap_rows.append({"pathway": term, "collection": collection_name(term),
                             "gene_set_n": len(genes), "rank_overlap_n": overlap_n,
                             "eligible": minimum_size <= overlap_n <= maximum_size})
    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(output / f"{label}_gene_set_overlap.csv", index=False)
    prerank = gp.prerank(
        rnk=rank, gene_sets=pathways, min_size=minimum_size, max_size=maximum_size,
        permutation_num=permutations, threads=threads, seed=seed, outdir=None,
        no_plot=True, verbose=False,
    ).res2d
    result = prerank.rename(columns={
        "Term": "pathway", "ES": "enrichment_score", "NOM p-val": "p_value",
        "FDR q-val": "fdr", "FWER p-val": "fwer", "Lead_genes": "leading_edge_genes",
        "Tag %": "tag_fraction", "Gene %": "rank_fraction",
    }).copy()
    if "pathway" not in result.columns or "NES" not in result.columns or "fdr" not in result.columns:
        raise RuntimeError(f"Unexpected GSEApy columns: {list(result.columns)}")
    for column in ("enrichment_score", "NES", "p_value", "fdr", "fwer"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["collection"] = result.pathway.map(collection_name)
    result["direction"] = np.where(result.NES >= 0, "robust_rejected_enriched",
                                   "robust_retained_enriched")
    result = result.sort_values(["fdr", "NES"], ascending=[True, False])
    result.to_csv(output / f"{label}_gsea_results.csv", index=False)
    return {"analysis": label, "ranked_gene_n": len(rank),
            "eligible_pathway_n": int(overlap.eligible.sum()), "tested_pathway_n": len(result),
            "fdr_005_pathway_n": int(result.fdr.lt(0.05).sum()),
            "rejected_enriched_fdr_005_n": int((result.fdr.lt(0.05) & result.NES.gt(0)).sum()),
            "retained_enriched_fdr_005_n": int((result.fdr.lt(0.05) & result.NES.lt(0)).sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pydeseq2_root", type=Path)
    parser.add_argument("human_pathways_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--minimum-size", type=int, default=10)
    parser.add_argument("--maximum-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pathways = read_gmt(args.human_pathways_gmt)
    summaries = []
    contrasts_root = args.pydeseq2_root / "contrasts"
    contrast_directories = sorted(path for path in contrasts_root.iterdir() if path.is_dir())
    if not contrast_directories:
        raise RuntimeError(f"No PyDESeq2 contrast directories found under {contrasts_root}")
    run_index = 0
    for contrast_directory in contrast_directories:
        contrast = contrast_directory.name
        destination = args.output_root / "contrasts" / contrast
        destination.mkdir(parents=True, exist_ok=True)
        inputs = {
            "all_gene_discovery": contrast_directory / "pydeseq2_all_gene_wald.rnk",
            "non_ot_gene_validation": contrast_directory / "pydeseq2_non_ot_gene_wald.rnk",
        }
        for label, path in inputs.items():
            if not path.exists():
                raise FileNotFoundError(path)
            summary = run_one(
                label, path, pathways, destination,
                permutations=args.permutations, threads=args.threads,
                minimum_size=args.minimum_size, maximum_size=args.maximum_size,
                seed=args.seed + run_index,
            )
            summary["contrast"] = contrast
            summaries.append(summary)
            run_index += 1
    report = {
        "engine": "GSEApy prerank", "ranking_metric": "paired PyDESeq2 Wald statistic",
        "positive_direction": "robust rejected malignant cells",
        "negative_direction": "robust retained malignant cells",
        "permutations": args.permutations, "analyses": summaries,
        "interpretation_limit": "Pathway findings are exploratory until robust to cell-complexity sensitivity analysis.",
    }
    pd.DataFrame(summaries).to_csv(args.output_root / "gsea_summary.csv", index=False)
    (args.output_root / "gsea_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
