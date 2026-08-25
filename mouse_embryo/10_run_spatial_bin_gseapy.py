"""Run section-level and descriptive-consensus GSEA for rejected-bin DEG."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd


SECTION_PATTERN = re.compile(r"^section_(.+)_bin_deg\.csv\.gz$")


def collection_name(pathway: str) -> str:
    upper = pathway.upper()
    if upper.startswith("HALLMARK_"):
        return "Mouse Hallmark"
    if upper.startswith(("GOBP_", "GO_BP_")):
        return "Mouse GO Biological Process"
    if upper.startswith(("WP_", "WIKIPATHWAYS_")):
        return "Mouse WikiPathways"
    return "Other"


def read_gmt(path: Path) -> dict[str, list[str]]:
    pathways: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"Malformed GMT line {line_number} in {path}")
            pathways[fields[0]] = list(dict.fromkeys(fields[2:]))
    if not pathways:
        raise RuntimeError(f"No pathways were read from {path}")
    return pathways


def stable_ranking(table: pd.DataFrame, gene_column: str, score_column: str) -> pd.DataFrame:
    rank = table[[gene_column, score_column]].rename(
        columns={gene_column: "gene", score_column: "score"}
    )
    rank = rank.replace([np.inf, -np.inf], np.nan).dropna().drop_duplicates("gene")
    rank = rank.sort_values(["score", "gene"], ascending=[False, True]).reset_index(drop=True)
    if rank.empty:
        raise RuntimeError("The DEG table contained no finite ranking values.")
    # Deterministically resolve exact ties only; the perturbation is far below
    # meaningful Wilcoxon-score precision and leaves all non-tied ordering intact.
    tie_order = rank.groupby("score", sort=False).cumcount().to_numpy(dtype=float)
    rank["score"] = rank.score.to_numpy(dtype=float) - tie_order * 1e-12
    return rank


def overlap_qc(
    rank: pd.DataFrame,
    pathways: dict[str, list[str]],
    minimum_size: int,
    maximum_size: int,
) -> tuple[pd.DataFrame, dict]:
    ranked_genes = set(rank.gene.astype(str))
    rows = []
    for pathway, genes in pathways.items():
        overlap = ranked_genes.intersection(genes)
        rows.append({
            "pathway": pathway,
            "collection": collection_name(pathway),
            "gene_set_n": len(genes),
            "rank_overlap_n": len(overlap),
            "rank_overlap_fraction": len(overlap) / len(genes) if genes else 0.0,
            "eligible_for_gsea": minimum_size <= len(overlap) <= maximum_size,
        })
    table = pd.DataFrame(rows)
    summary = {
        "ranked_gene_n": len(ranked_genes),
        "gmt_pathway_n": len(pathways),
        "gmt_unique_gene_n": len(set().union(*map(set, pathways.values()))),
        "eligible_pathway_n": int(table.eligible_for_gsea.sum()),
        "median_pathway_overlap_fraction": float(table.rank_overlap_fraction.median()),
    }
    return table, summary


def standardized_results(table: pd.DataFrame) -> pd.DataFrame:
    result = table.rename(columns={
        "Term": "pathway",
        "ES": "enrichment_score",
        "NES": "NES",
        "NOM p-val": "p_value",
        "FDR q-val": "fdr",
        "FWER p-val": "fwer",
        "Lead_genes": "leading_edge_genes",
        "Tag %": "tag_fraction",
        "Gene %": "rank_fraction",
    }).copy()
    required = ["pathway", "enrichment_score", "NES", "p_value", "fdr"]
    missing = [column for column in required if column not in result]
    if missing:
        raise RuntimeError(f"GSEApy result lacked expected columns: {missing}")
    result["collection"] = result.pathway.map(collection_name)
    numeric = ["enrichment_score", "NES", "p_value", "fdr"]
    for column in numeric:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["fdr", "NES"], ascending=[True, False])


def run_prerank(
    rank: pd.DataFrame,
    pathways: dict[str, list[str]],
    permutations: int,
    threads: int,
    minimum_size: int,
    maximum_size: int,
    seed: int,
) -> pd.DataFrame:
    result = gp.prerank(
        rnk=rank,
        gene_sets=pathways,
        min_size=minimum_size,
        max_size=maximum_size,
        permutation_num=permutations,
        threads=threads,
        seed=seed,
        outdir=None,
        no_plot=True,
        verbose=False,
    )
    return standardized_results(result.res2d)


def pathway_consensus(section_results: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    long = pd.concat(
        [table.assign(sample=sample) for sample, table in section_results],
        ignore_index=True,
    )
    rows = []
    expected_section_n = len(section_results)
    for pathway, values in long.groupby("pathway", sort=False):
        nes = values.NES.to_numpy(dtype=float)
        median_nes = float(np.nanmedian(nes))
        sign = np.sign(median_nes)
        consistency = float(np.mean(np.sign(nes) == sign)) if sign != 0 else 0.0
        observed_section_n = int(values["sample"].nunique())
        coverage = observed_section_n / expected_section_n
        rows.append({
            "pathway": pathway,
            "collection": values.collection.iloc[0],
            "median_NES": median_nes,
            "section_direction_consistency": consistency,
            "section_fdr_005_n": int(np.sum(values.fdr < 0.05)),
            "best_section_fdr": float(np.nanmin(values.fdr)),
            "section_n": observed_section_n,
            "expected_section_n": expected_section_n,
            "section_coverage_fraction": coverage,
            "pathway_consensus_score": median_nes * consistency * coverage,
            "inference_scope": "descriptive_across_section_consensus",
        })
    return pd.DataFrame(rows).sort_values("pathway_consensus_score", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deg_root", type=Path)
    parser.add_argument("mouse_pathways_gmt", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--permutations", type=int, default=2_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--minimum-size", type=int, default=10)
    parser.add_argument("--maximum-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    pathways = read_gmt(args.mouse_pathways_gmt)
    run_summaries: list[dict] = []

    for candidate_root in sorted(path for path in args.deg_root.iterdir() if path.is_dir()):
        consensus_path = candidate_root / "consensus_deg.csv"
        section_paths = sorted(candidate_root.glob("section_*_bin_deg.csv.gz"))
        if not consensus_path.exists() or not section_paths:
            continue
        destination = args.output_root / candidate_root.name
        destination.mkdir(exist_ok=True)

        consensus_rank = stable_ranking(
            pd.read_csv(consensus_path), "gene", "consensus_rank_score"
        )
        qc, qc_summary = overlap_qc(
            consensus_rank, pathways, args.minimum_size, args.maximum_size
        )
        qc.to_csv(destination / "gene_set_overlap_qc.csv", index=False)
        consensus_gsea = run_prerank(
            consensus_rank, pathways, args.permutations, args.threads,
            args.minimum_size, args.maximum_size, args.seed,
        )
        consensus_gsea.to_csv(destination / "consensus_gsea_results.csv", index=False)

        section_results: list[tuple[str, pd.DataFrame]] = []
        for section_index, section_path in enumerate(section_paths):
            match = SECTION_PATTERN.match(section_path.name)
            if match is None:
                continue
            sample = match.group(1)
            rank = stable_ranking(pd.read_csv(section_path), "gene", "wilcoxon_score")
            section_gsea = run_prerank(
                rank, pathways, args.permutations, args.threads,
                args.minimum_size, args.maximum_size, args.seed + section_index + 1,
            )
            section_gsea.to_csv(destination / f"section_{sample}_gsea_results.csv", index=False)
            section_results.append((sample, section_gsea))

        pathway_consensus(section_results).to_csv(
            destination / "pathway_consensus.csv", index=False
        )
        run_summaries.append({
            "candidate": candidate_root.name,
            "section_n": len(section_results),
            "consensus_pathway_n": len(consensus_gsea),
            "consensus_pathway_fdr_005_n": int(np.sum(consensus_gsea.fdr < 0.05)),
            "section_reproduced_pathway_n": int(np.sum(
                pd.read_csv(destination / "pathway_consensus.csv").section_fdr_005_n
                == len(section_results)
            )),
            "engine": "GSEApy prerank",
            "permutations": args.permutations,
            **qc_summary,
        })
        print(run_summaries[-1], flush=True)

    if not run_summaries:
        raise RuntimeError("No candidate DEG tables were found.")
    pd.DataFrame(run_summaries).to_csv(args.output_root / "gsea_summary.csv", index=False)
    print(f"GSEA outputs: {args.output_root.resolve()}", flush=True)


if __name__ == "__main__":
    main()
