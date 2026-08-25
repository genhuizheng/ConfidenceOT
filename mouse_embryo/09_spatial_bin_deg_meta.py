"""Spatial-bin DEG with block bootstrap and embryo-aware meta-analysis.

ConfidenceOT-rejected bins are compared with retained bins of the same MOSTA
annotation inside each section.  Bins are never treated as independent
biological replicates: uncertainty is estimated by resampling contiguous
spatial blocks, section effects are combined inside embryo, and embryo effects
are combined only when multiple embryos are available.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse, stats


PAIR_PATTERN = re.compile(r"^(E\d+\.\d+)_(.+)_to_(E\d+\.\d+)_(.+)$")
EMBRYO_PATTERN = re.compile(r"^(E\d+)S\d+$")


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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--consensus-threshold", type=float, default=0.5)
    parser.add_argument("--block-width-bins", type=float, default=8.0)
    parser.add_argument("--bootstrap-replicates", type=int, default=400)
    parser.add_argument("--minimum-bins", type=int, default=20)
    parser.add_argument("--minimum-blocks", type=int, default=4)
    parser.add_argument("--minimum-gene-total", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument(
        "--candidates", nargs="+", choices=[candidate.name for candidate in DEFAULT_CANDIDATES],
        help="Run only the named candidates; the default runs all predefined candidates.",
    )
    return parser.parse_args()


def pair_metadata(pair_id: str):
    match = PAIR_PATTERN.match(pair_id)
    if match is None:
        raise ValueError(f"Unrecognized pair id: {pair_id}")
    source_stage, source_sample, target_stage, target_sample = match.groups()
    return {
        "source_stage": source_stage, "source_sample": source_sample,
        "target_stage": target_stage, "target_sample": target_sample,
        "transition": f"{source_stage} → {target_stage}",
    }


def embryo_from_sample(sample: str) -> str:
    match = EMBRYO_PATTERN.match(sample)
    if match is None:
        raise ValueError(f"Cannot parse embryo from {sample}")
    return match.group(1)


def load_valid_results(run_root: Path):
    cells, transport_by_transition, certificates = [], {}, []
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
        prepared = np.load(pair_root / "preparation" / "prepared_pair.npz", allow_pickle=False)
        transport_by_transition.setdefault(info["transition"], set()).update(
            prepared["hvg_genes"].astype(str)
        )
    if not cells:
        raise RuntimeError("No calibration-valid pairs were found.")
    return pd.concat(cells, ignore_index=True), transport_by_transition, pd.DataFrame(certificates)


def consensus_for_candidate(cells: pd.DataFrame, candidate: Candidate) -> pd.DataFrame:
    sample_column = f"{candidate.side}_sample"
    selected = cells[cells.transition.eq(candidate.transition) & cells.side.eq(candidate.side)]
    result = selected.groupby(
        [sample_column, "observation_id", "annotation", "spatial_x", "spatial_y"], as_index=False
    ).agg(rejection_frequency=("rejected", "mean"), partner_pair_n=("pair_id", "nunique"))
    return result.rename(columns={sample_column: "sample"})


def nearest_neighbor_spacing(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 1.0
    rounded = np.unique(xy, axis=0)
    sample = rounded[: min(len(rounded), 2000)]
    distances = np.sum((sample[:, None, :] - sample[None, :, :]) ** 2, axis=2)
    distances[distances == 0] = np.inf
    spacing = float(np.median(np.sqrt(np.min(distances, axis=1))))
    return spacing if np.isfinite(spacing) and spacing > 0 else 1.0


def spatial_blocks(xy: np.ndarray, width_bins: float) -> np.ndarray:
    width = nearest_neighbor_spacing(xy) * width_bins
    origin = xy.min(axis=0)
    grid = np.floor((xy - origin) / width).astype(np.int64)
    _, labels = np.unique(grid, axis=0, return_inverse=True)
    return labels


def normalized_log_counts(counts: sparse.csr_matrix) -> sparse.csr_matrix:
    library = np.asarray(counts.sum(axis=1)).ravel()
    if np.any(library <= 0):
        raise ValueError("Selected MOSTA bin has zero library size.")
    result = counts.multiply((10_000.0 / library)[:, None]).tocsr()
    result.data = np.log1p(result.data)
    return result


def block_sums(expression, blocks, group, block_count):
    membership = sparse.csr_matrix(
        (np.ones(len(blocks)), (blocks, np.arange(len(blocks)))),
        shape=(block_count, len(blocks)),
    )
    selected = membership.multiply(group[None, :])
    sums = (selected @ expression).toarray().astype(np.float32)
    counts = np.asarray(selected.sum(axis=1)).ravel().astype(np.float32)
    return sums, counts


def bootstrap_section_effect(
    expression: sparse.csr_matrix,
    case: np.ndarray,
    control: np.ndarray,
    blocks: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
):
    block_count = int(blocks.max()) + 1
    case_sums, case_n = block_sums(expression, blocks, case, block_count)
    control_sums, control_n = block_sums(expression, blocks, control, block_count)
    observed = (
        case_sums.sum(axis=0) / max(case_n.sum(), 1)
        - control_sums.sum(axis=0) / max(control_n.sum(), 1)
    )
    draws = rng.integers(0, block_count, size=(replicates, block_count))
    weights = np.zeros((replicates, block_count), dtype=np.float32)
    rows = np.repeat(np.arange(replicates), block_count)
    np.add.at(weights, (rows, draws.ravel()), 1)
    case_denominator = weights @ case_n
    control_denominator = weights @ control_n
    usable = (case_denominator > 0) & (control_denominator > 0)
    boot = (
        (weights[usable] @ case_sums) / case_denominator[usable, None]
        - (weights[usable] @ control_sums) / control_denominator[usable, None]
    )
    standard_error = boot.std(axis=0, ddof=1)
    standard_error[standard_error < 1e-8] = np.nan
    lower, upper = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return observed, standard_error, lower, upper, int(block_count), int(usable.sum())


def analyze_section(
    path: Path,
    section: pd.DataFrame,
    candidate: Candidate,
    excluded_genes: set[str],
    threshold: float,
    block_width: float,
    replicates: int,
    minimum_bins: int,
    minimum_blocks: int,
    minimum_gene_total: int,
    rng: np.random.Generator,
):
    rejected = section.rejection_frequency.to_numpy() >= threshold
    case_rows = section[rejected & section.annotation.eq(candidate.annotation)]
    control_rows = section[(~rejected) & section.annotation.eq(candidate.annotation)]
    if min(len(case_rows), len(control_rows)) < minimum_bins:
        return None
    dataset = ad.read_h5ad(path, backed="r")
    try:
        ids = np.asarray(dataset.obs_names.astype(str), dtype=str)
        lookup = {value: index for index, value in enumerate(ids)}
        ordered_ids = pd.concat([case_rows, control_rows]).observation_id.tolist()
        rows = np.asarray([lookup[value] for value in ordered_ids], dtype=np.int64)
        raw = sparse.csr_matrix(dataset.layers["count"][rows], dtype=np.float64)
        genes = np.asarray(dataset.var_names.astype(str), dtype=str)
    finally:
        dataset.file.close()
    keep = (~np.isin(genes, list(excluded_genes))) & (np.asarray(raw.sum(axis=0)).ravel() >= minimum_gene_total)
    raw, genes = raw[:, keep], genes[keep]
    expression = normalized_log_counts(raw)
    case = np.arange(len(rows)) < len(case_rows)
    control = ~case
    xy = pd.concat([case_rows, control_rows])[["spatial_x", "spatial_y"]].to_numpy()
    blocks = spatial_blocks(xy, block_width)
    if np.unique(blocks).size < minimum_blocks:
        return None
    effect, standard_error, lower, upper, block_n, usable_bootstrap = bootstrap_section_effect(
        expression, case, control, blocks, replicates, rng
    )
    return pd.DataFrame({
        "gene": genes, "section_effect": effect, "section_se": standard_error,
        "bootstrap_ci_low": lower, "bootstrap_ci_high": upper,
        "case_bin_n": len(case_rows), "control_bin_n": len(control_rows),
        "spatial_block_n": block_n, "usable_bootstrap_n": usable_bootstrap,
    })


def combine_effects(effects: np.ndarray, errors: np.ndarray):
    variance = errors ** 2
    weights = 1.0 / variance
    fixed = np.sum(weights * effects, axis=0) / np.sum(weights, axis=0)
    k = effects.shape[0]
    if k == 1:
        return fixed, np.sqrt(1.0 / np.sum(weights, axis=0)), np.zeros_like(fixed)
    q = np.sum(weights * (effects - fixed) ** 2, axis=0)
    c = np.sum(weights, axis=0) - np.sum(weights ** 2, axis=0) / np.sum(weights, axis=0)
    tau = np.maximum((q - (k - 1)) / np.maximum(c, 1e-12), 0.0)
    random_weights = 1.0 / (variance + tau)
    combined = np.sum(random_weights * effects, axis=0) / np.sum(random_weights, axis=0)
    standard_error = np.sqrt(1.0 / np.sum(random_weights, axis=0))
    return combined, standard_error, tau


def bh(pvalues):
    result = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if finite.size == 0:
        return result
    order = finite[np.argsort(pvalues[finite])]
    values = pvalues[order] * len(finite) / np.arange(1, len(finite) + 1)
    result[order] = np.minimum(np.minimum.accumulate(values[::-1])[::-1], 1)
    return result


def hierarchical_meta(section_tables: list[pd.DataFrame]):
    common = sorted(set.intersection(*(set(table.gene) for table in section_tables)))
    indexed = [table.set_index("gene").loc[common] for table in section_tables]
    usable = np.ones(len(common), dtype=bool)
    for table in indexed:
        usable &= (
            np.isfinite(table.section_effect.to_numpy())
            & np.isfinite(table.section_se.to_numpy())
            & (table.section_se.to_numpy() > 0)
        )
    common = np.asarray(common)[usable].tolist()
    if not common:
        raise RuntimeError("No genes had finite block-bootstrap uncertainty in every section.")
    embryo_results = []
    for embryo in sorted({table.embryo.iloc[0] for table in section_tables}):
        selected = [table.set_index("gene").loc[common] for table in section_tables if table.embryo.iloc[0] == embryo]
        effects = np.stack([table.section_effect for table in selected])
        errors = np.stack([table.section_se for table in selected])
        effect, standard_error, _ = combine_effects(effects, errors)
        embryo_results.append((embryo, effect, standard_error))
    effects = np.stack([item[1] for item in embryo_results])
    errors = np.stack([item[2] for item in embryo_results])
    effect, standard_error, tau = combine_effects(effects, errors)
    z = effect / standard_error
    pvalue = 2 * stats.norm.sf(np.abs(z))
    direction = np.mean(np.sign(np.stack([
        table.set_index("gene").loc[common].section_effect for table in section_tables
    ])) == np.sign(effect), axis=0)
    result = pd.DataFrame({
        "gene": common, "meta_effect": effect, "meta_se": standard_error,
        "meta_z": z, "conditional_p_value": pvalue,
        "conditional_fdr": bh(pvalue), "tau_squared": tau,
        "section_direction_consistency": direction,
        "section_n": len(section_tables), "embryo_n": len(embryo_results),
        "inference_scope": (
            "two_embryo_exploratory" if len(embryo_results) == 2
            else "single_embryo_exploratory" if len(embryo_results) == 1
            else "multi_embryo"
        ),
    })
    return result.sort_values("meta_z", ascending=False)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells, transport, certificates = load_valid_results(args.run_root)
    certificates.to_csv(args.output_dir / "certificate_scope.csv", index=False)
    summary = []
    candidates = [
        candidate for candidate in DEFAULT_CANDIDATES
        if args.candidates is None or candidate.name in args.candidates
    ]
    for candidate_index, candidate in enumerate(candidates):
        root = args.output_dir / candidate.name
        root.mkdir(exist_ok=True)
        consensus = consensus_for_candidate(cells, candidate)
        consensus.to_csv(root / "cell_consensus.csv.gz", index=False, compression="gzip")
        source_stage, target_stage = candidate.transition.split(" → ")
        stage = source_stage if candidate.side == "source" else target_stage
        section_tables = []
        for section_index, sample in enumerate(sorted(consensus["sample"].unique())):
            table = analyze_section(
                args.data_root / f"{stage}_{sample}.MOSTA.h5ad",
                consensus[consensus["sample"].eq(sample)], candidate,
                transport[candidate.transition], args.consensus_threshold,
                args.block_width_bins, args.bootstrap_replicates, args.minimum_bins,
                args.minimum_blocks, args.minimum_gene_total,
                np.random.default_rng(args.seed + candidate_index * 1009 + section_index),
            )
            if table is None:
                continue
            table["sample"] = sample
            table["embryo"] = embryo_from_sample(sample)
            table.to_csv(root / f"section_{sample}_deg.csv.gz", index=False, compression="gzip")
            section_tables.append(table)
        if not section_tables:
            summary.append({"candidate": candidate.name, "status": "no_eligible_sections"})
            continue
        meta = hierarchical_meta(section_tables)
        meta.to_csv(root / "hierarchical_meta_deg.csv", index=False)
        meta[["gene", "meta_z"]].to_csv(root / "gsea_rank.rnk", sep="\t", index=False, header=False)
        embryo_n = int(meta.embryo_n.iloc[0])
        summary.append({
            "candidate": candidate.name, "annotation": candidate.annotation,
            "transition": candidate.transition, "side": candidate.side,
            "status": "complete", "section_n": len(section_tables), "embryo_n": embryo_n,
            "inference_scope": meta.inference_scope.iloc[0], "tested_gene_n": len(meta),
        })
        (root / "analysis_design.json").write_text(json.dumps({
            **candidate.__dict__, "consensus_threshold": args.consensus_threshold,
            "block_width_bins": args.block_width_bins,
            "bootstrap_replicates": args.bootstrap_replicates,
            "excluded_transport_gene_n": len(transport[candidate.transition]),
            "primary_control": "retained_bins_within_same_annotation",
        }, indent=2), encoding="utf-8")
        print(summary[-1])
    pd.DataFrame(summary).to_csv(args.output_dir / "spatial_bin_deg_summary.csv", index=False)
    print(f"Spatial-bin DEG outputs: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
