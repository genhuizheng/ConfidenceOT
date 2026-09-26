"""Turn one generated technical condition into its three biological cases.

One invocation handles one ``(replicate, technical level)``. The level's counts
were drawn by Splatter for that level alone: the three levels are independent
realizations of the same biological parameters, not one realization passed
through a chain of transforms. That matters for what a difference between
levels means. Derived from a single draw, part of the difference is the same
cells being altered; drawn independently, the whole of it is the observation
setting, which is what the benchmark claims to vary.

**What each level changes, and where.**

``L0_matched`` leaves the target alone. ``L1_depth`` thins the target's reads.
``L2_depth_dropout`` thins them and was generated with Splatter's own per-batch
dropout on the target side, so the extra detection loss is part of the
simulation rather than a mask applied here. L2 is not L1 with more dropout: it
is its own draw, with its own observation setting.

Thinning is binomial, the exact likelihood of sequencing the same library less
deeply: every read survives independently with probability theta. It stays
here, and is the only post-hoc operation on counts, because Splatter has one
global library size and no per-batch override.

There is no level that holds one readout while moving the other. At a fixed
library size nothing can change how many genes are detected except changing
what the counts land on, and that is a change to the cell rather than to the
measurement.

**The cases.** ``all_shared`` keeps every group on both sides.
``population_lost`` removes the last group from the target, so the source cells
of that group have no match. ``population_emerged`` removes it from the source
instead. Both directions are built, because the gate is called on both sides
and a benchmark that only ever asks one side to reject cannot see half of what
it claims to measure.

Ratios are measured after the fact and written into the manifest. The nominal
theta is recorded too, but the axis a result is plotted against is the realised
one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io as scipy_io
from scipy import sparse

LEVELS = ("L0_matched", "L1_depth", "L2_depth_dropout")
THINNED_LEVELS = ("L1_depth", "L2_depth_dropout")
CASES = ("all_shared", "population_lost", "population_emerged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=Path, required=True,
                        help="Directory written by 10_generate_splatter.R for "
                             "this replicate and this level")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-cells", type=int, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--level", required=True, choices=LEVELS)
    parser.add_argument("--theta", type=float, default=0.50,
                        help="Read survival probability for the target at the "
                             "thinned levels; the realised depth ratio is "
                             "measured, not assumed")
    parser.add_argument("--unmatched-group", default=None,
                        help="Group removed in the lost and emerged cases; "
                             "the last group by name when not given")
    parser.add_argument("--rank-top-n", type=int, default=256,
                        help="Recorded so the assertion that the rank cut "
                             "stays below the sparsest cell can be checked "
                             "per condition")
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def read_generated(root: Path) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame,
                                        dict]:
    """Genes x cells from R, returned cells x genes, with its record."""
    matrix = scipy_io.mmread(root / "counts.mtx")
    counts = np.asarray(sparse.csr_matrix(matrix).todense(), dtype=np.float64).T
    cells = pd.read_csv(root / "cells.csv")
    genes = pd.read_csv(root / "genes.csv")
    record = json.loads((root / "generation.json").read_text(encoding="utf-8"))
    if counts.shape != (len(cells), len(genes)):
        raise ValueError(
            f"counts {counts.shape} disagrees with cells {len(cells)} and "
            f"genes {len(genes)}")
    return counts, cells, genes, record


def thin(counts: np.ndarray, theta: float,
         rng: np.random.Generator) -> np.ndarray:
    """Sequence the same library less deeply: every read survives at theta."""
    if theta >= 1.0:
        return counts
    return rng.binomial(counts.astype(np.int64), theta).astype(np.float64)


def readouts(counts: np.ndarray) -> dict:
    total = counts.sum(axis=1)
    detected = (counts > 0).sum(axis=1)
    return {"median_nCount": float(np.median(total)),
            "median_nFeature": float(np.median(detected)),
            "min_nFeature": int(detected.min()),
            "q01_nFeature": float(np.quantile(detected, 0.01))}


def write_side(path: Path, counts: np.ndarray, cells: pd.DataFrame,
               genes: pd.DataFrame, sample: str) -> None:
    import anndata as ad

    frame = cells.reset_index(drop=True).copy()
    frame["sample_id"] = sample
    frame.index = pd.Index(
        [f"{sample}:{name}" for name in frame["cell_id"]], dtype=str)
    var = genes.set_index(pd.Index(genes["gene_id"].astype(str)))
    data = ad.AnnData(X=sparse.csr_matrix(counts), obs=frame, var=var)
    # The key the pipeline reads is "Expression matrix type"; anything else is
    # recorded as unknown, which is harmless for a label-driven run but wrong
    # in the provenance.
    data.uns["metadata"] = {"Expression matrix type": "raw counts",
                            "source": "splatter benchmark"}
    path.parent.mkdir(parents=True, exist_ok=True)
    data.write_h5ad(path, compression="gzip")


def main() -> None:
    args = parse_args()
    counts, cells, genes, record = read_generated(args.generated)
    if record.get("technical_level") != args.level:
        raise ValueError(
            f"--level {args.level!r} but the generation record says "
            f"{record.get('technical_level')!r}; the levels are independent "
            f"realizations, so this would mix two of them")
    rng = np.random.default_rng(
        args.seed + 7919 * args.replicate + 104729 * LEVELS.index(args.level))

    batches = sorted(cells["batch"].unique())
    if len(batches) != 2:
        raise ValueError(f"expected two batches, found {batches}")
    source_mask = (cells["batch"] == batches[0]).to_numpy()
    target_mask = ~source_mask
    group_names = sorted(cells["group"].unique())
    unmatched = args.unmatched_group or group_names[-1]
    if unmatched not in group_names:
        raise ValueError(f"{unmatched!r} is not one of {group_names}")

    source = counts[source_mask]
    target = counts[target_mask]
    theta = args.theta if args.level in THINNED_LEVELS else 1.0
    target = thin(target, theta, rng)
    source_groups = cells.loc[source_mask, "group"].to_numpy()
    target_groups = cells.loc[target_mask, "group"].to_numpy()

    rows = []
    for case in CASES:
        keep_source = np.ones(source.shape[0], dtype=bool)
        keep_target = np.ones(target.shape[0], dtype=bool)
        if case == "population_lost":
            keep_target = target_groups != unmatched
        elif case == "population_emerged":
            keep_source = source_groups != unmatched

        pair_id = f"N{args.n_cells}_rep{args.replicate}_{args.level}_{case}"
        directory = args.out / pair_id
        source_path = directory / "source.h5ad"
        target_path = directory / "target.h5ad"
        write_side(source_path, source[keep_source],
                   cells.loc[source_mask].loc[keep_source], genes,
                   f"{pair_id}_source")
        write_side(target_path, target[keep_target],
                   cells.loc[target_mask].loc[keep_target], genes,
                   f"{pair_id}_target")

        # Which cells have no match, by construction rather than by
        # inference: the group label came out of the simulator. The side that
        # rejects is the side that still *has* the group, which is the
        # opposite of the side the removal was applied to.
        source_kept = source_groups[keep_source]
        target_kept = target_groups[keep_target]
        source_reject = (source_kept == unmatched
                         if case == "population_lost"
                         else np.zeros(source_kept.size, dtype=bool))
        target_reject = (target_kept == unmatched
                         if case == "population_emerged"
                         else np.zeros(target_kept.size, dtype=bool))
        truth = pd.concat([
            pd.DataFrame({"side": "source", "group": source_kept,
                          "should_reject": source_reject}),
            pd.DataFrame({"side": "target", "group": target_kept,
                          "should_reject": target_reject}),
        ], ignore_index=True)
        truth.to_csv(directory / "truth.csv.gz", index=False)

        source_read = readouts(source[keep_source])
        target_read = readouts(target[keep_target])
        measured = {
            "R_nCount": target_read["median_nCount"] / source_read["median_nCount"],
            "R_nFeature": target_read["median_nFeature"] / source_read["median_nFeature"],
        }
        valid = min(source_read["min_nFeature"],
                    target_read["min_nFeature"]) > args.rank_top_n
        (directory / "readouts.json").write_text(json.dumps(
            {"source": source_read, "target": target_read,
             "measured": measured, "theta_nominal": theta,
             "technical_level": args.level,
             "dropout_from_splatter": record.get("dropout_type"),
             "dropout_mid_target": record.get("dropout_mid_target"),
             "generation_seed": record.get("seed"),
             "unmatched_group": unmatched,
             "rank_top_n": args.rank_top_n,
             "rank_cut_valid": bool(valid)},
            indent=2), encoding="utf-8")

        rows.append({
            "pair_id": pair_id,
            "dataset_id": "splatter_technical",
            "patient_id": f"N{args.n_cells}_rep{args.replicate}",
            "source_sample": f"{pair_id}_source",
            "target_sample": f"{pair_id}_target",
            "source_h5ad": str(source_path),
            "target_h5ad": str(target_path),
            "eligible": True,
            "n_cells": args.n_cells,
            "replicate": args.replicate,
            "technical_level": args.level,
            "biological_case": case,
            "unmatched_group": unmatched,
            "theta_nominal": theta,
            "dropout_mid_target": record.get("dropout_mid_target"),
            "generation_seed": record.get("seed"),
            "R_nCount": round(measured["R_nCount"], 4),
            "R_nFeature": round(measured["R_nFeature"], 4),
            "min_nFeature_source": source_read["min_nFeature"],
            "min_nFeature_target": target_read["min_nFeature"],
            "rank_top_n": args.rank_top_n,
            "rank_cut_valid": bool(valid),
            "truth_csv": str(directory / "truth.csv.gz"),
        })

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "manifest.csv", index=False)
    (args.out / "CONSTRUCTION_DONE").write_text("SUCCESS\n", encoding="utf-8")
    invalid = table.loc[~table["rank_cut_valid"], "pair_id"].tolist()
    print(f"wrote {len(table)} conditions for {args.level} to "
          f"{args.out / 'manifest.csv'}")
    if invalid:
        print("outside valid rank-encoding regime (min nFeature <= "
              f"{args.rank_top_n}): {invalid}")


if __name__ == "__main__":
    main()
