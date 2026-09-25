"""Check a rank-encoding cut against the cells it will actually be applied to.

``rank_value`` keeps each cell's top ``rank_top_n`` genes and lets the value
fall from 1.0 to 1/rank_top_n across them.  The fixed cut is what makes the
encoding depth-invariant, but ``rank_value_encode`` keeps
``min(top_n, detected)`` genes, so a cell detecting fewer genes than the cut
gets a shorter vector supported on fewer coordinates.  The cut therefore has to
stay below the shallowest cell's detected-gene count, or the encoding
manufactures its own low-content artefact -- the very thing it was chosen to
remove.

This matters most on depth-equalised counts, which is where the cut is least
safe: every cell has been subsampled to a low quantile of the pooled depth, so
the detected-gene distribution sits lower than in the raw object the cut was
first reasoned about.  The depth screen ran at 256 on simulated cells with
4,000 genes and near-uniform detection; that is not evidence about these cells.

Reported per pair, on the *gene intersection* the run uses and after the same
annotation scoping, because that is the matrix the encoder sees.  The number to
read is the minimum: a ``safe_top_n`` below the requested cut means some cell is
encoded short.

Under ``--cost cosine`` a short vector's smaller norm is normalised away, so
the magnitude half of the artefact goes; the support difference does not.  Read
a failure here as "this cut is doing something the screen did not test", not as
"the run is invalid".

Usage:
  python cancer_metastasis/tools/audit_rank_top_n.py MANIFEST.csv
      --rank-top-n 256 --analysis-scope malignant
      --include-annotation Malignant.cells --out audit.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cancer_metastasis.common import (  # noqa: E402
    cell_qc_table,
    expression_matrix,
    gene_keys,
    load_exact_side,
)


def scope_labels(data) -> np.ndarray:
    for column in ("cell_type", "annotation", "celltype", "cell_type_final"):
        if column in data.obs:
            return data.obs[column].astype(str).to_numpy()
    return np.repeat("unannotated", data.n_obs)


def paths_for(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def detected_per_cell(matrix) -> np.ndarray:
    """Nonzero genes per cell, which is what bounds the rank cut."""
    csr = sparse.csr_matrix(matrix)
    csr.eliminate_zeros()
    return np.diff(csr.indptr).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("--rank-top-n", type=int, default=None)
    parser.add_argument(
        "--preprocessing", default=None, metavar="LABEL",
        help="Take the cut from a configuration label instead, as the run "
             "itself does: 'rank256_ds_cos' audits a cut of 256. The label is "
             "parsed by confidenceot.Preprocessing, the same parser the pair "
             "runner uses, so the audited cut and the applied cut cannot "
             "disagree -- which is the whole point of auditing it.",
    )
    parser.add_argument("--analysis-scope", choices=("all", "malignant"),
                        default="all")
    parser.add_argument("--include-annotation", action="append", default=[])
    parser.add_argument("--malignant-column", default=None, metavar="COLUMN",
                        help="Select the malignant compartment by this obs column equalling --malignant-value, instead of by the deposit's own labels. One rule for every deposit.")
    parser.add_argument("--malignant-value", default="malignant")
    parser.add_argument("--cell-qc", action="store_true")
    parser.add_argument("--minimum-total-counts", type=int, default=0)
    parser.add_argument("--minimum-detected-genes", type=int, default=0)
    parser.add_argument("--maximum-mitochondrial-percent", type=float,
                        default=100.0)
    parser.add_argument("--index", type=int, action="append", default=[],
                        help="Audit only these manifest rows; default is all")
    parser.add_argument(
        "--max-fraction-short", type=float, default=0.01,
        help="Exit non-zero when more than this fraction of cells in any pair "
             "detect fewer genes than the cut. A strict minimum is the wrong "
             "gate: one shallow cell in ten thousand changes nothing, while a "
             "third of them makes the encoding partly a detection-breadth "
             "readout, which is the residual that survives subsampling to a "
             "common total. Set to 1.0 to report without gating.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if (args.analysis_scope == "malignant" and not args.include_annotation
            and not args.malignant_column):
        raise SystemExit("malignant scope requires at least one "
                         "--include-annotation or --malignant-column")
    if args.preprocessing is not None:
        if args.rank_top_n is not None:
            raise SystemExit("--preprocessing carries the cut; drop "
                             "--rank-top-n")
        from confidenceot import Preprocessing

        configuration = Preprocessing.from_label(args.preprocessing)
        if not configuration.normalisation.startswith("rank"):
            print(f"{args.preprocessing} is not a rank encoding, so there is "
                  f"no cut to audit. Nothing to check.")
            return
        args.rank_top_n = configuration.rank_top_n
        print(f"auditing {args.preprocessing}: cut {args.rank_top_n}")
    if args.rank_top_n is None:
        args.rank_top_n = 256

    manifest = pd.read_csv(args.manifest_csv)
    indices = args.index or list(range(len(manifest)))
    rows = []
    for index in indices:
        row = manifest.iloc[index]
        if "eligible" in row and not bool(row["eligible"]):
            continue
        pair_id = str(row["pair_id"])
        sides = {}
        for side in ("source", "target"):
            data = load_exact_side(paths_for(row, side),
                                   str(row[f"{side}_sample"]))
            if args.analysis_scope == "malignant":
                if args.malignant_column:
                    if args.malignant_column not in data.obs:
                        raise KeyError(
                            f"obs has no {args.malignant_column!r} column; use "
                            f"--include-annotation for files that predate the "
                            f"uniform call")
                    keep = (data.obs[args.malignant_column].astype(str).to_numpy()
                            == args.malignant_value)
                else:
                    keep = np.isin(scope_labels(data), args.include_annotation)
                data = data[keep].copy()
            if args.cell_qc and data.n_obs:
                table = cell_qc_table(
                    data,
                    minimum_total_counts=args.minimum_total_counts,
                    minimum_detected_genes=args.minimum_detected_genes,
                    maximum_mitochondrial_percent=(
                        args.maximum_mitochondrial_percent),
                )
                data = data[table["qc_pass"].to_numpy()].copy()
            sides[side] = data
        if not sides["source"].n_obs or not sides["target"].n_obs:
            print(f"{pair_id}: a side has no cells in scope; skipped")
            continue
        # The encoder runs on the stacked intersection, so the cut is bounded
        # by detection over the shared genes, not over each object's own.
        source_keys = pd.Index(gene_keys(sides["source"]).astype(str))
        target_keys = pd.Index(gene_keys(sides["target"]).astype(str))
        common = source_keys.intersection(target_keys)
        detected = []
        for side in ("source", "target"):
            keys = source_keys if side == "source" else target_keys
            position = keys.get_indexer(common)
            matrix = expression_matrix(sides[side])[:, position[position >= 0]]
            detected.append(detected_per_cell(matrix))
        joint = np.concatenate(detected)
        quantiles = np.quantile(joint, [0.0, 0.001, 0.01, 0.05, 0.50])
        # The actionable cut, as opposed to the strictest one: the largest cut
        # that leaves at most the tolerated fraction of cells encoded short.
        tolerated = int(np.quantile(joint, min(args.max_fraction_short, 1.0)))
        rows.append({
            "index": index,
            "pair_id": pair_id,
            "cells": int(joint.size),
            "common_genes": int(len(common)),
            "detected_min": int(joint.min()),
            "detected_p0.1": float(quantiles[1]),
            "detected_p1": float(quantiles[2]),
            "detected_p5": float(quantiles[3]),
            "detected_median": float(quantiles[4]),
            "cells_below_cut": int((joint < args.rank_top_n).sum()),
            "fraction_below_cut": float((joint < args.rank_top_n).mean()),
            "safe_top_n": int(joint.min()),
            "tolerated_top_n": tolerated,
        })
        del sides
        print(f"{pair_id}: {rows[-1]['cells']:,} cells, detected min "
              f"{rows[-1]['detected_min']:,}, "
              f"{rows[-1]['fraction_below_cut']:.1%} below a cut of "
              f"{args.rank_top_n}")
    if not rows:
        raise SystemExit("no auditable pairs")
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(table.to_string(index=False))
    worst = int(table["safe_top_n"].min())
    tolerated = int(table["tolerated_top_n"].min())
    affected = float(table["fraction_below_cut"].max())
    print()
    print(f"requested rank_top_n                : {args.rank_top_n}")
    print(f"largest cut every cell can fill     : {worst}")
    print(f"largest cut within the tolerance    : {tolerated} "
          f"(at most {args.max_fraction_short:.1%} short)")
    print(f"worst pair's fraction encoded short : {affected:.2%}")
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    if worst >= args.rank_top_n:
        print("VERDICT ok: no cell is encoded short at this cut.")
        return
    if affected <= args.max_fraction_short:
        print(f"VERDICT ok: {affected:.2%} of cells are encoded short, within "
              f"the tolerated {args.max_fraction_short:.1%}.")
        return
    print(f"VERDICT short: up to {affected:.2%} of cells detect fewer genes "
          f"than the cut, so their vectors are supported on fewer "
          f"coordinates and the encoding is partly a detection-breadth "
          f"readout. Lower the cut to {tolerated} (or {worst} for no short "
          f"cell at all), or raise --max-fraction-short to record deliberately "
          f"that this run differs from the screened configuration here.")
    raise SystemExit(3)


if __name__ == "__main__":
    main()
