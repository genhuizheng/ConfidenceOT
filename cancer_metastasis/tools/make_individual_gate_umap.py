"""One UMAP per pair, in the geometry that pair's gate was actually computed in.

The dataset-wide figures drew every pair onto one published embedding. That
embedding is not the one the gate used: each pair gets its own joint PCA over
its own two samples, and the gate is a threshold on a cost built from those
coordinates. A cell that looks misplaced on a dataset-wide UMAP may sit exactly
where its own pair's geometry puts it, and vice versa, so a per-pair embedding
is the only one in which the gate's split is a statement about what the method
saw.

Read from `joint_pca_coordinates.csv.gz`, which `02_run_pair.py` writes when
`--save-pairing-edges` is on. Nothing is recomputed: the coordinates are the
run's own, so this figure cannot disagree with the run about what the
representation was.

Three panels per pair, and the third is not decoration:

  (a) both sides on one embedding. If the primary and metastatic clouds do not
      overlap, every later panel is about a gate that had nothing to match.
  (b) the gate: primary cells retained or rejected, metastasis as a backdrop.
  (c) pre-equalisation depth. Depth still tracks the gate on this data -- a
      third of GSE180661's pairs and half of GSE271675's exceed the bound, in
      both directions -- so a reader has to be able to see whether this pair's
      split follows its depth gradient. Without panel (c), panel (b) invites a
      biological reading of something already known to be partly an artefact.

A contact sheet per dataset comes with the individual files, because 94
separate figures are not reviewable and the question "which pairs look wrong"
is asked of the set, not of one pair.

Usage:
  python cancer_metastasis/tools/make_individual_gate_umap.py OUT_DIR
      --dataset GSE180661=.../ot_GSE180661_rank256_ds_cos_20260921
      --predownsample-depth .../downsampled_GSE180661_20260914/predownsample_depth.csv.gz
"""

from __future__ import annotations

import argparse
import os

# Thread caps, set before anything imports numba. UMAP's nearest-neighbour
# search runs `joblib.Parallel(n_jobs=-1)` inside pynndescent, which on a
# 144-core node asks for 144 threads per call and accumulates across pairs:
# the third pair of a batch hit "can't start new thread" on a login node.
# Numba fixes its pool size at import, so this cannot be a command-line flag --
# it has to be in the environment before the import below, which is why it
# sits here rather than in main().
UMAP_THREADS = os.environ.get("CONFIDENCEOT_UMAP_THREADS", "4")
for _variable in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS",
                  "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, UMAP_THREADS)

from pathlib import Path  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

RETAINED = "#1f6f8b"
REJECTED = "#d1603d"
BACKDROP = "#c9c9c4"
PRIMARY = "#2f5d50"
METASTASIS = "#b07d2b"


def dataset_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=OT_ROOT")
    label, path = value.split("=", 1)
    return label, Path(path)


def marker_size(n: int) -> float:
    """Shrink the marker as the cloud fills up.

    A fixed size saturates a panel holding ten thousand cells and leaves a
    panel holding ninety nearly empty, and both failures hide the thing the
    panel exists to show.
    """
    return float(np.clip(2.6 * (4_000.0 / max(n, 1)) ** 0.35, 0.6, 14.0))


def embed(coordinates: np.ndarray, seed: int) -> np.ndarray:
    """UMAP of one pair's joint PCA, both sides together.

    Together, because the gate is a joint object: embedding the sides
    separately would place them on two unrelated coordinate systems and any
    apparent overlap or separation would be an artefact of the layout.

    ``n_neighbors`` is capped by the cloud: the default of 15 is larger than
    some pairs here have cells, and UMAP silently reduces it while warning,
    which is a warning nobody reads in a batch of ninety-four.
    """
    import joblib
    import umap

    n = len(coordinates)
    neighbours = int(np.clip(n - 1, 2, 15))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_neighbors=neighbours, min_dist=0.3,
                            random_state=seed, n_components=2)
        # The environment caps numba; this caps the joblib pool pynndescent
        # opens for its random-projection forest, which is the one that
        # actually ran out of threads. Both are needed: they are different
        # pools. `parallel_config` arrived in joblib 1.3 and
        # `parallel_backend` is what older versions have, so take whichever
        # exists rather than pinning a version for two lines.
        limit = (joblib.parallel_config if hasattr(joblib, "parallel_config")
                 else joblib.parallel_backend)
        keywords = ({"n_jobs": int(UMAP_THREADS)}
                    if hasattr(joblib, "parallel_config")
                    else {"backend": "threading",
                          "n_jobs": int(UMAP_THREADS)})
        with limit(**keywords):
            return reducer.fit_transform(
                np.asarray(coordinates, dtype=np.float32))


def draw(frame: pd.DataFrame, dataset: str, pair_id: str, out: Path,
         configuration: str) -> dict[str, float]:
    primary = frame[frame.side.eq("primary")]
    metastasis = frame[frame.side.eq("metastasis")]
    retained = primary[primary.retained]
    rejected = primary[~primary.retained]
    size = marker_size(len(frame))

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.5))
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#d8d8d2")

    axes[0].scatter(primary.x, primary.y, s=size, c=PRIMARY, linewidths=0,
                    label=f"primary ({len(primary):,})")
    axes[0].scatter(metastasis.x, metastasis.y, s=size, c=METASTASIS,
                    linewidths=0, label=f"metastasis ({len(metastasis):,})")
    axes[0].set_title("(a) both sides", loc="left", fontsize=11)
    axes[0].legend(loc="best", frameon=False, fontsize=8, markerscale=3)

    axes[1].scatter(metastasis.x, metastasis.y, s=size, c=BACKDROP,
                    linewidths=0)
    axes[1].scatter(rejected.x, rejected.y, s=size, c=REJECTED, linewidths=0)
    axes[1].scatter(retained.x, retained.y, s=size, c=RETAINED, linewidths=0)
    kept = len(retained) / max(len(primary), 1)
    axes[1].set_title(f"(b) gate: {kept:.0%} of primary retained", loc="left",
                      fontsize=11)
    axes[1].legend(handles=[
        Line2D([], [], marker="o", linestyle="", color=RETAINED,
               label=f"retained ({len(retained):,})", markersize=5),
        Line2D([], [], marker="o", linestyle="", color=REJECTED,
               label=f"rejected ({len(rejected):,})", markersize=5),
        Line2D([], [], marker="o", linestyle="", color=BACKDROP,
               label="metastasis", markersize=5),
    ], loc="best", frameon=False, fontsize=8)

    ratio = float("nan")
    if "depth" in frame and frame.depth.notna().any():
        depth = frame.depth.to_numpy(dtype=np.float64)
        finite = np.isfinite(depth) & (depth > 0)
        # Log scale and percentile clipping, because a few very deep cells
        # otherwise take the whole colour range and every other cell reads as
        # one shade, which would show no gradient where a gradient exists.
        values = np.full(len(frame), np.nan)
        values[finite] = np.log10(depth[finite])
        low, high = (np.nanpercentile(values, 2), np.nanpercentile(values, 98))
        points = axes[2].scatter(frame.x, frame.y, s=size, c=values,
                                 cmap="viridis", vmin=low, vmax=high,
                                 linewidths=0)
        bar = figure.colorbar(points, ax=axes[2], fraction=0.046, pad=0.02)
        bar.set_label("log10 pre-equalisation depth", fontsize=8)
        bar.ax.tick_params(labelsize=8)
        if len(retained) and len(rejected):
            retained_depth = np.median(retained.depth.dropna())
            rejected_depth = np.median(rejected.depth.dropna())
            if rejected_depth > 0:
                ratio = float(retained_depth / rejected_depth)
        axes[2].set_title(
            "(c) depth before equalisation"
            + (f": retained/rejected {ratio:.2f}x" if np.isfinite(ratio)
               else ""),
            loc="left", fontsize=11)
    else:
        axes[2].text(0.5, 0.5, "no pre-equalisation depth supplied",
                     ha="center", va="center", fontsize=9, color="#777770")
        axes[2].set_title("(c) depth", loc="left", fontsize=11)

    figure.suptitle(f"{dataset}  {pair_id}", fontsize=13,
                    fontweight="semibold", x=0.01, ha="left")
    figure.text(0.01, 0.005,
                f"UMAP of this pair's own joint PCA, the geometry its gate was "
                f"computed in. Configuration {configuration}. "
                f"Depth in (c) is each cell's depth before equalisation, which "
                f"is the covariate the gate must not track.",
                fontsize=7.5, color="#55554f", ha="left")
    figure.tight_layout(rect=(0, 0.03, 1, 0.94))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=170)
    plt.close(figure)
    return {"retained_fraction": kept, "depth_ratio": ratio}


def contact_sheet(panels: list[tuple[str, pd.DataFrame]], dataset: str,
                  out: Path) -> None:
    """Every pair's gate panel on one sheet, ordered as given."""
    if not panels:
        return
    columns = min(6, len(panels))
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns,
                                figsize=(2.3 * columns, 2.5 * rows),
                                squeeze=False)
    for index, axis in enumerate(axes.ravel()):
        axis.set_xticks([])
        axis.set_yticks([])
        if index >= len(panels):
            axis.axis("off")
            continue
        pair_id, frame = panels[index]
        primary = frame[frame.side.eq("primary")]
        metastasis = frame[frame.side.eq("metastasis")]
        size = marker_size(len(frame)) * 0.5
        axis.scatter(metastasis.x, metastasis.y, s=size, c=BACKDROP,
                     linewidths=0)
        axis.scatter(primary[~primary.retained].x, primary[~primary.retained].y,
                     s=size, c=REJECTED, linewidths=0)
        axis.scatter(primary[primary.retained].x, primary[primary.retained].y,
                     s=size, c=RETAINED, linewidths=0)
        kept = primary.retained.mean() if len(primary) else float("nan")
        # The pair id, shortened from the right: the informative part of these
        # names is the sample pair at the end, not the shared patient prefix.
        short = pair_id if len(pair_id) <= 26 else "..." + pair_id[-23:]
        axis.set_title(f"{short}\n{kept:.0%} kept", fontsize=6.5)
        for spine in axis.spines.values():
            spine.set_color("#e0e0da")
    figure.suptitle(f"{dataset}: every pair's gate, same colours as panel (b)",
                    fontsize=12, fontweight="semibold")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset", type=dataset_argument, action="append",
                        required=True, metavar="LABEL=OT_ROOT")
    parser.add_argument("--predownsample-depth", type=Path, action="append",
                        default=None,
                        help="one per --dataset, in the same order; omit to "
                             "draw without panel (c)")
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--configuration", default="rank256_ds_cos",
                        help="printed in the caption, so a figure names the "
                             "configuration that produced it")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--minimum-cells", type=int, default=20,
                        help="pairs smaller than this get no figure; a UMAP of "
                             "a dozen points is a decorative scatter plot")
    parser.add_argument("--pair", action="append", default=[],
                        help="restrict to these pair ids; default is all")
    parser.add_argument("--no-contact-sheet", action="store_true")
    args = parser.parse_args()
    depths_for = args.predownsample_depth or [None] * len(args.dataset)
    if len(depths_for) != len(args.dataset):
        raise SystemExit("--dataset and --predownsample-depth must pair up")

    summary, failures = [], []
    for (label, root), depth_path in zip(args.dataset, depths_for):
        depths = None
        if depth_path is not None:
            table = pd.read_csv(depth_path)
            needed = {"sample_id", "observation_id",
                      "predownsample_total_counts"}
            if not needed <= set(table.columns):
                raise SystemExit(f"{depth_path} is missing {needed - set(table.columns)}")
            table["sample_id"] = table["sample_id"].astype(str)
            table["observation_id"] = table["observation_id"].astype(str)
            depths = (table.drop_duplicates(["sample_id", "observation_id"])
                      [["sample_id", "observation_id",
                        "predownsample_total_counts"]])
        files = sorted(root.glob(
            f"*/{args.scope}/*/joint_pca_coordinates.csv.gz"))
        if not files:
            print(f"{label}: no joint_pca_coordinates.csv.gz under {root}. "
                  f"The run needs --save-pairing-edges for these to exist.")
            continue
        panels = []
        for path in files:
            pair_id = path.parent.parent.parent.name
            if args.pair and pair_id not in args.pair:
                continue
            coordinates = pd.read_csv(path)
            confidence_path = path.parent / "cell_confidence.csv"
            if not confidence_path.is_file():
                print(f"{label} {pair_id}: no cell_confidence.csv; skipped")
                continue
            confidence = pd.read_csv(confidence_path)
            confidence = confidence.loc[confidence["method"].eq(args.method)
                                        & confidence["side"].eq("source")]
            pca_columns = [c for c in coordinates.columns
                           if c.startswith("PC")]
            if len(pca_columns) < 2:
                print(f"{label} {pair_id}: fewer than two PCs; skipped")
                continue
            if len(coordinates) < args.minimum_cells:
                print(f"{label} {pair_id}: {len(coordinates)} cells, below "
                      f"--minimum-cells; skipped")
                continue
            # One pair failing must not lose the batch. Ninety-four pairs at
            # a few seconds each is a long enough run that an exception on
            # pair three would otherwise throw away everything after it, and
            # the failure that prompted this -- a thread limit -- is exactly
            # the kind that hits partway through.
            try:
                points = embed(coordinates[pca_columns].to_numpy(), args.seed)
                frame = pd.DataFrame({
                    "x": points[:, 0], "y": points[:, 1],
                    "side": coordinates["side"].astype(str),
                    "sample_id": coordinates["sample_id"].astype(str),
                    "observation_id": coordinates["observation_id"].astype(str),
                })
                gate = dict(zip(confidence["observation_id"].astype(str),
                                confidence["retained"].astype(bool)))
                # A metastatic cell has no source-side gate; False keeps it out
                # of the retained set, and it is only drawn as the backdrop.
                frame["retained"] = [gate.get(name, False)
                                     for name in frame.observation_id]
                if depths is not None:
                    frame = frame.merge(
                        depths, on=["sample_id", "observation_id"], how="left")
                    frame = frame.rename(
                        columns={"predownsample_total_counts": "depth"})
                out = args.output_dir / label / f"{pair_id}.png"
                measured = draw(frame, label, pair_id, out, args.configuration)
            except Exception as error:  # noqa: BLE001 - reported, not hidden
                failures.append({"dataset": label, "pair_id": pair_id,
                                 "error": f"{type(error).__name__}: {error}"})
                print(f"{label} {pair_id}: FAILED {type(error).__name__}: "
                      f"{error}")
                traceback.print_exc()
                continue
            summary.append({"dataset": label, "pair_id": pair_id,
                            "cells": len(frame), **measured,
                            "figure": str(out)})
            panels.append((pair_id, frame))
            print(f"{label} {pair_id}: {len(frame):,} cells, "
                  f"{measured['retained_fraction']:.0%} retained -> {out}")
        if panels and not args.no_contact_sheet:
            sheet = args.output_dir / f"{label}_contact_sheet.png"
            contact_sheet(panels, label, sheet)
            print(f"{label}: contact sheet -> {sheet}")
    if failures:
        print(f"\n{len(failures)} pair(s) failed:")
        for record in failures:
            print(f"  {record['dataset']} {record['pair_id']}: "
                  f"{record['error']}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(
            args.output_dir / "individual_umap_failures.csv", index=False)
    if not summary:
        raise SystemExit("no pair produced a figure")
    table = pd.DataFrame(summary)
    pd.set_option("display.width", 200)
    print()
    print(table.drop(columns=["figure"]).round(3).to_string(index=False))
    out = args.output_dir / "individual_umap_summary.csv"
    table.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
