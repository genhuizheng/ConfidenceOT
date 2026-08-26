"""Compose exported spatial-consensus panels into a wide PPT-ready figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("panel_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--transition", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    source = sorted(args.panel_dir.glob(f"{args.prefix}__source__*.png"))
    target = sorted(args.panel_dir.glob(f"{args.prefix}__target__*.png"))
    if not source or not target:
        raise FileNotFoundError("Could not find both source and target panel PNGs.")

    columns = max(len(source), len(target))
    figure, axes = plt.subplots(2, columns, figsize=(16.5, 8.3), squeeze=False)
    for row, panels in enumerate((source, target)):
        for column in range(columns):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(panels):
                axis.imshow(mpimg.imread(panels[column]))

    normalizer = mpl.colors.Normalize(vmin=0, vmax=1)
    scalar = mpl.cm.ScalarMappable(norm=normalizer, cmap="magma")
    colorbar_axis = figure.add_axes([0.925, 0.24, 0.014, 0.52])
    figure.colorbar(
        scalar,
        cax=colorbar_axis,
        label="Cross-section rejection\nfrequency (valid pairs)",
    )
    figure.suptitle(
        f"Spatial consensus of developmental exclusion — {args.transition}",
        fontsize=20,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.015, right=0.90, bottom=0.025, top=0.93,
        hspace=0.08, wspace=0.035,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=args.dpi, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    print(f"Wide spatial-consensus figure: {args.output.resolve()}")


if __name__ == "__main__":
    main()
