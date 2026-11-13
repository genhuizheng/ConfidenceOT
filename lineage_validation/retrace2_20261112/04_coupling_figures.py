"""RETrace2: coupling heatmaps of homopolymer-OT and methylation-OT per sample pair (outputs of 02_, no refits).

    python -I lineage_validation/retrace2_20261112/04_coupling_figures.py --out results/retrace2_20261112 \
        --data results/retrace2_20261112/extracted/2025_RETrace2_MSH2_Female_Combined

Per pair: two heatmaps side by side, rows = source cells, columns = target cells, both in the authors' tree order;
colour = coupling normalised per target column (where each target's mass comes from). Gate strips beside the rows / above the
columns: grey = retained, black = rejected. One quantity per heatmap. Writes OUT/figures/r8_coupling_<pair>.{png,pdf}.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    a = p.parse_args()
    fo = a.out / "figures"; st.apply()
    from Bio import Phylo
    tree = Phylo.read(a.data / "results_10000targets_3000cpgs" / "phylo" / "2025_MSH2_female_combined.buildPhylo.newick-bootstrap.txt", "newick")
    rank = {t.name: i for i, t in enumerate(tree.get_terminals())}
    C = pd.read_csv(a.out / "prep" / "cells.tsv", sep="\t").set_index("cell")
    F = pd.read_csv(a.out / "ot" / "fits.tsv", sep="\t")
    for pair in dict.fromkeys(F.pair):
        s, t = pair.split(" -> ")
        fig = plt.figure(figsize=(7.2, 3.3))
        for k, (cost, title) in enumerate((("lineage", "homopolymer-OT"), ("methylation", "methylation-OT"))):
            z = np.load(a.out / "ot" / "couplings" / f"{s}__{t}__{cost}.npz")
            so = np.argsort([rank[c] for c in z["source"]]); to = np.argsort([rank[c] for c in z["target"]])
            P = z["coupling"][np.ix_(so, to)].astype(float); col = P.sum(0); Q = P / np.where(col > 0, col, np.nan)[None, :]
            ax = fig.add_axes([0.07 + k * 0.47, 0.12, 0.36, 0.66])
            im = ax.imshow(Q, cmap="Greys", vmin=0, vmax=np.nanpercentile(Q, 99), aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            gs, gt = z["source_gate"][so], z["target_gate"][to]
            ax.set_xlabel(f"target cells ({t.replace('_', ' ')}, tree order)", fontsize=st.FS_MIN)
            ax.set_ylabel(f"source cells ({s.replace('_', ' ')})", fontsize=st.FS_MIN, labelpad=10)
            gx = fig.add_axes([0.07 + k * 0.47 - 0.015, 0.12, 0.01, 0.66]); gx.imshow(gs[:, None].astype(float), cmap=ListedColormap([st.INK, "#C9CDD1"]), vmin=0, vmax=1, aspect="auto"); gx.axis("off")
            gy = fig.add_axes([0.07 + k * 0.47, 0.79, 0.36, 0.02]); gy.imshow(gt[None, :].astype(float), cmap=ListedColormap([st.INK, "#C9CDD1"]), vmin=0, vmax=1, aspect="auto"); gy.axis("off")
            r = F[(F.pair == pair) & (F.cost == cost)].iloc[0]
            ax.set_title(f"{title}: retained {r.retained_source:.0%} / {r.retained_target:.0%}" + ("" if bool(r.m4e_inference_valid) else " (calibration flagged)"),
                         fontsize=st.FS_TITLE, pad=12)
            cb = fig.colorbar(im, cax=fig.add_axes([0.07 + k * 0.47 + 0.37, 0.3, 0.012, 0.3])); cb.ax.tick_params(labelsize=st.FS_MIN)
            cb.outline.set_visible(False)
        fig.text(0.01, 0.98, f"{pair.replace('_', ' ')}: where each target cell's coupling mass comes from (column-normalised); "
                 "strips: grey = retained, black = rejected", fontsize=st.FS_MIN, va="top")
        name = pair.replace(" -> ", "__to__").replace("_", "").replace("-", "")
        for ext in ("png", "pdf"):
            fig.savefig(fo / f"r8_coupling_{name}.{ext}", dpi=400 if ext == "png" else None)
        plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
