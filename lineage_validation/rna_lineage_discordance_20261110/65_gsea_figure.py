"""Hallmark GSEA panel of the discordance contrasts: mouse-mean NES (depth-thinned version, deg_gsea_v1_ds) of all
50 sets for KP / Quinn x divergent / convergent; the raw-count NES is in hallmark_consistency.tsv beside it.

    python lineage_validation/rna_lineage_discordance_20261110/65_gsea_figure.py --root results/rna_lineage_discordance_20261110
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "realdata_benchmark" / "v3_grid" / "figures"))
import nbt_style as st  # noqa: E402
from pdx_gsea_figures import CATEGORIES, CM, PATHWAYS, pretty  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    T = pd.read_csv(a.root / "deg_gsea_v1" / "hallmark_consistency.tsv", sep="\t")
    cols = [("KP", "divergent"), ("Quinn", "divergent"), ("KP", "convergent"), ("Quinn", "convergent")]
    M = np.full((len(cols), 50), np.nan); N = []
    for i, (ds, k) in enumerate(cols):
        g = T[(T.dataset == ds) & (T.kind == k)].set_index("term")
        M[i] = g.nes_ds.reindex(PATHWAYS).to_numpy()
        N.append(f"{ds} {k} ({int(g.contrasts.max())} contrasts, {int(g.mice.max())} mice)")
    st.apply()
    sh = st.Sheet(8.6, 2.6); ax = sh.ax(2.3, 1.25, 50 * 0.12, len(cols) * 0.2)
    im = ax.pcolormesh(np.arange(51) - 0.5, np.arange(len(cols) + 1) - 0.5, np.ma.masked_invalid(M), cmap=CM,
                       norm=TwoSlopeNorm(0, -2, 2), edgecolors="white", lw=0.3)
    ax.set_ylim(len(cols) - 0.5, -0.5); ax.set_yticks(range(len(cols))); ax.set_yticklabels(N, fontsize=st.FS_MIN)
    ax.set_xticks(range(50)); ax.set_xticklabels([pretty(x) for x in PATHWAYS], rotation=60, ha="right", rotation_mode="anchor", fontsize=st.FS_MIN)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    x0 = 0
    for cat, m in CATEGORIES:
        x1 = x0 + len(m) - 1
        ax.text((x0 + x1) / 2, -0.75, cat.replace("\n", " "), ha="center", va="bottom", fontsize=st.FS_MIN, clip_on=False)
        if x0:
            ax.axvline(x0 - 0.5, color="white", lw=1.2)
        x0 = x1 + 1
    cax = sh.ax(2.3 + 50 * 0.12 + 0.15, 1.25, 0.08, len(cols) * 0.2)
    cb = sh.fig.colorbar(im, cax=cax); cb.set_ticks([-2, -1, 0, 1, 2]); cb.ax.tick_params(labelsize=st.FS_MIN); cb.set_label("NES", fontsize=st.FS_MIN)
    sh.text(0.05, 2.55, "Hallmark NES, discordant vs concordant cells of the same lineage and site (mouse mean; depth-thinned counts; "
            "orange = higher in discordant cells)", fontsize=st.FS_TITLE, va="top")
    out = a.root / "figures" / "l5best"; sh.save(out, "f7_gsea_discordant")


if __name__ == "__main__":
    main()
