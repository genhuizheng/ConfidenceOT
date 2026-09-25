"""Define the problem with UMAPs: what the six scenarios are, and what corrupts them.

Two things a rejection number cannot say and a picture can.

**What the method is being asked.** Each scenario is a source pool and a target
pool drawn from the same seven populations, with one structured difference.
Source and target are embedded *together*, then drawn as two panels on one set
of axes, so a population sits in the same place on both sides and an arrow
between the panels is a real displacement rather than a line between two
unrelated coordinate systems. Two independent UMAPs cannot be compared this
way at all: the same population lands somewhere arbitrary in each, and an
arrow drawn between them is decoration.

The arrow is the claim. A population present on both sides gets one, and its
length is how far that population moved. A population with no counterpart gets
no arrow and a mark instead, and that absence is exactly what the method is
supposed to find:

* **S0 clean movement** -- every population is on both sides and B has an
  expression module added, so B moves a long way. Nothing should be rejected.
  This is the control that separates *moved* from *gone*, and a method that
  rejects a displaced population fails here while looking sensitive elsewhere.
* **S1 extinction** -- source A has no target. Reject source A.
* **S2 emergence** -- target G has no source. Reject target G.
* **S3 source outlier** -- source O has no target. Reject source O.
* **S4 bifurcation** -- target B splits into B1 and B2, each with its own
  module. One arrow in, two out. Nothing should be rejected: both descendants
  have an ancestor.
* **S5 abundance shift** -- the same populations on both sides in different
  proportions. Nothing should be rejected; only the arrow widths change.

**What corrupts it.** The second figure embeds one scenario under two
preprocessings and colours the same cells by three nuisances -- total counts,
detected genes, and dropout rate. Under log CPM the cells arrange along the
nuisance; the question the preprocessing round answers is whether they still
do afterwards. Depth and breadth are not the same quantity and neither is
dropout: breadth is how many genes were seen at all, dropout is the fraction
of genes a cell should express and did not.

**Counts.** ``--splatter-root`` reads the R generator's output
(``<scenario>/n_XXXX/rep_XX/{counts.mtx,cells.csv,genes.tsv}``). Without it the
script builds the same structure itself -- seven groups from one gamma-Poisson
pool, the same per-scenario quotas, the same added modules -- so the layout can
be settled before the real counts are in hand. The figure says which it used.

Usage:

    python scripts/make_scenario_umap.py --out benchmark_results/scenario_umap
    python scripts/make_scenario_umap.py --splatter-root /path/to/pools \\
        --n 1000 --replicate 1 --out benchmark_results/scenario_umap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import ConnectionPatch  # noqa: E402

SCENARIOS = ("S0_clean_movement", "S1_extinction", "S2_emergence",
             "S3_source_outlier", "S4_bifurcation", "S5_abundance_shift")

TITLES = {
    "S0_clean_movement": "S0  control",
    "S1_extinction": "S1  extinction",
    "S2_emergence": "S2  emergence",
    "S3_source_outlier": "S3  source outlier",
    "S4_bifurcation": "S4  bifurcation",
    "S5_abundance_shift": "S5  abundance shift",
}

READING = {
    "S0_clean_movement": "Every population is on both sides. Reject nothing.",
    "S1_extinction": "A has no target. Reject source A.",
    "S2_emergence": "G has no source. Reject target G.",
    "S3_source_outlier": "O has no target. Reject source O.",
    "S4_bifurcation": "B splits in two. Reject nothing.",
    "S5_abundance_shift": "Proportions change. Reject nothing.",
}

BASE = ("A", "B", "C", "D", "E", "F", "G")
STYLE = {
    "figure.dpi": 130, "savefig.dpi": 400, "font.size": 8,
    "axes.titlesize": 8.5, "axes.labelsize": 7.5,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
}
PALETTE = {
    "A": "#c0392b", "B": "#2e86c1", "B1": "#2e86c1", "B2": "#5dade2",
    "C": "#28b463", "D": "#b7950b", "E": "#7d3c98", "F": "#5d6d7e",
    "G": "#d35400", "O": "#c0392b",
}
C_REJECT = "#c0392b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splatter-root", type=Path, default=None,
                        help="Output root of generate_splatter_population_benchmark.R")
    parser.add_argument("--n", type=int, default=800,
                        help="Cells per condition")
    parser.add_argument("--replicate", type=int, default=1)
    parser.add_argument("--genes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--umap-min-dist", type=float, default=0.6,
                        help="Higher packs each cluster less tightly and "
                             "leaves less empty field between them")
    parser.add_argument("--scenarios", default=None,
                        help="Comma-separated subset, e.g. "
                             "S0_clean_movement,S1_extinction,S2_emergence")
    parser.add_argument("--layout", choices=("joint", "paired", "schematic",
                                             "shared"),
                        default="joint",
                        help="joint: one embedding per scenario. paired: two "
                             "panels on that embedding. schematic: drawn, no "
                             "counts.")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


# --------------------------------------------------------------------------
# Counts


def gamma_poisson_pool(n_cells: int, n_genes: int, groups: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Seven populations from one gamma-Poisson pool, as splatSimulateGroups does.

    Mirrors the R generator's structure rather than its exact draws: one shared
    gene-mean vector, per-group differential expression on 12% of genes, a
    per-cell library factor, Poisson sampling. The point of matching the
    structure is that the scenario layout and the figure can be settled before
    the real counts are in hand; the point of not claiming to match the draws
    is that this is not Splatter and should not be read as it.
    """
    base_mean = rng.gamma(shape=0.6, scale=4.0, size=n_genes) + 0.15
    assignment = rng.integers(0, groups, size=n_cells)
    factors = np.ones((groups, n_genes))
    for group in range(groups):
        affected = rng.random(n_genes) < 0.12
        factors[group, affected] = np.exp(
            rng.normal(0.0, 0.9, size=int(affected.sum())))
    library = rng.lognormal(mean=0.0, sigma=0.25, size=n_cells)
    rate = base_mean[None, :] * factors[assignment] * library[:, None]
    return rng.poisson(rate).astype(np.float64), assignment


def add_module(counts: np.ndarray, rows: np.ndarray, genes: np.ndarray,
               rate: float, rng: np.random.Generator) -> None:
    """Displace a population by adding expression on a gene module, in place."""
    if rows.size == 0 or genes.size == 0:
        return
    block = counts[np.ix_(rows, genes)]
    # Scaled by the module's own mean expression so the added signal is a
    # multiple of what those genes already carry, which is what makes the shift
    # survive log CPM and gene scaling. A fixed additive constant washes out
    # under the row normalisation and the population does not move at all.
    counts[np.ix_(rows, genes)] = block + rng.poisson(
        np.maximum(block.mean(axis=0, keepdims=True), 1.0) * rate * 6.0,
        size=block.shape)


def quotas(scenario: str, n: int) -> tuple[dict[str, int], dict[str, int]]:
    """Cells per population on each side, matching the R generator's tables."""
    def even(names: tuple[str, ...]) -> dict[str, int]:
        share = n // len(names)
        counts = {name: share for name in names}
        counts[names[0]] += n - share * len(names)
        return counts

    six = BASE[:6]
    if scenario == "S1_extinction":
        return even(six), even(six[1:])
    if scenario == "S2_emergence":
        return even(six), even(BASE)
    if scenario == "S3_source_outlier":
        return even(six + ("O",)), even(six)
    if scenario == "S5_abundance_shift":
        heavy = {"A": 0.30, "B": 0.25, "C": 0.20, "D": 0.13, "E": 0.07, "F": 0.05}
        light = {"A": 0.05, "B": 0.07, "C": 0.13, "D": 0.20, "E": 0.25, "F": 0.30}
        return ({k: int(round(v * n)) for k, v in heavy.items()},
                {k: int(round(v * n)) for k, v in light.items()})
    return even(six), even(six)


def synthesise(scenario: str, n: int, n_genes: int, seed: int
               ) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    source_quota, target_quota = quotas(scenario, n)
    names = sorted(set(source_quota) | set(target_quota))
    # O is G under another name, exactly as the R generator renames it.
    pool_names = [name if name != "O" else "G" for name in names]
    index = {name: BASE.index(pool) for name, pool in zip(names, pool_names)}

    total = sum(source_quota.values()) + sum(target_quota.values())
    counts, assignment = gamma_poisson_pool(total * 2, n_genes, len(BASE), rng)

    rows, condition, population = [], [], []
    for side, quota in (("source", source_quota), ("target", target_quota)):
        for name, wanted in quota.items():
            candidates = np.flatnonzero(assignment == index[name])
            chosen = rng.choice(candidates, size=wanted,
                                replace=candidates.size < wanted)
            rows.append(chosen)
            condition += [side] * wanted
            population += [name] * wanted
    selected = np.concatenate(rows)
    counts = counts[selected].copy()
    meta = pd.DataFrame({"condition": condition, "population": population})

    expressed = np.argsort(counts.mean(axis=0))[::-1]
    module = min(30, n_genes)
    first, second = expressed[:module], expressed[module:2 * module]
    target_b = np.flatnonzero(
        (meta["condition"] == "target") & (meta["population"] == "B"))
    if scenario == "S0_clean_movement":
        add_module(counts, target_b, first, 0.18, rng)
    if scenario == "S4_bifurcation":
        split = rng.permutation(np.array(["B1", "B2"] * ((len(target_b) + 1) // 2))[:len(target_b)])
        meta.loc[target_b, "population"] = split
        add_module(counts, target_b[split == "B1"], first, 0.20, rng)
        add_module(counts, target_b[split == "B2"], second, 0.20, rng)

    meta["expected_rejection"] = False
    for scen, side, name in (("S1_extinction", "source", "A"),
                             ("S2_emergence", "target", "G"),
                             ("S3_source_outlier", "source", "O")):
        if scenario == scen:
            meta.loc[(meta["condition"] == side)
                     & (meta["population"] == name), "expected_rejection"] = True
    return counts, meta


def load_splatter(root: Path, scenario: str, n: int, replicate: int
                  ) -> tuple[np.ndarray, pd.DataFrame]:
    from scipy.io import mmread

    directory = root / scenario / f"n_{n:04d}" / f"rep_{replicate:02d}"
    counts = np.asarray(mmread(directory / "counts.mtx").todense()).T
    meta = pd.read_csv(directory / "cells.csv")
    return counts.astype(np.float64), meta


# --------------------------------------------------------------------------
# Embedding


def embed(counts: np.ndarray, seed: int, representation: str = "logcpm",
          min_dist: float = 0.3) -> np.ndarray:
    """One joint embedding of both sides, through the standard route.

    log CPM, top-variance genes, gene scaling, PCA, then UMAP. The scaling step
    is the one a reader will ask about: it is here, mean-centred and divided by
    each gene's standard deviation, which is the order Seurat and Scanpy use
    and the same order ``confidenceot.Preprocessing`` uses.
    """
    from sklearn.decomposition import PCA
    import umap

    library = counts.sum(axis=1, keepdims=True)
    library[library == 0] = 1.0
    if representation == "rank":
        # Per-cell ranks of the top genes, which is what the rank arm feeds the
        # cost. Ties below the cut all become zero, so a cell is described by
        # which genes it expresses most rather than by how much it expressed.
        order = np.argsort(np.argsort(-counts, axis=1), axis=1)
        matrix = np.where(order < 256, 256 - order, 0).astype(np.float64)
    else:
        matrix = np.log1p(counts / library * 1e4)

    variance = matrix.var(axis=0)
    selected = np.argsort(variance)[::-1][:min(2000, matrix.shape[1])]
    dense = matrix[:, selected]
    dense = dense - dense.mean(axis=0)
    deviation = dense.std(axis=0)
    dense = dense / np.where(deviation > 1e-8, deviation, 1.0)
    components = PCA(n_components=min(30, dense.shape[0] - 1, dense.shape[1]),
                     random_state=seed).fit_transform(dense)
    return umap.UMAP(n_neighbors=30, min_dist=min_dist, random_state=seed,
                     verbose=False).fit_transform(components)


# --------------------------------------------------------------------------
# Drawing


def draw_scenario(axes, points, meta: pd.DataFrame, scenario: str) -> None:
    """One joint panel: both sides on one set of axes, arrows for displacement.

    An earlier version drew source and target as two panels sharing the
    embedding and joined them with arrows across the gap. The coordinates being
    shared is what makes that honest -- and also what makes it useless: a
    population in the same place on both sides produces a perfectly horizontal
    rail the width of the figure, every rail looks alike, and the displacement
    the arrow is supposed to carry is swamped by the panel spacing. Drawn in
    one frame, an arrow's length is the displacement, and a population that did
    not move has no arrow, which is the correct way to say it did not move.
    """
    from matplotlib import patheffects

    source = (meta["condition"] == "source").to_numpy()
    population = meta["population"].to_numpy()

    axes.scatter(points[source, 0], points[source, 1], s=11,
                 facecolors="none", edgecolors="#9a9a94", linewidths=0.45,
                 zorder=2)
    for name in sorted(set(population[~source])):
        rows = ~source & (population == name)
        axes.scatter(points[rows, 0], points[rows, 1], s=5,
                     c=PALETTE.get(name, "#333333"), linewidths=0, zorder=3)

    def centroid(side, name):
        rows = side & (population == name)
        return points[rows].mean(axis=0) if rows.sum() else None

    source_names = sorted(set(population[source]))
    target_names = sorted(set(population[~source]))
    descendants = {name: [name] for name in source_names}
    if scenario == "S4_bifurcation" and "B" in descendants:
        descendants["B"] = [n for n in ("B1", "B2") if n in target_names]
    matched = {child for children in descendants.values() for child in children}

    def spread(side, name) -> float:
        """How far the population's own cells sit from their centre."""
        rows = side & (population == name)
        if rows.sum() < 2:
            return float("inf")
        block = points[rows]
        return float(np.sqrt(((block - block.mean(axis=0)) ** 2).sum(axis=1).mean()))

    moves = []
    moved_names: set[str] = set()
    for name in source_names:
        start = centroid(source, name)
        children = [c for c in descendants[name] if c in target_names]
        if start is None:
            continue
        if not children:
            axes.scatter([start[0]], [start[1]], s=260, facecolors="none",
                         edgecolors=C_REJECT, linewidths=1.6, zorder=6)
            axes.annotate(name + " gone", xy=tuple(start), xytext=(0, -15),
                          textcoords="offset points", fontsize=6.4,
                          color=C_REJECT, ha="center", zorder=7)
            continue
        for child in children:
            end = centroid(~source, child)
            shift = float(np.linalg.norm(end - start))
            # Measured against the population's own scatter, not against the
            # panel. A panel-relative threshold hid exactly the two scenarios
            # that contain a displacement: the populations are far apart, so a
            # real move within one of them is small next to the figure and
            # large next to the cloud that moved.
            reference = min(spread(source, name), spread(~source, child))
            moves.append({"population": name, "child": child,
                          "shift": round(shift, 3),
                          "own_spread": round(reference, 3),
                          "drawn": bool(shift > reference)})
            if shift <= reference:
                continue
            moved_names.add(name)
            moved_names.add(child)
            axes.annotate("", xy=tuple(end), xytext=tuple(start),
                          arrowprops=dict(arrowstyle="-|>", color="#1a1a1f",
                                          linewidth=1.4, shrinkA=1, shrinkB=1,
                                          mutation_scale=9), zorder=5)
    for name in target_names:
        if name in matched:
            continue
        end = centroid(~source, name)
        axes.scatter([end[0]], [end[1]], s=260, facecolors="none",
                     edgecolors=C_REJECT, linewidths=1.6, zorder=6)
        axes.annotate(name + " new", xy=tuple(end), xytext=(0, -15),
                      textcoords="offset points", fontsize=6.4,
                      color=C_REJECT, ha="center", zorder=7)

    for name in sorted(set(population)):
        if name in moved_names and name not in source_names:
            continue  # named inside the inset, where the split is legible
        on_target = ~source & (population == name)
        rows = on_target if on_target.sum() else (population == name)
        centre = points[rows].mean(axis=0)
        axes.text(centre[0], centre[1], name, fontsize=7, fontweight="bold",
                  ha="center", va="center", zorder=8,
                  color=PALETTE.get(name, "#333333"),
                  path_effects=[patheffects.withStroke(linewidth=2.2,
                                                       foreground="white")])
    # The displacement is small next to the distance between populations --
    # a population that moves stays recognisably itself -- so at figure scale
    # the arrow is a few points long and a reader misses the one thing the
    # scenario is about. The inset shows the same cells and the same arrow,
    # magnified, with a rectangle marking where it came from.
    if moved_names:
        rows = np.isin(population, sorted(moved_names))
        block = points[rows]
        pad = 0.28 * max(np.ptp(block[:, 0]), np.ptp(block[:, 1]), 1e-6)
        window = (block[:, 0].min() - pad, block[:, 0].max() + pad,
                  block[:, 1].min() - pad, block[:, 1].max() + pad)
        # Put it where there are fewest cells. A fixed corner covered a
        # population in one scenario and ran off the panel in another, and the
        # emptiest quadrant is a property of the data rather than a guess.
        limits = (points[:, 0].min(), points[:, 0].max(),
                  points[:, 1].min(), points[:, 1].max())
        width, height = 0.40, 0.38
        best, fewest = (0.58, 0.60), None
        for x0 in (0.02, 0.58):
            for y0 in (0.02, 0.60):
                left = limits[0] + x0 * (limits[1] - limits[0])
                right = limits[0] + (x0 + width) * (limits[1] - limits[0])
                bottom = limits[2] + y0 * (limits[3] - limits[2])
                top = limits[2] + (y0 + height) * (limits[3] - limits[2])
                inside = int(((points[:, 0] >= left) & (points[:, 0] <= right)
                              & (points[:, 1] >= bottom)
                              & (points[:, 1] <= top)).sum())
                if fewest is None or inside < fewest:
                    best, fewest = (x0, y0), inside
        inset = axes.inset_axes([best[0], best[1], width, height])
        inset.scatter(points[source & rows, 0], points[source & rows, 1], s=26,
                      facecolors="none", edgecolors="#9a9a94", linewidths=0.6)
        for name in sorted(moved_names):
            child_rows = ~source & (population == name)
            if child_rows.sum():
                inset.scatter(points[child_rows, 0], points[child_rows, 1],
                              s=13, c=PALETTE.get(name, "#333333"),
                              linewidths=0)
        for move in moves:
            if not move["drawn"]:
                continue
            start = centroid(source, move["population"])
            end = centroid(~source, move["child"])
            inset.annotate("", xy=tuple(end), xytext=tuple(start),
                           arrowprops=dict(arrowstyle="-|>", color="#1a1a1f",
                                           linewidth=1.5, shrinkA=2, shrinkB=2,
                                           mutation_scale=13))
            inset.annotate(move["child"], xy=tuple(end), xytext=(7, -7),
                           textcoords="offset points", fontsize=6.8,
                           fontweight="bold",
                           color=PALETTE.get(move["child"], "#333333"))
        inset.set_xlim(window[0], window[1])
        inset.set_ylim(window[2], window[3])
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color("#9a9a94")
            spine.set_linewidth(0.8)
        axes.indicate_inset_zoom(inset, edgecolor="#9a9a94", linewidth=0.8,
                                 alpha=0.9)

    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(True)
        spine.set_color("#d8d8d2")
        spine.set_linewidth(0.6)
    return moves



def draw_paired(axes_left, axes_right, points, meta: pd.DataFrame,
                scenario: str) -> list[dict]:
    """Two panels on one embedding, and arrows only where something moved.

    The two-panel layout produced a rail per population when every population
    got an arrow: shared coordinates mean an unmoved population is the same
    point in both panels, and the line between them is horizontal and as wide
    as the gap. Drawing only the displacements leaves at most two arrows in any
    scenario, so the rails are gone and what is left is the movement. Presence
    and absence are then read by comparing the panels, which is the one thing
    this layout does better than a single frame.
    """
    from matplotlib import patheffects

    source = (meta["condition"] == "source").to_numpy()
    population = meta["population"].to_numpy()
    limits = (points[:, 0].min() - 1, points[:, 0].max() + 1,
              points[:, 1].min() - 1, points[:, 1].max() + 1)

    for axes, mask, label in ((axes_left, source, "source"),
                              (axes_right, ~source, "target")):
        axes.scatter(points[~mask, 0], points[~mask, 1], s=2.5, c="#eeeeea",
                     linewidths=0, zorder=1)
        for name in sorted(set(population[mask])):
            rows = mask & (population == name)
            axes.scatter(points[rows, 0], points[rows, 1], s=5,
                         c=PALETTE.get(name, "#333333"), linewidths=0, zorder=3)
            centre = points[rows].mean(axis=0)
            axes.text(centre[0], centre[1], name, fontsize=7,
                      fontweight="bold", ha="center", va="center", zorder=8,
                      color=PALETTE.get(name, "#333333"),
                      path_effects=[patheffects.withStroke(linewidth=2.2,
                                                           foreground="white")])
        axes.set_xlim(limits[0], limits[1])
        axes.set_ylim(limits[2], limits[3])
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_title(label, fontsize=7, color="#55555a", pad=2)
        for spine in axes.spines.values():
            spine.set_color("#d8d8d2")
            spine.set_linewidth(0.6)

    def centroid(side, name):
        rows = side & (population == name)
        return points[rows].mean(axis=0) if rows.sum() else None

    def spread(side, name) -> float:
        rows = side & (population == name)
        if rows.sum() < 2:
            return float("inf")
        block = points[rows]
        return float(np.sqrt(((block - block.mean(axis=0)) ** 2).sum(axis=1).mean()))

    source_names = sorted(set(population[source]))
    target_names = sorted(set(population[~source]))
    descendants = {name: [name] for name in source_names}
    if scenario == "S4_bifurcation" and "B" in descendants:
        descendants["B"] = [n for n in ("B1", "B2") if n in target_names]
    matched = {c for children in descendants.values() for c in children}

    moves = []
    for name in source_names:
        start = centroid(source, name)
        children = [c for c in descendants[name] if c in target_names]
        if start is None:
            continue
        if not children:
            axes_left.scatter([start[0]], [start[1]], s=230, facecolors="none",
                              edgecolors=C_REJECT, linewidths=1.5, zorder=6)
            axes_left.annotate(name + " gone", xy=tuple(start), xytext=(0, -14),
                               textcoords="offset points", fontsize=6.4,
                               color=C_REJECT, ha="center", zorder=7)
            continue
        for child in children:
            end = centroid(~source, child)
            shift = float(np.linalg.norm(end - start))
            reference = min(spread(source, name), spread(~source, child))
            moves.append({"population": name, "child": child,
                          "shift": round(shift, 3),
                          "own_spread": round(reference, 3),
                          "drawn": bool(shift > reference)})
            if shift <= reference:
                continue
            axes_left.add_artist(ConnectionPatch(
                xyA=tuple(start), coordsA=axes_left.transData,
                xyB=tuple(end), coordsB=axes_right.transData,
                arrowstyle="-|>", mutation_scale=11, linewidth=1.4,
                color="#1a1a1f", zorder=9, clip_on=False))
    for name in target_names:
        if name in matched:
            continue
        end = centroid(~source, name)
        axes_right.scatter([end[0]], [end[1]], s=230, facecolors="none",
                           edgecolors=C_REJECT, linewidths=1.5, zorder=6)
        axes_right.annotate(name + " new", xy=tuple(end), xytext=(0, -14),
                            textcoords="offset points", fontsize=6.4,
                            color=C_REJECT, ha="center", zorder=7)
    return moves


def draw_schematic(axes, scenario: str) -> None:
    """The same six scenarios drawn rather than computed.

    No counts, no embedding: populations are discs on a ring at fixed places,
    so every scenario has the same geometry and the only thing that differs
    between panels is what the scenario says. A UMAP spends most of its area on
    where the clusters happened to land, which is noise for a figure whose job
    is to say what the question is. This version cannot show that the method
    works and is not meant to; it says what it is being asked.
    """
    from matplotlib.patches import Circle

    ring = {name: (np.cos(angle), np.sin(angle)) for name, angle in
            zip(BASE[:6], np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, 6,
                                      endpoint=False))}
    ring["G"] = (0.0, 0.0)
    ring["O"] = (-0.34, 0.30)
    present_source = set(BASE[:6])
    present_target = set(BASE[:6])
    moved: dict[str, list[str]] = {}
    ringed_source: set[str] = set()
    ringed_target: set[str] = set()

    if scenario == "S0_clean_movement":
        moved = {"B": ["B"]}
    elif scenario == "S1_extinction":
        present_target.discard("A")
        ringed_source.add("A")
    elif scenario == "S2_emergence":
        present_target.add("G")
        ringed_target.add("G")
    elif scenario == "S3_source_outlier":
        present_source.add("O")
        ringed_source.add("O")
    elif scenario == "S4_bifurcation":
        moved = {"B": ["B1", "B2"]}
        present_target.discard("B")
        present_target.update({"B1", "B2"})
        base = np.asarray(ring["B"])
        ring["B1"] = tuple(base + np.array([-0.30, -0.62]))
        ring["B2"] = tuple(base + np.array([0.52, -0.40]))

    sizes = {name: 0.16 for name in ring}
    if scenario == "S5_abundance_shift":
        heavy = {"A": 0.23, "B": 0.20, "C": 0.17, "D": 0.13, "E": 0.10, "F": 0.09}

    for name in sorted(present_source):
        centre = ring[name]
        radius = (heavy[name] if scenario == "S5_abundance_shift" and name in heavy
                  else sizes[name])
        axes.add_patch(Circle(centre, radius, facecolor="none",
                              edgecolor="#9a9a94", linewidth=1.1, zorder=3))
    for name in sorted(present_target):
        centre = ring[name]
        radius = sizes[name] * (0.62 if scenario == "S5_abundance_shift"
                                and name in ("A", "B", "C") else 1.0)
        if scenario == "S5_abundance_shift" and name in ("D", "E", "F"):
            radius = sizes[name] * 1.35
        axes.add_patch(Circle(centre, radius * 0.72,
                              facecolor=PALETTE.get(name, "#333333"),
                              edgecolor="none", alpha=0.9, zorder=4))
        axes.text(centre[0], centre[1], name, fontsize=7.6, fontweight="bold",
                  ha="center", va="center", color="white", zorder=6)
    for name in sorted(present_source - present_target):
        centre = ring[name]
        axes.text(centre[0], centre[1], name, fontsize=7.6, fontweight="bold",
                  ha="center", va="center", color="#9a9a94", zorder=6)

    for parent, children in moved.items():
        for child in children:
            start = np.asarray(ring[parent], dtype=float)
            end = np.asarray(ring[child], dtype=float)
            if np.allclose(start, end):
                end = start + np.array([0.52, -0.46])
                ring[child] = tuple(end)
                axes.add_patch(Circle(tuple(end), sizes[parent] * 0.72,
                                      facecolor=PALETTE.get(child, "#333333"),
                                      edgecolor="none", alpha=0.9, zorder=4))
                axes.text(end[0], end[1], child, fontsize=7.6,
                          fontweight="bold", ha="center", va="center",
                          color="white", zorder=6)
            axes.annotate("", xy=tuple(end), xytext=tuple(start),
                          arrowprops=dict(arrowstyle="-|>", color="#1a1a1f",
                                          linewidth=1.5, shrinkA=13, shrinkB=13,
                                          mutation_scale=13), zorder=5)
    for name, target_side in ((n, False) for n in ringed_source):
        centre = ring[name]
        axes.add_patch(Circle(centre, 0.24, facecolor="none",
                              edgecolor=C_REJECT, linewidth=1.7, zorder=7))
        axes.text(centre[0], centre[1] - 0.33, name + " gone", fontsize=6.6,
                  color=C_REJECT, ha="center", zorder=7)
    for name in ringed_target:
        centre = ring[name]
        axes.add_patch(Circle(centre, 0.24, facecolor="none",
                              edgecolor=C_REJECT, linewidth=1.7, zorder=7))
        axes.text(centre[0], centre[1] - 0.33, name + " new", fontsize=6.6,
                  color=C_REJECT, ha="center", zorder=7)

    axes.set_xlim(-1.62, 1.62)
    axes.set_ylim(-1.52, 1.42)
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color("#d8d8d2")
        spine.set_linewidth(0.6)



# The population each scenario is about. Everything else is background that
# behaves the same way in all three, and labelling it six times over says
# nothing three times over.
SUBJECT = {
    "S0_clean_movement": None,
    "S1_extinction": "A",
    "S2_emergence": "G",
    "S3_source_outlier": "O",
    "S4_bifurcation": "B",
    "S5_abundance_shift": None,
}


def draw_shared(figure, grid, scenarios, loaded, seed: int,
                min_dist: float = 0.6) -> dict:
    """Source on the left, target on the right, one row per scenario.

    The two panels share one embedding, which is what makes them comparable:
    the R generator builds a single Splatter pool per (N, replicate) and every
    scenario selects from it, so a population occupies the same coordinates
    wherever it is drawn. A cluster in the left panel with nothing at that
    position on the right is a population the target does not have, and that
    absence is the whole question.

    No arrows between the panels. An arrow joining the same coordinates in two
    panels is horizontal and as long as the gap between them whatever the data
    does, so it says nothing while looking like it says something. The reading
    here is a comparison of two pictures, which needs no line drawn on it.

    The position a population has vacated is marked, because an empty patch of
    white is not visible as an absence unless the eye is sent there.
    """
    from matplotlib import patheffects

    counts = np.vstack([block[0] for block in loaded])
    offsets, cursor = [], 0
    for block in loaded:
        offsets.append((cursor, cursor + block[0].shape[0]))
        cursor += block[0].shape[0]
    points = embed(counts, seed, min_dist=min_dist)

    margin = 0.045 * max(np.ptp(points[:, 0]), np.ptp(points[:, 1]))
    limits = (points[:, 0].min() - margin, points[:, 0].max() + margin,
              points[:, 1].min() - margin, points[:, 1].max() + margin)

    report: dict = {}
    for row, scenario in enumerate(scenarios):
        lo, hi = offsets[row]
        local, meta = points[lo:hi], loaded[row][1]
        source = (meta["condition"] == "source").to_numpy()
        population = meta["population"].to_numpy()
        subject = SUBJECT.get(scenario)
        colour = PALETTE.get(subject, C_REJECT)
        held_by = "source" if subject in set(population[source]) else "target"

        panels = {}
        for column, (side, mask) in enumerate((("source", source),
                                               ("target", ~source))):
            axes = figure.add_subplot(grid[row, column])
            panels[side] = axes
            rest = mask & (population != subject)
            axes.scatter(local[rest, 0], local[rest, 1], s=13, c="#cfcfc8",
                         linewidths=0, zorder=2)
            here = mask & (population == subject)
            if here.sum():
                axes.scatter(local[here, 0], local[here, 1], s=26, c=colour,
                             linewidths=0, zorder=5)
                centre = local[here].mean(axis=0)
                axes.text(centre[0], centre[1] + 2.0, subject, fontsize=11,
                          fontweight="bold", ha="center", va="bottom",
                          color=colour, zorder=9,
                          path_effects=[patheffects.withStroke(
                              linewidth=3.0, foreground="white")])
            else:
                # The vacated position. Without the mark the panel is simply
                # missing a cluster, and a missing cluster among six is not
                # something a reader finds by looking.
                where = local[population == subject].mean(axis=0)
                axes.scatter([where[0]], [where[1]], s=620, facecolors="none",
                             edgecolors=C_REJECT, linewidths=2.0,
                             linestyle=(0, (3, 2)), zorder=6)
                axes.text(where[0], where[1] + 2.0,
                          f"no {subject}", fontsize=10.5, fontweight="bold",
                          ha="center", va="bottom", color=C_REJECT, zorder=9,
                          path_effects=[patheffects.withStroke(
                              linewidth=3.0, foreground="white")])
            axes.set_xlim(limits[0], limits[1])
            axes.set_ylim(limits[2], limits[3])
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_color("#dcdcd6")
                spine.set_linewidth(0.7)
            if row == 0:
                axes.text(0.5, 1.045, side, transform=axes.transAxes,
                          fontsize=10, color="#44444a", ha="center",
                          va="bottom")

        panels["source"].text(
            0.0, 1.115 if row == 0 else 1.02, TITLES[scenario],
            transform=panels["source"].transAxes, fontsize=10.5,
            fontweight="semibold", ha="left", va="bottom")
        # One arrow, between the panels, for the direction of the question --
        # not one per population, which is the version that produced rails.
        figure.add_artist(ConnectionPatch(
            xyA=(1.008, 0.5), coordsA=panels["source"].transAxes,
            xyB=(-0.008, 0.5), coordsB=panels["target"].transAxes,
            arrowstyle="-|>", mutation_scale=26, linewidth=2.4,
            color="#6a6a62"))
        report[scenario] = {
            "subject": subject, "present_on": held_by,
            "cells": int((population == subject).sum()),
            "reading": (f"{subject} is in the {held_by} and absent from the "
                        f"other side")}
    return report


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    provenance = ("Splatter" if args.splatter_root
                  else "structure-matched synthetic counts, not Splatter")
    report: dict[str, object] = {"counts_source": provenance, "n": args.n,
                                 "genes": args.genes, "seed": args.seed}

    chosen = (tuple(s.strip() for s in args.scenarios.split(","))
              if args.scenarios else SCENARIOS)
    unknown = [s for s in chosen if s not in SCENARIOS]
    if unknown:
        raise SystemExit(f"unknown scenario {unknown}; known: {list(SCENARIOS)}")

    if args.layout == "shared":
        loaded = [
            load_splatter(args.splatter_root, scenario, args.n, args.replicate)
            if args.splatter_root
            else synthesise(scenario, args.n, args.genes, args.seed + index)
            for index, scenario in enumerate(chosen)]
        with mpl.rc_context(STYLE):
            figure = plt.figure(figsize=(7.4, 3.5 * len(chosen)))
            grid = figure.add_gridspec(
                len(chosen), 2, wspace=0.055, hspace=0.14,
                left=0.010, right=0.990,
                top=1 - 0.085 / len(chosen), bottom=0.015)
            report["shared_embedding"] = draw_shared(
                figure, grid, chosen, loaded, args.seed, args.umap_min_dist)
            report["umap_min_dist"] = args.umap_min_dist
            for suffix in ("png", "pdf"):
                figure.savefig(args.out / f"scenario_umap.{suffix}",
                               bbox_inches="tight")
            plt.close(figure)
        (args.out / "diagnostics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        print(f"\n{args.out / 'scenario_umap.png'}")
        return

    with mpl.rc_context(STYLE):
        wide = args.layout == "paired"
        figure = plt.figure(figsize=(7.4, 11.0) if wide else (7.4, 9.6))
        grid = figure.add_gridspec(6 if wide else 3, 2,
                                   wspace=0.05 if wide else 0.07,
                                   hspace=0.55 if wide else 0.22,
                                   left=0.02, right=0.98,
                                   top=0.925 if wide else 0.895, bottom=0.03)
        for row, scenario in enumerate(SCENARIOS):
            if args.splatter_root:
                counts, meta = load_splatter(args.splatter_root, scenario,
                                             args.n, args.replicate)
            else:
                counts, meta = synthesise(scenario, args.n, args.genes,
                                          args.seed + row)
            if args.layout == "schematic":
                axes = figure.add_subplot(grid[row // 2, row % 2])
                draw_schematic(axes, scenario)
                moves = []
            elif args.layout == "paired":
                points = embed(counts, args.seed)
                left = figure.add_subplot(grid[row, 0])
                right = figure.add_subplot(grid[row, 1])
                moves = draw_paired(left, right, points, meta, scenario)
                axes = left
            else:
                points = embed(counts, args.seed)
                axes = figure.add_subplot(grid[row // 2, row % 2])
                moves = draw_scenario(axes, points, meta, scenario)
            axes.set_title(TITLES[scenario], loc="left", fontsize=8.8,
                           fontweight="semibold", pad=16 if args.layout == "paired" else 12)
            axes.text(0.0, 1.075 if args.layout == "paired" else 1.015,
                      READING[scenario], transform=axes.transAxes,
                      fontsize=7, color="#55555a", va="bottom")
            report[scenario] = {
                "cells": int(len(meta)),
                "displacements": moves,
                "source_populations": sorted(set(meta.loc[meta.condition.eq("source"), "population"])),
                "target_populations": sorted(set(meta.loc[meta.condition.eq("target"), "population"])),
                "expected_rejection_n": int(meta["expected_rejection"].sum()),
            }
        figure.legend(handles=[
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor="#9a9a94", markeredgewidth=0.7,
                   label="source cell", markersize=6),
            Line2D([], [], marker="o", linestyle="", color="#7f8c8d",
                   label="target cell", markersize=4),
            Line2D([], [], marker=">", linestyle="", color="#33333a",
                   label="the population moved", markersize=7),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=C_REJECT, markeredgewidth=1.6,
                   label="no counterpart: reject", markersize=9),
        ], loc="upper center", ncol=4, frameon=False, fontsize=7.4,
            bbox_to_anchor=(0.5, 0.965), columnspacing=2.0)
        figure.text(0.5, 0.992,
                    "One joint embedding per scenario. Source and target are "
                    "placed by the same coordinates, so an arrow is a real\n"
                    "displacement and a population that did not move has none.",
                    fontsize=7.6, color="#55555a", ha="center", va="top",
                    linespacing=1.5)
        for suffix in ("png", "pdf"):
            figure.savefig(args.out / f"scenario_umap.{suffix}",
                           bbox_inches="tight")
        plt.close(figure)

    (args.out / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n{args.out / 'scenario_umap.png'}")


if __name__ == "__main__":
    main()
