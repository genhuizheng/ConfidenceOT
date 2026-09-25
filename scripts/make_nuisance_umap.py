"""What sequencing depth and the low-gene effect actually do to an embedding.

The ablation reports a number called "axis vs total counts". It is the
correlation between the leading axis of the representation and each cell's
sequencing depth, and as a number it says nothing to anyone. This is the
picture it was measured on.

**Every cell here is drawn from one population.** There is no biology to find:
the cells differ only in how deeply they were sequenced, or in how many genes
were detected. So any structure that appears in the embedding is the artefact
and nothing else, and the question "is the artefact still there after the
preprocessing" is something a reader can answer by looking rather than by
trusting a correlation coefficient.

Two nuisances, and they are not the same thing.

**Sequencing depth.** Cells get different total counts -- the spread the
benchmark uses is sd(log depth) 0.9, roughly a twenty-fold range between the
shallowest and deepest cell. This is the thing most normalisations are built
to remove.

**The low-gene effect, or dropout.** Total counts are held *fixed* and the
cells differ in how many distinct genes those counts land on: the same library
concentrated in a few genes, or spread over many. Depth normalisation does not
touch this, because the totals already agree. It is the covariate the
ablation's third column was about and the one nothing in the factorial
removes.

Each row is one preprocessing. Each column colours the same embedding by one
nuisance. A gradient means the embedding is arranging cells by that nuisance;
an even mix of colours means it is not.

Usage:

    python scripts/make_nuisance_umap.py --out benchmark_results/nuisance_umap
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from make_journal_figure import STYLE  # noqa: E402

ARMS = (
    ("logcpm", "log CPM"),
    ("rank256_ds_cos", "rank + DS + cos"),
)
NUISANCES = (
    ("total_counts", "sequencing depth", "total counts per cell"),
    ("detected_genes", "low-gene effect", "genes detected per cell"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=900)
    parser.add_argument("--genes", type=int, default=1200)
    parser.add_argument("--depth-sd", type=float, default=0.9,
                        help="sd of log depth, the spread the benchmark uses")
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def one_population(nuisance: str, n_cells: int, n_genes: int, depth_sd: float,
                   rng: np.random.Generator, uniform: bool = False,
                   profile: np.ndarray | None = None) -> np.ndarray:
    """One population carrying exactly one nuisance, and nothing else.

    Every cell draws from the same relative expression profile, so a figure
    built on this has no biology in it at all: whatever structure the
    embedding shows is the nuisance. One nuisance per population rather than
    both in one, because a reader looking at a gradient has to be able to say
    which of the two put it there.

    ``depth``: the totals vary, sd(log depth) 0.9 by default, which is the
    spread the benchmark uses and about a twenty-fold range end to end. The
    genes a cell lands on are unrestricted.

    ``breadth``: the totals are **identical** and the cells differ in how many
    distinct genes those counts fall on. Depth normalisation cannot touch this
    -- the totals already agree -- which is what makes it a separate problem
    rather than a symptom of the first.
    """
    # Passed in when two draws have to come from the *same* population.
    # Drawn afresh on each call, two draws are two different populations, and
    # an embedding of them separates the sides -- which is a picture of a
    # mistake rather than of the benchmark arm.
    if profile is None:
        profile = rng.gamma(shape=0.5, scale=3.0, size=n_genes) + 0.05
        profile = profile / profile.sum()
    order = np.argsort(-profile)

    counts = np.zeros((n_cells, n_genes))
    for index in range(n_cells):
        if nuisance == "depth":
            library = (3000.0 if uniform
                       else 3000.0 * rng.lognormal(0.0, depth_sd))
            weights = profile
        else:
            # The ideal arm holds the nuisance fixed rather than removing the
            # cells that carry it: what is being shown is the measurement
            # without the artefact, not a different set of cells.
            fraction = 0.30 if uniform else float(
                np.clip(rng.lognormal(np.log(0.30), 0.60), 0.03, 1.0))
            width = int(fraction * n_genes)
            weights = np.zeros(n_genes)
            weights[order[:width]] = profile[order[:width]]
            weights /= weights.sum()
            library = 3000.0
        counts[index] = rng.poisson(weights * library)
    return counts


def embed(counts: np.ndarray, label: str, seed: int) -> np.ndarray:
    """The project's own configuration, not a second copy of it.

    ``confidenceot.Preprocessing`` is what the cancer runs and the benchmark
    both use, so the embedding here is the geometry the gate actually sees.
    Writing the transform out again in this file is how the depth screen and
    the production runner came to measure different things.
    """
    import umap
    from confidenceot import Preprocessing

    configuration = Preprocessing.from_label(label)
    half = counts.shape[0] // 2
    source, target = counts[:half], counts[half:]
    if configuration.equalise_depth:
        source, target, _ = configuration.equalise(
            source, target, rng=np.random.default_rng(seed))
    representation = configuration.representation(source, target, seed=seed)
    joint = np.vstack([np.asarray(representation.source),
                       np.asarray(representation.target)])
    return umap.UMAP(n_neighbors=60, min_dist=1.4, spread=1.8,
                     random_state=seed, verbose=False).fit_transform(joint)


NUISANCE = {
    "depth": {
        "title": "sequencing depth",
        "ideal": "every cell sequenced equally",
        "affected": "cells sequenced to different depths",
        "colour_by": "total counts per cell",
        "construction": ("the same depth spread on both sides,\n"
                         "sd(log depth) 0 to 0.9"),
        "why_both": ("On one side only this would be a batch effect between\n"
                     "samples, which is a different problem with its own\n"
                     "methods. Here both sides carry it, so no cell lacks a\n"
                     "counterpart and the correct answer stays: reject nothing."),
    },
    "breadth": {
        "title": "low-gene effect",
        "ideal": "every cell detects the same genes",
        "affected": "same total counts, different genes detected",
        "colour_by": "genes detected per cell",
        "construction": ("the same breadth spread on both sides,\n"
                         "total counts held at 3,000"),
        "why_both": ("Depth normalisation cannot touch this: the totals\n"
                     "already agree. It is a separate problem, not a symptom\n"
                     "of the first, and nothing in the preprocessing\n"
                     "factorial removes it."),
    },
}


def measure(points: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """How strongly the embedding orders cells by the nuisance.

    The largest absolute correlation between the nuisance and any direction in
    the plane. Taking the first coordinate alone would report a small number
    whenever UMAP happened to lay the gradient out diagonally.
    """
    if np.ptp(values) == 0:
        return 0.0, 0.0
    centred = points - points.mean(axis=0)
    best, best_angle = 0.0, 0.0
    for angle in np.linspace(0.0, np.pi, 180, endpoint=False):
        projection = (centred[:, 0] * np.cos(angle)
                      + centred[:, 1] * np.sin(angle))
        value = float(np.corrcoef(projection, values)[0, 1])
        if abs(value) > best:
            # Point the arrow the way the nuisance increases, so its direction
            # carries the sign and not only the axis.
            best, best_angle = abs(value), angle + (np.pi if value < 0 else 0.0)
    return best, best_angle


def problem_figure(out: Path, cells: int, genes: int, depth_sd: float,
                   seed: int) -> dict:
    """The two problems, each as the data one wants beside the data one has.

    No preprocessing appears here and none should. A problem statement that
    names log CPM is asking the reader to care about a method before they have
    been told what the method is for; the left panel is what the measurement
    would look like if the nuisance did not exist, and the right panel is what
    it looks like because it does.

    Every cell in every panel is from one population, so the left panel is a
    single featureless cloud and the right panel's structure is entirely the
    nuisance. Nothing is labelled beyond the titles: a gradient either appears
    or it does not, and a caption saying which would be doing the reader's
    looking for them.
    """
    report: dict = {}
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(6.6, 6.0))
        outer = figure.add_gridspec(2, 1, hspace=0.30, left=0.015,
                                    right=0.885, top=0.945, bottom=0.02)
        for row, nuisance in enumerate(("depth", "breadth")):
            facts = NUISANCE[nuisance]
            inner = outer[row].subgridspec(1, 2, wspace=0.05)
            for column, kind in enumerate(("ideal", "affected")):
                rng = np.random.default_rng(seed + row)
                spread = depth_sd if kind == "affected" else 0.0
                counts = one_population(nuisance, cells, genes, spread, rng,
                                        uniform=kind == "ideal")
                values = (counts.sum(axis=1) if nuisance == "depth"
                          else (counts > 0).sum(axis=1))
                points = embed(counts, "logcpm", seed)
                strength, angle = measure(points, values.astype(float))
                report[f"{nuisance}/{kind}"] = round(strength, 3)
                axes = figure.add_subplot(inner[0, column])
                colour = np.log10(values + 1.0)
                dots = axes.scatter(points[:, 0], points[:, 1], c=colour,
                                    s=11, cmap="viridis", linewidths=0,
                                    vmin=colour.min() - 1e-9,
                                    vmax=colour.max() + 1e-9)
                axes.set_xticks([])
                axes.set_yticks([])
                for spine in axes.spines.values():
                    spine.set_color("#d8d8d2")
                    spine.set_linewidth(0.6)
                axes.set_title(facts[kind], loc="center", fontsize=8,
                               color="#55555a", pad=4)
                if column == 1:
                    bar = figure.colorbar(dots, ax=axes, fraction=0.055,
                                          pad=0.025)
                    bar.set_label(facts["colour_by"], fontsize=7)
                    bar.ax.tick_params(labelsize=6.5)
                    bar.outline.set_linewidth(0.5)
                if column == 0:
                    axes.text(0.0, 1.14, f"({chr(97 + row)})  {facts['title']}",
                              transform=axes.transAxes, fontsize=8.4,
                              fontweight="semibold", ha="left", va="bottom")
        for suffix in ("png", "pdf"):
            figure.savefig(out / f"problem.{suffix}", bbox_inches="tight")
        plt.close(figure)
    return report


def benchmark_figure(out: Path, cells: int, genes: int, depth_sd: float,
                     seed: int) -> dict:
    """The arms as a transport problem, because that is what is being tested.

    An earlier version drew two point clouds and said the nuisance was on both
    sides. It never showed the coupling, which is the only thing the method
    produces and the only thing the benchmark scores. This draws it: source
    cells on the upper row, target cells on the lower one, a line for every
    matched pair.

    Both rows are drawn from one population and both carry the same nuisance,
    so a complete coupling exists -- every source cell has a target cell it
    belongs with. The correct answer is therefore to reject nothing, and the
    failure the arm is built to catch is a method that reads the nuisance as
    structure and refuses to match across it: the deep cells of one side
    declining the shallow cells of the other, leaving cells unmatched that
    have perfectly good partners.

    Cells are ordered along each row by the nuisance, so the gradient is
    visible on both rows and a coupling that respects it would slant while a
    correct one runs straight across.
    """
    shown = 46
    report: dict = {}
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(6.9, 3.3))
        for row, nuisance in enumerate(("depth", "breadth")):
            facts = NUISANCE[nuisance]
            rng = np.random.default_rng(seed + 200 + row)
            profile = rng.gamma(shape=0.5, scale=3.0, size=genes) + 0.05
            profile = profile / profile.sum()
            source = one_population(nuisance, shown, genes, depth_sd, rng,
                                    profile=profile)
            target = one_population(nuisance, shown, genes, depth_sd, rng,
                                    profile=profile)

            def nuisance_of(counts: np.ndarray) -> np.ndarray:
                return (counts.sum(axis=1) if nuisance == "depth"
                        else (counts > 0).sum(axis=1)).astype(float)

            # Sorted so the gradient reads along each row. The coupling is
            # then legible as a shape: straight across if the match ignores
            # the nuisance, slanting if it does not.
            source_values = np.sort(nuisance_of(source))
            target_values = np.sort(nuisance_of(target))
            bounds = (min(source_values.min(), target_values.min()),
                      max(source_values.max(), target_values.max()))
            scale = np.log10(np.asarray(bounds) + 1.0)

            # Tighter rows with a reserved band at the foot: the
            # closing note, the lower panel's verdict and the second
            # colour bar were all landing on each other.
            axes = figure.add_axes([0.015, 0.615 - 0.350 * row,
                                    0.785, 0.250])
            axes.set_xlim(-1.5, shown + 0.5)
            axes.set_ylim(-0.66, 1.52)
            axes.axis("off")
            x = np.arange(shown)
            for index in range(shown):
                axes.plot([x[index], x[index]], [1.0, 0.0], color="#c6c6c0",
                          linewidth=0.7, zorder=1)
            for y, values, side in ((1.0, source_values, "source"),
                                    (0.0, target_values, "target")):
                dots = axes.scatter(
                    x, np.full(shown, y), c=np.log10(values + 1.0), s=34,
                    cmap="viridis", linewidths=0, zorder=3,
                    vmin=scale[0], vmax=scale[1])
                axes.text(-0.9, y, side, fontsize=7.6, ha="right",
                          va="center", color="#55555a")
            axes.text(0.0, 1.40, f"({chr(97 + row)})  {facts['title']}",
                      fontsize=8.4, fontweight="semibold", ha="left",
                      transform=axes.get_yaxis_transform(which="grid"))
            axes.text(shown - 0.5, 1.40, facts["construction"].replace("\n", "  "),
                      fontsize=7.4, ha="right", va="bottom",
                      color="#b07d2b" if nuisance == "depth" else "#1f6f8b")
            axes.text(shown * 0.5, -0.62, "every cell matched: reject nothing",
                      fontsize=7.8, fontweight="bold", color="#2f7d4f",
                      ha="center", va="bottom")
            bar = figure.colorbar(
                dots, ax=axes, fraction=0.035, pad=0.015, aspect=9)
            bar.set_label(facts["colour_by"], fontsize=7)
            bar.ax.tick_params(labelsize=6.5)
            bar.outline.set_linewidth(0.5)
            report[nuisance] = {"pairs_drawn": shown,
                                "correct_answer": "reject nothing",
                                "nuisance_on": "both sides"}

        figure.text(0.015, 0.035,
                    "Carried by one side only, either spread would be a batch "
                    "effect between samples: a different problem with its own "
                    "methods.\nCarried by both, a complete coupling exists, "
                    "and the failure to catch is a method that reads the "
                    "gradient as structure and refuses to match across it.",
                    fontsize=7.4, color="#55555a", va="bottom",
                    linespacing=1.55)
        for suffix in ("png", "pdf"):
            figure.savefig(out / f"benchmark.{suffix}", bbox_inches="tight")
        plt.close(figure)
    return report


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "cells": int(args.cells), "genes": int(args.genes),
        "depth_sd_log": args.depth_sd,
        "reading": ("the correlation is the largest over any direction in the "
                    "plane, which is what the ablation's axis-versus-covariate "
                    "column reports"),
    }
    report["problem"] = problem_figure(args.out, args.cells, args.genes,
                                       args.depth_sd, args.seed)
    report["benchmark"] = benchmark_figure(args.out, args.cells, args.genes,
                                           args.depth_sd, args.seed)
    (args.out / "diagnostics.json").write_text(json.dumps(report, indent=2),
                                               encoding="utf-8")
    print(json.dumps(report, indent=2))
    for name in ("problem", "benchmark"):
        print(args.out / f"{name}.png")


if __name__ == "__main__":
    main()
