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


C_REJECT = "#c4553b"

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
    """The task definition: what is generated, what is true, what is scored.

    An earlier version drew a line between the i-th source cell and the i-th
    target cell. **There is no such pairing.** The two sides are independent
    draws from one population, so a source cell has no particular target cell
    it belongs to, and a line saying otherwise describes a benchmark that does
    not exist. The ground truth here is a statement about *sets*: every source
    cell has some valid partner in the target set, or it has none at all.

    Three things a benchmark figure has to carry, and this one carries them as
    three columns.

    **Generation.** Colour is biology and size is the technical nuisance, so
    the two kinds of variable are separable by eye. In the specificity arm
    there is one population and no biological difference between the sides at
    all; in the power arm a subpopulation is added to the source only.

    **Ground truth.** Which cells should be rejected, drawn as a ring. Nothing
    in the specificity arm; exactly the planted cells in the power arm. Both
    arms are needed and neither is sufficient: a method that never rejects
    scores perfectly on the first, and one that rejects everything scores
    perfectly on the recall of the second.

    **Evaluation.** What is computed from the method's output. Rejection is
    scored per cell against the ring, and technical invariance is scored as
    the correlation between the gate and the nuisance, which should be zero
    whatever the arm.
    """
    rng = np.random.default_rng(seed)
    n = 34
    planted = 10
    report: dict = {}

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.1, 3.8))
        for row, (arm, subtitle) in enumerate((
                ("specificity", "one population, drawn twice"),
                ("power", "a subpopulation added to the source only"))):
            axes = figure.add_axes([0.015, 0.545 - 0.455 * row, 0.62, 0.335])
            axes.set_xlim(-2.6, 12.4)
            axes.set_ylim(-1.18, 1.34)
            axes.axis("off")

            for column, side in enumerate(("source", "target")):
                x0 = 0.6 + 6.2 * column
                extra = planted if (arm == "power" and side == "source") else 0
                total = n + extra
                spread = rng.uniform(-1.0, 1.0, size=(total, 2))
                spread[:, 0] *= 2.1
                spread[:, 1] = 0.30 + 0.52 * spread[:, 1]
                # Size is the technical nuisance; colour is biology. Keeping
                # them in different channels is the first thing the figure has
                # to say, because a reader cannot judge a benchmark without
                # knowing which variation is supposed to be there.
                sizes = 9.0 + 52.0 * rng.beta(1.6, 2.2, size=total)
                colours = np.array(["#0072B2"] * n + ["#D55E00"] * extra)
                axes.scatter(spread[:, 0] + x0, spread[:, 1], s=sizes,
                             c=colours, linewidths=0, alpha=0.9, zorder=3)
                if extra:
                    for index in range(n, total):
                        axes.scatter([spread[index, 0] + x0], [spread[index, 1]],
                                     s=sizes[index] + 90, facecolors="none",
                                     edgecolors=C_REJECT, linewidths=1.1,
                                     zorder=5)
                axes.text(x0, 0.95, side, fontsize=7.6, ha="center",
                          color="#55555a")
            axes.text(-2.5, 1.14, f"({chr(97 + row)})  {arm} arm",
                      fontsize=8.4, fontweight="semibold", ha="left")
            axes.text(-2.5, 0.80, subtitle, fontsize=7.2, ha="left",
                      color="#55555a")

            truth = ("no cell is rejected" if arm == "specificity"
                     else f"exactly the {planted} planted cells are rejected")
            scored = ("false rejection rate" if arm == "specificity"
                      else "precision, recall, F1 against the ring")
            axes.text(-2.5, -0.62, "ground truth", fontsize=7.4,
                      fontweight="semibold", ha="left")
            axes.text(-2.5, -0.98, truth, fontsize=7.2, ha="left",
                      color="#2f7d4f" if arm == "specificity" else C_REJECT)
            axes.text(5.4, -0.62, "scored by", fontsize=7.4,
                      fontweight="semibold", ha="left")
            axes.text(5.4, -0.98, scored, fontsize=7.2, ha="left",
                      color="#55555a")
            report[arm] = {"ground_truth": truth, "scored_by": scored}

        # Its own 0-1 data coordinates rather than transAxes with clipping
        # off. Drawn the other way, the markers sat outside the axes as far as
        # matplotlib was concerned, and bbox_inches="tight" grew the canvas to
        # contain them -- a figure eleven thousand pixels tall.
        legend = figure.add_axes([0.655, 0.08, 0.335, 0.80])
        legend.set_xlim(0, 1)
        legend.set_ylim(0, 1)
        legend.axis("off")
        legend.text(0.0, 0.99, "generation", fontsize=8.4,
                    fontweight="semibold", va="top")
        legend.text(0.0, 0.91, "colour is biology", fontsize=7.0,
                    color="#8c8c86", va="top", style="italic")
        for y, colour, text in ((0.815, "#0072B2", "the shared population"),
                                (0.735, "#D55E00", "the planted subpopulation")):
            legend.scatter([0.04], [y], s=46, c=colour, linewidths=0)
            legend.text(0.12, y, text, fontsize=7.2, va="center")

        legend.text(0.0, 0.64,
                    "size is the technical nuisance,\napplied to both sides",
                    fontsize=7.0, color="#8c8c86", va="top", style="italic",
                    linespacing=1.6)
        for y, size, text in ((0.505, 14, "shallow, or few genes"),
                              (0.425, 58, "deep, or many genes")):
            legend.scatter([0.04], [y], s=size, c="#8c8c86", linewidths=0)
            legend.text(0.12, y, text, fontsize=7.2, va="center")
        legend.text(0.0, 0.345,
                    "depth sd(log) 0 to 0.9, or\nbreadth at fixed total counts",
                    fontsize=7.0, color="#8c8c86", va="top", linespacing=1.6)

        legend.text(0.0, 0.215, "ground truth", fontsize=8.4,
                    fontweight="semibold", va="top")
        legend.scatter([0.04], [0.135], s=46, facecolors="none",
                       edgecolors=C_REJECT, linewidths=1.1)
        legend.text(0.12, 0.135, "should be rejected", fontsize=7.2,
                    va="center", color=C_REJECT)
        legend.text(0.0, 0.055,
                    "There is no cell-to-cell pairing:\nthe sides are "
                    "independent draws,\nso the truth is which cells have a\n"
                    "partner in the other set at all.",
                    fontsize=7.0, color="#8c8c86", va="top", linespacing=1.6)

        figure.text(0.015, 0.012,
                    "Both arms carry the nuisance, and both are needed: a "
                    "method that never rejects is perfect on (a), and one "
                    "that rejects everything is perfect\non (b)'s recall. "
                    "Technical invariance is scored across both, as the "
                    "correlation between the gate and the nuisance.",
                    fontsize=7.2, color="#55555a", va="bottom",
                    linespacing=1.55)
        for suffix in ("png", "pdf"):
            figure.savefig(out / f"benchmark.{suffix}", bbox_inches="tight")
        plt.close(figure)
    report["technical_invariance"] = ("|corr(gate, nuisance)|, scored in both "
                                      "arms, expected zero")
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
