"""The two technical nuisances, as a problem statement and as a benchmark.

Every cell in the problem figure is drawn from one population, so there is no
biology to find: the cells differ only in how deeply they were sequenced, or
in how many genes were detected. Any structure the embedding shows is the
artefact and nothing else, which is something a reader can check by looking.

The two are not the same thing and the figures keep them apart.

**Sequencing depth, nCount.** Cells get different total counts. This is what
most normalisations are built to remove.

**Gene detection, nFeature.** Total counts are held *fixed* and the cells
differ in how many distinct genes those counts land on. Depth normalisation
cannot touch it, because the totals it would divide by already agree.

Two figures are written. ``problem.png`` puts the embedding one wants beside
the embedding one gets, for each nuisance. ``benchmark.png`` states the
setting the method is scored in: counts, an OT coupling, a rejection gate and
the mask that is correct, in the case where the same populations are in both
samples and in the case where one is missing from the target. The coupling and
the gate there are drawn, not run.

Usage:

    python scripts/make_nuisance_umap.py
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



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=int, default=900)
    parser.add_argument("--genes", type=int, default=1200)
    parser.add_argument("--depth-sd", type=float, default=0.9,
                        help="sd of log depth, the spread the benchmark uses")
    parser.add_argument("--seed", type=int, default=7300)
    parser.add_argument("--out", type=Path, default=Path("figures/schematic"),
                        help="Every schematic is written to one folder, "
                             "apart from the results")
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
        # Named for what the panel is rather than for what was done to it, and
        # in the same words as the benchmark figure: nCount and nFeature.
        "ideal": "ideal: no nCount variation",
        "affected": "real: nCount varies",
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
        "ideal": "ideal: no nFeature variation",
        "affected": "real: nFeature varies",
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


def schematic_coupling(grid: int, rng: np.random.Generator) -> np.ndarray:
    """A transport plan with uniform marginals, and no message in its shape.

    Sinkhorn-scaled noise rather than anything from a run: the marginals come
    out flat, which is what makes it a coupling, while the interior carries no
    structure. Drawing a band down the diagonal here would assert the answer
    the result figure is meant to measure -- with both sides ordered by the
    nuisance, a band along the diagonal is exactly what "the coupling followed
    the nuisance" looks like.
    """
    plan = rng.gamma(2.0, 1.0, size=(grid, grid))
    for _ in range(4):
        plan /= plan.sum(axis=1, keepdims=True)
        plan /= plan.sum(axis=0, keepdims=True)
    return plan


def degradation_levels(n_cells: int, n_genes: int,
                       rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """One reference sample and three targets, each worse than the last.

    The three levels are stages of the same degradation, not independent
    axes. Level 1 moves the library size and lets detection follow it. Level 2
    keeps that shift and drops counts on top, so both readouts fall further --
    which is what dropout does, and why the earlier framing of a
    detection-only arm at a fixed total was wrong.

    Dropout is weighted towards the low expressers, the way real capture loss
    is; a uniform mask would take the housekeeping genes at the same rate and
    would not look like anything.
    """
    profile = rng.gamma(shape=0.5, scale=3.0, size=n_genes) + 0.05
    profile = profile / profile.sum()
    loss = 0.45 + 0.55 / (1.0 + profile / np.median(profile))

    def sample(median_library: float, dropout: float) -> np.ndarray:
        counts = np.array([
            rng.poisson(profile * median_library * rng.lognormal(0.0, 0.35))
            for _ in range(n_cells)], dtype=float)
        if dropout > 0.0:
            counts *= rng.random(counts.shape) > dropout * loss[None, :]
        return counts

    return profile, {
        "reference source": sample(3000.0, 0.0),
        "matched": sample(3000.0, 0.0),
        "depth mismatch": sample(1350.0, 0.0),
        "depth + dropout": sample(1350.0, 0.55),
    }


def benchmark_figure(out: Path, cells: int, genes: int, depth_sd: float,
                     seed: int) -> dict:
    """The benchmark setting: one reference sample against three targets.

    Technical degradation arrives in stages -- matched observation, a
    sequencing-depth mismatch, and that mismatch with dropout on top -- rather
    than as two axes that can be varied independently. nCount is a direct
    readout of sequencing depth; nFeature is a downstream readout of both
    depth and the extra detection loss, which is why the realised ratio under
    each column is printed for both and why neither is called a knob.

    The source is drawn once and shared, so the three gate strips are the same
    cells in the same order and a reader can watch rejections appear as the
    measurement gets worse. The coupling and the gates are drawn, not run.

    The biology is identical in all four samples, so the correct answer is the
    same at every level and the ground-truth bar spans all three.
    """
    rng = np.random.default_rng(seed)
    shown_cells, shown_genes, grid = 40, 60, 40
    profile, samples = degradation_levels(shown_cells, genes, rng)
    gene_order = np.argsort(-profile)[:shown_genes * 16:16]

    C_COUNT, C_FEATURE = "#b07d2b", "#1f6f8b"
    C_KEEP, C_DROP = "#2f7d4f", "#c4553b"
    panels: dict[str, dict] = {}
    for name, counts in samples.items():
        total = counts.sum(axis=1)
        detected = (counts > 0).sum(axis=1)
        order = np.argsort(total)
        panels[name] = {
            "matrix": np.log1p(counts[np.ix_(order, gene_order)]).T,
            "total": total[order], "detected": detected[order]}

    reference = panels["reference source"]
    top = float(np.percentile(
        np.concatenate([q["matrix"].ravel() for q in panels.values()]), 99.5))
    limits = {key: (min(float(q[key].min()) for q in panels.values()),
                    max(float(q[key].max()) for q in panels.values()))
              for key in ("total", "detected")}
    # Illustrative gates, not measurements: a method that follows the
    # measurement rejects at the shallow end of its own ordering, and it does
    # so more as the target degrades. The source cells are shared, so the
    # three strips are directly comparable cell by cell.
    gates = {"matched": 0, "depth mismatch": 5, "depth + dropout": 11}

    levels = ("matched", "depth mismatch", "depth + dropout")
    column_w, column_x = 0.185, (0.115, 0.328, 0.541, 0.754)
    report: dict = {}

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.6, 7.2))
        for index, name in enumerate(("reference source",) + levels):
            panel, x0 = panels[name], column_x[index]
            figure.text(x0 + column_w / 2, 0.975, name, fontsize=8.0,
                        fontweight="semibold" if index else "normal",
                        color="#2b2b2b" if index else "#55555a",
                        ha="center", va="top")
            if index:
                # Measured, not set. Printing both ratios is what says that
                # detection is not being held anywhere.
                for line, key in enumerate(("total", "detected")):
                    ratio = (np.median(panel[key])
                             / np.median(reference[key]))
                    figure.text(
                        x0 + column_w / 2, 0.950 - 0.022 * line,
                        f"{'nCount' if key == 'total' else 'nFeature'}"
                        f"  x{ratio:.2f}", fontsize=7.0,
                        color=C_COUNT if key == "total" else C_FEATURE,
                        ha="center", va="top")
                report[name] = {
                    "nCount_ratio": round(float(
                        np.median(panel["total"]) / np.median(reference["total"])), 3),
                    "nFeature_ratio": round(float(
                        np.median(panel["detected"]) / np.median(reference["detected"])), 3)}

            for line, (key, colour, label) in enumerate((
                    ("total", C_COUNT, "nCount"),
                    ("detected", C_FEATURE, "nFeature"))):
                strip = figure.add_axes(
                    [x0, 0.898 - 0.026 * line, column_w, 0.015])
                strip.imshow(panel[key][None, :], aspect="auto", cmap="viridis",
                             vmin=limits[key][0], vmax=limits[key][1],
                             interpolation="nearest")
                strip.set_xticks([])
                strip.set_yticks([])
                for spine in strip.spines.values():
                    spine.set_color("#d8d8d2")
                    spine.set_linewidth(0.6)
                if index == 0:
                    strip.set_ylabel(label, fontsize=7.2, color=colour,
                                     rotation=0, ha="right", va="center",
                                     labelpad=6)

            heat = figure.add_axes([x0, 0.718, column_w, 0.144])
            image = heat.imshow(panel["matrix"], aspect="auto", cmap="Blues",
                                vmin=0.0, vmax=top, interpolation="nearest")
            heat.set_xticks([])
            heat.set_yticks([])
            for spine in heat.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            if index == 0:
                heat.set_ylabel("expression", fontsize=7.2, color="#55555a",
                                labelpad=6)
                # Left-aligned and narrow: centred, it ran into the
                # coupling's rotated y label in the column beside it.
                figure.text(x0, 0.655,
                            "one source,\nshared by\nall three levels",
                            fontsize=7.2, color="#55555a", ha="left",
                            va="top", linespacing=1.5)
            heat.set_xlabel("ordered by nCount", fontsize=6.8,
                            color="#55555a", labelpad=3)

        for index, name in enumerate(levels):
            x0 = column_x[index + 1]
            plan = figure.add_axes([x0, 0.489, column_w, 0.195])
            plan.imshow(schematic_coupling(grid, rng), cmap="Greys",
                        aspect="auto", interpolation="bicubic")
            plan.set_xticks([])
            plan.set_yticks([])
            for spine in plan.spines.values():
                spine.set_color("#9a9a94")
                spine.set_linewidth(0.7)
            if index == 0:
                plan.set_ylabel("OT coupling  $\\pi$\nsource x target",
                                fontsize=7.2, color="#55555a", labelpad=6,
                                linespacing=1.5)

            keep = np.ones(shown_cells, dtype=bool)
            if gates[name]:
                keep[:gates[name]] = False
            gate = figure.add_axes([x0, 0.428, column_w, 0.018])
            gate.imshow(keep[None, :], aspect="auto",
                        cmap=mpl.colors.ListedColormap([C_DROP, C_KEEP]),
                        vmin=0, vmax=1, interpolation="nearest")
            gate.set_xticks([])
            gate.set_yticks([])
            for spine in gate.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            if index == 0:
                gate.set_ylabel("example gate", fontsize=7.2, color="#55555a",
                                rotation=0, ha="right", va="center",
                                labelpad=6)

        span_x = column_x[1]
        span_w = column_x[3] + column_w - span_x
        figure.text(span_x, 0.392, "ground truth", fontsize=7.2,
                    color="#55555a", va="bottom")
        truth = figure.add_axes([span_x, 0.366, span_w, 0.020])
        truth.imshow(np.ones((1, shown_cells * 3)), aspect="auto",
                     cmap=mpl.colors.ListedColormap([C_KEEP]),
                     interpolation="nearest")
        truth.set_xticks([])
        truth.set_yticks([])
        for spine in truth.spines.values():
            spine.set_color("#d8d8d2")
            spine.set_linewidth(0.6)
        figure.text(span_x + span_w / 2, 0.340,
                    "all biological populations remain matchable at every level",
                    fontsize=7.4, color="#2f7d4f", ha="center", va="top")

        bar = figure.colorbar(image, cax=figure.add_axes(
            [column_x[0], 0.286, 0.125, 0.011]), orientation="horizontal")
        ticks = [value for value in (0, 1, 3, 10, 30, 100, 300)
                 if value <= float(np.expm1(top))]
        bar.set_ticks(np.log1p(ticks))
        bar.set_ticklabels([f"{value:,}" for value in ticks])
        bar.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        bar.outline.set_linewidth(0.5)
        figure.text(column_x[0] + 0.140, 0.292, "counts (white = 0)",
                    fontsize=7.2, color="#55555a", va="center")
        for index, (colour, text) in enumerate(((C_KEEP, "kept"),
                                                (C_DROP, "rejected"))):
            swatch = figure.add_axes([0.600 + 0.120 * index, 0.286, 0.018,
                                      0.011])
            swatch.set_xticks([])
            swatch.set_yticks([])
            swatch.set_facecolor(colour)
            for spine in swatch.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            figure.text(0.624 + 0.120 * index, 0.292, text, fontsize=7.2,
                        color="#55555a", va="center")
        figure.text(column_x[0], 0.256,
                    "nCount reads out sequencing depth; nFeature reads out "
                    "both depth and the extra detection loss, so they are not "
                    "independent axes", fontsize=7.2, color="#8c8c86",
                    va="center", style="italic")

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
