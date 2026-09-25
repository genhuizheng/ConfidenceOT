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
        "affected": "real: nFeature varies, nCount fixed",
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


def benchmark_figure(out: Path, cells: int, genes: int, depth_sd: float,
                     seed: int) -> dict:
    """The benchmark as a pipeline: counts, coupling, gate, answer.

    Each row reads left to right and then down. Two count matrices, one per
    sample, with the perturbation named on the strips above them. Between them
    the object the method actually produces -- the coupling, how much of each
    source cell is carried onto each target cell. Below it the two things
    that get compared: the gate the coupling and its costs induce, one call
    per source cell, and the mask that is correct -- in two cases. In one the
    same populations are in both samples, so the correct decision is to keep
    every cell. In the other a population is missing from the target, and the
    correct decision is to reject only that population. One case without the
    other is a task that can be passed by never rejecting, or by rejecting
    everything.

    The coupling and the gate here are schematic. This is the setting, not the
    result: what a real run puts in those two places is the measurement, and
    it belongs in a figure of its own, with the real n x n plan and both axes
    ordered by the nuisance.

    The perturbations stay as they were. In (a) both strips run the full scale
    together -- sequence a cell deeper and it detects more genes. In (b) the
    nCount strip is one flat colour and the nFeature strip still runs, so the
    white in the matrix is the only thing that changed, and no depth
    normalisation can reach it because the totals it would divide by already
    agree.

    The biology is identical in all four matrices -- one population, one
    expression profile, both samples -- so every cell has a counterpart and
    the ground-truth mask is solid: reject nothing.
    """
    rng = np.random.default_rng(seed)
    shown_cells, shown_genes, grid = 40, 60, 40
    profile = rng.gamma(shape=0.5, scale=3.0, size=genes) + 0.05
    profile = profile / profile.sum()
    # One profile for the whole figure, so the gene order is the same in all
    # four matrices. Every sixteenth gene by expression rather than a band off
    # the top: the genes a sparse cell loses are the low expressers, so a band
    # off the top would show none of it.
    gene_order = np.argsort(-profile)[:shown_genes * 16:16]

    C_COUNT, C_FEATURE = "#b07d2b", "#1f6f8b"
    C_KEEP, C_DROP = "#2f7d4f", "#c4553b"
    panels: dict[tuple[str, str], dict] = {}
    for nuisance in ("depth", "breadth"):
        for side in ("source", "target"):
            counts = one_population(nuisance, shown_cells, genes, depth_sd,
                                    rng, profile=profile)
            total = counts.sum(axis=1)
            detected = (counts > 0).sum(axis=1)
            order = np.argsort(total if nuisance == "depth" else detected)
            panels[(nuisance, side)] = {
                "matrix": np.log1p(counts[np.ix_(order, gene_order)]).T,
                "total": total[order], "detected": detected[order]}

    top = float(np.percentile(
        np.concatenate([q["matrix"].ravel() for q in panels.values()]), 99.5))
    # One scale per variable across all four matrices. Scaled per matrix
    # instead, the fixed nCount in (b) would be stretched over the full colour
    # map and its Poisson noise would look like the gradient in (a).
    limits = {key: (min(float(q[key].min()) for q in panels.values()),
                    max(float(q[key].max()) for q in panels.values()))
              for key in ("total", "detected")}
    # An illustrative disagreement, not a measurement: a method that follows
    # the nuisance rejects at the low end of the ordering, so that is where
    # the schematic puts it. Without any disagreement the two strips would be
    # the same solid bar and the comparison would not be legible.
    returned = np.ones(shown_cells, dtype=bool)
    returned[[0, 1, 2, 4, 5, 7, 10, 15]] = False
    truth = np.ones(shown_cells, dtype=bool)
    # A population with no counterpart is a fact about the biology, so its
    # cells sit wherever the nuisance ordering happens to put them. Scattered
    # is what a real unmatched population looks like against this axis, and it
    # is what distinguishes it from a gate that has followed the nuisance and
    # rejects at one end.
    unmatched = np.zeros(shown_cells, dtype=bool)
    unmatched[rng.choice(shown_cells, 9, replace=False)] = True
    truth_unmatched = ~unmatched
    returned_unmatched = truth_unmatched.copy()
    returned_unmatched[[1, 4]] = False
    returned_unmatched[int(np.flatnonzero(unmatched)[0])] = True
    report: dict = {}

    x_source, x_target, matrix_w = 0.100, 0.665, 0.290
    plan_w = 0.128
    x_plan = (x_source + matrix_w + x_target) / 2 - plan_w / 2
    case_x, case_w = (0.200, 0.590), 0.360
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.2, 8.0))
        for row, (nuisance, title) in enumerate((
                ("depth", "sequencing depth (nCount)"),
                ("breadth",
                 "gene detection (nFeature), nCount held fixed"))):
            base = 0.400 * row
            # No subtitle and no restatement of the answer: the title
            # names the variable and the solid ground-truth bar is the answer.
            figure.text(x_source - 0.058, 0.972 - base,
                        f"({chr(97 + row)})  {title}", fontsize=8.6,
                        fontweight="semibold", ha="left", va="top")

            for x0, side in ((x_source, "source"), (x_target, "target")):
                panel = panels[(nuisance, side)]
                figure.text(x0 + matrix_w / 2, 0.936 - base, side, fontsize=7.8,
                            color="#55555a", ha="center", va="bottom")

                for index, (key, colour, name) in enumerate((
                        ("total", C_COUNT, "nCount"),
                        ("detected", C_FEATURE, "nFeature"))):
                    strip = figure.add_axes(
                        [x0, 0.898 - base - 0.036 * index, matrix_w, 0.014])
                    strip.imshow(panel[key][None, :], aspect="auto",
                                 cmap="viridis", vmin=limits[key][0],
                                 vmax=limits[key][1], interpolation="nearest")
                    strip.set_xticks([])
                    strip.set_yticks([])
                    for spine in strip.spines.values():
                        spine.set_color("#d8d8d2")
                        spine.set_linewidth(0.6)
                    if x0 == x_source:
                        strip.set_ylabel(name, fontsize=7.0, color=colour,
                                         rotation=0, ha="right", va="center",
                                         labelpad=5)
                    low, high = panel[key].min(), panel[key].max()
                    if key == "total" and high - low < 0.10 * high:
                        # Held fixed by construction, which the row title
                        # already says. Printing the mean on a bar whose point
                        # is that it has no range invited it to be read as one.
                        pass
                    else:
                        # The cells are sorted, so the ends of the strip are
                        # the ends of the range and no colour bar is needed.
                        for at, value, align in ((0.0, low, "left"),
                                                 (1.0, high, "right")):
                            strip.text(at, 1.45, f"{value:,.0f}",
                                       transform=strip.transAxes, fontsize=6.6,
                                       ha=align, va="bottom", color=colour)

                heat = figure.add_axes([x0, 0.726 - base, matrix_w, 0.115])
                image = heat.imshow(panel["matrix"], aspect="auto",
                                    cmap="Blues", vmin=0.0, vmax=top,
                                    interpolation="nearest")
                heat.set_xticks([])
                heat.set_yticks([])
                for spine in heat.spines.values():
                    spine.set_color("#d8d8d2")
                    spine.set_linewidth(0.6)
                if x0 == x_source:
                    heat.set_ylabel("genes", fontsize=7.0,
                                    color="#55555a", labelpad=5)
                heat.set_xlabel(
                    f"ordered by "
                    f"{'nCount' if nuisance == 'depth' else 'nFeature'}",
                    fontsize=7.0, color="#55555a", labelpad=3)

            # The object the method produces, between the two samples it is
            # built from.
            plan = figure.add_axes([x_plan, 0.726 - base, plan_w, 0.115])
            plan.imshow(schematic_coupling(grid, rng), cmap="Greys",
                        aspect="auto", interpolation="bicubic")
            plan.set_xticks([])
            plan.set_yticks([])
            for spine in plan.spines.values():
                spine.set_color("#9a9a94")
                spine.set_linewidth(0.7)
            plan.set_ylabel("source cells", fontsize=6.8, color="#55555a",
                            labelpad=4)
            plan.set_xlabel("target cells", fontsize=6.8, color="#55555a",
                            labelpad=3)
            plan.set_title("OT coupling  $\\pi_{ij}$", fontsize=7.8, pad=3)
            # Clear of the coupling's own y label, which the arrow used to
            # run straight through.
            for x_from, x_to in (
                    (x_source + matrix_w + 0.008, x_plan - 0.030),
                    (x_plan + plan_w + 0.008, x_target - 0.008)):
                figure.add_artist(mpl.patches.FancyArrowPatch(
                    (x_from, 0.784 - base), (x_to, 0.784 - base),
                    transform=figure.transFigure, arrowstyle="-|>",
                    mutation_scale=8, linewidth=0.9, color="#9a9a94"))

            # Coupling and costs, down to one call per source cell, against
            # the call that is correct -- in the two cases the perturbation
            # above has to be read against. Without the second column a method
            # that never rejects would be right about everything on the page.
            figure.add_artist(mpl.patches.FancyArrowPatch(
                (x_plan + plan_w / 2, 0.708 - base),
                (x_plan + plan_w / 2, 0.688 - base),
                transform=figure.transFigure, arrowstyle="-|>",
                mutation_scale=7, linewidth=0.9, color="#9a9a94"))
            for index, label in enumerate(("gate (schematic)", "ground truth")):
                figure.text(case_x[0] - 0.012, 0.657 - base - 0.042 * index,
                            label, fontsize=7.0, color="#55555a", ha="right",
                            va="center")
            # Two layers, because they are two different statements. The
            # header says what is in the samples; the line above the mask says
            # what the correct decision is. Run together as one phrase, a
            # reader cannot tell which part is the setting and which part is
            # the answer being scored against.
            for column, (header, decision, pair) in enumerate((
                    ("the same populations in source and target",
                     "keep every cell", (returned, truth)),
                    ("one population is missing from the target",
                     "reject only that population",
                     (returned_unmatched, truth_unmatched)))):
                figure.text(case_x[column] + case_w / 2, 0.682 - base, header,
                            fontsize=7.0, color="#55555a", ha="center",
                            va="top")
                figure.text(case_x[column] + case_w / 2, 0.627 - base,
                            decision, fontsize=7.0, color="#55555a",
                            ha="center", va="bottom")
                for index, values in enumerate(pair):
                    gate = figure.add_axes(
                        [case_x[column], 0.648 - base - 0.042 * index,
                         case_w, 0.018])
                    gate.imshow(values[None, :], aspect="auto",
                                cmap=mpl.colors.ListedColormap([C_DROP, C_KEEP]),
                                vmin=0, vmax=1, interpolation="nearest")
                    gate.set_xticks([])
                    gate.set_yticks([])
                    for spine in gate.spines.values():
                        spine.set_color("#d8d8d2")
                        spine.set_linewidth(0.6)

            reference = panels[(nuisance, "source")]
            report[title] = {
                "biology": "one population, identical in both samples",
                "correct_answer": "reject nothing",
                "nCount_range": [float(reference["total"].min()),
                                 float(reference["total"].max())],
                "nFeature_range": [int(reference["detected"].min()),
                                   int(reference["detected"].max())],
                "coupling": "schematic, Sinkhorn-scaled noise",
                "gate": "schematic, an illustrative disagreement",
                "scored_by": ("fraction wrongly rejected, and whether "
                              "rejection follows the nuisance")}

        bar = figure.colorbar(image, cax=figure.add_axes(
            [x_source, 0.150, 0.135, 0.011]), orientation="horizontal")
        ticks = [value for value in (0, 1, 3, 10, 30, 100, 300)
                 if value <= float(np.expm1(top))]
        bar.set_ticks(np.log1p(ticks))
        bar.set_ticklabels([f"{value:,}" for value in ticks])
        bar.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        bar.outline.set_linewidth(0.5)
        figure.text(x_source + 0.150, 0.156, "counts (white = 0)",
                    fontsize=7.2, color="#55555a", va="center")
        for index, (colour, text) in enumerate(((C_KEEP, "kept"),
                                                (C_DROP, "rejected"))):
            swatch = figure.add_axes([0.640 + 0.128 * index, 0.150, 0.020,
                                      0.011])
            swatch.set_xticks([])
            swatch.set_yticks([])
            swatch.set_facecolor(colour)
            for spine in swatch.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            figure.text(0.666 + 0.128 * index, 0.156, text, fontsize=7.2,
                        color="#55555a", va="center")
        # Nothing is written under the legend. The metrics are definitions
        # and live in BENCHMARK_METRICS.md; that the coupling and the gate are
        # drawn rather than run belongs in the caption of whatever the figure
        # is placed in, not inside the figure.

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
