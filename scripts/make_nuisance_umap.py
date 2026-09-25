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
    primary cell is carried onto each metastatic cell. Below it the two things
    that get compared: the gate the coupling and its costs induce, one call
    per primary cell, and the mask that is correct.

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
        for side in ("primary", "metastasis"):
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
    report: dict = {}

    x_primary, x_meta, matrix_w = 0.100, 0.665, 0.290
    plan_w = 0.128
    x_plan = (x_primary + matrix_w + x_meta) / 2 - plan_w / 2
    x_gate = x_plan + plan_w / 2 - matrix_w / 2
    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.2, 8.0))
        for row, (nuisance, title) in enumerate((
                ("depth", "sequencing depth (nCount)"),
                ("breadth",
                 "gene detection (nFeature), nCount held fixed"))):
            base = 0.445 * row
            # No subtitle and no restatement of the answer: the title
            # names the variable and the solid ground-truth bar is the answer.
            figure.text(x_primary - 0.058, 0.972 - base,
                        f"({chr(97 + row)})  {title}", fontsize=8.6,
                        fontweight="semibold", ha="left", va="top")

            for x0, side in ((x_primary, "primary"), (x_meta, "metastasis")):
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
                    if x0 == x_primary:
                        strip.set_ylabel(name, fontsize=7.0, color=colour,
                                         rotation=0, ha="right", va="center",
                                         labelpad=5)
                    low, high = panel[key].min(), panel[key].max()
                    if key == "total" and high - low < 0.10 * high:
                        # Held fixed by construction: the residual spread is
                        # Poisson noise on the total, so quoting the extremes
                        # would read as a range the arm does not have.
                        strip.text(0.5, 1.45,
                                   f"{round(panel[key].mean(), -2):,.0f} in "
                                   "every cell", transform=strip.transAxes,
                                   fontsize=6.6, ha="center", va="bottom",
                                   color=colour)
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
                if x0 == x_primary:
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
            plan.set_ylabel("primary cells", fontsize=6.8, color="#55555a",
                            labelpad=4)
            plan.set_xlabel("metastatic cells", fontsize=6.8, color="#55555a",
                            labelpad=3)
            plan.set_title("OT coupling  $\\pi_{ij}$", fontsize=7.8, pad=3)
            # Clear of the coupling's own y label, which the arrow used to
            # run straight through.
            for x_from, x_to in (
                    (x_primary + matrix_w + 0.008, x_plan - 0.030),
                    (x_plan + plan_w + 0.008, x_meta - 0.008)):
                figure.add_artist(mpl.patches.FancyArrowPatch(
                    (x_from, 0.784 - base), (x_to, 0.784 - base),
                    transform=figure.transFigure, arrowstyle="-|>",
                    mutation_scale=8, linewidth=0.9, color="#9a9a94"))

            # Coupling and costs, down to one call per primary cell, against
            # the call that is correct.
            for index, (values, label) in enumerate((
                    (returned, "rejection gate"),
                    (truth, "ground truth"))):
                y = 0.640 - base - 0.072 * index
                figure.add_artist(mpl.patches.FancyArrowPatch(
                    (x_plan + plan_w / 2, y + 0.066),
                    (x_plan + plan_w / 2, y + 0.042),
                    transform=figure.transFigure, arrowstyle="-|>",
                    mutation_scale=7, linewidth=0.9, color="#9a9a94"))
                figure.text(x_plan + plan_w / 2, y + 0.036, label, fontsize=7.0,
                            color="#55555a", ha="center", va="top")
                gate = figure.add_axes([x_gate, y, matrix_w, 0.018])
                gate.imshow(values[None, :], aspect="auto",
                            cmap=mpl.colors.ListedColormap([C_DROP, C_KEEP]),
                            vmin=0, vmax=1, interpolation="nearest")
                gate.set_xticks([])
                gate.set_yticks([])
                for spine in gate.spines.values():
                    spine.set_color("#d8d8d2")
                    spine.set_linewidth(0.6)

            reference = panels[(nuisance, "primary")]
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
            [x_primary, 0.082, 0.135, 0.011]), orientation="horizontal")
        ticks = [value for value in (0, 1, 3, 10, 30, 100, 300)
                 if value <= float(np.expm1(top))]
        bar.set_ticks(np.log1p(ticks))
        bar.set_ticklabels([f"{value:,}" for value in ticks])
        bar.ax.tick_params(labelsize=6.5, length=2, pad=1.5)
        bar.outline.set_linewidth(0.5)
        figure.text(x_primary + 0.150, 0.088, "counts (white = 0)",
                    fontsize=7.2, color="#55555a", va="center")
        for index, (colour, text) in enumerate(((C_KEEP, "kept"),
                                                (C_DROP, "rejected"))):
            swatch = figure.add_axes([0.640 + 0.128 * index, 0.082, 0.020,
                                      0.011])
            swatch.set_xticks([])
            swatch.set_yticks([])
            swatch.set_facecolor(colour)
            for spine in swatch.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            figure.text(0.666 + 0.128 * index, 0.088, text, fontsize=7.2,
                        color="#55555a", va="center")
        # What the arm returns as numbers. "How many cells are wrongly
        # rejected" was prose; these are the four columns the run writes.
        figure.text(x_primary, 0.046,
                    "metrics, against the nuisance of each row:   false "
                    "rejections $\\to$ 0    AUC(rejected | nuisance) "
                    "$\\to$ 0.5", fontsize=7.2, color="#55555a", va="center")
        figure.text(x_primary + 0.196, 0.024,
                    "$\\rho$(cost, nuisance) $\\to$ 0    "
                    "$\\rho$(coupling, nuisance) $\\to$ 0",
                    fontsize=7.2, color="#55555a", va="center")
        figure.text(x_primary, 0.002, "coupling and gate are schematic",
                    fontsize=7.2, color="#8c8c86", va="center", style="italic")

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
