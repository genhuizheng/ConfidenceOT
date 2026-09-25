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


def dropout_weight(profile: np.ndarray) -> np.ndarray:
    """Per-gene capture loss, tilted towards the low expressers.

    A uniform mask would take the housekeeping genes at the same rate and
    would not look like dropout. Tilted too steeply, though, it removes almost
    no counts -- the low expressers carry little of the library -- and the
    sample comes out with its total intact, which is the fixed-total
    construction this design exists to avoid. This sits between the two: the
    top of the distribution loses about half as much as the bottom.
    """
    return 0.45 + 0.55 / (1.0 + profile / np.median(profile))


def sample_counts(profile: np.ndarray, n_cells: int,
                  rng: np.random.Generator, *, median_library: float = 3000.0,
                  depth_sd: float = 0.0, dropout: float = 0.0) -> np.ndarray:
    """One sample of one population under one observation setting.

    The three knobs are the observation, not the biology: every cell draws
    from the same ``profile``, so whatever structure an embedding of this
    shows is the measurement. ``depth_sd`` spreads the library size,
    ``dropout`` removes counts on top of it, and the two are meant to be used
    in that order -- dropout is an addition to a depth mismatch, not an
    alternative to one.
    """
    counts = np.array([
        rng.poisson(profile * median_library * rng.lognormal(0.0, depth_sd))
        for _ in range(n_cells)], dtype=float)
    if dropout > 0.0:
        counts *= rng.random(counts.shape) > dropout * dropout_weight(profile)
    return counts


def shared_profile(n_genes: int, rng: np.random.Generator) -> np.ndarray:
    """One relative expression profile, shared by every cell in a figure."""
    profile = rng.gamma(shape=0.5, scale=3.0, size=n_genes) + 0.05
    return profile / profile.sum()


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
    "dropout": {
        "title": "depth + dropout",
        "ideal": "ideal: no depth variation, no dropout",
        "affected": "real: depth varies, plus dropout",
        "colour_by": "genes detected per cell",
        "construction": ("the depth spread of the row above, with\n"
                         "capture loss applied on top of it"),
        "why_both": ("The second row is the first row plus dropout, not a\n"
                     "different axis. Both readouts fall, and detection\n"
                     "falls further than the total, which is why nFeature\n"
                     "cannot be treated as a knob of its own."),
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
        for row, nuisance in enumerate(("depth", "dropout")):
            facts = NUISANCE[nuisance]
            inner = outer[row].subgridspec(1, 2, wspace=0.05)
            for column, kind in enumerate(("ideal", "affected")):
                rng = np.random.default_rng(seed + row)
                # Both rows are compared against the same clean ideal, so the
                # second panel of the second row shows what depth and dropout
                # do together rather than what dropout adds in isolation.
                affected = kind == "affected"
                counts = sample_counts(
                    shared_profile(genes, rng), cells, rng,
                    depth_sd=depth_sd if affected else 0.0,
                    dropout=(0.55 if affected and nuisance == "dropout"
                             else 0.0))
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
    profile = shared_profile(n_genes, rng)

    def sample(median_library: float, dropout: float) -> np.ndarray:
        return sample_counts(profile, n_cells, rng,
                             median_library=median_library,
                             depth_sd=0.35, dropout=dropout)

    return profile, {
        "reference source": sample(3000.0, 0.0),
        "matched": sample(3000.0, 0.0),
        "depth mismatch": sample(1350.0, 0.0),
        "depth + dropout": sample(1350.0, 0.55),
    }


def strip_row(figure, x0: float, y: float, width: float, height: float,
              values: np.ndarray, cmap, vmin: float = 0.0,
              vmax: float = 1.0) -> None:
    """One horizontal track, drawn the same way everywhere it appears."""
    axes = figure.add_axes([x0, y, width, height])
    axes.imshow(values[None, :], aspect="auto", cmap=cmap, vmin=vmin,
                vmax=vmax, interpolation="nearest")
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_color("#d8d8d2")
        spine.set_linewidth(0.6)


def benchmark_figure(out: Path, cells: int, genes: int, depth_sd: float,
                     seed: int) -> dict:
    """The setting, and the two answers it is scored against.

    (a) defines the technical setting: one source against three targets, each
    measured worse than the last. Degradation arrives in stages because nCount
    and nFeature are not separable -- moving the library size moves detection
    with it, and dropout moves both again.

    (b) is not a second pipeline drawn over again. It is the pair of
    ground-truth regimes the same framework is evaluated under: one where the
    biology is shared and nothing should be rejected, and one where a
    population exists on one side only and just those cells should be. The
    example gates are drawn, not run; they are there so the comparison has
    something to compare against.
    """
    rng = np.random.default_rng(seed)
    shown_cells, shown_genes = 40, 60
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

    levels = ("matched", "depth mismatch", "depth + dropout")
    column_w, column_x = 0.185, (0.165, 0.370, 0.575, 0.780)
    report: dict = {}

    with mpl.rc_context(STYLE):
        figure = plt.figure(figsize=(7.6, 6.6))

        # ---- (a) the technical setting --------------------------------------
        figure.text(0.035, 0.983, "(a)  technical degradation", fontsize=8.6,
                    fontweight="semibold", ha="left", va="top")
        figure.text(0.035, 0.960,
                    "the biology is unchanged; only the measurement degrades",
                    fontsize=7.4, color="#55555a", ha="left", va="top")

        for index, name in enumerate(("reference source",) + levels):
            panel, x0 = panels[name], column_x[index]
            figure.text(x0 + column_w / 2, 0.926, name, fontsize=7.8,
                        fontweight="semibold" if index else "normal",
                        color="#2b2b2b" if index else "#55555a",
                        ha="center", va="top")
            if index:
                for line, key in enumerate(("total", "detected")):
                    ratio = np.median(panel[key]) / np.median(reference[key])
                    figure.text(
                        x0 + column_w / 2, 0.902 - 0.022 * line,
                        f"{'nCount' if key == 'total' else 'nFeature'}"
                        f"  x{ratio:.2f}", fontsize=6.8,
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
                    [x0, 0.845 - 0.030 * line, column_w, 0.018])
                strip.imshow(panel[key][None, :], aspect="auto", cmap="viridis",
                             vmin=limits[key][0], vmax=limits[key][1],
                             interpolation="nearest")
                strip.set_xticks([])
                strip.set_yticks([])
                for spine in strip.spines.values():
                    spine.set_color("#d8d8d2")
                    spine.set_linewidth(0.6)
                if index == 0:
                    strip.set_ylabel(label, fontsize=7.0, color=colour,
                                     rotation=0, ha="right", va="center",
                                     labelpad=6)

            heat = figure.add_axes([x0, 0.596, column_w, 0.194])
            image = heat.imshow(panel["matrix"], aspect="auto", cmap="Blues",
                                vmin=0.0, vmax=top, interpolation="nearest")
            heat.set_xticks([])
            heat.set_yticks([])
            for spine in heat.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            if index == 0:
                heat.set_ylabel("expression", fontsize=7.0, color="#55555a",
                                labelpad=6)
                figure.text(x0, 0.563, "one source, shared by all three levels",
                            fontsize=6.8, color="#55555a", ha="left", va="top")
            heat.set_xlabel("ordered by nCount", fontsize=6.6,
                            color="#55555a", labelpad=3)

        figure.text(0.035, 0.535,
                    "all biological populations remain matchable at every "
                    "level", fontsize=7.6, color=C_KEEP, ha="left", va="top")

        # ---- (b) the two ground-truth regimes --------------------------------
        figure.text(0.035, 0.480, "(b)  evaluation cases", fontsize=8.6,
                    fontweight="semibold", ha="left", va="top")

        n = 48
        case_w, case_x = 0.380, (0.165, 0.585)
        gate_map = mpl.colors.ListedColormap([C_DROP, C_KEEP])
        unmatched = np.zeros(n, dtype=bool)
        unmatched[8:16] = True          # one population, cells sit together

        shared_gate = np.ones(n, dtype=bool)
        shared_gate[[2, 3, 29, 41]] = False
        unmatched_gate = ~unmatched
        unmatched_gate[[6, 7]] = False
        unmatched_gate[15] = True

        for column, (title, gate, truth, reading) in enumerate((
                ("Case 1: all populations shared", shared_gate,
                 np.ones(n, dtype=bool), "no cell should be rejected"),
                ("Case 2: one population unmatched", unmatched_gate,
                 ~unmatched,
                 "reject only cells from the unmatched population"))):
            x0 = case_x[column]
            figure.text(x0, 0.452, title, fontsize=7.8,
                        fontweight="semibold", ha="left", va="top")
            for line, values in enumerate((gate, truth)):
                y = 0.400 - 0.040 * line
                strip_row(figure, x0, y, case_w, 0.024,
                          values.astype(float), gate_map)
                if column == 0:
                    figure.text(x0 - 0.010, y + 0.012,
                                ("example gate", "ground truth")[line],
                                fontsize=7.0, color="#55555a", ha="right",
                                va="center")
            figure.text(x0, 0.348, reading, fontsize=7.4,
                        color=C_KEEP if column == 0 else C_DROP,
                        ha="left", va="top")

        figure.text(0.035, 0.300,
                    "two ground-truth regimes are evaluated: all shared, and "
                    "one unmatched", fontsize=7.6, color="#55555a",
                    ha="left", va="top")

        # ---- legend ----------------------------------------------------------
        bar = figure.colorbar(image, cax=figure.add_axes(
            [0.165, 0.245, 0.110, 0.013]), orientation="horizontal")
        ticks = [value for value in (0, 1, 3, 10, 30, 100, 300)
                 if value <= float(np.expm1(top))]
        bar.set_ticks(np.log1p(ticks))
        bar.set_ticklabels([f"{value:,}" for value in ticks])
        bar.ax.tick_params(labelsize=6.2, length=2, pad=1.5)
        bar.outline.set_linewidth(0.5)
        figure.text(0.290, 0.2515, "counts (white = 0)", fontsize=7.0,
                    color="#55555a", va="center")
        for index, (colour, text) in enumerate(((C_KEEP, "kept"),
                                                (C_DROP, "rejected"))):
            swatch = figure.add_axes([0.470 + 0.130 * index, 0.245, 0.018,
                                      0.013])
            swatch.set_xticks([])
            swatch.set_yticks([])
            swatch.set_facecolor(colour)
            for spine in swatch.spines.values():
                spine.set_color("#d8d8d2")
                spine.set_linewidth(0.6)
            figure.text(0.494 + 0.130 * index, 0.2515, text, fontsize=7.0,
                        color="#55555a", va="center")
        figure.text(0.035, 0.210,
                    "nCount reflects sequencing depth; nFeature reflects both "
                    "depth and additional detection loss", fontsize=6.8,
                    color="#8c8c86", ha="left", va="top", style="italic")

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
