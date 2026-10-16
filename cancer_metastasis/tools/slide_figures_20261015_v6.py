"""Slide-size redraws of the deck's matplotlib figures (ConfidenceOT_20261015 slides 9-18).

The original figures were drawn large (the GSEA dot plots about 17.5 x 12.8 in) and shrunk
onto the slide, so their 7-10 pt text showed at 4-6 pt. Here every figure is drawn at the
size it occupies on the slide, so font sizes are the sizes seen. Nothing is recomputed:
each figure is replotted from the value table its original wrote, and the encodings
(colour scales, limits, size bins, outlines, row and column order) are the original's.

* comparator heatmap: method_comparison_heatmap_values.csv, split into ConfidenceOT
  (6 strategies x 3 arms) and the comparators (7 x 3), one colour scale;
* directional F1: method_comparison_f1.csv;
* GSEA dot plots: gsea_dotplot_<dataset>.csv of the factorial (32 arms, split 16 + 16)
  and cycling x CNV layouts; all 50 hallmark sets on every panel.

    python cancer_metastasis/tools/slide_figures_20261015_v6.py

Writes PNG and PDF to figures/slides_20261015_v6/ and a manifest listing every figure's
source table and md5. Existing figures are not touched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "slides_20261015_v6"
COMPARATOR = REPO / "figures" / "slides_20261015" / "comparator"
DOWN = REPO / "results" / "downstream_20261007" / "figures"
DATASETS = (("ovarian", "gsea_dotplot_ovarian_gse180661"), ("colorectal", "gsea_dotplot_colorectal_gse225857"),
            ("prostate", "gsea_dotplot_prostate_gse271675"))

# Slide content area below the title, inches.
AREA_W, AREA_H = 12.6, 6.2
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})


def load(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


comparison = load("plot55", REPO / "benchmark" / "55_plot_method_comparison.py")
dotplot = load("dotplot47", REPO / "cancer_metastasis" / "47_gsea_dotplot.py")
INK, MUTED, RULE = comparison.INK, comparison.MUTED, comparison.RULE


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def save(fig, stem: str, sources: list[Path], manifest: list) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)
    manifest.append({"figure": f"{stem}.png", "size_in": None,
                     "sources": {str(p.relative_to(REPO)): md5(p) for p in sources}})


# --- comparator heatmap -----------------------------------------------------------------------

def comparator_heatmap(manifest: list) -> None:
    source = COMPARATOR / "method_comparison_heatmap_values.csv"
    values = pd.read_csv(source).set_index(["strategy", "arm"])
    strategies, arms = comparison.STRATEGIES, comparison.ARMS
    assert list(values.index) == [(s, a) for s in strategies for a in arms], "row order changed"
    readouts, levels = comparison.READOUTS, comparison.LEVELS
    n_cols = len(readouts) * len(levels)
    assert values.shape[1] == n_cols
    cmap = LinearSegmentedColormap.from_list("rejected", comparison.RAMP)
    cmap.set_bad(comparison.UNDEFINED)
    parts = (("confidenceot", [s for s in strategies if s.startswith("ConfidenceOT")]),
             ("comparators", [s for s in strategies if not s.startswith("ConfidenceOT")]))
    assert sum(len(p) for _, p in parts) == len(strategies)
    for stem, chosen in parts:
        rows = [(s, a) for s in chosen for a in arms]
        matrix = values.loc[rows].to_numpy(float)
        n_rows = len(rows)
        method_w, arm_w, right = 2.55, 1.85, 0.95
        top, bottom = 1.7, 0.35
        cell_w = (AREA_W - method_w - arm_w - right) / n_cols
        cell_h = min(0.25, (AREA_H - top - bottom) / n_rows)
        fig_h = top + n_rows * cell_h + bottom
        fig = plt.figure(figsize=(AREA_W, fig_h))
        left = method_w + arm_w
        ax = fig.add_axes([left / AREA_W, bottom / fig_h, n_cols * cell_w / AREA_W, n_rows * cell_h / fig_h])
        mesh = ax.pcolormesh(np.ma.masked_invalid(matrix), cmap=cmap, vmin=0.0, vmax=1.0,
                             edgecolors="#ffffff", linewidth=1.4)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(n_rows, 0)
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels([arm for _, arm in rows], fontsize=10, color=INK)
        ax.tick_params(axis="y", length=0, pad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for boundary in range(len(arms), n_rows, len(arms)):
            ax.axhline(boundary, color="#ffffff", linewidth=5)
        for boundary in range(len(levels), n_cols, len(levels)):
            ax.axvline(boundary, color="#ffffff", linewidth=5)
        for g, strategy in enumerate(chosen):
            method, rule = strategy.split(": ", 1)
            middle = bottom + (n_rows - g * len(arms) - len(arms) / 2) * cell_h
            # Name, its setting and the decision rule on three short lines, clear of the arm names.
            name, _, setting = method.replace("acceptance >= ", "acceptance ≥ ").partition(" (")
            setting = setting.rstrip(")")
            detail = f"{setting}, {rule}" if setting and len(setting) + len(rule) < 30 else None
            lines = [setting, rule] if setting and detail is None else [detail or rule]
            fig.text(0.05 / AREA_W, (middle + 0.10) / fig_h, name, ha="left", va="bottom", fontsize=10.5,
                     fontweight="semibold", color=INK)
            fig.text(0.05 / AREA_W, (middle + 0.07) / fig_h, "\n".join(lines), ha="left", va="top",
                     fontsize=10, color=MUTED, linespacing=1.1)
        for col in range(n_cols):
            ax.text(col + 0.5, n_rows + 0.25, levels[col % len(levels)][1], ha="center", va="top",
                    fontsize=9.5, color=MUTED)
        groups: dict[str, list[int]] = {}
        units = 1.0 / cell_h  # data units per inch, vertically
        for j, (group, title, _case, _column, truth) in enumerate(readouts):
            start = j * len(levels)
            centre = start + len(levels) / 2
            ax.text(centre, -0.08 * units, f"ground truth {truth}", ha="center", va="bottom", fontsize=9.5,
                    color=MUTED)
            wrapped = "\n".join(textwrap.wrap(title.replace("\n", " "), 13))
            ax.text(centre, -0.30 * units, wrapped, ha="center", va="bottom", fontsize=10, color=INK,
                    linespacing=1.0)
            groups.setdefault(group, []).append(start)
        for group, starts in groups.items():
            a, b = min(starts), max(starts) + len(levels)
            ax.plot([a + 0.15, b - 0.15], [-0.98 * units] * 2, color=RULE, linewidth=1.0, clip_on=False)
            label = group if len(group) * 0.085 < (b - a) * cell_w else group.replace(": ", ":\n", 1)
            ax.text((a + b) / 2, -1.03 * units, label, ha="center", va="bottom", fontsize=10.5,
                    fontweight="semibold", color=INK, linespacing=1.05)
        cax = fig.add_axes([(left + n_cols * cell_w + 0.35) / AREA_W, bottom / fig_h, 0.16 / AREA_W,
                            min(2.4, n_rows * cell_h) / fig_h])
        bar = fig.colorbar(mesh, cax=cax)
        bar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        bar.ax.tick_params(labelsize=9.5, colors=INK, length=2)
        bar.outline.set_visible(False)
        cax.set_title("fraction\nrejected", fontsize=10, color=INK, loc="left", pad=6)
        save(fig, f"comparator_heatmap_{stem}", [source], manifest)
        manifest[-1]["size_in"] = [AREA_W, round(fig_h, 3)]


# --- directional F1 ----------------------------------------------------------------------------

def comparator_f1(manifest: list, height: float = 3.45) -> None:
    source = COMPARATOR / "method_comparison_f1.csv"
    summary = pd.read_csv(source)
    strategies, arms = comparison.STRATEGIES, comparison.ARMS
    fig, ax = plt.subplots(figsize=(AREA_W, height))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.27)
    width, gap = 0.26, 0.02
    handles = []
    for k, (arm, colour) in enumerate(zip(arms, comparison.ARM_COLOURS)):
        part = summary[summary["arm"] == arm].set_index("strategy").reindex(strategies)
        x = np.arange(len(strategies)) + (k - 1) * (width + gap)
        mean, spread = part["mean"].to_numpy(float), part["std"].fillna(0).to_numpy(float)
        handles.append(ax.bar(x, np.nan_to_num(mean), width=width, color=colour, label=arm,
                              edgecolor="#ffffff", linewidth=1.0, zorder=2))
        lower, upper = np.minimum(spread, mean), np.minimum(spread, 1.0 - mean)
        ax.errorbar(x, mean, yerr=[lower, upper], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        for xi, value, top in zip(x, mean, upper):
            if np.isfinite(value):
                ax.text(xi, value + top + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8.5,
                        color=INK, rotation=90)
    ax.set_xticks(np.arange(len(strategies)))
    labels = []
    for strategy in strategies:
        method, rule = strategy.split(": ", 1)
        method = method.replace("acceptance >= ", "≥ ")
        name, _, detail = method.partition(" (")
        lines = textwrap.wrap(name, 14) + (textwrap.wrap("(" + detail, 14) if detail else [])
        labels.append("\n".join(lines + textwrap.wrap(rule, 14)))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK, linespacing=1.05)
    ax.set_xlim(-0.55, len(strategies) - 0.45)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="y", labelsize=10, colors=INK, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("directional F1", fontsize=10.5, color=INK)
    ax.grid(axis="y", color="#e6e6e1", linewidth=0.6, zorder=0)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    fig.legend(handles=handles, frameon=False, fontsize=10.5, ncol=3, loc="upper left",
               bbox_to_anchor=(0.05, 1.0))
    save(fig, "comparator_f1", [source], manifest)
    manifest[-1]["size_in"] = [AREA_W, height]


# --- GSEA dot plots ------------------------------------------------------------------------------

# Cycling x CNV rows, written out: what each representation is and what each arm removed.
CYCLE_TEXT = {"A": "A: nothing removed", "B": "B: non-G1 cells removed",
              "C": "C: 97 cell-cycle genes removed", "D": "D: non-G1 cells + cycle genes removed"}
REP_TEXT = {"rna": "RNA", "rna_cnv": "RNA + CNV", "cnv": "CNV only"}


def cycling_rows(arms: list[str]):
    """Row labels (the cycle arm) and row groups (representation and RNA preprocessing)."""
    labels, groups = [], []
    for k, arm in enumerate(arms):
        rep, cycle, label = arm.split("/")
        labels.append(CYCLE_TEXT[cycle])
        if rep == "cnv":
            text = "CNV only\n(each cell's inferCNV\nprofile; no RNA)"
        else:
            text = f"{REP_TEXT[rep]}\nRNA preprocessing:\n{label}"
        if groups and groups[-1][0] == text:
            groups[-1][2] = k
        else:
            groups.append([text, k, k])
    return labels, groups


def gsea(frame: pd.DataFrame, arms: list[str], stem: str, sources: list[Path], manifest: list,
         limit: float = 2.0, row_labels: list[str] | None = None, groups: list | None = None) -> None:
    pathways, categories = dotplot.PATHWAYS, dotplot.CATEGORIES
    x_of, x, spans = {}, 0.0, []
    for name, members in categories:
        start = x
        for pathway in members:
            x_of[pathway] = x
            x += 1.0
        spans.append((name, start, x - 1.0))
    frame = frame[frame["arm"].isin(arms)]
    counts = frame.groupby("arm")["pathway"].nunique()
    assert len(counts) == len(arms) and (counts == 50).all(), "every arm must carry all 50 hallmark sets"
    y_of = {arm: float(i) for i, arm in enumerate(arms)}
    n_rows, n_cols = len(arms), len(pathways)

    grouped = groups is not None
    row_labels = row_labels or [dotplot.arm_label(a) for a in arms]
    if grouped:
        # Group column (representation, RNA preprocessing) left of the row labels; legend bottom left.
        group_w = 2.0
        left = group_w + 0.2 + 0.066 * max(len(t) for t in row_labels)
        legend_w, top, bottom = 0.15, 0.62, 2.3
    else:
        left = 0.25 + 0.083 * max(len(t) for t in row_labels)
        legend_w, top, bottom = 1.45, 0.62, 2.3
    cell_w = (AREA_W - left - legend_w) / n_cols
    cell_h = min(0.32, (AREA_H - top - bottom) / n_rows)
    fig_h = top + n_rows * cell_h + bottom
    fig = plt.figure(figsize=(AREA_W, fig_h))
    ax = fig.add_axes([left / AREA_W, bottom / fig_h, cell_w * n_cols / AREA_W, cell_h * n_rows / fig_h])
    cmap = LinearSegmentedColormap.from_list("nes", [dotplot.NEGATIVE, dotplot.MIDPOINT, dotplot.POSITIVE])
    norm = Normalize(vmin=-limit, vmax=limit)
    # The original's four marker areas were set for 0.30 in cells; scale them to this cell.
    scale = (min(cell_w, cell_h) / 0.30) ** 2
    areas = np.asarray(dotplot.SIZE_AREAS) * scale

    strength = -np.log10(frame["fdr"].clip(lower=1.0 / dotplot.PERMUTATIONS))
    sizes = areas[np.digitize(strength.to_numpy(float), dotplot.SIZE_BREAKS)]
    called = (frame["fdr"] < dotplot.FDR_CALL).to_numpy()
    xs = frame["pathway"].map(x_of).to_numpy()
    ys = frame["arm"].map(y_of).to_numpy()
    colours = frame["NES"].clip(-limit, limit).to_numpy()
    ax.scatter(xs[~called], ys[~called], s=sizes[~called], c=colours[~called], cmap=cmap, norm=norm,
               linewidths=0, zorder=2)
    ax.scatter(xs[called], ys[called], s=sizes[called], c=colours[called], cmap=cmap, norm=norm,
               edgecolors="black", linewidths=0.8, zorder=3)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_xticks([x_of[p] for p in pathways])
    ax.set_xticklabels([dotplot.heatmap43.display_name("HALLMARK_" + p) for p in pathways], rotation=60,
                       ha="right", rotation_mode="anchor", fontsize=9, color="black")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=9 if grouped else 10.5, color="black")
    if grouped:
        for text, first, last in groups:
            if first > 0:
                ax.axhline(first - 0.5, color="black", linewidth=0.9, zorder=1)
            y_mid = bottom + (n_rows - (first + last + 1) / 2) * cell_h
            fig.text(0.08 / AREA_W, y_mid / fig_h, text, ha="left", va="center", fontsize=9.5, color="black",
                     linespacing=1.15, fontweight="normal")
            # A bracket spanning the group's rows, between the group text and the row labels.
            y0 = bottom + (n_rows - first - 0.1) * cell_h
            y1 = bottom + (n_rows - last - 0.9) * cell_h
            xb = (group_w - 0.05) / AREA_W
            fig.add_artist(plt.Line2D([xb, xb], [y0 / fig_h, y1 / fig_h], color="black", linewidth=1.0))
    ax.tick_params(length=3, width=1, color="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.2)
    units = 1.0 / cell_h
    for name, start, end in spans:
        y_line = -0.6 - 0.10 * units
        ax.plot([start - 0.35, end + 0.35], [y_line, y_line], color="black", lw=1.0, clip_on=False)
        for x_end in (start - 0.35, end + 0.35):
            ax.plot([x_end, x_end], [y_line, y_line + 0.07 * units], color="black", lw=1.0, clip_on=False)
        text = name if (end - start + 1) * cell_w < 0.95 else name.replace("\n", " ")
        ax.text((start + end) / 2, y_line - 0.03 * units, text, ha="center", va="bottom",
                fontsize=(10 if "\n" not in text else 8.5) if cell_w >= 0.17 else (9 if "\n" not in text else 7.5),
                color="black", linespacing=1.0, clip_on=False)

    if grouped:
        legend_bottom_left(fig, fig_h, areas, norm, cmap, limit)
        save(fig, stem, sources, manifest)
        manifest[-1]["size_in"] = [AREA_W, round(fig_h, 3)]
        manifest[-1]["arms"] = arms
        return
    # Legend in the right column: size bins, outline, colour.
    lx = (left + cell_w * n_cols + 0.22) / AREA_W
    legend_h = min(fig_h - top - 0.2, 2.75)
    lax = fig.add_axes([lx, (fig_h - top - legend_h) / fig_h, (legend_w - 0.3) / AREA_W, legend_h / fig_h])
    lax.set_axis_off()
    lax.set_xlim(0, 1)
    lax.set_ylim(0, legend_h)
    y = legend_h - 0.05
    lax.text(0.0, y, "−log10 (FDR)", fontsize=10, va="top", color="black")
    y -= 0.36
    for area, label in zip(areas, dotplot.SIZE_LABELS):
        lax.scatter([0.12], [y], s=area, facecolor="#a6a6a6", edgecolor="black", linewidths=0.8, clip_on=False)
        lax.text(0.32, y, label, fontsize=10, va="center", color="black")
        y -= 0.24
    for outlined, label in ((False, "NS"), (True, f"FDR < {dotplot.FDR_CALL:g}")):
        lax.scatter([0.12], [y], s=areas[2], facecolor="#a6a6a6", edgecolor="black" if outlined else "none",
                    linewidths=0.8 if outlined else 0, clip_on=False)
        lax.text(0.32, y, label, fontsize=10, va="center", color="black")
        y -= 0.24
    cax_h = 0.13
    cax_y = (fig_h - top - legend_h) + max(0.0, y - 0.55)
    cax = fig.add_axes([lx, cax_y / fig_h, (legend_w - 0.35) / AREA_W, cax_h / fig_h])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                       ticks=np.arange(-limit, limit + 0.5, 1.0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9.5, length=0, colors="black")
    bar.ax.xaxis.set_ticks_position("top")
    cax.set_title("NES", loc="left", fontsize=10, pad=17, color="black")
    save(fig, stem, sources, manifest)
    manifest[-1]["size_in"] = [AREA_W, round(fig_h, 3)]
    manifest[-1]["arms"] = arms


def legend_bottom_left(fig, fig_h: float, areas, norm, cmap, limit: float) -> None:
    """Size bins, outline and colour bar in the free corner below the row labels."""
    lax = fig.add_axes([0.08 / AREA_W, 0.15 / fig_h, 3.3 / AREA_W, 1.75 / fig_h])
    lax.set_axis_off()
    lax.set_xlim(0, 3.3)
    lax.set_ylim(0, 1.75)
    lax.text(0.0, 1.72, "−log10 (FDR)", fontsize=10, va="top", color="black")
    for k, (area, label) in enumerate(zip(areas, dotplot.SIZE_LABELS)):
        y = 1.35 - 0.3 * k
        lax.scatter([0.12], [y], s=area, facecolor="#a6a6a6", edgecolor="black", linewidths=0.8, clip_on=False)
        lax.text(0.3, y, label, fontsize=10, va="center", color="black")
    for k, (outlined, label) in enumerate(((False, "NS"), (True, f"FDR < {dotplot.FDR_CALL:g}"))):
        y = 1.35 - 0.3 * k
        lax.scatter([1.05], [y], s=areas[2], facecolor="#a6a6a6", edgecolor="black" if outlined else "none",
                    linewidths=0.8 if outlined else 0, clip_on=False)
        lax.text(1.23, y, label, fontsize=10, va="center", color="black")
    cax = fig.add_axes([1.15 / AREA_W, 0.45 / fig_h, 1.25 / AREA_W, 0.13 / fig_h])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                       ticks=np.arange(-limit, limit + 0.5, 1.0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9.5, length=0, colors="black")
    bar.ax.xaxis.set_ticks_position("top")
    cax.set_title("NES", loc="left", fontsize=10, pad=17, color="black")


def gsea_figures(manifest: list) -> None:
    dotplot.LAYOUT = "factorial"
    factorial = dotplot.layout().arms
    assert len(factorial) == 32
    halves = (factorial[:16], factorial[16:])
    for dataset, stem in DATASETS:
        source = DOWN / "gsea_dotplot_factorial" / f"{stem}.csv"
        frame = pd.read_csv(source)
        assert set(frame["arm"]) == set(factorial)
        for part, arms in enumerate(halves, 1):
            gsea(frame, list(arms), f"gsea_factorial_{dataset}_part{part}", [source], manifest)
    dotplot.LAYOUT = "cycling"
    cycling = dotplot.layout().arms
    for dataset, stem in DATASETS:
        source = DOWN / "gsea_dotplot" / f"{stem}.csv"
        frame = pd.read_csv(source)
        arms = [a for a in cycling if a in set(frame["arm"])]
        labels, groups = cycling_rows(arms)
        gsea(frame, arms, f"gsea_cycling_{dataset}", [source], manifest, row_labels=labels, groups=groups)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    comparator_heatmap(manifest)
    comparator_f1(manifest)
    gsea_figures(manifest)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for item in manifest:
        print(item["figure"], item["size_in"])


if __name__ == "__main__":
    main()
