"""What is higher on one side of the gate, for every preprocessing arm.

Features as rows -- hallmark pathways (``--readout gsea``) or genes
(``--readout deg``) -- and the 32 arms of the factorial as columns, one panel
per analysis. One side is drawn at a time. The contrast is
primary_rejected_vs_primary_retained, so a feature is higher in retained
primary cells where its NES or its log2 fold change is negative, and higher in
rejected ones where it is positive. ``--side retained``, the default, colours
-NES or -log2FC; ``--side rejected`` colours NES or log2FC. Where a feature is
not higher on the side drawn the cell is white, so each figure answers one
question -- what that side's cells carry -- and the two sides never share a
scale. Retained is blue and rejected red, the ends 41_'s RdBu_r gives the same
contrast, at matched lightness steps.

Every label reads horizontally. Pathways are written the way papers write
them -- "E2F targets", not HALLMARK_E2F_TARGETS -- from a fixed table of the 50
hallmark sets, so a name on the figure is MSigDB's own and nothing is coined.
The arms are not written as 32 rotated labels: a design strip under the grid
marks which of noscale, ds and cos each column carries, under the name of its
normalisation.

One quantity on one scale, as everywhere else in the project: the colour is
the effect on the side drawn and nothing else. Significance is a dot, and the
number right of a row is how many of the 32 arms carry it -- the agreement
between arms, which is what a factorial over preprocessing exists to measure.

* **gsea** -- the dot is FDR < 0.05, the call 41_ made, on the side drawn. A
  panel shows the pathways with a dot in at least half of its arms
  (--min-arms-fraction).
* **deg** -- the dot is FDR < 0.05 *and* at least two-fold higher on the side
  drawn (--minimum-log2fc), the floor the DEG prespecification sets. A panel
  shows the --top genes with the most dots, ties broken by the median effect:
  on the retained side no gene carries that dot in half the arms of any
  analysis, so the pathways' rule would leave those panels empty.

Under each panel is the arm's fraction of primary cells on the side drawn:
the median over patients of retained (or rejected) / (retained + rejected)
cells in the pseudobulk. It is there because the arms split the same cells
very differently -- 0.12 to 0.73 retained in ovarian -- so each column
compares a different pair of groups, and a column that disagrees with its
neighbours is read with its split in view. A grey cell has no value in that
arm.

The matrices behind the figure are the ones
41_preprocessing_strategy_heatmaps.py wrote, unfiltered.

    python cancer_metastasis/43_retained_side_heatmap.py OUT_DIR --readout gsea \\
        --analysis "Ovarian GSE180661::<41_ output dir>::<downstream analysis dir>" \\
        --analysis ...

``--layout cycling`` draws the arms of the cycling x representation design
instead: RNA, RNA+CNV and CNV as the groups, arms A-D under each, the two RNA
labels side by side, and a strip marking cells removed, genes removed and the
label. DOWNSTREAM_DIR is then a template the arm fills in,
``<down>/{rep}/arm_{cycle}/<block>/<accession>/{label}``, and an arm a panel
lacks -- prostate has no CNV arms -- is a grey column. The panel subtitle gives
the patients in the DEG beside the pseudobulk's count when the downstream run
wrote its pydeseq2_report.json.

--separate draws one figure per analysis instead of one with a panel each;
give it --vmax so they share a scale. --slide draws the same figure to sit on
a 16:9 slide at its own size, so its type is as large on the slide as it is
here: no header, which the slide's own text replaces, and shorter rows.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CONTRAST = "primary_rejected_vs_primary_retained"
NORMALISATIONS = ("raw", "logcpm", "rank256", "ranknm256")
# The factorial's order as the rest of the project writes it.
ARMS = [a for n in NORMALISATIONS for a in (
    n, f"{n}_cos", f"{n}_ds", f"{n}_ds_cos", f"{n}_noscale", f"{n}_noscale_cos",
    f"{n}_noscale_ds", f"{n}_noscale_ds_cos")]
TAGS = ("noscale", "ds", "cos")


@dataclass
class Layout:
    """The columns of a figure: which arms, how they group, what the strip marks.

    ``tag_rows`` are the rows of the design strip, each a name and a test of an
    arm: True draws a filled mark, False a faint one, None nothing. Under
    ``every_arm`` an arm missing from a panel stops the run; otherwise its
    column is grey in that panel.
    """
    arms: list[str]
    groups: list[tuple[str, int]]
    tag_rows: list[tuple[str, Callable[[str], bool | None]]]
    every_arm: bool
    arm_dir: Callable[[str, str], Path]
    note: str = ""
    per: str = "preprocessing arm"


def factorial_layout() -> Layout:
    return Layout(
        arms=ARMS, groups=[(norm, 8) for norm in NORMALISATIONS],
        tag_rows=[(tag, lambda arm, tag=tag: f"_{tag}" in arm) for tag in TAGS],
        every_arm=True, arm_dir=lambda spec, arm: Path(spec) / arm)


# The cycling x representation arms of submit_cycling_cnv_factorial.sh. An arm
# is <rep>/<A-D>/<label>; CNV ran arms A and B only, under the label raw, which
# names no RNA preprocessing there because the cost uses no RNA.
CYCLING_REPS = (("rna", "RNA"), ("rna_cnv", "RNA+CNV"), ("cnv", "CNV"))
CYCLING_LABELS = ("ranknm256_noscale_ds_cos", "raw")


def cycling_layout() -> Layout:
    arms, groups = [], []
    for rep, title in CYCLING_REPS:
        cycles, labels = ("AB", ("raw",)) if rep == "cnv" else ("ABCD", CYCLING_LABELS)
        block = [f"{rep}/{cycle}/{label}" for label in labels for cycle in cycles]
        arms += block
        groups.append((title, len(block)))

    def cycle(arm: str) -> str:
        return arm.split("/")[1]

    def template(spec: str, arm: str) -> Path:
        rep, letter, label = arm.split("/")
        return Path(spec.format(rep=rep, cycle=letter, label=label))

    return Layout(
        arms=arms, groups=groups,
        tag_rows=[("cells removed", lambda arm: cycle(arm) in "BD"),
                  ("genes removed", lambda arm: cycle(arm) in "CD"),
                  (CYCLING_LABELS[0], lambda arm: None if arm.startswith("cnv/")
                   else arm.endswith(f"/{CYCLING_LABELS[0]}"))],
        every_arm=False, arm_dir=template, per="arm",
        note=("Arms: A nothing removed; B malignant cells outside G1 removed on both sides; "
              "C the 97 cell-cycle genes removed; D both. A column without a mark in the "
              f"last strip row is raw; CNV columns use no RNA preprocessing."))


LAYOUTS = {"factorial": factorial_layout, "cycling": cycling_layout}
SIDES = {"retained": -1, "rejected": 1}  # the sign that makes the side drawn positive
# One hue per side. The red has the blue's OKLab lightness and chroma at every
# step, so the same value looks as strong on either side.
RAMPS = {
    "retained": ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
                 "#184f95", "#0d366b"],
    "rejected": ["#ffffff", "#fad6d2", "#f1aea8", "#e4857e", "#d75853", "#b13f3c",
                 "#892b2a", "#621b1a"],
}
UNDEFINED = "#bdbdbd"
INK, MUTED, FAINT = "#1a1a1a", "#5f5f5a", "#d0d0cc"

# The 50 MSigDB hallmark sets as papers write them. Fixed rather than derived,
# so that every name on the figure is checkable against MSigDB's own.
HALLMARK = {
    "ADIPOGENESIS": "Adipogenesis",
    "ALLOGRAFT_REJECTION": "Allograft rejection",
    "ANDROGEN_RESPONSE": "Androgen response",
    "ANGIOGENESIS": "Angiogenesis",
    "APICAL_JUNCTION": "Apical junction",
    "APICAL_SURFACE": "Apical surface",
    "APOPTOSIS": "Apoptosis",
    "BILE_ACID_METABOLISM": "Bile acid metabolism",
    "CHOLESTEROL_HOMEOSTASIS": "Cholesterol homeostasis",
    "COAGULATION": "Coagulation",
    "COMPLEMENT": "Complement",
    "DNA_REPAIR": "DNA repair",
    "E2F_TARGETS": "E2F targets",
    "EPITHELIAL_MESENCHYMAL_TRANSITION": "Epithelial-mesenchymal transition",
    "ESTROGEN_RESPONSE_EARLY": "Estrogen response early",
    "ESTROGEN_RESPONSE_LATE": "Estrogen response late",
    "FATTY_ACID_METABOLISM": "Fatty acid metabolism",
    "G2M_CHECKPOINT": "G2M checkpoint",
    "GLYCOLYSIS": "Glycolysis",
    "HEDGEHOG_SIGNALING": "Hedgehog signaling",
    "HEME_METABOLISM": "Heme metabolism",
    "HYPOXIA": "Hypoxia",
    "IL2_STAT5_SIGNALING": "IL-2/STAT5 signaling",
    "IL6_JAK_STAT3_SIGNALING": "IL-6/JAK/STAT3 signaling",
    "INFLAMMATORY_RESPONSE": "Inflammatory response",
    "INTERFERON_ALPHA_RESPONSE": "Interferon-\u03b1 response",
    "INTERFERON_GAMMA_RESPONSE": "Interferon-\u03b3 response",
    "KRAS_SIGNALING_DN": "KRAS signaling down",
    "KRAS_SIGNALING_UP": "KRAS signaling up",
    "MITOTIC_SPINDLE": "Mitotic spindle",
    "MTORC1_SIGNALING": "mTORC1 signaling",
    "MYC_TARGETS_V1": "MYC targets V1",
    "MYC_TARGETS_V2": "MYC targets V2",
    "MYOGENESIS": "Myogenesis",
    "NOTCH_SIGNALING": "Notch signaling",
    "OXIDATIVE_PHOSPHORYLATION": "Oxidative phosphorylation",
    "P53_PATHWAY": "p53 pathway",
    "PANCREAS_BETA_CELLS": "Pancreas \u03b2 cells",
    "PEROXISOME": "Peroxisome",
    "PI3K_AKT_MTOR_SIGNALING": "PI3K/AKT/mTOR signaling",
    "PROTEIN_SECRETION": "Protein secretion",
    "REACTIVE_OXYGEN_SPECIES_PATHWAY": "Reactive oxygen species pathway",
    "SPERMATOGENESIS": "Spermatogenesis",
    "TGF_BETA_SIGNALING": "TGF-\u03b2 signaling",
    "TNFA_SIGNALING_VIA_NFKB": "TNF-\u03b1 signaling via NF-\u03baB",
    "UNFOLDED_PROTEIN_RESPONSE": "Unfolded protein response",
    "UV_RESPONSE_DN": "UV response down",
    "UV_RESPONSE_UP": "UV response up",
    "WNT_BETA_CATENIN_SIGNALING": "WNT/\u03b2-catenin signaling",
    "XENOBIOTIC_METABOLISM": "Xenobiotic metabolism",
}
READOUTS = {
    "gsea": {"prefix": "gsea", "effect": "NES", "noun": "Hallmark pathways"},
    "deg": {"prefix": "deg", "effect": "log2FC", "noun": "Genes"},
}


def readout_spec(readout: str, side: str, per: str = "preprocessing arm") -> dict:
    spec = dict(READOUTS[readout], side=side, sign=SIDES[side])
    spec["colour"] = ("-" if SIDES[side] < 0 else "") + spec["effect"]
    spec["title"] = f"{spec['noun']} higher in {side} primary cells, per {per}"
    return spec


def display_name(feature: str) -> str:
    if feature.startswith("HALLMARK_"):
        key = feature[len("HALLMARK_"):]
        return HALLMARK.get(key, key.replace("_", " ").capitalize())
    return feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="factorial",
                        help="The columns: the 32 arms of the preprocessing factorial, or "
                             "the cycling x representation arms. With cycling, "
                             "DOWNSTREAM_DIR is a template with {rep}, {cycle} and {label}, "
                             "and an arm a panel lacks is a grey column")
    parser.add_argument("--readout", choices=sorted(READOUTS), default="gsea")
    parser.add_argument("--side", choices=sorted(SIDES), default="retained",
                        help="Which side of the gate to draw: what is higher in retained "
                             "or in rejected primary cells")
    parser.add_argument("--analysis", action="append", required=True,
                        metavar="TITLE::STRATEGY_DIR::DOWNSTREAM_DIR",
                        help="41_'s output directory and the downstream analysis "
                             "directory it was drawn from; repeat per panel")
    parser.add_argument("--min-arms-fraction", type=float, default=0.5,
                        help="gsea: rows need a dot in this fraction of the arms")
    parser.add_argument("--top", type=int, default=30,
                        help="deg: genes per panel, by how many arms carry a dot")
    parser.add_argument("--minimum-log2fc", type=float, default=1.0,
                        help="deg: the dot also needs log2FC <= -this")
    parser.add_argument("--vmax", type=float, default=None,
                        help="Top of the colour scale; larger values saturate. "
                             "Defaults to the largest value shown. Pass it when "
                             "drawing --separate, so the figures share one scale")
    parser.add_argument("--separate", action="store_true",
                        help="One figure per --analysis instead of one with a panel each")
    parser.add_argument("--slide", action="store_true",
                        help="Lay the figure out to sit on a 16:9 slide at its own size: no "
                             "header, since the slide carries the title and the notes, and "
                             "shorter rows. Writes <name>_slide.png beside the full figure")
    return parser.parse_args()


def load(spec: str, prefix: str, side: str, layout: Layout) -> dict:
    parts = spec.split("::")
    if len(parts) != 3:
        raise SystemExit(f"--analysis wants TITLE::STRATEGY_DIR::DOWNSTREAM_DIR, got {spec!r}")
    title, strategy, downstream = parts[0], Path(parts[1]), parts[2]
    effect = pd.read_csv(strategy / f"{prefix}_value_matrix.csv", index_col=0)
    called = pd.read_csv(strategy / f"{prefix}_called_matrix.csv", index_col=0).astype(bool)
    missing = [arm for arm in layout.arms if arm not in effect.index]
    if missing and layout.every_arm:
        raise SystemExit(f"{strategy} lacks arms {missing}")
    effect = effect.reindex(layout.arms)
    called = called.reindex(layout.arms, fill_value=False).astype(bool)
    fraction, patients, in_deg = {}, set(), []
    for arm in layout.arms:
        if arm in missing:
            fraction[arm] = float("nan")
            continue
        directory = layout.arm_dir(downstream, arm)
        meta = pd.read_csv(directory / "deg/contrasts" / CONTRAST
                           / "pseudobulk_sample_metadata.csv")
        cells = meta.pivot_table(index="patient_id", columns="comparison_status",
                                 values="cell_n", aggfunc="sum")
        # reference is retained and case is rejected in this contrast.
        status = "reference" if side == "retained" else "case"
        fraction[arm] = float((cells[status] / cells.sum(axis=1)).median())
        done = (directory / "DONE").read_text(encoding="utf-8")
        patients.update(line.split("=", 1)[1] for line in done.splitlines()
                        if line.startswith("patients="))
        # The patients the fit actually had, which can be fewer than the
        # pseudobulk's: one without a cell in both states has no pair to enter.
        report = directory / "deg" / "pydeseq2_report.json"
        if report.is_file():
            for contrast in json.loads(report.read_text(encoding="utf-8")).get("contrasts", []):
                if contrast.get("contrast") == CONTRAST:
                    in_deg.append(int(contrast["patient_n"]))
    return {"title": title, "effect": effect, "called": called,
            "fraction": pd.Series(fraction), "patients": "/".join(sorted(patients)),
            "in_deg": in_deg, "available": len(layout.arms) - len(missing)}


def choose_features(panel: dict, args: argparse.Namespace) -> None:
    # Signed so that the side drawn is positive: -effect for retained.
    effect = SIDES[args.side] * panel["effect"]
    called = panel["called"]
    if args.readout == "gsea":
        dots = called & (effect > 0)
    else:
        dots = called & (effect >= args.minimum_log2fc)
    counts = dots.sum(axis=0)
    order = pd.DataFrame({"n": counts, "median": effect.median(axis=0)})
    if args.readout == "gsea":
        # Of the arms this panel has: a panel missing some is not held to the
        # columns it could not fill.
        order = order[order["n"] >= args.min_arms_fraction * panel["available"]]
    else:
        order = order[order["n"] > 0]
    order = order.sort_values(["n", "median"], ascending=[False, False])
    if args.readout == "deg":
        # How many genes share the last count shown, so a cut through a tie
        # is reported rather than hidden.
        shown = order.head(args.top)
        if len(shown):
            last = int(shown["n"].iloc[-1])
            panel["tie_note"] = (f"{int((order['n'] == last).sum())} genes have {last} arm(s), "
                                 f"{int((shown['n'] == last).sum())} shown")
        order = shown
    panel["features"] = list(order.index)
    panel["side_effect"] = effect
    panel["counts"] = counts
    panel["dots"] = dots
    panel["eligible"] = int((counts > 0).sum())


def header_lines(panels: list[dict], args: argparse.Namespace, spec: dict,
                 layout: Layout) -> list[str]:
    side = spec["side"]
    if args.readout == "gsea":
        dot_text = "Dot: FDR < 0.05."
        rows_text = f"Rows: pathways with a dot in at least {args.min_arms_fraction:.0%} of the arms."
    else:
        dot_text = (f"Dot: FDR < 0.05 and at least {2 ** args.minimum_log2fc:g}-fold "
                    f"higher in {side}.")
        rows_text = f"Rows: the genes with the most dots, at most {args.top}, ties by median log2FC."
    method = "paired PyDESeq2" + (", hallmark GSEA" if args.readout == "gsea" else "")
    if any(p["side_effect"][p["features"]].isna().any().any() for p in panels if p["features"]):
        dot_text += " Grey: no value in that arm."
    lines = [f"Primary rejected vs primary retained ({method}). Colour: {spec['colour']} where it "
             f"is higher in {side}, white otherwise. {dot_text}",
             f"{rows_text} Number right of a row: arms with a dot. Under a panel: each arm's "
             f"{side} fraction. Columns: the {len(layout.arms)} arms, marked by the strip at "
             f"the bottom."]
    if layout.note:
        lines.append(layout.note)
    ties = [f"{p['title']}: {p['tie_note']}" for p in panels if p.get("tie_note")]
    if ties:
        lines.append("Cut at the last count shown -- " + "; ".join(ties) + ".")
    return lines


def main() -> None:
    args = parse_args()
    layout = LAYOUTS[args.layout]()
    spec = readout_spec(args.readout, args.side, layout.per)
    panels = [load(item, spec["prefix"], args.side, layout) for item in args.analysis]
    for panel in panels:
        choose_features(panel, args)
    stem = f"{args.side}_side_{args.readout}"
    suffix = "_slide" if args.slide else ""
    if args.separate:
        for panel in panels:
            slug = "".join(c if c.isalnum() else "_" for c in panel["title"].lower()).strip("_")
            draw([panel], args, spec, f"{stem}_{slug}{suffix}", layout)
    else:
        draw(panels, args, spec, f"{stem}{suffix}", layout)


def draw(panels: list[dict], args: argparse.Namespace, spec: dict, stem: str,
         layout: Layout) -> None:
    arms = layout.arms
    # Where each group of columns starts, and how wide it is.
    spans, start = [], 0
    for title, width in layout.groups:
        spans.append((title, start, width))
        start += width
    n_arms = len(arms)
    shown = [p for p in panels if p["features"]]
    largest = max((float(p["side_effect"][p["features"]].clip(lower=0).max().max()) for p in shown),
                  default=1.0)
    vmax = args.vmax if args.vmax is not None else largest

    # Inches. On a slide the figure is shown at its own size, so the type sizes
    # stay as they are and the rows and gaps shrink instead.
    slide = args.slide
    cell_w, cell_h = (0.165, 0.12) if slide else (0.24, 0.22)
    names = [display_name(f) for p in panels for f in p["features"]] or ["none"]
    names += [name for name, _ in layout.tag_rows]
    left = max(1.6, 0.35 + 0.062 * max(len(name) for name in names))
    count_w, right = 0.45, (0.8 if slide else 1.05)
    grid_w = n_arms * cell_w
    fig_w = left + grid_w + count_w + right
    text_x = 0.3
    chars = max(40, int((fig_w - text_x - 0.3) / 0.062))
    wrapped = [] if slide else [part for line in header_lines(panels, args, spec, layout)
                                for part in textwrap.wrap(line, chars)]
    header_h = 0.08 if slide else 0.40 + 0.135 * len(wrapped) + 0.1
    title_h, fraction_h, panel_gap = (0.36, 0.19, 0.07) if slide else (0.58, 0.24, 0.22)
    heights = [max(len(p["features"]), 1) * cell_h for p in panels]
    strip_row, strip_bottom = (0.13, 0.12) if slide else (0.17, 0.2)
    strip_h = 0.22 + len(layout.tag_rows) * strip_row + strip_bottom
    fig_h = header_h + sum(title_h + h + fraction_h + panel_gap for h in heights) + strip_h
    fig = plt.figure(figsize=(fig_w, fig_h))
    side = spec["side"]
    cmap = LinearSegmentedColormap.from_list(side, RAMPS[side])
    cmap.set_bad(UNDEFINED)

    def axes_at(x0: float, y_top: float, width: float, height: float):
        """Axes from inches, with y measured down from the top of the figure."""
        return fig.add_axes([x0 / fig_w, (fig_h - y_top - height) / fig_h,
                             width / fig_w, height / fig_h])

    if not slide:
        fig.text(text_x / fig_w, 1 - 0.15 / fig_h, spec["title"],
                 fontsize=10.5, fontweight="semibold", color=INK, va="top")
        fig.text(text_x / fig_w, 1 - 0.40 / fig_h, "\n".join(wrapped),
                 fontsize=7.2, color=MUTED, va="top", linespacing=1.35)

    y = header_h
    mesh, first_grid = None, None
    for panel, height in zip(panels, heights):
        features = panel["features"]
        subtitle = f"{panel['patients']} patients"
        if panel["in_deg"]:
            low, high = min(panel["in_deg"]), max(panel["in_deg"])
            subtitle += f", {low}" + ("" if low == high else f"-{high}") + " in the DEG"
        if args.readout == "deg":
            subtitle += f", {panel['eligible']} genes with a dot in any arm"
        if slide:
            fig.text(left / fig_w, 1 - (y + 0.02) / fig_h, f"{panel['title']} ({subtitle})",
                     fontsize=8.6, fontweight="semibold", color=INK, va="top")
        else:
            fig.text(left / fig_w, 1 - (y + 0.06) / fig_h, panel["title"], fontsize=8.6,
                     fontweight="semibold", color=INK, va="top")
            fig.text(left / fig_w, 1 - (y + 0.24) / fig_h, subtitle, fontsize=7.0,
                     color=MUTED, va="top")
        y += title_h
        ax = axes_at(left, y, grid_w, height)
        values = (panel["side_effect"][features].clip(lower=0).to_numpy(dtype=float).T
                  if features else np.zeros((1, n_arms)))
        mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=vmax,
                             edgecolors="#ffffff", linewidth=1.0)
        if features:
            dots = panel["dots"][features].to_numpy().T
            for i, j in zip(*np.nonzero(dots)):
                ax.plot(j + 0.5, i + 0.5, marker="o", markersize=2.3, linestyle="none",
                        color="#ffffff" if values[i, j] > 0.55 * vmax else INK)
        ax.set_xlim(0, n_arms)
        ax.set_ylim(values.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks(np.arange(values.shape[0]) + 0.5)
        ax.set_yticklabels([display_name(f) for f in features] or ["none"],
                           fontsize=7.2, color=INK)
        ax.tick_params(axis="y", length=0, pad=4)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for _, begin, _width in spans[1:]:
            ax.axvline(begin, color="#ffffff", linewidth=3.5)
        # Each block of columns is one group, named above it.
        for title, begin, width in spans:
            ax.text(begin + width / 2, -0.25, title, ha="center", va="bottom", fontsize=6.8,
                    color=MUTED, clip_on=False)
        # Agreement between arms, right of each row.
        for i, feature in enumerate(features):
            ax.text(n_arms + 0.4, i + 0.5, f"{int(panel['counts'][feature])}", ha="left",
                    va="center", fontsize=6.8, color=MUTED, clip_on=False)
        ax.text(n_arms + 0.4, -0.3, "arms", ha="left", va="bottom", fontsize=6.2,
                color=MUTED, clip_on=False)
        first_grid = first_grid or (y, height)
        y += height
        # Each arm's fraction of cells on the side drawn, under its column.
        rax = axes_at(left, y + 0.03, grid_w, fraction_h - 0.06)
        rax.set_xlim(0, n_arms)
        rax.set_ylim(0, 1)
        rax.axis("off")
        for j, arm in enumerate(arms):
            value = panel["fraction"][arm]
            if np.isfinite(value):
                rax.text(j + 0.5, 0.5, f"{value:.2f}".lstrip("0"), ha="center",
                         va="center", fontsize=5.6, color=MUTED)
        rax.text(-0.3, 0.5, f"{side} fraction", ha="right", va="center", fontsize=6.4,
                 color=MUTED)
        y += fraction_h + panel_gap

    # The design strip: which options each column carries, under its group.
    y += 0.02
    sax = axes_at(left, y, grid_w, strip_h - strip_bottom)
    sax.set_xlim(0, n_arms)
    rows = 1 + len(layout.tag_rows)
    sax.set_ylim(rows, 0)
    sax.axis("off")
    for title, begin, width in spans:
        sax.text(begin + width / 2, 0.45, title, ha="center", va="center", fontsize=7.4,
                 fontweight="semibold", color=INK)
        sax.plot([begin + 0.3, begin + width - 0.3], [0.9, 0.9], color=FAINT, linewidth=0.8)
    for t, (tag, carries) in enumerate(layout.tag_rows):
        sax.text(-0.3, t + 1.5, tag, ha="right", va="center", fontsize=6.8, color=INK)
        for j, arm in enumerate(arms):
            on = carries(arm)
            if on is None:
                continue
            sax.plot(j + 0.5, t + 1.5, marker="o", linestyle="none",
                     markersize=4.0 if on else 2.4, color=INK if on else FAINT)

    if mesh is not None and first_grid is not None:
        top_in, height = first_grid
        bar_h = min(2.2, max(height, 1.2))
        cax = axes_at(left + grid_w + count_w + 0.15, top_in, 0.12, bar_h)
        bar = fig.colorbar(mesh, cax=cax, extend="max" if largest > vmax else "neither")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=6.0, colors=MUTED, length=2)
        # Above the bar rather than along it, so that it reads horizontally too.
        cax.set_title(f"{spec['colour']}\nhigher in\n{side}", fontsize=6.6, color=INK,
                      loc="left", pad=6)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)
    out_rows = []
    for panel in panels:
        for feature in panel["features"]:
            out_rows.append({"analysis": panel["title"], "feature": feature,
                             "name_on_figure": display_name(feature),
                             "arms_with_dot": int(panel["counts"][feature]),
                             f"median_{spec['effect']}": round(float(panel["effect"][feature].median()), 3)})
    pd.DataFrame(out_rows).to_csv(args.out_dir / f"{stem}_rows.csv", index=False)
    print(f"wrote {args.out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
