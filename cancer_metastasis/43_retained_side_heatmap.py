"""What is higher in retained primary cells, for every preprocessing arm.

One panel per analysis, the 32 arms of the factorial as rows, pathways
(``--readout gsea``) or genes (``--readout deg``) as columns. Only the retained
side is drawn: the contrast is primary_rejected_vs_primary_retained, so a
feature is higher in retained where its NES or its log2 fold change is
negative, and the colour is that magnitude, -NES or -log2FC. Where a feature is
not higher in retained the cell is white; nothing about the rejected side is
shown, because the question here is what the retained cells carry.

One quantity on one scale, as everywhere else in the project: the colour is
the retained-side effect and nothing else. Significance is a dot, and the
number under each name is how many of the 32 arms carry it -- the agreement
between arms, which is what a factorial over preprocessing exists to measure.

* **gsea** -- the dot is FDR < 0.05, the call 41_ made. A panel shows the
  pathways with a dot in at least half of its arms (--min-arms-fraction).
* **deg** -- the dot is FDR < 0.05 *and* at least two-fold higher in retained
  (--minimum-log2fc), the floor the DEG prespecification sets. No gene carries
  that dot in half the arms of any analysis, so a panel shows the --top genes
  with the most dots instead, ties broken by the median effect.

The matrices behind the figure are the ones
41_preprocessing_strategy_heatmaps.py wrote, unfiltered.

Beside each panel is the arm's retained fraction, the median over patients of
retained / (retained + rejected) primary cells in the pseudobulk. It is there
because it explains the rows that disagree: the arms that retain most of the
cells are the ones whose retained side looks different.

    python cancer_metastasis/43_retained_side_heatmap.py OUT_DIR --readout gsea \\
        --analysis "Ovarian GSE180661::<41_ output dir>::<downstream analysis dir>" \\
        --analysis ...

Genes run to thirty columns a panel, so --separate draws one figure per
analysis instead of one wide figure; give it --vmax so they share a scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

CONTRAST = "primary_rejected_vs_primary_retained"
# The factorial's order as the rest of the project writes it.
ARMS = [a for n in ("raw", "logcpm", "rank256", "ranknm256") for a in (
    n, f"{n}_cos", f"{n}_ds", f"{n}_ds_cos", f"{n}_noscale", f"{n}_noscale_cos",
    f"{n}_noscale_ds", f"{n}_noscale_ds_cos")]
RAMP = ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf",
        "#184f95", "#0d366b"]
UNDEFINED = "#bdbdbd"
INK, MUTED = "#1a1a1a", "#5f5f5a"
READOUTS = {
    "gsea": {"prefix": "gsea", "effect": "NES", "colour": "-NES (higher in retained)",
             "title": "Hallmark pathways higher in retained primary cells, per preprocessing arm"},
    "deg": {"prefix": "deg", "effect": "log2FC", "colour": "-log2 fold change (higher in retained)",
            "title": "Genes higher in retained primary cells, per preprocessing arm"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--readout", choices=sorted(READOUTS), default="gsea")
    parser.add_argument("--analysis", action="append", required=True,
                        metavar="TITLE::STRATEGY_DIR::DOWNSTREAM_DIR",
                        help="41_'s output directory and the downstream analysis "
                             "directory it was drawn from; repeat per panel")
    parser.add_argument("--min-arms-fraction", type=float, default=0.5,
                        help="gsea: columns need a dot in this fraction of the arms")
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
    return parser.parse_args()


def load(spec: str, prefix: str) -> dict:
    parts = spec.split("::")
    if len(parts) != 3:
        raise SystemExit(f"--analysis wants TITLE::STRATEGY_DIR::DOWNSTREAM_DIR, got {spec!r}")
    title, strategy, downstream = parts[0], Path(parts[1]), Path(parts[2])
    effect = pd.read_csv(strategy / f"{prefix}_value_matrix.csv", index_col=0)
    called = pd.read_csv(strategy / f"{prefix}_called_matrix.csv", index_col=0).astype(bool)
    missing = [arm for arm in ARMS if arm not in effect.index]
    if missing:
        raise SystemExit(f"{strategy} lacks arms {missing}")
    effect, called = effect.loc[ARMS], called.loc[ARMS]
    fraction, patients = {}, set()
    for arm in ARMS:
        meta = pd.read_csv(downstream / arm / "deg/contrasts" / CONTRAST
                           / "pseudobulk_sample_metadata.csv")
        cells = meta.pivot_table(index="patient_id", columns="comparison_status",
                                 values="cell_n", aggfunc="sum")
        # reference is retained and case is rejected in this contrast.
        fraction[arm] = float((cells["reference"] / cells.sum(axis=1)).median())
        done = (downstream / arm / "DONE").read_text(encoding="utf-8")
        patients.update(line.split("=", 1)[1] for line in done.splitlines()
                        if line.startswith("patients="))
    return {"title": title, "effect": effect, "called": called,
            "fraction": pd.Series(fraction), "patients": "/".join(sorted(patients))}


def choose_columns(panel: dict, args: argparse.Namespace) -> None:
    effect, called = panel["effect"], panel["called"]
    if args.readout == "gsea":
        dots = called & (effect < 0)
    else:
        dots = called & (effect <= -args.minimum_log2fc)
    counts = dots.sum(axis=0)
    order = pd.DataFrame({"n": counts, "median": effect.median(axis=0)})
    if args.readout == "gsea":
        order = order[order["n"] >= args.min_arms_fraction * len(ARMS)]
    else:
        order = order[order["n"] > 0]
    order = order.sort_values(["n", "median"], ascending=[False, True])
    if args.readout == "deg":
        # How many genes share the last count shown, so a cut through a tie
        # is reported rather than hidden.
        shown = order.head(args.top)
        if len(shown):
            last = int(shown["n"].iloc[-1])
            panel["tie_note"] = (f"{int((order['n'] == last).sum())} genes have {last} arm(s); "
                                 f"{int((shown['n'] == last).sum())} shown")
        order = shown
    panel["columns"] = list(order.index)
    panel["counts"] = counts
    panel["dots"] = dots
    panel["eligible"] = int((counts > 0).sum())


def main() -> None:
    args = parse_args()
    spec = READOUTS[args.readout]
    panels = [load(item, spec["prefix"]) for item in args.analysis]
    for panel in panels:
        choose_columns(panel, args)
    if args.separate:
        for panel in panels:
            slug = "".join(c if c.isalnum() else "_" for c in panel["title"].lower()).strip("_")
            draw([panel], args, spec, f"retained_side_{args.readout}_{slug}")
    else:
        draw(panels, args, spec, f"retained_side_{args.readout}")


def header_lines(panels: list[dict], args: argparse.Namespace, spec: dict) -> list[str]:
    if args.readout == "gsea":
        dot_text = "Dot: FDR < 0.05."
        columns_text = (f"Columns: pathways with a dot in at least {args.min_arms_fraction:.0%} "
                        "of the arms.")
    else:
        dot_text = (f"Dot: FDR < 0.05 and at least {2 ** args.minimum_log2fc:g}-fold "
                    "higher in retained.")
        columns_text = (f"Columns: the genes with the most dots, at most {args.top}, ties by "
                        "median log2FC.")
    method = "paired PyDESeq2" + (", hallmark GSEA" if args.readout == "gsea" else "")
    lines = [f"Primary rejected vs primary retained ({method}). Colour: -{spec['effect']} where it "
             f"is higher in retained, white otherwise. {dot_text}",
             f"{columns_text} Number under a name: arms with a dot. Right of a panel: the arm's "
             "retained fraction."]
    ties = [f"{p['title']}: {p['tie_note']}" for p in panels if p.get("tie_note")]
    if ties:
        lines.append("Cut at the last count shown -- " + "; ".join(ties))
    return lines


def draw(panels: list[dict], args: argparse.Namespace, spec: dict, stem: str) -> None:
    import textwrap

    n_arms = len(ARMS)
    shown = [p for p in panels if p["columns"]]
    largest = max((float((-p["effect"][p["columns"]]).clip(lower=0).max().max()) for p in shown),
                  default=1.0)
    vmax = args.vmax if args.vmax is not None else largest

    cell_w, cell_h = 0.27, 0.2
    left, right, bottom = 2.0, 0.9, 0.35
    gap, frac_w = 0.25, 0.42
    widths = [max(len(p["columns"]), 1) * cell_w for p in panels]
    # A narrow panel still needs room for the header, which is wrapped to the
    # width rather than allowed to run off the edge.
    fig_w = max(6.5, left + sum(widths) + len(panels) * (frac_w + gap) + right)
    text_x = 0.35
    chars = max(40, int((fig_w - text_x - 0.3) / 0.062))
    wrapped = [part for line in header_lines(panels, args, spec)
               for part in textwrap.wrap(line, chars)]
    # Heights above the grid, in inches. The rotated names rise as far as the
    # longest one, so the panel title sits just above that rather than above
    # room kept for pathway names when the columns are genes.
    longest = max((len(f.replace("HALLMARK_", "")) for p in panels for f in p["columns"]),
                  default=4)
    names_top = 0.31 + 0.052 * longest
    patients_in, title_in = names_top + 0.14, names_top + 0.36
    header_bottom = 0.40 + 0.135 * len(wrapped)
    top = header_bottom + 0.14 + title_in + 0.16
    title_y, patients_y = -title_in / cell_h, -patients_in / cell_h
    fig_h = top + n_arms * cell_h + bottom
    fig = plt.figure(figsize=(fig_w, fig_h))
    cmap = LinearSegmentedColormap.from_list("retained", RAMP)
    cmap.set_bad(UNDEFINED)

    x = left
    mesh = None
    for index, (panel, width) in enumerate(zip(panels, widths)):
        ax = fig.add_axes([x / fig_w, bottom / fig_h, width / fig_w, n_arms * cell_h / fig_h])
        columns = panel["columns"]
        values = ((-panel["effect"][columns]).clip(lower=0).to_numpy(dtype=float)
                  if columns else np.zeros((n_arms, 1)))
        mesh = ax.pcolormesh(np.ma.masked_invalid(values), cmap=cmap, vmin=0.0, vmax=vmax,
                             edgecolors="#ffffff", linewidth=1.0)
        if columns:
            dots = panel["dots"][columns].to_numpy()
            for i, j in zip(*np.nonzero(dots)):
                ax.plot(j + 0.5, i + 0.5, marker="o", markersize=2.2, linestyle="none",
                        color="#ffffff" if values[i, j] > 0.55 * vmax else INK)
        ax.set_xlim(0, values.shape[1])
        ax.set_ylim(n_arms, 0)
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_arms) + 0.5)
        ax.set_yticklabels(ARMS if index == 0 else [], fontsize=6.8, color=INK)
        ax.tick_params(axis="y", length=0, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for boundary in range(8, n_arms, 8):
            ax.axhline(boundary, color="#ffffff", linewidth=3.5)
        # Names, then how many arms carry the dot, then the panel title.
        for j, feature in enumerate(columns):
            ax.text(j + 0.5, -1.55, feature.replace("HALLMARK_", ""), rotation=90,
                    ha="center", va="bottom", fontsize=6.0, color=INK)
            ax.text(j + 0.5, -0.35, f"{int(panel['counts'][feature])}", ha="center",
                    va="bottom", fontsize=6.0, color=MUTED)
        if not columns:
            ax.text(0.5, n_arms / 2, "none", ha="center", va="center", fontsize=7, color=MUTED)
        ax.text(0, title_y, panel["title"], ha="left", va="bottom", fontsize=8.2,
                fontweight="semibold", color=INK)
        note = f"{panel['patients']} patients"
        if args.readout == "deg":
            note += f"; {panel['eligible']} genes with a dot in any arm"
        ax.text(0, patients_y, note, ha="left", va="bottom", fontsize=6.8, color=MUTED)
        # The retained fraction beside the panel.
        fx = (x + width + 0.08) / fig_w
        fax = fig.add_axes([fx, bottom / fig_h, frac_w / fig_w, n_arms * cell_h / fig_h])
        fax.set_xlim(0, 1)
        fax.set_ylim(n_arms, 0)
        fax.axis("off")
        for i, arm in enumerate(ARMS):
            fax.text(0.05, i + 0.5, f"{panel['fraction'][arm]:.2f}", ha="left", va="center",
                     fontsize=6.0, color=MUTED)
        fax.text(0.05, -0.35, "retained", ha="left", va="bottom", fontsize=6.0, color=MUTED,
                 rotation=90)
        x += width + frac_w + gap

    fig.text(text_x / fig_w, 1 - 0.15 / fig_h, spec["title"],
             fontsize=10.5, fontweight="semibold", color=INK, va="top")
    fig.text(text_x / fig_w, 1 - 0.40 / fig_h, "\n".join(wrapped),
             fontsize=7.2, color=MUTED, va="top", linespacing=1.35)
    if mesh is not None:
        # Beside the last panel, wherever the figure's own edge is.
        cax = fig.add_axes([(x + 0.05) / fig_w, bottom / fig_h,
                            0.12 / fig_w, min(2.4, n_arms * cell_h) / fig_h])
        bar = fig.colorbar(mesh, cax=cax, extend="max" if largest > vmax else "neither")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=6.0, colors=MUTED, length=2)
        bar.set_label(spec["colour"], fontsize=6.6, color=INK)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"{stem}.{suffix}", dpi=220)
    plt.close(fig)
    rows = []
    for panel in panels:
        for feature in panel["columns"]:
            rows.append({"analysis": panel["title"], "feature": feature,
                         "arms_with_dot": int(panel["counts"][feature]),
                         f"median_{spec['effect']}": round(float(panel["effect"][feature].median()), 3)})
    pd.DataFrame(rows).to_csv(args.out_dir / f"{stem}_columns.csv", index=False)
    print(f"wrote {args.out_dir / (stem + '.png')}")


if __name__ == "__main__":
    main()
