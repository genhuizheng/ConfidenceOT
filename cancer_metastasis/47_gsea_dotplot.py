"""One signed GSEA dot plot per analysis: every arm, every hallmark pathway.

    python cancer_metastasis/47_gsea_dotplot.py OUT_DIR \\
        --analysis "Ovarian GSE180661::<down>/{rep}/arm_{cycle}/uniform/GSE180661/{label}" \\
        --analysis ...

This replaces the two one-sided figures of 43_retained_side_heatmap.py, which
split one GSEA into "higher in retained" and "higher in rejected" panels and
showed each side as an unsigned value. Here one figure holds the GSEA as it was
computed:

* **Colour** is NES of primary_rejected_vs_primary_retained, with retained as the
  reference: NES < 0 (blue) is higher in retained cells, NES > 0 (orange) higher
  in rejected cells. Blue-white-orange with limits +/-2, the reference figure's
  scale, shared by every figure; a point beyond them takes the end colour and
  its exact NES is in the CSV.
* **Size** is -log10 FDR. GSEApy reports FDR = 0 below its permutation
  resolution; the runs used 2,000 permutations (tacc_factorial_downstream.slurm),
  so 0 is drawn at 1/2000 (-log10 = 3.3).
* **Outline** marks FDR < 0.05, the call 41_ made.

The figure carries no title or note -- row and column labels, the category
brackets and the legends only -- so a slide or a figure legend says the rest.

Rows are the arms of the cycling x representation design (43_'s cycling layout:
RNA, RNA+CNV and CNV; arms A-D; ranknm256_noscale_ds_cos and raw). An arm a
panel lacks -- prostate has no CNV arms -- is not drawn; gsea_dotplot.json lists
it.

Columns are all 50 hallmark pathways, none dropped, grouped by the eight process
categories of the hallmark collection (Liberzon et al., Cell Systems 2015,
Fig. 2): cellular component, development, DNA damage, immune, metabolic,
pathway, proliferation, signalling. Within a category pathways keep the
category's order below, which is fixed and does not depend on any result.

Values are read from each arm's own all_gene_discovery_gsea_results.csv, not
from 41_'s matrices.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("heatmap43", HERE / "43_retained_side_heatmap.py")
heatmap43 = importlib.util.module_from_spec(_spec)
sys.modules["heatmap43"] = heatmap43  # dataclasses look the module up by name
_spec.loader.exec_module(heatmap43)

CONTRAST = heatmap43.CONTRAST
PERMUTATIONS = 2000
FDR_CALL = 0.05

# Liberzon et al. 2015, Fig. 2: the hallmark sets by process category.
CATEGORIES = [
    ("Cellular\ncomponent", ["APICAL_JUNCTION", "APICAL_SURFACE", "PEROXISOME"]),
    ("Development", ["ADIPOGENESIS", "ANGIOGENESIS", "EPITHELIAL_MESENCHYMAL_TRANSITION",
                     "MYOGENESIS", "SPERMATOGENESIS", "PANCREAS_BETA_CELLS"]),
    ("DNA\ndamage", ["DNA_REPAIR", "UV_RESPONSE_DN", "UV_RESPONSE_UP"]),
    ("Immune", ["ALLOGRAFT_REJECTION", "COAGULATION", "COMPLEMENT", "INTERFERON_ALPHA_RESPONSE",
                "INTERFERON_GAMMA_RESPONSE", "IL6_JAK_STAT3_SIGNALING", "INFLAMMATORY_RESPONSE"]),
    ("Metabolic", ["BILE_ACID_METABOLISM", "CHOLESTEROL_HOMEOSTASIS", "FATTY_ACID_METABOLISM",
                   "GLYCOLYSIS", "HEME_METABOLISM", "OXIDATIVE_PHOSPHORYLATION", "XENOBIOTIC_METABOLISM"]),
    ("Pathway", ["APOPTOSIS", "HYPOXIA", "PROTEIN_SECRETION", "UNFOLDED_PROTEIN_RESPONSE",
                 "REACTIVE_OXYGEN_SPECIES_PATHWAY"]),
    ("Proliferation", ["E2F_TARGETS", "G2M_CHECKPOINT", "MYC_TARGETS_V1", "MYC_TARGETS_V2",
                       "P53_PATHWAY", "MITOTIC_SPINDLE"]),
    ("Signaling", ["ANDROGEN_RESPONSE", "ESTROGEN_RESPONSE_EARLY", "ESTROGEN_RESPONSE_LATE",
                   "IL2_STAT5_SIGNALING", "KRAS_SIGNALING_UP", "KRAS_SIGNALING_DN", "MTORC1_SIGNALING",
                   "NOTCH_SIGNALING", "PI3K_AKT_MTOR_SIGNALING", "HEDGEHOG_SIGNALING",
                   "TGF_BETA_SIGNALING", "TNFA_SIGNALING_VIA_NFKB", "WNT_BETA_CATENIN_SIGNALING"]),
]
PATHWAYS = [p for _, members in CATEGORIES for p in members]
assert len(PATHWAYS) == 50 and len(set(PATHWAYS)) == 50

# Blue, white, orange, as in the reference figure the user supplied.
NEGATIVE = "#2b6cb4"
MIDPOINT = "#ffffff"
POSITIVE = "#f28c1c"
# -log10 FDR values the size legend shows; FDR 0 is drawn at 1/PERMUTATIONS.
SIZE_LEGEND = (1.0, 2.0, 3.0)


def size_of(strength):
    """Marker area for -log10 FDR, from a visible dot at 0 to the cap 1/PERMUTATIONS."""
    top = -np.log10(1.0 / PERMUTATIONS)
    return 12 + 230 * np.clip(np.asarray(strength, dtype=float), 0, top) / top




def arm_label(arm: str) -> str:
    rep, cycle, label = arm.split("/")
    rep_text = dict(heatmap43.CYCLING_REPS)[rep]
    preprocessing = "ds_cos" if label == "ranknm256_noscale_ds_cos" else "raw"
    if rep == "cnv":
        return f"{rep_text}  {cycle}"
    return f"{rep_text}  {preprocessing}  {cycle}"


def load(template: str) -> tuple[pd.DataFrame, list[str]]:
    layout = heatmap43.cycling_layout()
    rows, missing = [], []
    for arm in layout.arms:
        path = layout.arm_dir(template, arm) / "gsea" / "contrasts" / CONTRAST / "all_gene_discovery_gsea_results.csv"
        if not path.is_file():
            missing.append(arm)
            continue
        table = pd.read_csv(path)
        table = table[table["collection"].astype(str).eq("Hallmark")]
        table["pathway"] = table["pathway"].astype(str).str.replace("^HALLMARK_", "", regex=True)
        absent = sorted(set(PATHWAYS) - set(table["pathway"]))
        if absent:
            raise RuntimeError(f"{path}: hallmark pathways missing {absent}")
        for record in table.itertuples(index=False):
            if record.pathway in PATHWAYS:
                rows.append({"arm": arm, "pathway": record.pathway, "NES": float(record.NES),
                             "fdr": float(record.fdr), "source": str(path)})
    return pd.DataFrame(rows), missing


def draw(frame: pd.DataFrame, title: str, missing: list[str], limit: float, path: Path) -> None:
    """The figure alone: no title, subtitle or note; legends only."""
    arms = [arm for arm in heatmap43.cycling_layout().arms if arm in set(frame["arm"])]
    y_of = {arm: float(i) for i, arm in enumerate(arms)}
    x_of, x, spans = {}, 0.0, []
    for name, members in CATEGORIES:
        start = x
        for pathway in members:
            x_of[pathway] = x
            x += 1.0
        spans.append((name, start, x - 1.0))
    n_rows, n_cols = len(arms), len(PATHWAYS)

    cell = 0.30
    left, right, top, bottom = 1.75, 0.25, 0.80, 2.35
    width = left + cell * n_cols + right
    height = top + cell * n_rows + bottom
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([left / width, bottom / height, cell * n_cols / width, cell * n_rows / height])
    cmap = LinearSegmentedColormap.from_list("nes", [NEGATIVE, MIDPOINT, POSITIVE])
    norm = Normalize(vmin=-limit, vmax=limit)

    strength = -np.log10(frame["fdr"].clip(lower=1.0 / PERMUTATIONS))
    sizes = size_of(strength)
    called = (frame["fdr"] < FDR_CALL).to_numpy()
    xs = frame["pathway"].map(x_of).to_numpy()
    ys = frame["arm"].map(y_of).to_numpy()
    colours = frame["NES"].clip(-limit, limit).to_numpy()
    ax.scatter(xs[~called], ys[~called], s=sizes[~called], c=colours[~called], cmap=cmap, norm=norm,
               linewidths=0, zorder=2)
    ax.scatter(xs[called], ys[called], s=sizes[called], c=colours[called], cmap=cmap, norm=norm,
               edgecolors="black", linewidths=1.0, zorder=3)
    ax.set_xlim(-0.6, n_cols - 0.4)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_xticks([x_of[p] for p in PATHWAYS])
    ax.set_xticklabels([heatmap43.display_name("HALLMARK_" + p) for p in PATHWAYS], rotation=45,
                       ha="right", rotation_mode="anchor", fontsize=10, color="black")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([arm_label(a) for a in arms], fontsize=10.5, color="black")
    ax.tick_params(length=3, width=1, color="black")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.4)

    # Category brackets above the frame, as in the reference figure.
    for name, start, end in spans:
        y_line = -0.95
        ax.plot([start - 0.35, end + 0.35], [y_line, y_line], color="black", lw=1.1, clip_on=False)
        for x_end in (start - 0.35, end + 0.35):
            ax.plot([x_end, x_end], [y_line, y_line + 0.28], color="black", lw=1.1, clip_on=False)
        # Two lines only where the bracket is too narrow for one.
        text = name if end - start < 3 else name.replace("\n", " ")
        ax.text((start + end) / 2, y_line - 0.12, text, ha="center", va="bottom",
                fontsize=10.5, color="black", linespacing=1.0, clip_on=False)

    # Legends in one column, bottom left, below the row labels: size, outline, colour.
    legend_ax = fig.add_axes([0.010, 0.70 / height, (left - 0.15) / width, (bottom - 0.80) / height])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.text(0.02, 0.99, "−log10 (FDR)", fontsize=10, va="top", color="black")
    for row, value in enumerate(SIZE_LEGEND):
        y = 0.84 - row * 0.155
        legend_ax.scatter([0.13], [y], s=size_of(np.asarray([value])), facecolor="#a6a6a6",
                          edgecolor="black", linewidths=1.0, clip_on=False)
        legend_ax.text(0.30, y, f"{value:g}", fontsize=10, va="center", color="black")
    for y, outlined, text in ((0.30, False, "NS"), (0.14, True, f"FDR < {FDR_CALL:g}")):
        legend_ax.scatter([0.13], [y], s=size_of(np.asarray([2.0])), facecolor="#a6a6a6",
                          edgecolor="black" if outlined else "none", linewidths=1.0 if outlined else 0,
                          clip_on=False)
        legend_ax.text(0.30, y, text, fontsize=10, va="center", color="black")
    cax = fig.add_axes([0.20 / width, 0.22 / height, 1.45 / width, 0.15 / height])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                       ticks=np.arange(-limit, limit + 0.5, 1.0))
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=9.5, length=0, colors="black")
    bar.ax.xaxis.set_ticks_position("top")
    cax.set_title("NES", loc="left", fontsize=10, pad=16, color="black")

    fig.savefig(path.with_suffix(".png"), dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--analysis", action="append", required=True,
                        help="TITLE::TEMPLATE, the template as 43_'s cycling layout takes it")
    parser.add_argument("--limit", type=float, default=2.0,
                        help="Colour limit |NES|, shared by every figure; points beyond it take the end "
                             "colour and the colour bar shows an arrow. The values are in the CSV")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = []
    for spec in args.analysis:
        title, template = spec.split("::", 1)
        frame, missing = load(template)
        if frame.empty:
            raise RuntimeError(f"{title}: no GSEA table found under {template}")
        loaded.append((title, frame, missing))
    limit = float(args.limit)
    record = {"limit": limit, "permutations": PERMUTATIONS, "figures": []}
    for title, frame, missing in loaded:
        stem = "gsea_dotplot_" + "".join(c if c.isalnum() else "_" for c in title).strip("_").lower()
        draw(frame, title, missing, limit, args.out / stem)
        frame.to_csv(args.out / f"{stem}.csv", index=False)
        record["figures"].append({"title": title, "stem": stem, "arms": int(frame["arm"].nunique()),
                                  "points_beyond_limit": int((frame["NES"].abs() > limit).sum()),
                                  "largest_abs_nes": float(frame["NES"].abs().max()),
                                  "missing_arms": missing, "points": int(len(frame))})
    (args.out / "gsea_dotplot.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
