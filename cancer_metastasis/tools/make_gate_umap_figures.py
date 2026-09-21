"""One gate UMAP per dataset, from whichever export each dataset has.

The deck wired these one at a time, and only colorectal ever got the two-panel
version, because the call sites named files rather than discovering them. This
discovers every dataset with a UMAP export and draws the same figure for each:
the metastasis as a grey backdrop with the retained and rejected primary cells
over it, and beside it whatever the dataset can actually support as a second
panel.

Two export formats exist and both are read:

  replication   `<dir>/<GSE>/umap_coordinates_and_labels.csv.gz`, carrying
                UMAP1/UMAP2, side, primary_gate and the authors' own subtype
                labels. The second panel is the authors' proliferation subtype.
  report input  `umap_<tissue>.csv.gz`, carrying umap_1/umap_2, side, a boolean
                `retained`, and signature scores but no author labels. The
                second panel is the cell-division score.

A dataset whose second panel has nothing to show gets one panel and says so,
rather than a panel that looks informative and is not. GSE181919 is that case:
its authors provide a single malignant label, so a subtype panel there would
colour every cell identically.

Both formats carry the gate as it was run -- cap enforced, depth uncorrected --
so the caption says so. See `SEQUENCING_DEPTH_RESOLUTION.md`: with that
configuration the retained cells are about twice as deep as the rejected ones,
so a reader should treat the split as substantially a depth readout until the
pipeline is re-run.

Usage:
  python cancer_metastasis/tools/make_gate_umap_figures.py OUT_DIR \
      --replication-dir .../confidenceot_analysis/replication_umap \
      --report-inputs   .../report_inputs_20260916
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_report_figures import (  # noqa: E402
    AQUA,
    CJK,
    LABEL_BOX,
    ORANGE,
    finish,
    plt,
)

# Subtype labels that name a proliferating population. Matched case-insensitively
# against the authors' own strings, so a dataset that never labelled one simply
# does not get that panel.
PROLIFERATION = r"MKI67|MK167|PCNA|TOP2A|prolifer"
DIVISION_COLOUR = "#c2453a"
BACKDROP = "#d9d9d4"

# Tissue names for the report-input exports, whose filenames carry the tissue
# and not the accession.
TISSUE_ACCESSION = {"ovarian": "GSE180661", "prostate": "GSE271675"}
# Keyed by accession, because that is the name that reaches draw() once the
# report-input filenames have been mapped through TISSUE_ACCESSION.
TISSUE_LABEL = {"GSE180661": "Ovarian cancer",
                "GSE271675": "Prostate cancer",
                "GSE225857": "Colorectal cancer",
                "GSE181919": "Head and neck cancer"}


def normalise_replication(path: Path) -> pd.DataFrame | None:
    """The older export: UMAP1/UMAP2, side, primary_gate, author_subtype."""
    cells = pd.read_csv(path, keep_default_na=False, na_values=[""])
    needed = {"UMAP1", "UMAP2", "side", "primary_gate"}
    if not needed <= set(cells.columns):
        return None
    frame = pd.DataFrame({
        "x": pd.to_numeric(cells["UMAP1"], errors="coerce"),
        "y": pd.to_numeric(cells["UMAP2"], errors="coerce"),
        "side": cells["side"].astype(str),
        "gate": cells["primary_gate"].astype(str),
    })
    frame["subtype"] = (cells["author_subtype"].astype(str)
                        if "author_subtype" in cells else pd.NA)
    frame["division"] = np.nan
    return frame.dropna(subset=["x", "y"])


def normalise_report_input(path: Path) -> pd.DataFrame | None:
    """The newer export: umap_1/umap_2, side, a boolean retained, scores."""
    cells = pd.read_csv(path)
    needed = {"umap_1", "umap_2", "side", "retained"}
    if not needed <= set(cells.columns):
        return None
    retained = cells["retained"]
    if retained.dtype == object:
        retained = retained.astype(str).str.lower().isin(
            {"true", "1", "yes", "retained"})
    frame = pd.DataFrame({
        "x": pd.to_numeric(cells["umap_1"], errors="coerce"),
        "y": pd.to_numeric(cells["umap_2"], errors="coerce"),
        "side": cells["side"].astype(str),
        "gate": np.where(retained.to_numpy(dtype=bool), "retained", "rejected"),
    })
    frame["subtype"] = pd.NA
    frame["division"] = (pd.to_numeric(cells["cell_division"], errors="coerce")
                         if "cell_division" in cells else np.nan)
    return frame.dropna(subset=["x", "y"])


def discover(replication: Path | None,
             report_inputs: Path | None) -> list[tuple[str, pd.DataFrame]]:
    found: list[tuple[str, pd.DataFrame]] = []
    if replication and replication.is_dir():
        for directory in sorted(p for p in replication.iterdir() if p.is_dir()):
            path = directory / "umap_coordinates_and_labels.csv.gz"
            if not path.exists():
                continue
            frame = normalise_replication(path)
            if frame is None:
                print(f"  skip {directory.name}: replication columns absent")
                continue
            found.append((directory.name, frame))
    if report_inputs and report_inputs.is_dir():
        for path in sorted(report_inputs.glob("umap_*.csv.gz")):
            tissue = path.stem.removeprefix("umap_").removesuffix(".csv")
            frame = normalise_report_input(path)
            if frame is None:
                print(f"  skip {tissue}: report-input columns absent")
                continue
            found.append((TISSUE_ACCESSION.get(tissue, tissue), frame))
    return found


def second_panel_plan(frame: pd.DataFrame) -> tuple[str, str]:
    """What the second panel can honestly show, and its label."""
    subtypes = frame.loc[frame.side.eq("primary"), "subtype"].dropna()
    distinct = set(subtypes.astype(str)) - {"", "nan", "<NA>"}
    if len(distinct) > 1:
        if subtypes.astype(str).str.contains(PROLIFERATION, case=False).any():
            return "subtype", ("作者注释的增殖亚型"
                               "（MKI67 / PCNA）")
        return "none", (f"authors label {len(distinct)} subtypes, none of them "
                        f"a proliferating one")
    if frame["division"].notna().any():
        return "division", "细胞分裂评分"
    if len(distinct) == 1:
        return "none", (f"authors provide one malignant label only "
                        f"({next(iter(distinct))}), so a subtype panel would "
                        f"colour every cell alike")
    return "none", "no author subtypes and no division score in this export"


def draw(name: str, frame: pd.DataFrame, out: Path) -> dict:
    primary = frame[frame.side.eq("primary")]
    metastasis = frame[frame.side.eq("metastasis")]
    kept = primary[primary.gate.eq("retained")]
    dropped = primary[primary.gate.eq("rejected")]
    if primary.empty:
        print(f"  skip {name}: no primary cells")
        return {"dataset": name, "drawn": False, "reason": "no primary cells"}

    plan, label = second_panel_plan(frame)
    panels = 2 if plan != "none" else 1

    span_x = np.percentile(frame.x.to_numpy(float), [0.5, 99.5])
    span_y = np.percentile(frame.y.to_numpy(float), [0.5, 99.5])
    pad_x, pad_y = 0.04 * np.ptp(span_x), 0.04 * np.ptp(span_y)
    aspect = float(np.ptp(span_x) / np.ptp(span_y))
    width = max(4.6, 5.4 * aspect)

    marker = float(np.clip(2.3 * (10_000.0 / max(len(frame), 1)) ** 0.35,
                           0.5, 3.0))
    figure, axes = plt.subplots(1, panels,
                                figsize=(panels * width + 0.4, 5.6))
    axes = np.atleast_1d(axes)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlim(span_x[0] - pad_x, span_x[1] + pad_x)
        axis.set_ylim(span_y[0] - pad_y, span_y[1] + pad_y)
        axis.set_aspect("equal", adjustable="box")
        for spine in axis.spines.values():
            spine.set_visible(False)

    axes[0].scatter(metastasis.x, metastasis.y, s=marker, c=BACKDROP,
                    linewidth=0, rasterized=True, zorder=2)
    # Shuffled, so neither group is drawn entirely on top of the other.
    mixed = pd.concat([kept, dropped]).sample(frac=1.0, random_state=5)
    axes[0].scatter(mixed.x, mixed.y, s=marker,
                    c=np.where(mixed.gate.eq("retained"), ORANGE, AQUA),
                    linewidth=0, rasterized=True, zorder=3)
    axes[0].set_title("留下的细胞落在转移"
                      "灶的哪一侧", fontsize=14)
    for index, (text, colour, count) in enumerate((
            ("primary rejected", AQUA, len(dropped)),
            ("primary retained", ORANGE, len(kept)),
            ("metastasis", "#a8a8a2", len(metastasis)))):
        axes[0].text(0.98, 0.02 + index * 0.058, f"{text}  n = {count:,}",
                     color=colour, fontsize=12.5, transform=axes[0].transAxes,
                     va="bottom", ha="right", fontweight="semibold",
                     bbox=LABEL_BOX)

    comparison = ""
    if plan == "subtype":
        flagged = (primary.subtype.astype(str)
                   .str.contains(PROLIFERATION, case=False, na=False))
        axes[1].scatter(primary.loc[~flagged, "x"], primary.loc[~flagged, "y"],
                        s=marker, c=BACKDROP, linewidth=0,
                        rasterized=True, zorder=2)
        axes[1].scatter(primary.loc[flagged, "x"], primary.loc[flagged, "y"],
                        s=marker, c=DIVISION_COLOUR, linewidth=0,
                        rasterized=True, zorder=3)
        inside = float(primary.loc[flagged, "gate"].eq("retained").mean())
        outside = float(primary.loc[~flagged, "gate"].eq("retained").mean())
        comparison = f"kept {inside:.0%}  vs  {outside:.0%}"
    elif plan == "division":
        # Shown as the continuous score it is. Cutting it at a tertile and
        # colouring that block the way an author subtype is coloured paints a
        # third of every cluster and reads as a finding. The quantity that
        # matters is whether the retained cells score higher, so that is
        # stated as a difference of means.
        score = primary["division"].to_numpy(float)
        low, high = np.nanpercentile(score, [2, 98])
        drawn = axes[1].scatter(primary.x, primary.y, s=marker, c=score,
                                cmap="RdGy_r", vmin=low, vmax=high,
                                linewidth=0, rasterized=True, zorder=3)
        bar = figure.colorbar(drawn, ax=axes[1], fraction=0.035, pad=0.02)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=10, length=2)
        kept_score = float(np.nanmean(
            primary.loc[primary.gate.eq("retained"), "division"]))
        drop_score = float(np.nanmean(
            primary.loc[primary.gate.eq("rejected"), "division"]))
        comparison = (f"retained {kept_score:+.3f}  vs  rejected "
                      f"{drop_score:+.3f}")
    if panels == 2:
        axes[1].set_title(label, fontsize=14)
        axes[1].text(0.98, 0.02, comparison, color=DIVISION_COLOUR,
                     fontsize=12.5, transform=axes[1].transAxes, va="bottom",
                     ha="right", fontweight="semibold", bbox=LABEL_BOX)

    plt.rcParams["font.sans-serif"] = CJK
    title = TISSUE_LABEL.get(name, name)
    figure.suptitle(f"{title} — {name}" if title != name else name,
                    fontsize=16, fontweight="semibold", y=1.01)
    note = (f"{len(primary):,} primary and {len(metastasis):,} metastatic "
            f"malignant cells. 这是运行时的 gate"
            f"（cap 生效、深度未修正）,"
            f"该配置下留下的细胞深度约为被拒细胞的两倍。")
    if panels == 1:
        note += f" 第二个面板省略:{label}"
    finish(figure, out / f"fig_gatemap_{name.lower()}.png", note)
    return {"dataset": name, "drawn": True, "panels": panels,
            "second_panel": plan, "primary": len(primary),
            "metastasis": len(metastasis), "retained": len(kept),
            "rejected": len(dropped),
            "retained_share": round(len(kept) / max(len(primary), 1), 3),
            "comparison": comparison, "reason": "" if panels == 2 else label}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--replication-dir", type=Path, default=None)
    parser.add_argument("--report-inputs", type=Path, default=None)
    arguments = parser.parse_args()
    if not (arguments.replication_dir or arguments.report_inputs):
        raise SystemExit("give --replication-dir, --report-inputs, or both")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    datasets = discover(arguments.replication_dir, arguments.report_inputs)
    if not datasets:
        raise SystemExit("no UMAP exports found under the given directories")

    rows = []
    for name, frame in datasets:
        print(f"== {name}")
        rows.append(draw(name, frame, arguments.output_dir))

    report = pd.DataFrame(rows)
    report.to_csv(arguments.output_dir / "gate_umap_report.csv", index=False)
    print()
    print(report.to_string(index=False))
    print(f"\nwrote {arguments.output_dir / 'gate_umap_report.csv'}")


if __name__ == "__main__":
    main()
