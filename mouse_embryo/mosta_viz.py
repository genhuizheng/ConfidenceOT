"""Shared pieces of the MOSTA figures (30_mosta_figures.py).

Reading: everything is read from a finished run (25_ layout) and never
written there.  Slices are read one at a time from slices/<sample>_bins.csv.gz;
the per-pair tables from pairs/<pair_id>/.

Writing: every file goes under one visualisation root through ``Outputs``,
which refuses to replace a file it did not write itself (its own manifest,
manifests/OUTPUT_MANIFEST.tsv, lists what it wrote), so an existing file of
anything else is never overwritten.
"""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.patches import PathPatch, Rectangle  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

STAGES = ("E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5")
EXCLUDED = "Cavity"
RETAINED = "#1A9E8F"
REJECTED = "#E4572E"
EXCLUDED_COLOUR = "#D9D9D9"
CONTEXT = "#C8C8C8"
TEXT = "#222222"
MUTED = "#6B6B6B"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.edgecolor": "#444444",
})


def stage_of(sample: str) -> str:
    return sample.split("_")[0]


def stage_order(stage: str) -> int:
    return STAGES.index(stage)


def group_order(group: str) -> float:
    return float(group.split("->")[0].lstrip("E"))


# ---------------------------------------------------------------- outputs

class Outputs:
    """Writes under one root, records every file, never replaces a stranger's file."""

    FIELDS = ("path", "bytes", "sha256", "stage", "written", "job_id", "commit")

    def __init__(self, root: Path, commit: str) -> None:
        self.root = Path(root)
        self.commit = commit
        for part in ("figures/main", "figures/supplementary", "figures/individual_panels", "tables", "scripts",
                     "logs", "reports", "manifests"):
            (self.root / part).mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifests" / "OUTPUT_MANIFEST.tsv"
        self.rows: dict[str, dict] = {}
        if self.manifest.is_file():
            with self.manifest.open(encoding="utf-8") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    self.rows[row["path"]] = row
        self.stage = "unset"

    def path(self, relative: str) -> Path:
        target = self.root / relative
        key = target.relative_to(self.root).as_posix()
        if target.exists() and key not in self.rows:
            raise FileExistsError(f"{target} exists and was not written by this script; not replaced")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def record(self, target: Path) -> None:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        key = target.relative_to(self.root).as_posix()
        self.rows[key] = {"path": key, "bytes": target.stat().st_size, "sha256": digest, "stage": self.stage,
                          "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "job_id": os.environ.get("SLURM_JOB_ID", ""), "commit": self.commit}
        self.flush()

    def flush(self) -> None:
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS, delimiter="\t")
            writer.writeheader()
            for key in sorted(self.rows):
                writer.writerow(self.rows[key])

    def table(self, frame: pd.DataFrame, relative: str) -> Path:
        target = self.path(relative)
        frame.to_csv(target, sep="\t", index=False)
        self.record(target)
        return target

    def text(self, content: str, relative: str) -> Path:
        target = self.path(relative)
        target.write_text(content, encoding="utf-8")
        self.record(target)
        return target

    def figure(self, figure, relative_stem: str, dpi: int = 300) -> list[Path]:
        written = []
        for suffix in (".pdf", ".png"):
            target = self.path(relative_stem + suffix)
            figure.savefig(target, dpi=dpi, bbox_inches="tight")
            self.record(target)
            written.append(target)
        plt.close(figure)
        return written


# ---------------------------------------------------------------- data

class Run:
    """Read-only access to a finished MOSTA run."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._slices: dict[str, pd.DataFrame] = {}

    def manifest_slices(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "manifest" / "slices.csv")

    def manifest_pairs(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "manifest" / "pairs.csv")

    def per_slice(self) -> pd.DataFrame:
        return pd.read_csv(self.root / "equalisation" / "per_slice.csv")

    def audit(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "audit" / name)

    def slice_table(self, sample: str, keep: int = 6) -> pd.DataFrame:
        """Every bin of one slice (Cavity included); a few slices are cached."""
        if sample not in self._slices:
            if len(self._slices) >= keep:
                self._slices.pop(next(iter(self._slices)))
            table = pd.read_csv(self.root / "slices" / f"{sample}_bins.csv.gz",
                                usecols=["bin_id", "x", "y", "annotation", "analysed", "equalised_row"])
            table["annotation"] = table["annotation"].astype(str)
            self._slices[sample] = table
        return self._slices[sample]

    def analysed(self, sample: str) -> pd.DataFrame:
        """The analysed bins in the row order of the per-pair tables (bin_row)."""
        table = self.slice_table(sample)
        table = table[table["equalised_row"] >= 0].sort_values("equalised_row").reset_index(drop=True)
        if not np.array_equal(table["equalised_row"].to_numpy(), np.arange(len(table))):
            raise ValueError(f"{sample}: equalised_row is not 0..n-1")
        return table

    def pair_bins(self, pair_id: str, side: str, columns: list[str] | None = None) -> pd.DataFrame:
        table = pd.read_csv(self.root / "pairs" / pair_id / f"bins_{side}.csv.gz", usecols=columns)
        if "bin_row" in table and not np.array_equal(table["bin_row"].to_numpy(), np.arange(len(table))):
            raise ValueError(f"{pair_id} {side}: bin_row is not 0..n-1")
        return table

    def transport(self, pair_id: str, side: str) -> dict:
        with np.load(self.root / "pairs" / pair_id / f"transport_{side}.npz", allow_pickle=False) as stored:
            return {key: stored[key] for key in stored.files}


# ---------------------------------------------------------------- palette

def build_palette(annotations: list[str]) -> dict[str, str]:
    """One colour per annotation for every figure; Cavity is the excluded grey."""
    from matplotlib import colormaps

    colours = []
    for name in ("tab20", "tab20b", "tab20c", "Set3", "Pastel1", "Dark2", "Accent"):
        cmap = colormaps[name]
        colours += [matplotlib.colors.to_hex(cmap(i)) for i in range(cmap.N)]
    # Drop greys so no annotation looks like the excluded or context grey.
    colours = [c for c in colours if len({c[1:3], c[3:5], c[5:7]}) > 1]
    palette = {}
    for annotation, colour in zip([a for a in annotations if a != EXCLUDED], colours):
        palette[annotation] = colour
    missing = [a for a in annotations if a != EXCLUDED and a not in palette]
    if missing:
        raise ValueError(f"{len(missing)} annotations have no colour left: {missing[:5]}")
    palette[EXCLUDED] = EXCLUDED_COLOUR
    return palette


# ---------------------------------------------------------------- drawing

def grid_pitch(x: np.ndarray, y: np.ndarray, sample: int = 20000) -> float:
    """Side of an axis-aligned square that covers one bin of the grid.

    The bin spacing is the median nearest-neighbour distance and the grid angle
    the median angle of those neighbour vectors modulo 90 degrees, so registered
    (rotated, non-integer) coordinates tile as well as integer ones; spacing
    from the unique x values alone collapses to ~0 on a rotated grid.
    """
    from scipy.spatial import cKDTree
    points = np.column_stack([x, y]).astype(float)
    if len(points) < 2:
        return 1.0
    query = points
    if len(points) > sample:
        query = points[np.random.default_rng(0).choice(len(points), sample, replace=False)]
    distance, index = cKDTree(points).query(query, k=2)
    vector = points[index[:, 1]] - query
    angle = float(np.median(np.mod(np.arctan2(vector[:, 1], vector[:, 0]), np.pi / 2)))
    return float(np.median(distance[:, 1])) * (abs(np.cos(angle)) + abs(np.sin(angle)))


def marker_size(ax, x: np.ndarray, y: np.ndarray) -> float:
    """Square marker area (pt^2) that tiles the bin grid at the axis' current scale."""
    pitch = grid_pitch(x, y)
    figure = ax.figure
    figure.canvas.draw_idle()
    bbox = ax.get_window_extent().transformed(figure.dpi_scale_trans.inverted())
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    points_per_unit = min(bbox.width * 72 / abs(x1 - x0), bbox.height * 72 / abs(y1 - y0))
    return max((pitch * points_per_unit) ** 2, 0.01)


def slice_axes(ax, x: np.ndarray, y: np.ndarray, pad: float = 0.03) -> None:
    """Equal aspect, the slice's own extent, image orientation (y down), no axes."""
    dx, dy = np.ptp(x), np.ptp(y)
    ax.set_xlim(x.min() - pad * dx, x.max() + pad * dx)
    ax.set_ylim(y.max() + pad * dy, y.min() - pad * dy)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_bins(ax, x, y, colours, size=None, order=None, alpha=1.0, extent=None):
    """Bins as rasterised squares; ``extent`` = (x, y) arrays that fix the axis (default: these bins)."""
    if extent is None:
        extent = (x, y)
    slice_axes(ax, *extent)
    if size is None:
        size = marker_size(ax, *extent)
    if order is not None:
        x, y, colours = x[order], y[order], np.asarray(colours, dtype=object)[order]
    ax.scatter(x, y, c=list(colours), s=size, marker="s", linewidths=0, alpha=alpha, rasterized=True)
    return size


def legend_handles(palette: dict[str, str], names: list[str]):
    return [Rectangle((0, 0), 1, 1, facecolor=palette[name], edgecolor="none") for name in names]


def status_colours(retained: np.ndarray) -> np.ndarray:
    return np.where(retained, RETAINED, REJECTED)


def alluvial(ax, left: pd.Series, flows: pd.DataFrame, right: pd.Series, palette: dict[str, str],
             stub: pd.Series | None = None, stub_label: str = "", gap: float = 0.006,
             min_label: float = 0.012) -> None:
    """A two-column alluvial drawn with Bezier ribbons.

    left: node height per source node; flows: columns source, target, mass;
    right: node height per target node; stub: per source node, mass that
    leaves to no target (drawn as a grey band to a single node ``stub_label``).
    Heights are in the same unit everywhere, so widths compare directly.
    """
    ax.set_xlim(0, 1)
    ax.axis("off")
    left_nodes = list(left.index)
    right_nodes = list(right.index) + ([stub_label] if stub is not None and stub.sum() > 0 else [])
    right_heights = right.to_dict()
    if stub is not None and stub.sum() > 0:
        right_heights[stub_label] = float(stub.sum())

    def positions(nodes, heights):
        out, y = {}, 0.0
        for node in nodes:
            out[node] = [y, y + heights[node]]
            y += heights[node] + gap
        return out, y

    lpos, ltotal = positions(left_nodes, left.to_dict())
    rpos, rtotal = positions(right_nodes, right_heights)
    total = max(ltotal, rtotal)
    ax.set_ylim(total, 0)
    width = 0.035
    for node, (y0, y1) in lpos.items():
        ax.add_patch(Rectangle((0.12, y0), width, y1 - y0, facecolor=palette.get(node, CONTEXT), edgecolor="none"))
        if y1 - y0 >= min_label * total:
            ax.text(0.115, (y0 + y1) / 2, node, ha="right", va="center", fontsize=5.5, color=TEXT)
    for node, (y0, y1) in rpos.items():
        colour = MUTED if node == stub_label else palette.get(node, CONTEXT)
        ax.add_patch(Rectangle((0.845, y0), width, y1 - y0, facecolor=colour, edgecolor="none"))
        if y1 - y0 >= min_label * total:
            ax.text(0.885, (y0 + y1) / 2, node, ha="left", va="center", fontsize=5.5, color=TEXT)
    lcursor = {node: lpos[node][0] for node in left_nodes}
    rcursor = {node: rpos[node][0] for node in right_nodes}
    rows = [(r.source, r.target, r.mass) for r in flows.itertuples()]
    if stub is not None:
        rows += [(node, stub_label, float(mass)) for node, mass in stub.items() if mass > 0]
    order = {node: i for i, node in enumerate(right_nodes)}
    rows.sort(key=lambda r: (left_nodes.index(r[0]), order[r[1]]))
    for source, target, mass in rows:
        if mass <= 0:
            continue
        a0 = lcursor[source]
        a1 = a0 + mass
        lcursor[source] = a1
        b0 = rcursor[target]
        b1 = b0 + mass
        rcursor[target] = b1
        xa, xb = 0.12 + width, 0.845
        xm = (xa + xb) / 2
        verts = [(xa, a0), (xm, a0), (xm, b0), (xb, b0), (xb, b1), (xm, b1), (xm, a1), (xa, a1), (xa, a0)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.LINETO,
                 MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        colour = MUTED if target == stub_label else palette.get(source, CONTEXT)
        alpha = 0.35 if target == stub_label else 0.55
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colour, edgecolor="none", alpha=alpha))


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right",
            color=TEXT)
