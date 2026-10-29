"""MOSTA figures from the finished all-stage run, drawn on TACC.

Reads a finished run (25_ layout; default the ds production run) and writes
figures, source tables, reports and manifests under one new visualisation
root.  Nothing in any run directory is written, moved or removed; no OT is
computed; no coupling is rebuilt (the ds production run saved none).

Stages (``--stages``, comma-separated; ``all`` = every stage in this order):

  audit     the three run roots, the inputs of the ds run (counts, bytes,
            access times), the AnnData schema of every raw MOSTA file, the
            environment; manifests/INPUT_INVENTORY.tsv lists every input that
            exists only on scratch
  select    representative slice per stage and representative pair per
            displayed group, by fixed rules, every candidate listed
  fig1      embryogenesis overview: real sections at common scale on the
            stage axis, dataset composition, workflow
  fig2      tissue atlas: annotation maps of the representative slices
  s1        all 53 slices as thumbnails
  fig3      M4-E retain/reject maps of the selected pairs, same-stage
            controls beside the transitions (and S7: M4-E against M4-R)
  fig4      where retained mass goes: partner centroids, push-forward
            densities and partner spread, ConfidenceOT against balanced OT on
            the same retained bins
  fig5      annotation correspondence: heatmaps, alluvials, source-side mass
            and target-side rejection (and S8: heatmaps of every transition)
  supp      S2 rejection summary, S3 source/target asymmetry, S4 rejection
            by annotation and partner presence, S5 balanced OT on the same
            bins, S6 calibration, convergence and validity
  reports   FIGURE_DESIGN.md, FIGURE_CAPTIONS.md, VISUALIZATION_REPORT.md,
            README.md, DATA_PROVENANCE.tsv, PAIR_SELECTION.tsv; scripts copied
  package   one tar.gz of figures, tables, reports, scripts and manifests

Every panel is written as its own PDF + PNG (figures/<main|supplementary>/<figure>/...),
legends as separate files; no composite pages.

A stage that fails is recorded in logs/BLOCKERS.tsv with its traceback and
the remaining stages still run.  Selection rules (fixed, no visual choice):

* representative slice of a stage: most annotations present (Cavity not
  counted), then most analysed bins, then the sample name;
* representative pair of a group: the pair at the median M4-E source rejected
  fraction of that group (lower middle for an even count), ties by pair id.

    python mouse_embryo/30_mosta_figures.py --out <viz root> --stages all
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import traceback

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mosta_viz as mv  # noqa: E402
from mosta_viz import plt  # noqa: E402

DS_ROOT = "/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_all_stages"
OTHER_ROOTS = ("/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_all_stages_ds_potentials_20261028",
               "/scratch/10119/ghzheng/OT_project/mouse_embryo_results/mosta_all_stages_logcpm")
RAW_DATA = "/scratch/10119/ghzheng/OT_project/data"
ATLAS_STAGES = ("E9.5", "E11.5", "E12.5", "E14.5", "E16.5")
DISPLAY_GROUPS = (  # (kind, group, role)
    ("adjacent", "E9.5->E10.5", "early transition"),
    ("adjacent", "E11.5->E12.5", "mid transition"),
    ("adjacent", "E14.5->E15.5", "late transition"),
    ("same_stage", "E11.5", "same-stage control"),
    ("same_stage", "E14.5", "same-stage control"),
)
HEATMAP_GROUP = "E14.5->E15.5"
SANKEY_MIN_NODE = 0.01   # annotations under 1% of the slice's mass are pooled into "Other"
SANKEY_MIN_LINK = 0.002  # links under 0.2% of the slice's mass go to one pooled grey node
SMALL_LINKS = "links < 0.2% each (pooled)"
STAGE_ORDER = ["audit", "select", "fig1", "fig2", "s1", "fig3", "fig4", "fig5", "supp", "reports", "package"]


def arrow(group: str) -> str:
    return group.replace("->", "→")


class Context:
    def __init__(self, args) -> None:
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.commit = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip() or "unknown"
        self.run = mv.Run(Path(args.run))
        out = Path(args.out).resolve()
        if out == self.run.root.resolve() or self.run.root.resolve() in out.parents:
            raise SystemExit(f"--out {out} is inside the run {self.run.root}; refused")
        self.out = mv.Outputs(out, self.commit)
        self._palette = None
        self._selection = None

    # ------------------------------------------------------------ shared tables
    def palette(self) -> dict[str, str]:
        if self._palette is None:
            path = self.out.root / "tables" / "annotation_palette.tsv"
            if path.is_file():
                table = pd.read_csv(path, sep="\t")
            else:
                slices = self.run.manifest_slices()
                first = {}
                for sample in slices["sample"]:
                    for annotation in self.run.slice_table(sample)["annotation"].unique():
                        first.setdefault(annotation, (mv.stage_order(mv.stage_of(sample)), annotation))
                ordered = sorted(first, key=lambda a: first[a])
                palette = mv.build_palette(ordered)
                table = pd.DataFrame({"annotation": ordered, "colour": [palette[a] for a in ordered],
                                      "first_stage": [mv.STAGES[first[a][0]] for a in ordered]})
                self.out.table(table, "tables/annotation_palette.tsv")
            self._palette = dict(zip(table["annotation"], table["colour"]))
        return self._palette

    def annotation_order(self) -> list[str]:
        return list(self.palette())

    def selection(self) -> dict:
        if self._selection is None:
            slices_path = self.out.root / "tables" / "representative_slices.tsv"
            pairs_path = self.out.root / "reports" / "PAIR_SELECTION.tsv"
            if not (slices_path.is_file() and pairs_path.is_file()):
                stage_select(self)
            slices = pd.read_csv(slices_path, sep="\t")
            pairs = pd.read_csv(pairs_path, sep="\t")
            self._selection = {
                "slices": slices, "pairs": pairs,
                "slice_of_stage": dict(zip(slices.loc[slices["selected"], "stage"],
                                           slices.loc[slices["selected"], "sample"])),
                "pair_of_group": {row.group: row for row in pairs[pairs["selected"]].itertuples()},
            }
        return self._selection


# ---------------------------------------------------------------- audit

def stage_audit(ctx: Context) -> None:
    rows = []
    for root in (ctx.run.root, *map(Path, ctx.args.other_runs)):
        record = {"root": str(root), "exists": root.is_dir()}
        if root.is_dir():
            record.update({
                "pairs_success": len(list(root.glob("pairs/*/SUCCESS"))),
                "pairs_failed": len(list(root.glob("pairs/*.FAILED"))),
                "representations_success": len(list(root.glob("representations/*/REPRESENTATION_SUCCESS"))),
                "equivalence": ",".join(sorted(p.name for p in root.glob("equivalence/EQUIVALENCE_*"))),
                "coupling_check": ",".join(sorted(p.name for p in root.glob("coupling_verification/COUPLING_*"))),
                "pairs_with_potentials": len(list(root.glob("pairs/*/potentials.npz"))),
                "audit_summary": (root / "audit" / "summary.json").is_file(),
                "representations_is_link": (root / "representations").is_symlink(),
            })
        rows.append(record)
    ctx.out.table(pd.DataFrame(rows), "tables/audit_runs.tsv")

    root = ctx.run.root
    inventory = []

    def add(category, path):
        try:
            info = path.stat()
            inventory.append({"category": category, "path": str(path), "bytes": info.st_size,
                              "modified": time.strftime("%Y-%m-%d", time.localtime(info.st_mtime)),
                              "accessed": time.strftime("%Y-%m-%d", time.localtime(info.st_atime)), "exists": True})
        except FileNotFoundError:
            inventory.append({"category": category, "path": str(path), "bytes": 0, "modified": "", "accessed": "",
                              "exists": False})

    package_list = root / "package" / "PACKAGE_FILES.txt"
    if package_list.is_file():
        for line in package_list.read_text(encoding="utf-8").split():
            add("package", root / line)
    for path in sorted(root.glob("representations/*/representation.npz")):
        add("representation_npz", path)
    for path in sorted(root.glob("equalised/*.h5ad")):
        add("equalised_h5ad", path)
    for path in sorted(Path(ctx.args.raw).glob("*.MOSTA.h5ad")):
        add("raw_mosta_h5ad", path)
    inventory = pd.DataFrame(inventory)
    ctx.out.table(inventory, "manifests/INPUT_INVENTORY.tsv")
    summary = inventory.groupby("category").agg(files=("path", "size"), missing=("exists", lambda s: int((~s).sum())),
                                                 gigabytes=("bytes", lambda s: round(s.sum() / 1e9, 2)),
                                                 oldest_access=("accessed", lambda s: min([v for v in s if v] or [""])),
                                                 newest_access=("accessed", lambda s: max([v for v in s if v] or [""])))
    summary = summary.reset_index()
    summary["storage"] = "scratch only (no copy known elsewhere)"
    ctx.out.table(summary, "manifests/INPUT_SUMMARY.tsv")

    schema = []
    try:
        import anndata as ad
    except ImportError as error:
        schema.append({"file": "", "error": f"anndata not importable: {error}"})
    else:
        for path in sorted(Path(ctx.args.raw).glob("*.MOSTA.h5ad")):
            record = {"file": path.name}
            try:
                data = ad.read_h5ad(path, backed="r")
                record.update({"n_obs": data.n_obs, "n_vars": data.n_vars,
                               "obs_columns": ";".join(map(str, data.obs.columns)),
                               "obsm_keys": ";".join(map(str, data.obsm.keys())),
                               "layers": ";".join(map(str, data.layers.keys())) if hasattr(data, "layers") else ""})
                if "annotation" in data.obs:
                    record["annotations"] = int(data.obs["annotation"].nunique())
                if "spatial" in data.obsm:
                    xy = np.asarray(data.obsm["spatial"])[:, :2]
                    record.update({"x_min": float(xy[:, 0].min()), "x_max": float(xy[:, 0].max()),
                                   "y_min": float(xy[:, 1].min()), "y_max": float(xy[:, 1].max())})
                data.file.close()
            except Exception as error:  # recorded, never repaired
                record["error"] = repr(error)
            schema.append(record)
    ctx.out.table(pd.DataFrame(schema), "tables/audit_raw_anndata_schema.tsv")

    columns = []
    for sample in ctx.run.manifest_slices()["sample"]:
        head = pd.read_csv(root / "slices" / f"{sample}_bins.csv.gz", nrows=2)
        table = ctx.run.slice_table(sample)
        columns.append({"sample": sample, "columns": ";".join(head.columns), "bins": len(table),
                        "analysed": int(table["analysed"].sum()), "cavity": int((table["annotation"] == mv.EXCLUDED).sum()),
                        "annotations_present": int(table.loc[table["annotation"] != mv.EXCLUDED, "annotation"].nunique()),
                        "x_range": float(np.ptp(table["x"])), "y_range": float(np.ptp(table["y"]))})
    ctx.out.table(pd.DataFrame(columns), "tables/audit_slice_tables.tsv")

    environment = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node(),
                   "commit": ctx.commit, "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
                   "cpus": os.environ.get("SLURM_CPUS_ON_NODE", str(os.cpu_count()))}
    from importlib import metadata
    for module in ("numpy", "pandas", "scipy", "matplotlib", "anndata", "scanpy", "plotly"):
        try:
            environment[module] = metadata.version(module)
        except metadata.PackageNotFoundError:
            environment[module] = "not installed"
    ctx.out.table(pd.DataFrame(list(environment.items()), columns=["item", "value"]), "reports/ENVIRONMENT.tsv")


# ---------------------------------------------------------------- select

def stage_select(ctx: Context) -> None:
    audit = ctx.run.root / "audit"
    slices = pd.read_csv(ctx.out.root / "tables" / "audit_slice_tables.tsv", sep="\t") \
        if (ctx.out.root / "tables" / "audit_slice_tables.tsv").is_file() else None
    if slices is None:
        rows = []
        for sample in ctx.run.manifest_slices()["sample"]:
            table = ctx.run.slice_table(sample)
            rows.append({"sample": sample, "analysed": int(table["analysed"].sum()),
                         "annotations_present": int(table.loc[table["annotation"] != mv.EXCLUDED, "annotation"].nunique())})
        slices = pd.DataFrame(rows)
    slices = slices[["sample", "analysed", "annotations_present"]].copy()
    slices["stage"] = slices["sample"].map(mv.stage_of)
    slices = slices.sort_values(["stage", "annotations_present", "analysed", "sample"],
                                ascending=[True, False, False, True], key=lambda s: s.map(mv.stage_order) if s.name == "stage" else s)
    slices["rank_in_stage"] = slices.groupby("stage").cumcount() + 1
    slices["selected"] = slices["rank_in_stage"] == 1
    slices["rule"] = "most annotations present (Cavity not counted), then most analysed bins, then sample name"
    ctx.out.table(slices, "tables/representative_slices.tsv")

    metrics = pd.read_csv(audit / "pair_metrics.csv")
    m4e = metrics[metrics["method"] == "M4-E"]
    fractions = pd.read_csv(audit / "annotation_fractions.csv.gz", usecols=["pair_id", "side", "annotation", "n_bins"])
    present = fractions[fractions["n_bins"] > 0].groupby(["pair_id", "side"])["annotation"].nunique().unstack()
    rows = []
    for kind, group, role in DISPLAY_GROUPS:
        part = m4e[(m4e["kind"] == kind) & (m4e["group"] == group)].sort_values(
            ["source_final_rejection_rate", "pair_id"]).reset_index(drop=True)
        pick = (len(part) - 1) // 2
        for i, row in part.iterrows():
            rows.append({"group": group, "kind": kind, "role": role, "pair_id": row["pair_id"],
                         "source_sample": row["source_sample"], "target_sample": row["target_sample"],
                         "source_bins": row["source_bins"], "target_bins": row["target_bins"],
                         "source_annotations": present.loc[row["pair_id"], "source"] if row["pair_id"] in present.index else np.nan,
                         "target_annotations": present.loc[row["pair_id"], "target"] if row["pair_id"] in present.index else np.nan,
                         "m4e_source_rejected": row["source_final_rejection_rate"],
                         "m4e_target_rejected": row["target_final_rejection_rate"],
                         "rank": i + 1, "pairs_in_group": len(part), "selected": i == pick,
                         "rule": "median M4-E source rejected fraction in the group (lower middle for an even count), "
                                 "ties by pair id; annotation counts and sizes reported, not used"})
    ctx.out.table(pd.DataFrame(rows), "reports/PAIR_SELECTION.tsv")
    ctx._selection = None


# ---------------------------------------------------------------- fig1

def map_figure(table: pd.DataFrame, width: float = 3.0, cap: float = 4.2):
    """One figure sized to the section's aspect ratio."""
    ratio = np.ptp(table["y"]) / max(np.ptp(table["x"]), 1e-9)
    figure = plt.figure(figsize=(width, min(width * ratio, cap) + 0.45))
    ax = figure.add_axes([0.02, 0.02, 0.96, 0.96 * (min(width * ratio, cap)) / (min(width * ratio, cap) + 0.45)])
    return figure, ax


def legend_figure(ctx: Context, handles, names, stem: str, ncol: int, title: str = "") -> None:
    rows = int(np.ceil(len(names) / ncol))
    figure = plt.figure(figsize=(7.2, 0.18 * rows + 0.35))
    ax = figure.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.legend(handles, names, ncol=ncol, frameon=False, loc="center", handlelength=1.0, handleheight=1.0,
              columnspacing=1.2, fontsize=6, title=title or None, title_fontsize=6.5)
    ctx.out.figure(figure, stem)


def annotation_legend_file(ctx: Context, names: list[str], stem: str) -> None:
    palette = ctx.palette()
    legend_figure(ctx, mv.legend_handles(palette, names), names, stem, ncol=5, title="MOSTA annotation")


STATUS_NAMES = ["retained (supported correspondence)", "rejected (no supported correspondence)",
                "Cavity (excluded, not analysed)"]


def status_legend_file(ctx: Context, stem: str) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (mv.RETAINED, mv.REJECTED, mv.EXCLUDED_COLOUR)]
    legend_figure(ctx, handles, STATUS_NAMES, stem, ncol=3)


def annotation_map_file(ctx: Context, table: pd.DataFrame, title: str, stem: str) -> None:
    figure, ax = map_figure(table)
    mv.draw_bins(ax, table["x"].to_numpy(), table["y"].to_numpy(), table["annotation"].map(ctx.palette()).to_numpy())
    ax.set_title(title, fontsize=7)
    ctx.out.figure(figure, stem)


def stage_fig1(ctx: Context) -> None:
    folder = "figures/main/Fig1_embryogenesis_overview"
    palette = ctx.palette()
    selection = ctx.selection()
    per_slice = ctx.run.per_slice()
    pairs = ctx.run.manifest_pairs()
    report = json.loads((ctx.run.root / "equalisation" / "report.json").read_text(encoding="utf-8"))
    target_depth = report.get("target_depth", report.get("target", "n/a"))

    # A: one real section per stage, one common scale, along the stage axis
    tables = {stage: ctx.run.slice_table(selection["slice_of_stage"][stage]) for stage in mv.STAGES}
    width_in, height_in = 7.2, 2.6
    slot = 0.92 / len(mv.STAGES)
    scale = min(min(slot * 0.92 * width_in / np.ptp(t["x"]) for t in tables.values()),
                min(0.62 * height_in / np.ptp(t["y"]) for t in tables.values()))
    figure = plt.figure(figsize=(width_in, height_in))
    baseline = 0.3
    for i, stage in enumerate(mv.STAGES):
        table = tables[stage]
        w, h = np.ptp(table["x"]) * scale / width_in, np.ptp(table["y"]) * scale / height_in
        cx = 0.04 + slot * (i + 0.5)
        ax = figure.add_axes([cx - w / 2, baseline, w, h])
        mv.draw_bins(ax, table["x"].to_numpy(), table["y"].to_numpy(), table["annotation"].map(palette).to_numpy())
        sample = selection["slice_of_stage"][stage]
        n_annotations = table.loc[table["annotation"] != mv.EXCLUDED, "annotation"].nunique()
        figure.text(cx, baseline - 0.1, stage, ha="center", fontsize=8, fontweight="bold")
        figure.text(cx, baseline - 0.2, f"{sample.split('_')[1]}\n{n_annotations} annotations", ha="center",
                    fontsize=5.5, color=mv.MUTED, linespacing=1.1)
    axis = figure.add_axes([0.04, baseline - 0.04, 0.92, 0.02])
    axis.axis("off")
    axis.annotate("", xy=(1, 0.5), xytext=(0, 0.5), xycoords="axes fraction",
                  arrowprops=dict(arrowstyle="-|>", color=mv.TEXT, lw=0.8))
    figure.text(0.04, 0.95, "MOSTA mouse embryo, E9.5–E16.5: one real section per stage at a common scale "
                "(representative section rule in tables/representative_slices.tsv)", fontsize=7)
    ctx.out.figure(figure, f"{folder}/Fig1A_stage_series")
    shown = sorted(set().union(*[set(t["annotation"]) for t in tables.values()]), key=ctx.annotation_order().index)
    annotation_legend_file(ctx, shown, f"{folder}/Fig1A_legend_annotations")

    # B: composition, three separate panels
    per_slice["stage"] = per_slice["sample"].map(mv.stage_of)
    by_stage = per_slice.groupby("stage").agg(slices=("sample", "size"), analysed=("bins_analysed", "sum"),
                                              cavity=("bins_excluded_cavity", "sum")).reindex(mv.STAGES)
    ctx.out.table(by_stage.reset_index(), "tables/Fig1B_composition_by_stage.tsv")
    x = np.arange(len(mv.STAGES))
    figure, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.bar(x, by_stage["slices"], color="#5B7FA6", width=0.7)
    for xi, value in zip(x, by_stage["slices"]):
        ax.text(xi, value + 0.3, str(int(value)), ha="center", fontsize=6)
    ax.set_xticks(x, mv.STAGES, rotation=45, ha="right")
    ax.set_ylabel("sections")
    ax.set_title(f"Sections per stage ({int(by_stage['slices'].sum())} total)")
    ctx.out.figure(figure, f"{folder}/Fig1B_sections_per_stage")
    figure, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.bar(x, by_stage["analysed"] / 1e3, color="#5B7FA6", width=0.7, label="analysed")
    ax.bar(x, by_stage["cavity"] / 1e3, bottom=by_stage["analysed"] / 1e3, color=mv.EXCLUDED_COLOUR, width=0.7,
           label="Cavity (excluded)")
    ax.set_xticks(x, mv.STAGES, rotation=45, ha="right")
    ax.set_ylabel("bins (thousands)")
    ax.set_title(f"Bins per stage ({int(by_stage['analysed'].sum()):,} analysed)")
    ax.legend(frameon=False, loc="upper left")
    ctx.out.figure(figure, f"{folder}/Fig1B_bins_per_stage")
    adjacent = pairs[pairs["kind"] == "adjacent"].groupby("group").size()
    same = pairs[pairs["kind"] == "same_stage"].groupby("group").size().reindex(mv.STAGES).fillna(0)
    figure, ax = plt.subplots(figsize=(2.6, 2.0))
    ax.bar(x, same.to_numpy(), color="#B0B0B0", width=0.55, label="same-stage pairs")
    for i in range(len(mv.STAGES) - 1):
        value = adjacent.get(f"{mv.STAGES[i]}->{mv.STAGES[i + 1]}", 0)
        ax.bar(i + 0.5, value, color="#5B7FA6", width=0.45, label="adjacent-stage pairs" if i == 0 else None)
    ax.set_xticks(x, mv.STAGES, rotation=45, ha="right")
    ax.set_ylabel("OT comparisons")
    ax.set_title(f"{int(adjacent.sum())} adjacent + {int(same.sum())} same-stage")
    ax.legend(frameon=False, loc="upper left")
    ctx.out.figure(figure, f"{folder}/Fig1B_comparisons")

    # C: workflow
    figure = plt.figure(figsize=(7.2, 1.7))
    ax = figure.add_axes([0, 0, 1, 1])
    ax.axis("off")
    steps = [
        ("MOSTA Stereo-seq\nsections", f"{len(per_slice)} sections\nE9.5–E16.5"),
        ("Bins and\nrepresentation", f"Cavity removed\ndepth equalised to {target_depth}\nranknm256_noscale_ds_cos"),
        ("Pairwise OT", f"{int(adjacent.sum())} adjacent + {int(same.sum())} same-stage\nM4-E, M4-R, balanced OT"),
        ("M4-E gate", "each bin retained\nor rejected\n(calibrated, acceptance 0.99)"),
        ("Summaries", "spatial maps\npartner positions\nannotation correspondence"),
    ]
    width = 0.165
    for i, (head, body) in enumerate(steps):
        x0 = 0.01 + i * 0.2
        ax.add_patch(plt.Rectangle((x0, 0.2), width, 0.7, facecolor="#F2F5F8", edgecolor="#5B7FA6", lw=0.8))
        ax.text(x0 + width / 2, 0.76, head, ha="center", va="center", fontsize=7, fontweight="bold")
        ax.text(x0 + width / 2, 0.44, body, ha="center", va="center", fontsize=5.8, color=mv.TEXT)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x0 + 0.2, 0.55), xytext=(x0 + width, 0.55),
                        arrowprops=dict(arrowstyle="-|>", color=mv.TEXT, lw=0.8))
    ax.text(0.5, 0.06, "Correspondence between sections of different embryos; not lineage tracing, not a 3D "
            "reconstruction.", ha="center", fontsize=6, color=mv.MUTED, style="italic")
    ctx.out.figure(figure, f"{folder}/Fig1C_workflow")


def stage_fig2(ctx: Context) -> None:
    folder = "figures/main/Fig2_spatial_atlas"
    selection = ctx.selection()
    shown = set()
    for stage in ATLAS_STAGES:
        sample = selection["slice_of_stage"][stage]
        table = ctx.run.slice_table(sample)
        shown |= set(table["annotation"])
        annotation_map_file(ctx, table, f"{stage}  {sample.split('_')[1]}: {int(table['analysed'].sum()):,} analysed "
                            f"bins, {int((table['annotation'] == mv.EXCLUDED).sum()):,} Cavity", f"{folder}/Fig2_{sample}")
    annotation_legend_file(ctx, [a for a in ctx.annotation_order() if a in shown], f"{folder}/Fig2_legend_annotations")


def stage_s1(ctx: Context) -> None:
    folder = "figures/supplementary/FigS1_all_stage_atlas"
    palette = ctx.palette()
    slices = ctx.run.manifest_slices()
    shown = set()
    for stage in mv.STAGES:
        samples = slices.loc[slices["stage"] == stage, "sample"].tolist()
        columns = min(len(samples), 6)
        rows = int(np.ceil(len(samples) / columns))
        figure = plt.figure(figsize=(7.2, 1.45 * rows + 0.3))
        grid = figure.add_gridspec(rows, columns, left=0.01, right=0.99, top=1 - 0.3 / (1.45 * rows + 0.3),
                                   bottom=0.01, hspace=0.25, wspace=0.05)
        for k, sample in enumerate(samples):
            table = ctx.run.slice_table(sample)
            shown |= set(table["annotation"])
            ax = figure.add_subplot(grid[k // columns, k % columns])
            mv.draw_bins(ax, table["x"].to_numpy(), table["y"].to_numpy(), table["annotation"].map(palette).to_numpy())
            ax.set_title(f"{sample.split('_')[1]}: {int(table['analysed'].sum()):,} bins", fontsize=5.5)
        figure.suptitle(f"{stage}: all {len(samples)} sections (each at its own scale)", fontsize=7, y=0.995)
        ctx.out.figure(figure, f"{folder}/FigS1_{stage}", dpi=250)
    annotation_legend_file(ctx, [a for a in ctx.annotation_order() if a in shown], f"{folder}/FigS1_legend_annotations")


# ---------------------------------------------------------------- fig3

def side_frame(ctx: Context, pair_id: str, side: str, sample: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All bins of the slice, and its analysed bins joined to the pair's per-bin table."""
    table = ctx.run.slice_table(sample)
    analysed = ctx.run.analysed(sample)
    bins = ctx.run.pair_bins(pair_id, side)
    if len(bins) != len(analysed):
        raise ValueError(f"{pair_id} {side}: {len(bins)} rows against {len(analysed)} analysed bins of {sample}")
    return table, pd.concat([analysed.reset_index(drop=True), bins.reset_index(drop=True)], axis=1)


def rows_in_slice(table: pd.DataFrame, joined: pd.DataFrame) -> np.ndarray:
    return pd.Series(np.arange(len(table)), index=table["bin_id"]).loc[joined["bin_id"]].to_numpy()


def gate_map_file(ctx: Context, table, joined, column, title, stem) -> None:
    colours = np.full(len(table), mv.EXCLUDED_COLOUR, dtype=object)
    colours[rows_in_slice(table, joined)] = mv.status_colours(joined[column].to_numpy().astype(bool))
    figure, ax = map_figure(table)
    mv.draw_bins(ax, table["x"].to_numpy(), table["y"].to_numpy(), colours)
    ax.set_title(title, fontsize=7)
    ctx.out.figure(figure, stem)


def pair_label(row) -> str:
    kind = "same-stage control" if row.kind == "same_stage" else row.role
    return f"{kind} {arrow(row.group)}"


def stage_fig3(ctx: Context) -> None:
    folder = "figures/main/Fig3_spatial_retain_reject"
    s7 = "figures/supplementary/FigS7_m4e_vs_m4r_gates"
    chosen = ctx.selection()["pair_of_group"]
    shown = set()
    agreement = []
    for _, group, _ in DISPLAY_GROUPS:
        row = chosen[group]
        prefix = f"{folder}/{group.replace('->', '_to_')}_{'same_stage' if row.kind == 'same_stage' else 'adjacent'}"
        for side in ("source", "target"):
            sample = getattr(row, f"{side}_sample")
            table, joined = side_frame(ctx, row.pair_id, side, sample)
            shown |= set(table["annotation"])
            annotation_map_file(ctx, table, f"{pair_label(row)}\n{side}: {sample}, {len(joined):,} analysed bins",
                                f"{prefix}/Fig3_{side}_annotation_{sample}")
            gate_map_file(ctx, table, joined, "m4e_retained",
                          f"{pair_label(row)}\n{side} {sample}: M4-E rejected {1 - joined['m4e_retained'].mean():.3f}",
                          f"{prefix}/Fig3_{side}_m4e_gate_{sample}")
            gate_map_file(ctx, table, joined, "m4r_retained",
                          f"{pair_label(row)}\n{side} {sample}: M4-R rejected {1 - joined['m4r_retained'].mean():.3f}",
                          f"{s7}/{row.pair_id}/FigS7_{side}_m4r_gate_{sample}")
            e = ~joined["m4e_retained"].astype(bool)
            m = ~joined["m4r_retained"].astype(bool)
            union = int((e | m).sum())
            agreement.append({"group": group, "pair_id": row.pair_id, "side": side, "m4e_rejected": float(e.mean()),
                              "m4r_rejected": float(m.mean()),
                              "jaccard_rejected": float((e & m).sum() / union) if union else np.nan})
    status_legend_file(ctx, f"{folder}/Fig3_legend_status")
    annotation_legend_file(ctx, [a for a in ctx.annotation_order() if a in shown], f"{folder}/Fig3_legend_annotations")
    ctx.out.table(pd.DataFrame(agreement), "tables/FigS7_m4e_m4r_agreement.tsv")


# ---------------------------------------------------------------- fig4

def total_push(arrays: dict, key: str) -> np.ndarray:
    value = np.asarray(arrays[key], dtype=np.float64)
    return value.sum(axis=1) if value.ndim == 2 else value


def stage_fig4(ctx: Context) -> None:
    folder = "figures/main/Fig4_spatial_correspondence"
    palette = ctx.palette()
    chosen = ctx.selection()["pair_of_group"]
    summary = []
    shown = set()
    for kind, group, _ in DISPLAY_GROUPS:
        if kind != "adjacent":
            continue
        row = chosen[group]
        prefix = f"{folder}/{group.replace('->', '_to_')}"
        s_table, source = side_frame(ctx, row.pair_id, "source", row.source_sample)
        t_table, target = side_frame(ctx, row.pair_id, "target", row.target_sample)
        transport = ctx.run.transport(row.pair_id, "target")
        retained = source["m4e_retained"].astype(bool).to_numpy()
        supported = retained & (source["co_partner_mass"].to_numpy() > 0)
        shown |= set(source.loc[retained, "annotation"])
        head = f"{arrow(group)}  {row.source_sample} → {row.target_sample}"

        colours = np.full(len(s_table), mv.EXCLUDED_COLOUR, dtype=object)
        colours[rows_in_slice(s_table, source)] = np.where(retained, source["annotation"].map(palette).to_numpy(), "#F6C9BA")
        figure, ax = map_figure(s_table)
        mv.draw_bins(ax, s_table["x"].to_numpy(), s_table["y"].to_numpy(), colours)
        ax.set_title(f"{head}\nsource: retained bins by annotation (retained {retained.mean():.3f}); pale = rejected",
                     fontsize=6.5)
        ctx.out.figure(figure, f"{prefix}/Fig4A_source_retained")

        tx, ty = t_table["x"].to_numpy(), t_table["y"].to_numpy()
        point_colours = source.loc[supported, "annotation"].map(palette).to_numpy()
        for letter, prefix_name, label in (("B", "co", "ConfidenceOT"), ("D", "bot", "balanced OT, same bins")):
            figure, ax = map_figure(t_table)
            mv.draw_bins(ax, tx, ty, np.full(len(tx), "#EDEDED", dtype=object))
            ax.scatter(source.loc[supported, f"{prefix_name}_partner_x"], source.loc[supported, f"{prefix_name}_partner_y"],
                       c=list(point_colours), s=0.8, linewidths=0, alpha=0.6, rasterized=True)
            ax.set_title(f"{head}\n{label}: partner centroid of each retained source bin on the target "
                         f"(transport-weighted, not a tracked position)", fontsize=6)
            ctx.out.figure(figure, f"{prefix}/Fig4{letter}_{prefix_name}_partner_centroids")

        co_push = total_push(transport, "co_pushforward")
        bot_push = total_push(transport, "bot_pushforward") - total_push(transport, "bot_pushforward_co_rejected")
        vmax = float(np.nanpercentile(np.concatenate([co_push, bot_push]), 99)) or 1.0
        target_rows = rows_in_slice(t_table, target)
        for letter, values, label in (("C", co_push, "ConfidenceOT"), ("E", bot_push, "balanced OT, same bins")):
            density = np.full(len(t_table), np.nan)
            density[target_rows] = values
            figure, ax = map_figure(t_table)
            mv.slice_axes(ax, tx, ty)
            size = mv.marker_size(ax, tx, ty)
            ax.scatter(tx, ty, c=mv.EXCLUDED_COLOUR, s=size, marker="s", linewidths=0, rasterized=True)
            ok = np.isfinite(density)
            image = ax.scatter(tx[ok], ty[ok], c=density[ok], s=size, marker="s", linewidths=0, cmap="magma_r",
                               vmin=0, vmax=vmax, rasterized=True)
            bar = figure.colorbar(image, ax=ax, fraction=0.035, pad=0.01)
            bar.set_label("mass received per target bin\n(C and E share this scale)", fontsize=5)
            bar.ax.tick_params(labelsize=5)
            ax.set_title(f"{head}\n{label}: push-forward of the retained source mass", fontsize=6.5)
            ctx.out.figure(figure, f"{prefix}/Fig4{letter}_pushforward_{'co' if letter == 'C' else 'bot'}")

        co_spread = source.loc[supported, "co_partner_spread"].to_numpy()
        bot_spread = source.loc[supported, "bot_partner_spread"].to_numpy()
        figure, ax = plt.subplots(figsize=(2.8, 2.1))
        bins = np.linspace(0, np.nanpercentile(np.concatenate([co_spread, bot_spread]), 99.5), 40)
        ax.hist(co_spread, bins=bins, histtype="step", color=mv.RETAINED, label="ConfidenceOT", density=True)
        ax.hist(bot_spread, bins=bins, histtype="step", color="#7B4FA0", label="balanced OT, same bins", density=True)
        ax.set_xlabel("partner spread (coordinate units)")
        ax.set_ylabel("density")
        ax.set_title(f"{arrow(group)}: spread of partners, {int(supported.sum()):,} retained bins", fontsize=6.5)
        ax.legend(frameon=False)
        ctx.out.figure(figure, f"{prefix}/Fig4F_partner_spread")
        summary.append({"pair_id": row.pair_id, "group": group, "source_retained": float(retained.mean()),
                        "retained_with_supported_mass": int(supported.sum()),
                        "median_spread_confidenceot": float(np.nanmedian(co_spread)),
                        "median_spread_balanced_same_bins": float(np.nanmedian(bot_spread)),
                        "pushed_mass_confidenceot": float(co_push.sum()),
                        "pushed_mass_balanced_same_bins": float(bot_push.sum())})
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#F6C9BA", mv.EXCLUDED_COLOUR, "#EDEDED")]
    legend_figure(ctx, handles, ["source bin rejected by M4-E", "Cavity (excluded)", "target section (context)"],
                  f"{folder}/Fig4_legend_context", ncol=3)
    annotation_legend_file(ctx, [a for a in ctx.annotation_order() if a in shown], f"{folder}/Fig4_legend_annotations")
    ctx.out.table(pd.DataFrame(summary), "tables/Fig4_correspondence_summary.tsv")


# ---------------------------------------------------------------- fig5

def group_flows(group: str, transitions: pd.DataFrame, fractions: pd.DataFrame) -> dict:
    """Mean over the group's pairs, in units of the source slice's nominal mass (absent = 0)."""
    side = fractions[(fractions["group"] == group) & (fractions["side"] == "source")].copy()
    n = side["pair_id"].nunique()
    totals = side.groupby("pair_id")["n_bins"].sum()
    side["share"] = side["n_bins"] / side["pair_id"].map(totals)
    share = side.groupby("annotation")["share"].sum() / n
    rejected = (side["m4e_rejected_fraction"] * side["share"]).groupby(side["annotation"]).sum() / n
    unsupported = (side["co_retained_without_support_share"] * side["share"]).groupby(side["annotation"]).sum() / n
    forward = transitions[(transitions["group"] == group) & (transitions["direction"] == "forward")]
    out = {"pairs": n, "share": share, "rejected": rejected, "unsupported": unsupported}
    for method in ("co", "bot"):
        part = forward[forward["method"] == method]
        out[method] = (part.groupby(["annotation", "partner_annotation"])["nominal_mass"].sum() / n).rename("mass")
    return out


def pooled(flows: pd.Series, share: pd.Series, minimum: float) -> tuple[pd.DataFrame, list[str]]:
    keep = set(share[share >= minimum].index)
    frame = flows.reset_index()
    frame.columns = ["source", "target", "mass"]
    received = frame.groupby("target")["mass"].sum()
    keep_target = set(received[received >= minimum].index)
    frame["source"] = frame["source"].where(frame["source"].isin(keep), "Other")
    frame["target"] = frame["target"].where(frame["target"].isin(keep_target), "Other")
    frame = frame.groupby(["source", "target"], as_index=False)["mass"].sum()
    frame["target"] = frame["target"].where(frame["mass"] >= SANKEY_MIN_LINK, SMALL_LINKS)
    return frame.groupby(["source", "target"], as_index=False)["mass"].sum(), sorted(keep)


def heatmap_file(ctx: Context, item: dict, method: str, group: str, stem: str) -> pd.DataFrame:
    order = ctx.annotation_order()
    share = item["share"][item["share"] > 0]
    rows = [a for a in order if a in share.index]
    matrix = item[method].unstack(fill_value=0.0)
    columns = [a for a in order if a in matrix.columns and matrix[a].sum() > 0]
    values = matrix.reindex(index=rows, columns=columns, fill_value=0.0).div(share.reindex(rows), axis=0)
    if method == "co":
        values["(rejected)"] = (item["rejected"].reindex(rows).fillna(0) / share.reindex(rows)).to_numpy()
    figure, ax = plt.subplots(figsize=(0.13 * values.shape[1] + 1.8, 0.13 * len(rows) + 1.4))
    image = ax.imshow(values.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(rows)), rows, fontsize=5)
    ax.set_xticks(range(values.shape[1]), values.columns, rotation=90, fontsize=5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    name = "ConfidenceOT (M4-E): retained mass by target annotation, and rejected" if method == "co" \
        else "balanced OT: all mass by target annotation"
    ax.set_title(f"{arrow(group)} ({item['pairs']} pairs, mean)\n{name}", fontsize=6.5)
    ax.set_xlabel("target annotation")
    ax.set_ylabel("source annotation")
    bar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("fraction of source annotation mass", fontsize=5.5)
    bar.ax.tick_params(labelsize=5)
    ctx.out.figure(figure, stem)
    return values


def stage_fig5(ctx: Context) -> None:
    folder = "figures/main/Fig5_annotation_transitions"
    s8 = "figures/supplementary/FigS8_annotation_heatmaps"
    palette = dict(ctx.palette())
    palette["Other"] = mv.CONTEXT
    palette[SMALL_LINKS] = "#E3E3E3"
    order = ctx.annotation_order() + ["Other", SMALL_LINKS]
    audit = ctx.run.root / "audit"
    transitions = pd.read_csv(audit / "annotation_transitions.csv.gz")
    transitions = transitions[transitions["kind"] == "adjacent"]
    fractions = pd.read_csv(audit / "annotation_fractions.csv.gz")
    fractions = fractions[fractions["kind"] == "adjacent"]
    groups = sorted(fractions["group"].unique(), key=mv.group_order)
    flows = {group: group_flows(group, transitions, fractions) for group in groups}

    records = []
    for group, item in flows.items():
        for method in ("co", "bot"):
            for (a, b), mass in item[method].items():
                records.append({"group": group, "method": method, "source_annotation": a, "target_annotation": b,
                                "mean_nominal_mass": mass, "source_annotation_share": item["share"].get(a, np.nan)})
        for a in item["share"].index:
            records.append({"group": group, "method": "co", "source_annotation": a, "target_annotation": "(rejected)",
                            "mean_nominal_mass": item["rejected"].get(a, 0.0), "source_annotation_share": item["share"][a]})
    records = pd.DataFrame(records)
    records["conditional_share_of_source_annotation"] = records["mean_nominal_mass"] / records["source_annotation_share"]
    ctx.out.table(records, "tables/Fig5_annotation_flows_all_transitions.tsv")

    for method in ("co", "bot"):
        heatmap_file(ctx, flows[HEATMAP_GROUP], method, HEATMAP_GROUP, f"{folder}/Fig5A_heatmap_{method}_{HEATMAP_GROUP.replace('->', '_to_')}")
        for group in groups:
            heatmap_file(ctx, flows[group], method, group, f"{s8}/FigS8_heatmap_{method}_{group.replace('->', '_to_')}")

    for kind, group, _ in DISPLAY_GROUPS:
        if kind != "adjacent":
            continue
        item = flows[group]
        for method in ("co", "bot"):
            frame, keep = pooled(item[method], item["share"], SANKEY_MIN_NODE)
            left = frame.groupby("source")["mass"].sum()
            stub = None
            if method == "co":
                rejected = item["rejected"] + item["unsupported"].reindex(item["rejected"].index).fillna(0)
                rejected.index = [a if a in keep else "Other" for a in rejected.index]
                stub = rejected.groupby(level=0).sum()
                left = left.add(stub, fill_value=0.0)
            left = left.reindex([a for a in order if a in left.index])
            right = frame.groupby("target")["mass"].sum()
            right = right.reindex([a for a in order if a in right.index])
            figure, ax = plt.subplots(figsize=(4.2, 5.2))
            mv.alluvial(ax, left, frame, right, palette, stub=stub, stub_label="rejected (no supported partner)")
            label = "ConfidenceOT (M4-E)" if method == "co" else "balanced OT"
            ax.set_title(f"{label}  {arrow(group)}  (mean of {item['pairs']} pairs)\nannotations < "
                         f"{SANKEY_MIN_NODE:.0%} of the section pooled as Other; links < {SANKEY_MIN_LINK:.1%} pooled",
                         fontsize=6.5)
            ctx.out.figure(figure, f"{folder}/Fig5B_alluvial_{method}_{group.replace('->', '_to_')}")

    breakdown = []
    for group in groups:
        item = flows[group]
        same = sum(m for (a, b), m in item["co"].items() if a == b)
        total = float(item["co"].sum())
        breakdown.append({"group": group, "retained_same": same, "retained_other": total - same,
                          "retained_without_support": float(item["unsupported"].sum()),
                          "rejected": float(item["rejected"].sum())})
    breakdown = pd.DataFrame(breakdown)
    ctx.out.table(breakdown, "tables/Fig5C_source_mass_breakdown.tsv")
    x = np.arange(len(groups))
    figure, ax = plt.subplots(figsize=(3.4, 2.4))
    bottom = np.zeros(len(groups))
    for column, colour, name in (("retained_same", "#0F6E64", "retained, same annotation"),
                                 ("retained_other", "#7CC6BB", "retained, other annotation"),
                                 ("retained_without_support", "#BBBBBB", "retained, no supported mass"),
                                 ("rejected", mv.REJECTED, "rejected")):
        ax.bar(x, breakdown[column], bottom=bottom, color=colour, width=0.7, label=name)
        bottom += breakdown[column].to_numpy()
    ax.set_xticks(x, [arrow(g) for g in groups], rotation=45, ha="right")
    ax.set_ylabel("source section mass (mean over pairs)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Source side: where each section's mass goes")
    ctx.out.figure(figure, f"{folder}/Fig5C_source_mass_breakdown")
    metrics = pd.read_csv(audit / "pair_metrics.csv")
    m4e = metrics[(metrics["method"] == "M4-E") & (metrics["kind"] == "adjacent")]
    figure, ax = plt.subplots(figsize=(3.0, 2.4))
    strip(ax, [m4e.loc[m4e["group"] == g, "target_final_rejection_rate"].to_numpy() for g in groups], x, mv.REJECTED)
    ax.set_xticks(x, [arrow(g) for g in groups], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("target rejected fraction")
    ax.set_title("Target side: M4-E rejected fraction per pair")
    ctx.out.figure(figure, f"{folder}/Fig5D_target_rejection")


# ---------------------------------------------------------------- supplementary

def strip(ax, data, positions, colour):
    ax.boxplot(data, positions=positions, widths=0.55, showfliers=False, medianprops=dict(color=mv.TEXT, lw=0.8),
               boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6))
    rng = np.random.default_rng(1)
    for xi, values in zip(positions, data):
        ax.scatter(xi + rng.uniform(-0.18, 0.18, len(values)), values, s=2.5, color=colour, alpha=0.55, linewidths=0)


def stage_supp(ctx: Context) -> None:
    audit = ctx.run.root / "audit"
    metrics = pd.read_csv(audit / "pair_metrics.csv")
    m4e = metrics[metrics["method"] == "M4-E"].copy()
    fractions = pd.read_csv(audit / "annotation_fractions.csv.gz")
    blockers = []

    # S2 rejection summary: one panel per kind and side
    folder = "figures/supplementary/FigS2_rejection_summary"
    for kind in ("adjacent", "same_stage"):
        part = m4e[m4e["kind"] == kind]
        groups = sorted(part["group"].unique(), key=mv.group_order)
        for side in ("source", "target"):
            figure, ax = plt.subplots(figsize=(3.4, 2.4))
            strip(ax, [part.loc[part["group"] == g, f"{side}_final_rejection_rate"].to_numpy() for g in groups],
                  np.arange(len(groups)), mv.REJECTED if kind == "adjacent" else "#8C8C8C")
            ax.set_xticks(range(len(groups)), [arrow(g) for g in groups], rotation=45, ha="right")
            ax.set_ylim(0, 1)
            ax.set_ylabel("M4-E rejected fraction")
            ax.set_title(f"{'adjacent-stage' if kind == 'adjacent' else 'same-stage'} pairs, {side} side "
                         f"({len(part)} pairs)")
            ctx.out.figure(figure, f"{folder}/FigS2_{kind}_{side}")
    ctx.out.table(m4e[["pair_id", "kind", "group", "source_bins", "target_bins", "source_final_rejection_rate",
                       "target_final_rejection_rate"]], "tables/FigS2_rejection_per_pair.tsv")

    # S3 asymmetry: three panels
    folder = "figures/supplementary/FigS3_source_target_asymmetry"
    present = fractions[fractions["n_bins"] > 0].groupby(["pair_id", "side"])["annotation"].nunique().unstack()
    m4e["difference"] = m4e["source_final_rejection_rate"] - m4e["target_final_rejection_rate"]
    m4e["log2_size_ratio"] = np.log2(m4e["source_bins"] / m4e["target_bins"])
    m4e["annotation_difference"] = m4e["pair_id"].map(present["source"] - present["target"])
    kinds = (("adjacent", mv.REJECTED), ("same_stage", "#5B7FA6"))
    figure, ax = plt.subplots(figsize=(2.6, 2.6))
    for kind, colour in kinds:
        part = m4e[m4e["kind"] == kind]
        ax.scatter(part["target_final_rejection_rate"], part["source_final_rejection_rate"], s=3, color=colour,
                   alpha=0.6, linewidths=0, label=kind.replace("_", "-"))
    ax.plot([0, 1], [0, 1], color=mv.MUTED, lw=0.6, ls="--")
    ax.set_xlabel("target rejected fraction")
    ax.set_ylabel("source rejected fraction")
    ax.set_title("M4-E, every pair (same-stage source = earlier section name)", fontsize=6)
    ax.legend(frameon=False)
    ctx.out.figure(figure, f"{folder}/FigS3A_source_vs_target")
    rows = []
    for column, name, letter in (("log2_size_ratio", "log2(source bins / target bins)", "B"),
                                 ("annotation_difference", "source − target annotations present", "C")):
        figure, ax = plt.subplots(figsize=(2.6, 2.6))
        lines = []
        for kind, colour in kinds:
            part = m4e[m4e["kind"] == kind]
            ax.scatter(part[column], part["difference"], s=3, color=colour, alpha=0.6, linewidths=0,
                       label=kind.replace("_", "-"))
            rho, p = stats.spearmanr(part[column], part["difference"], nan_policy="omit")
            rows.append({"kind": kind, "variable": column, "spearman_rho": rho, "p_value": p, "pairs": len(part)})
            lines.append(f"{kind.replace('_', '-')}: Spearman ρ = {rho:.2f}")
        ax.axhline(0, color=mv.MUTED, lw=0.6, ls="--")
        ax.set_xlabel(name)
        ax.set_ylabel("source − target rejected")
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=5.5)
        ax.set_title("association only, not a cause", fontsize=6)
        ctx.out.figure(figure, f"{folder}/FigS3{letter}_{column}")
    ctx.out.table(pd.DataFrame(rows), "tables/FigS3_asymmetry_correlations.tsv")

    # S4 annotation rejection by partner presence: four panels
    folder = "figures/supplementary/FigS4_annotation_rejection"
    adjacent = fractions[(fractions["kind"] == "adjacent") & (fractions["n_bins"] > 0)]
    groups = sorted(adjacent["group"].unique(), key=mv.group_order)
    order = [a for a in ctx.annotation_order() if a in set(adjacent["annotation"])]
    table_rows = []
    for side in ("source", "target"):
        for presence in (True, False):
            part = adjacent[(adjacent["side"] == side) & (adjacent["partner_present"].astype(bool) == presence)]
            mean = part.groupby(["annotation", "group"])["m4e_rejected_fraction"].mean().unstack().reindex(index=order, columns=groups)
            count = part.groupby(["annotation", "group"])["pair_id"].nunique().unstack().reindex(index=order, columns=groups)
            figure, ax = plt.subplots(figsize=(3.0, 0.11 * len(order) + 1.3))
            image = ax.imshow(mean.to_numpy(dtype=float), cmap="Oranges", vmin=0, vmax=1, aspect="auto",
                              interpolation="nearest")
            ax.set_facecolor("#F4F4F4")
            ax.set_xticks(range(len(groups)), [arrow(g) for g in groups], rotation=90, fontsize=5.5)
            ax.set_yticks(range(len(order)), order, fontsize=5)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.set_title(f"{side} side, partner section {'has' if presence else 'lacks'} the annotation\n"
                         "mean M4-E rejected fraction (grey = no pair)", fontsize=6)
            bar = figure.colorbar(image, ax=ax, fraction=0.05, pad=0.02)
            bar.ax.tick_params(labelsize=5)
            ctx.out.figure(figure, f"{folder}/FigS4_{side}_partner_{'present' if presence else 'absent'}")
            for (a, g), value in mean.stack().items():
                table_rows.append({"side": side, "partner_present": presence, "annotation": a, "group": g,
                                   "mean_rejected": value, "pairs": int(count.loc[a, g])})
    ctx.out.table(pd.DataFrame(table_rows), "tables/FigS4_annotation_rejection.tsv")

    # S5 balanced OT on the same bins: three panels
    folder = "figures/supplementary/FigS5_balanced_comparison"
    source = fractions[(fractions["kind"] == "adjacent") & (fractions["side"] == "source")].copy()
    n, rej = source["n_bins"], source["m4e_rejected"]
    agg = pd.DataFrame({
        "pair_id": source["pair_id"], "group": source["group"],
        "co_same": source["co_same_share"] * n, "co_supported": (source["co_same_share"] + source["co_other_share"]) * n,
        "bot_same_retained": source["bot_same_share"].fillna(0) * n - source["bot_co_rejected_same_share"].fillna(0) * rej,
        "kept": n - rej, "bot_same_rejected": source["bot_co_rejected_same_share"].fillna(0) * rej, "rejected": rej})
    agg = agg.groupby(["group", "pair_id"]).sum().reset_index()
    columns = ["ConfidenceOT, retained bins", "balanced OT, same retained bins", "balanced OT, rejected bins"]
    with np.errstate(divide="ignore", invalid="ignore"):
        agg[columns[0]] = agg["co_same"] / agg["co_supported"]
        agg[columns[1]] = agg["bot_same_retained"] / agg["kept"]
        agg[columns[2]] = agg["bot_same_rejected"] / agg["rejected"]
    ctx.out.table(agg, "tables/FigS5_same_annotation_shares.tsv")
    groups = sorted(agg["group"].unique(), key=mv.group_order)
    colours = [mv.RETAINED, "#7B4FA0", "#C9A3E0"]
    figure, ax = plt.subplots(figsize=(4.4, 2.6))
    for k, (column, colour) in enumerate(zip(columns, colours)):
        data = [agg.loc[agg["group"] == g, column].dropna().to_numpy() for g in groups]
        ax.boxplot(data, positions=np.arange(len(groups)) * 4 + k, widths=0.8, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=colour, lw=0.5), medianprops=dict(color=mv.TEXT, lw=0.7),
                   whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5))
    ax.set_xticks(np.arange(len(groups)) * 4 + 1, [arrow(g) for g in groups], rotation=45, ha="right")
    ax.set_ylabel("source mass on the same annotation")
    ax.set_ylim(0, 1)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colours], columns, frameon=False, fontsize=5.5,
              loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Same-annotation share, adjacent pairs (one box = pairs of a transition)", fontsize=6.5)
    ctx.out.figure(figure, f"{folder}/FigS5A_same_annotation_share")
    spread = []
    for pair_id in agg["pair_id"]:
        try:
            bins = ctx.run.pair_bins(pair_id, "source", ["m4e_retained", "co_partner_mass", "co_partner_spread",
                                                         "bot_partner_spread"])
        except Exception as error:
            blockers.append({"stage": "supp/S5", "item": pair_id, "reason": repr(error)})
            continue
        keep = bins["m4e_retained"].astype(bool) & (bins["co_partner_mass"] > 0)
        spread.append({"pair_id": pair_id, "median_spread_co": float(bins.loc[keep, "co_partner_spread"].median()),
                       "median_spread_bot_same_bins": float(bins.loc[keep, "bot_partner_spread"].median()),
                       "retained_fraction": float(bins["m4e_retained"].mean()),
                       "co_transported_mass": float(bins["co_partner_mass"].sum())})
    if spread:
        spread = pd.DataFrame(spread).merge(agg[["pair_id", "group"]], on="pair_id")
        ctx.out.table(spread, "tables/FigS5_spread_and_coverage.tsv")
        figure, ax = plt.subplots(figsize=(2.8, 2.6))
        for group in groups:
            part = spread[spread["group"] == group]
            ax.scatter(part["median_spread_bot_same_bins"], part["median_spread_co"], s=5, linewidths=0, label=arrow(group))
        top = float(np.nanmax(spread[["median_spread_co", "median_spread_bot_same_bins"]].to_numpy()))
        ax.plot([0, top], [0, top], ls="--", lw=0.6, color=mv.MUTED)
        ax.set_xlabel("balanced OT, same bins")
        ax.set_ylabel("ConfidenceOT")
        ax.set_title("median partner spread of retained bins, per pair", fontsize=6.5)
        ax.legend(frameon=False, fontsize=5, markerscale=2)
        ctx.out.figure(figure, f"{folder}/FigS5B_partner_spread")
        figure, ax = plt.subplots(figsize=(2.6, 2.6))
        ax.scatter(spread["retained_fraction"], spread["co_transported_mass"], s=5, linewidths=0, color=mv.RETAINED)
        ax.plot([0, 1], [0, 1], ls="--", lw=0.6, color=mv.MUTED)
        ax.set_xlabel("retained fraction (1 − rejected)")
        ax.set_ylabel("ConfidenceOT transported mass")
        ax.set_title("supported mass coverage, per pair", fontsize=6.5)
        ctx.out.figure(figure, f"{folder}/FigS5C_supported_mass")

    # S6 calibration, convergence, validity: six panels
    folder = "figures/supplementary/FigS6_calibration_diagnostics"
    calibration = pd.read_csv(audit / "calibration.csv")
    ctx.out.table(calibration, "tables/FigS6_calibration_per_pair.tsv")
    groups_all = sorted(calibration["group"].unique(), key=lambda g: ("->" not in g, mv.group_order(g)))
    figure, ax = plt.subplots(figsize=(3.6, 2.4))
    strip(ax, [calibration.loc[calibration["group"] == g, "rejection_cost"].to_numpy() for g in groups_all],
          np.arange(len(groups_all)), "#5B7FA6")
    ax.set_xticks(range(len(groups_all)), [arrow(g) for g in groups_all], rotation=90)
    ax.set_ylabel("rejection cost")
    ax.set_title("Calibrated rejection cost per pair")
    ctx.out.figure(figure, f"{folder}/FigS6A_rejection_cost")
    figure, ax = plt.subplots(figsize=(2.8, 2.2))
    for side, colour in (("source", "#5B7FA6"), ("target", "#E4A72E")):
        ax.hist(calibration[f"validation_{side}_raw_acceptance"], bins=30, histtype="step", color=colour, label=side)
    ax.axvline(0.99, color=mv.MUTED, ls="--", lw=0.6)
    ax.set_xlabel("held-out within-side acceptance")
    ax.set_ylabel("pairs")
    ax.set_title("Within-side calibration (target 0.99)")
    ax.legend(frameon=False)
    ctx.out.figure(figure, f"{folder}/FigS6B_within_side_acceptance")
    flags = [f for f in ("feasible_cost_found", "m4e_calibration_clean", "m4e_inference_valid",
                         "validation_aggregate_valid", "calibration_valid", "m4r_validation_clean") if f in calibration]
    true = [int(calibration[f].astype(bool).sum()) for f in flags]
    figure, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.barh(range(len(flags)), true, color="#5B7FA6", label="true")
    ax.barh(range(len(flags)), [len(calibration) - t for t in true], left=true, color=mv.EXCLUDED_COLOUR, label="false")
    for i, t in enumerate(true):
        ax.text(t, i, f" {t}", va="center", fontsize=5.5)
    ax.set_yticks(range(len(flags)), flags)
    ax.set_xlabel("pairs")
    ax.set_title(f"Calibration flags (reported, nothing removed; {len(calibration)} pairs)", fontsize=6.5)
    ctx.out.figure(figure, f"{folder}/FigS6C_calibration_flags")
    for letter, method in (("D", "M4-E"), ("E", "M4-R")):
        part = metrics[metrics["method"] == method]
        figure, ax = plt.subplots(figsize=(2.8, 2.2))
        ax.hist(part["outer_iterations"], bins=np.arange(0.5, part["outer_iterations"].max() + 1.5), color="#5B7FA6")
        ax.set_xlabel("outer iterations")
        ax.set_ylabel("pairs")
        ax.set_title(f"{method}: outer converged {int(part['outer_converged'].astype(bool).sum())}/{len(part)}, "
                     f"cycles {int(part['cycle_detected'].astype(bool).sum())}", fontsize=6.5)
        ctx.out.figure(figure, f"{folder}/FigS6{letter}_{method}_convergence")
    invalid = calibration[~calibration["m4e_inference_valid"].astype(bool)] if "m4e_inference_valid" in calibration \
        else calibration.iloc[0:0]
    reasons = invalid["warnings"].fillna("(no warning recorded)").astype(str).str.split(r" \| ").explode().str.strip()
    counts = reasons.value_counts()
    ctx.out.table(counts.rename_axis("warning").reset_index(name="pairs"),
                  "tables/FigS6_m4e_inference_invalid_warnings.tsv")
    top = counts.head(8)
    figure, ax = plt.subplots(figsize=(5.0, 0.25 * len(top) + 0.9))
    ax.barh(range(len(top)), top.to_numpy(), color=mv.REJECTED)
    ax.set_yticks(range(len(top)), [w[:70] + ("…" if len(w) > 70 else "") for w in top.index], fontsize=5)
    ax.invert_yaxis()
    ax.set_xlabel("pairs")
    ax.set_title(f"Warnings recorded for the {len(invalid)} pairs with m4e_inference_valid false "
                 "(validity is not convergence; full text in the table)", fontsize=6)
    ctx.out.figure(figure, f"{folder}/FigS6F_inference_validity_warnings")
    if blockers:
        record_blockers(ctx, blockers)


# ---------------------------------------------------------------- reports

FIGURE_DESIGN = """# MOSTA figure design (as built)

## References (checked 2026-10-29)

| Reference | Citation | Figures consulted | What we borrowed | What we did not borrow |
|---|---|---|---|---|
| moscot | Klein, D., Palla, G., Lange, M. et al. Mapping cells through time and space with moscot. *Nature* 638, 1065–1075 (2025). doi:10.1038/s41586-024-08453-2 | Fig. 4a (schematic of spatiotemporal trajectory inference across E9.5–E16.5), 4b (accuracy of curated transitions, Supplementary Table 5), 4c (heart cells mapped across time points on the embryo), 4d–g; Extended Data Fig. 5a–e | A left-to-right stage series of embryo sections (Fig. 1A, drawn from our own data); one quantity per row of small multiples on real sections (Fig. 3); correspondence shown as mass landing on the next section (push-forward density, Fig. 4) | CellRank fate probabilities, driver genes, annotation transfer, UMAPs; the curated-transition score (Supplementary Table 5 not downloaded) |
| 3d-OT | Dai, B., Yi, L., Wang, P. et al. 3d-OT: a deep geometry-aware framework for heterogeneous slices alignment of spatial multi-omics. *Nat. Methods* 23, 760–771 (2026). doi:10.1038/s41592-026-03034-9 | Fig. 5a–b (alignment visualisations), Fig. 6a (3D reconstruction of heart and liver, E11.5–E16.5), 6b–c (E11.5→E12.5 heart; urogenital ridge → kidney/ovary), 6d (Sankey E11.5–E16.5, links under 1% omitted); Supplementary Figs. 20, 23, 25 (annotation maps) | Faint annotation context with one quantity on top (Fig. 4); annotation-to-annotation alluvial with a stated display threshold (Fig. 5B) | Stacking sections as if registered, 3D reconstruction, chamfer distance: our sections come from different embryos and our cost has no spatial term |

Nothing in our figures is copied from either paper. Fig. 1A shows real MOSTA sections, not an illustration.

## Figures

| Figure | Biological question | Reference inspiration | Data | Visualisation |
|---|---|---|---|---|
| Fig. 1 | Which stages and how much data? | moscot Fig. 4a | manifest, equalisation per-slice table, real sections | stage axis with one real section per stage at common scale; composition bars; workflow |
| Fig. 2 | What do the tissues look like? | 3d-OT Supp. Fig. 25, MOSTA | slices/<sample>_bins.csv.gz | annotation maps, one palette for all stages |
| Fig. 3 | Where does M4-E reject, against same-stage controls? | moscot Fig. 4c layout | per-pair bins tables | 4 maps per pair: source/target annotation and retain/reject |
| Fig. 4 | Where does retained mass land? | moscot Fig. 4c, 3d-OT Fig. 6b | per-bin partner centroids and spread, push-forward vectors | centroids, densities, spread; balanced OT on the same bins |
| Fig. 5 | Which annotations correspond? | 3d-OT Fig. 6d | audit annotation transitions and fractions | heatmaps, alluvials with a rejected band, source/target side summaries |
| S1–S8 | Robustness and controls | ours | audit tables, per-pair bins | atlas, summaries, asymmetry, partner presence, balanced OT, calibration, M4-R, all heatmaps |

## Fixed rules

- Every panel is its own file (user 2026-10-29): each figure is a folder of panels (PDF + PNG) with its legends as separate files; nothing is assembled into a composite page.
- Representative section per stage: most annotations present (Cavity not counted), then most analysed bins, then name.
- Representative pair per displayed group: median M4-E source rejected fraction (lower middle for even counts), ties by pair id. Every candidate is in PAIR_SELECTION.tsv.
- Alluvial: annotations under 1% of the section's mass are pooled into "Other", links under 0.2% into one grey node; widths are means over all pairs of the transition (absent annotation = 0).
- Colours: one annotation palette everywhere (tables/annotation_palette.tsv); retained teal, rejected orange-red, Cavity light grey.

## Not drawn, and why

- Bin-to-bin connection lines: the ds production run saved no dual potentials, so no bin-level coupling exists; only per-bin summaries are drawn.
- Curated developmental transitions: moscot Supplementary Table 5 is not downloaded; cross-annotation flow is shown but not called a developmental transition.
"""


def record_blockers(ctx: Context, rows: list[dict]) -> None:
    path = ctx.out.root / "logs" / "BLOCKERS.tsv"
    existing = pd.read_csv(path, sep="\t") if path.is_file() else pd.DataFrame(columns=["stage", "item", "reason"])
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    ctx.out.table(frame, "logs/BLOCKERS.tsv")


def stage_reports(ctx: Context) -> None:
    ctx.out.text(FIGURE_DESIGN, "reports/FIGURE_DESIGN.md")
    selection = ctx.selection()
    pairs = selection["pairs"]
    chosen = pairs[pairs["selected"]]
    lines = ["# Figure captions (generated from the tables; numbers are read, not typed)", ""]
    pair_text = "; ".join(f"{r.role} {arrow(r.group)}: {r.pair_id} (source rejected {r.m4e_source_rejected:.3f}, "
                          f"target {r.m4e_target_rejected:.3f}, rank {r.rank}/{r.pairs_in_group})" for r in chosen.itertuples())
    slices = selection["slices"]
    slice_text = ", ".join(f"{r.stage} {r.sample} ({r.annotations_present} annotations)"
                           for r in slices[slices["selected"]].itertuples())
    captions = {
        "Fig1_embryogenesis_overview": f"A, one real MOSTA section per stage at a common scale ({slice_text}). B, sections, bins "
            "(analysed and Cavity) and OT comparisons per stage. C, the analysis workflow. Data: manifest/, equalisation/, slices/.",
        "Fig2_spatial_atlas": f"Annotation maps of the representative sections of {', '.join(ATLAS_STAGES)}; each panel at its "
            "own scale, x/y equal aspect, Cavity grey. Separate sections, not a 3D reconstruction.",
        "Fig3_spatial_retain_reject": f"M4-E retain/reject on the selected pairs ({pair_text}). Same-stage controls compare two "
            "sections of the same stage. Rejected means no supported correspondence in that comparison, under this "
            "representation and calibration; it does not mean absence of developmental descendants.",
        "Fig4_spatial_correspondence": "For the three adjacent pairs of Fig. 3: retained source bins; their transport-weighted "
            "partner centroids and the push-forward density of their mass on the target section, for ConfidenceOT and for "
            "balanced OT on the same bins; distribution of partner spread. Summaries, not tracked positions.",
        "Fig5_annotation_transitions": f"A, fraction of each source annotation's mass by target annotation, {arrow(HEATMAP_GROUP)}, "
            "ConfidenceOT (with the rejected share) and balanced OT. B, alluvials, mean over all pairs of a transition. C, "
            "source-side mass breakdown per transition and target-side rejected fraction per pair. Source and target "
            "rejection are shown separately, never as one conserved flow.",
    }
    for name, text in captions.items():
        lines += [f"## {name}", "", text, ""]
    lines += ["Supplementary: S1 all sections; S2 rejection per pair; S3 source/target asymmetry (association only); "
              "S4 rejection by annotation and partner presence; S5 balanced OT on the same bins; S6 calibration and "
              "validity; S7 M4-E vs M4-R gates; S8 annotation heatmaps for every transition.", ""]
    ctx.out.text("\n".join(lines), "reports/FIGURE_CAPTIONS.md")

    provenance = [
        ("Fig1", "manifest/slices.csv; manifest/pairs.csv; equalisation/per_slice.csv; equalisation/report.json; slices/<representative>_bins.csv.gz"),
        ("Fig2, S1", "slices/<sample>_bins.csv.gz (x, y, annotation, analysed)"),
        ("Fig3, S7", "pairs/<pair_id>/bins_{source,target}.csv.gz (m4e_retained, m4r_retained) joined on equalised_row; slices/"),
        ("Fig4", "pairs/<pair_id>/bins_*.csv.gz (co_/bot_partner_x, _y, _spread, _mass); pairs/<pair_id>/transport_target.npz (co_pushforward, bot_pushforward, bot_pushforward_co_rejected)"),
        ("Fig5, S8", "audit/annotation_transitions.csv.gz (forward, co and bot nominal_mass); audit/annotation_fractions.csv.gz; audit/pair_metrics.csv"),
        ("S2, S3", "audit/pair_metrics.csv; audit/annotation_fractions.csv.gz"),
        ("S4", "audit/annotation_fractions.csv.gz (m4e_rejected_fraction, partner_present)"),
        ("S5", "audit/annotation_fractions.csv.gz; pairs/<adjacent pair>/bins_source.csv.gz"),
        ("S6", "audit/calibration.csv; audit/pair_metrics.csv"),
    ]
    frame = pd.DataFrame(provenance, columns=["figure", "inputs"])
    frame["run_root"] = str(ctx.run.root)
    frame["commit"] = ctx.commit
    ctx.out.table(frame, "reports/DATA_PROVENANCE.tsv")

    blockers_path = ctx.out.root / "logs" / "BLOCKERS.tsv"
    blockers = pd.read_csv(blockers_path, sep="\t") if blockers_path.is_file() else pd.DataFrame()
    figures = sorted(p for p in ctx.out.rows if p.startswith("figures/") and p.endswith(".png"))
    report = ["# Visualisation report", "", f"Run: `{ctx.run.root}`  ", f"Commit: `{ctx.commit}`  ",
              f"Job: {os.environ.get('SLURM_JOB_ID', 'n/a')}  ", f"Written: {time.strftime('%Y-%m-%d %H:%M')}", "",
              f"## Figures ({len(figures)} PNG, each with a PDF)", ""] + [f"- `{p}`" for p in figures] + [
              "", "## Blockers", ""]
    report += ["None recorded."] if blockers.empty else [f"- {r.stage} / {r.item}: {str(r.reason)[:300]}" for r in blockers.itertuples()]
    report += ["", "## Selected pairs", "", pair_text, "",
               "Observations are written after the rendered figures are inspected; nothing here is a conclusion."]
    ctx.out.text("\n".join(report), "reports/VISUALIZATION_REPORT.md")
    ctx.out.text("\n".join([
        "# MOSTA visualisation", "",
        "Drawn by mouse_embryo/30_mosta_figures.py (copied to scripts/) from the finished ds run; nothing in the run "
        "was written. figures/main, figures/supplementary, figures/individual_panels: PDF + PNG. tables/: the source "
        "table of every figure. reports/: design, captions, report, pair selection, provenance, environment. "
        "manifests/: OUTPUT_MANIFEST.tsv (every file written, with SHA-256), INPUT_INVENTORY.tsv and INPUT_SUMMARY.tsv "
        "(inputs, sizes, access dates; all on scratch). logs/: job logs and BLOCKERS.tsv.", ""]), "reports/README.md")
    for source in (Path(__file__), Path(mv.__file__), Path(__file__).with_name("tacc_mosta_figures.slurm")):
        if source.is_file():
            target = ctx.out.path(f"scripts/{source.name}")
            shutil.copyfile(source, target)
            ctx.out.record(target)


def stage_package(ctx: Context) -> None:
    target = ctx.out.path("package/mosta_figures_20261029.tar.gz")
    with tarfile.open(target, "w:gz") as archive:
        for part in ("figures", "tables", "reports", "scripts", "manifests", "logs"):
            if (ctx.out.root / part).is_dir():
                archive.add(ctx.out.root / part, arcname=f"mosta_visualization_20261029/{part}")
    ctx.out.record(target)
    print(f"package {target}: {target.stat().st_size / 1e6:.1f} MB")


STAGES = {"audit": stage_audit, "select": stage_select, "fig1": stage_fig1, "fig2": stage_fig2, "s1": stage_s1,
          "fig3": stage_fig3, "fig4": stage_fig4, "fig5": stage_fig5, "supp": stage_supp,
          "reports": stage_reports, "package": stage_package}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=DS_ROOT)
    parser.add_argument("--other-runs", nargs="*", default=list(OTHER_ROOTS))
    parser.add_argument("--raw", default=RAW_DATA)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stages", default="all")
    args = parser.parse_args()
    ctx = Context(args)
    names = STAGE_ORDER if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in names if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages {unknown}; known: {STAGE_ORDER}")
    failed = []
    for name in names:
        started = time.perf_counter()
        ctx.out.stage = name
        print(f"[{time.strftime('%H:%M:%S')}] stage {name}", flush=True)
        try:
            STAGES[name](ctx)
            print(f"  done in {time.perf_counter() - started:.0f}s", flush=True)
        except Exception as error:  # recorded and reported, the next stage still runs
            failed.append(name)
            print(traceback.format_exc(), flush=True)
            record_blockers(ctx, [{"stage": name, "item": "stage", "reason": f"{error!r}"}])
    print("failed stages:", failed or "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
