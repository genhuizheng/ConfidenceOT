"""Figures of the lineage validation (VALIDATION_PROTOCOL.md §13), from ROOT/summary, ROOT/scores and ROOT/umap.

    python lineage_validation/06_figures.py --root ROOT [--method M4-E] [--skip-umap]

Writes ROOT/figures/ (a new figures_<time>/ if figures/ exists):
  retention/<block>__<mouse>.png    one panel per strategy; rows = runs, columns = the sites a run ranks
                                    (D_A: primaries; D_B: metastases or Quinn tissues; D_C3: both sides);
                                    one quantity, the site's rejected fraction (1 - retained); KP-Tracer
                                    related sites outlined in black, Quinn tissues labelled with f (§7.3)
  group_coupling/<block>/<pair>.png one panel per strategy; rows = source lineages (family / clone),
                                    columns = target sites; one quantity, the share of the target site's
                                    asserted mass that comes from each lineage (column sums 1)
  curves/<block>__<mouse>.png       §8.4 coverage-accuracy, one panel per run, one line per strategy, the
                                    gate's operating point as a dot
  strategies/<block>.png            the five strategies side by side for the T1/T2 metrics and their
                                    companions: one dot per mouse (mean over its runs), bar = mean of mice
  umap/<block>/<pair>.png           the fixed joint UMAP (48_reference_umap.py, ROOT/umap/<block>): site,
                                    lineage, then retain / reject / exclude per strategy
  figures_index.tsv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lv_common as lv  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
NAMES = {"rna": "RNA only", "cnv": "CNV only", "rna_cnv": "CNV + RNA", "cnv_then_rna": "CNV -> RNA", "rna_then_cnv": "RNA -> CNV"}
STATE_COLOURS = {"retained": "#2b8cbe", "rejected": "#d7301f", "excluded": "#8c8c8c"}
OTHER = "#e3e3e3"
STRATEGY_METRICS = ["e_M", "site_top1", "site_auroc", "E2E_minus_CH", "COND_minus_ind", "top1_COND",
                    "asserted_precision", "asserted_F", "cov_T", "cov_S", "target_rejection_auroc", "E2E_bal", "tree_dS_norm"]


def short(pair: str) -> str:
    parts = pair.split("__")
    return f"{parts[2]} -> {parts[3]}" if len(parts) == 4 else pair


def save(fig, path: Path, index: list, **info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    index.append({"figure": str(path), **info})


def score_dir(root: Path, method: str, strategy: str, block: str, pair: str) -> Path:
    return root / "scores" / method / strategy / block / pair


def retention_figures(root: Path, runs: pd.DataFrame, method: str, out: Path, index: list) -> None:
    for (block, mouse), part in runs.groupby(["block", "mouse"]):
        design = lv.DESIGN[block]
        pairs = sorted(part["pair"].unique())
        tables = {}
        for strategy in lv.STRATEGIES:
            frames = []
            for pair in pairs:
                path = score_dir(root, method, strategy, block, pair) / "sites.tsv"
                if path.exists():
                    sites = pd.read_csv(path, sep="\t", keep_default_na=False)
                    if design == "D_A":
                        sites = sites[sites["side"].eq("source")]
                    elif design == "D_B":
                        sites = sites[sites["side"].eq("target")]
                    frames.append(sites.assign(pair=pair))
            tables[strategy] = pd.concat(frames, ignore_index=True) if frames else None
        columns = sorted({(r.side, r.site) for t in tables.values() if t is not None for r in t.itertuples()})
        if not columns:
            continue
        fig, axes = plt.subplots(1, len(lv.STRATEGIES), figsize=(1.6 + 0.45 * len(columns) * len(lv.STRATEGIES),
                                                                  1.8 + 0.32 * len(pairs)), squeeze=False)
        for ax, strategy in zip(axes[0], lv.STRATEGIES):
            table = tables[strategy]
            grid = np.full((len(pairs), len(columns)), np.nan)
            related = np.zeros_like(grid, dtype=bool)
            labels = {}
            if table is not None:
                for r in table.itertuples():
                    i, j = pairs.index(r.pair), columns.index((r.side, r.site))
                    grid[i, j] = 1 - float(r.retained)
                    related[i, j] = str(getattr(r, "related", "")).lower() == "true"
                    if hasattr(r, "f_shared") and str(r.f_shared) not in ("", "nan"):
                        labels[(r.side, r.site)] = f"f={float(r.f_shared):.2f}"
            ax.set_facecolor(OTHER)
            mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap="Reds", vmin=0, vmax=1, edgecolors="white", linewidth=0.5)
            for i, j in zip(*np.nonzero(related)):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="black", linewidth=1.6))
            ax.set_ylim(len(pairs), 0)
            ax.set_xticks(np.arange(len(columns)) + 0.5)
            ax.set_xticklabels([f"{site}\n({'P' if side == 'source' else 'M'})" + (f"\n{labels[(side, site)]}" if (side, site) in labels else "")
                                for side, site in columns], rotation=90, fontsize=7)
            ax.set_yticks(np.arange(len(pairs)) + 0.5)
            ax.set_yticklabels([short(p) for p in pairs] if ax is axes[0][0] else [], fontsize=7)
            ax.set_title(NAMES[strategy] + ("" if table is not None else "\n(no scored run)"), fontsize=9)
        fig.colorbar(mesh, ax=axes[0].tolist(), fraction=0.02, label="site's rejected fraction\n(1 - retained)")
        fig.suptitle(f"{block}  mouse {mouse}  {method}; black outline = related site (same family); grey = no scored run",
                     fontsize=10, y=1.06)
        save(fig, out / "retention" / f"{block}__{mouse}.png", index, kind="retention", block=block, mouse=mouse)


def group_coupling_figures(root: Path, runs: pd.DataFrame, method: str, out: Path, index: list) -> None:
    for (block, pair), _ in runs.groupby(["block", "pair"]):
        tables = {}
        for strategy in lv.STRATEGIES:
            path = score_dir(root, method, strategy, block, pair) / "group_site_mass.tsv"
            if path.exists():
                tables[strategy] = pd.read_csv(path, sep="\t", keep_default_na=False)
        if not tables:
            continue
        first = next(iter(tables.values()))
        rows = (first.drop_duplicates("source_lineage").sort_values("source_cells", ascending=False)["source_lineage"].astype(str).tolist())
        cols = sorted(first["target_site"].astype(str).unique())
        fig, axes = plt.subplots(1, len(lv.STRATEGIES), figsize=(1.5 + 0.5 * len(cols) * len(lv.STRATEGIES),
                                                                  1.8 + 0.18 * len(rows)), squeeze=False)
        for ax, strategy in zip(axes[0], lv.STRATEGIES):
            grid = np.full((len(rows), len(cols)), np.nan)
            if strategy in tables:
                wide = tables[strategy].pivot_table(index="source_lineage", columns="target_site", values="asserted_mass", aggfunc="sum")
                wide.index = wide.index.astype(str); wide.columns = wide.columns.astype(str)
                wide = wide.reindex(index=rows, columns=cols).fillna(0.0)
                total = wide.sum(axis=0)
                grid = (wide / total.replace(0, np.nan)).to_numpy(float)
            ax.set_facecolor(OTHER)
            mesh = ax.pcolormesh(np.ma.masked_invalid(grid), cmap="Purples", vmin=0, vmax=1, edgecolors="white", linewidth=0.3)
            ax.set_ylim(len(rows), 0)
            ax.set_xticks(np.arange(len(cols)) + 0.5); ax.set_xticklabels(cols, rotation=90, fontsize=7)
            ax.set_yticks(np.arange(len(rows)) + 0.5)
            ax.set_yticklabels(rows if ax is axes[0][0] else [], fontsize=max(3, 7 - len(rows) // 25))
            ax.set_title(NAMES[strategy] + ("" if strategy in tables else "\n(no scored run)"), fontsize=9)
        fig.colorbar(mesh, ax=axes[0].tolist(), fraction=0.02, label="share of the target\nsite's asserted mass")
        fig.suptitle(f"{pair}  {method}: source lineage x target site (grey = no scored run or no asserted mass)", fontsize=9, y=1.06)
        save(fig, out / "group_coupling" / block / f"{pair}.png", index, kind="group_coupling", block=block, pair=pair)


def curve_figures(root: Path, runs: pd.DataFrame, method: str, out: Path, index: list) -> None:
    colours = dict(zip(lv.STRATEGIES, plt.get_cmap("tab10").colors))
    for (block, mouse), part in runs.groupby(["block", "mouse"]):
        pairs = sorted(part["pair"].unique())
        ncol = min(4, len(pairs))
        nrow = int(np.ceil(len(pairs) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.8 * nrow), squeeze=False)
        drawn = False
        for ax, pair in zip(axes.ravel(), pairs):
            for strategy in lv.STRATEGIES:
                folder = score_dir(root, method, strategy, block, pair)
                path = folder / "curve.tsv"
                if not path.exists() or path.stat().st_size < 10:
                    continue
                curve = pd.read_csv(path, sep="\t")
                if curve.empty:
                    continue
                summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
                ax.plot(curve["fraction"], curve["top1"], color=colours[strategy], label=NAMES[strategy], lw=1.2)
                if summary.get("cov_T") is not None and summary.get("top1_COND") is not None:
                    ax.scatter([summary["cov_T"]], [summary["top1_COND"]], color=colours[strategy], s=18, zorder=3)
                drawn = True
            ax.set_title(short(pair), fontsize=8)
            ax.set_xlim(0, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("fraction of matchable target cells", fontsize=7); ax.set_ylabel("Top-1", fontsize=7)
            ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(pairs):]:
            ax.axis("off")
        if not drawn:
            plt.close(fig)
            continue
        handles = [plt.Line2D([], [], color=colours[s], label=NAMES[s]) for s in lv.STRATEGIES]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
        fig.suptitle(f"{block}  mouse {mouse}  {method}: Top-1 among the most confidently retained cells; dot = gate's operating point", fontsize=9)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        save(fig, out / "curves" / f"{block}__{mouse}.png", index, kind="curves", block=block, mouse=mouse)


def strategy_figures(values: pd.DataFrame, method: str, out: Path, index: list) -> None:
    part = values[values["method"].eq(method)]
    for block, sub in part.groupby("block"):
        metrics = [m for m in STRATEGY_METRICS if m in set(sub["metric"])]
        if not metrics:
            continue
        ncol = 4
        nrow = int(np.ceil(len(metrics) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow), squeeze=False)
        for ax, metric in zip(axes.ravel(), metrics):
            data = sub[sub["metric"].eq(metric)]
            for x, strategy in enumerate(lv.STRATEGIES):
                y = data.loc[data["strategy"].eq(strategy), "value"].to_numpy(float)
                if y.size:
                    jitter = (np.arange(y.size) - (y.size - 1) / 2) * 0.05
                    ax.scatter(x + jitter, y, s=12, color="#525252", zorder=3)
                    ax.hlines(y.mean(), x - 0.3, x + 0.3, color="#2b8cbe", lw=2)
            ax.axhline(0, color="#bdbdbd", lw=0.8)
            ax.set_xticks(range(len(lv.STRATEGIES)))
            ax.set_xticklabels([NAMES[s] for s in lv.STRATEGIES], rotation=45, ha="right", fontsize=7)
            ax.set_title(metric, fontsize=9); ax.tick_params(labelsize=7)
        for ax in axes.ravel()[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"{block}  {method}: dot = one mouse (mean of its runs), bar = mean of mice", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, out / "strategies" / f"{block}.png", index, kind="strategies", block=block)


def gate_states(root: Path, acceptance: str, strategy: str, block: str, pair: str, method: str) -> pd.Series | None:
    try:
        directory = lv.run_directory(root / "runs" / f"acc{acceptance}", strategy, block, pair)
    except lv.RunMissing:
        return None
    gate = lv.read_gate(directory, method)
    state = pd.Series(np.where(gate["retained"], "retained", "rejected"),
                      index=gate["side"].map({"source": "source", "target": "target"}) + ":" + gate["observation_id"])
    if strategy in lv.STEP1:
        try:
            first = lv.read_gate(lv.run_directory(root / "runs" / f"acc{acceptance}", lv.STEP1[strategy], block, pair), "M4-E")
        except lv.RunMissing:
            return None
        base = pd.Series("excluded", index=first["side"] + ":" + first["observation_id"])
        base.update(state)
        state = base
    return state


def umap_figures(root: Path, runs: pd.DataFrame, method: str, acceptance: str, out: Path, index: list) -> None:
    cells = pd.read_csv(root / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    lineage = {k: g.set_index("cell")["lineage"] for k, g in cells.groupby(["dataset", "mouse"])}
    for (block, pair, mouse, dataset), _ in runs.groupby(["block", "pair", "mouse", "dataset"]):
        path = root / "umap" / block / "pairs" / pair / "reference_umap.csv.gz"
        if not path.exists():
            continue
        table = pd.read_csv(path, dtype={"observation_id": str})
        key = table["side"] + ":" + table["observation_id"]
        x, y = table["umap_1"].to_numpy(float), table["umap_2"].to_numpy(float)
        size = max(0.5, min(6.0, 6000 / max(len(table), 1)))
        site = cells[(cells["dataset"] == dataset) & (cells["mouse"] == mouse)].set_index("cell")["site"]
        sites = site.reindex(table["observation_id"]).fillna("?").to_numpy()
        lin = lineage[(dataset, mouse)].reindex(table["observation_id"]).fillna("").to_numpy()
        panels = [("site", sites), ("lineage", lin)] + [(NAMES[s], gate_states(root, acceptance, s, block, pair, method)) for s in lv.STRATEGIES]
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.2), squeeze=False)
        for ax, (title, values) in zip(axes[0], panels):
            ax.scatter(x, y, s=size * 0.6, c="#f2f2f2", linewidths=0, rasterized=True)
            if title in ("site", "lineage"):
                vals = pd.Series(values)
                order = vals.value_counts().index.tolist()
                shown = order[:19]
                palette = dict(zip(shown, plt.get_cmap("tab20").colors))
                colour = [palette.get(v, "#bdbdbd") for v in vals]
                ax.scatter(x, y, s=size, c=colour, linewidths=0, rasterized=True)
                note = f"{len(order)} values" + (f"; the {len(order) - 19} smallest in grey" if len(order) > 19 else "")
                if title == "site":
                    handles = [plt.Line2D([], [], marker="o", ls="", color=palette[v], label=v) for v in shown]
                    ax.legend(handles=handles, fontsize=5, frameon=False, loc="best", markerscale=0.6)
                ax.set_title(f"{title} ({note})", fontsize=8)
            else:
                if values is None:
                    ax.set_title(f"{title}\n(not run)", fontsize=8)
                else:
                    state = values.reindex(key).fillna("").to_numpy()
                    for name, c in STATE_COLOURS.items():
                        hit = state == name
                        ax.scatter(x[hit], y[hit], s=size, c=c, linewidths=0, rasterized=True)
                    tally = pd.Series(state).value_counts()
                    ax.set_title(f"{title}\n" + "  ".join(f"{k} {int(tally.get(k, 0))}" for k in STATE_COLOURS), fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{pair}  {method}  (fixed joint UMAP, both sides)", fontsize=9, y=1.08)
        save(fig, out / "umap" / block / f"{pair}.png", index, kind="umap", block=block, pair=pair)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", default="M4-E")
    parser.add_argument("--acceptance", default="0.99")
    parser.add_argument("--skip-umap", action="store_true")
    args = parser.parse_args()
    lv.check_frozen()
    summary = args.root / "summary"
    if not (summary / "SUCCESS").exists():
        sys.exit("summary/SUCCESS missing; run 05_aggregate.py first")
    out = args.root / "figures"
    if out.exists():
        out = args.root / f"figures_{time.strftime('%Y%m%d_%H%M%S')}"
    runs = pd.read_csv(summary / "runs.tsv", sep="\t", dtype={"mouse": str})
    runs = runs[runs["method"].eq(args.method)]
    values = pd.read_csv(summary / "mouse_values.tsv", sep="\t", dtype={"mouse": str})
    index: list = []
    retention_figures(args.root, runs, args.method, out, index)
    group_coupling_figures(args.root, runs, args.method, out, index)
    curve_figures(args.root, runs, args.method, out, index)
    strategy_figures(values, args.method, out, index)
    if not args.skip_umap:
        umap_figures(args.root, runs, args.method, args.acceptance, out, index)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(out / "figures_index.tsv", sep="\t", index=False, lineterminator="\n")
    print(f"{len(index)} figures in {out}", flush=True)


if __name__ == "__main__":
    main()
