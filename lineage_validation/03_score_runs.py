"""Score every lineage-validation run against lineage (VALIDATION_PROTOCOL.md §7-§10).

    python lineage_validation/03_score_runs.py --root ROOT --task-index I --task-count N

Runs = 5 strategies x every pair of inputs/manifests/<block>_raw.csv, each for M4-E and M4-R;
task I scores runs I, I+N, ... For each it writes ROOT/scores/<method>/<strategy>/<block>/<pair>/:
  summary.json     every §7/§8 quantity, coverage, shuffle means and p values, checks
  sites.tsv        §7 per side and site: cells, retained fraction, lineage, related, asserted share, f (Quinn)
  groups.tsv       §8.5 per source group (family / clone): sizes, retention, E2E, COND, Top-1
  bins.tsv         §8.5 size bins; curve.tsv §8.4; rejection.tsv §8.4 (both sides)
  shuffle.tsv.gz   §10, one row per draw; depth.tsv §10; tree_blocks.tsv §9
  group_site_mass.tsv  asserted mass (fraction of the plan) from each source lineage to each target site
  cells.tsv.gz     per target cell: id, site, class, lineage, retained, retained-source mass, w, pi*, margin
  SUCCESS          only after the checks pass
A run that was not completed gets MISSING (reason), one that fails here gets FAILED (traceback);
neither is replaced by anything (§4). Done runs are skipped.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lv_common as lv  # noqa: E402

KP_ROOT = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")


def run_items(root: Path) -> list[dict]:
    items = []
    for block, dataset in lv.BLOCKS.items():
        manifest = root / "inputs" / "manifests" / f"{block}_raw.csv"
        table = pd.read_csv(manifest, dtype=str)
        for _, row in table.iterrows():
            for strategy in lv.STRATEGIES:
                items.append({"block": block, "dataset": dataset, "strategy": strategy, "pair": row["pair_id"],
                              "mouse": row["patient_id"], "source_sample": row["source_sample"],
                              "target_sample": row["target_sample"]})
    return sorted(items, key=lambda r: (r["block"], r["pair"], r["strategy"]))


class Labels:
    def __init__(self, root: Path):
        cells = pd.read_csv(root / "labels" / "cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
        self.lineage = {k: g.set_index("cell")["lineage"] for k, g in cells.groupby(["dataset", "mouse"])}
        trees = pd.read_csv(root / "labels" / "trees.tsv", sep="\t", dtype=str, keep_default_na=False)
        self.trees = {(r["dataset"], r["mouse"], r["lineage"]): r for _, r in trees.iterrows()}
        leaves = pd.read_csv(root / "labels" / "tree_leaves.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
        self.leaves = {k: g.set_index("cell")["leaf"] for k, g in leaves.groupby(["dataset", "mouse", "lineage"])}
        self.index_cache: dict[str, lv.TreeIndex] = {}

    def of(self, dataset, mouse, cells) -> np.ndarray:
        table = self.lineage.get((dataset, mouse))
        if table is None:
            raise KeyError(f"no labels for {dataset} {mouse}")
        values = table.reindex(cells)
        if values.isna().any():
            raise KeyError(f"{int(values.isna().sum())} analysed cells have no label row, e.g. {list(values[values.isna()].index[:3])}")
        return values.to_numpy(object)

    def tree_index(self, path: str) -> lv.TreeIndex:
        if path not in self.index_cache:
            if len(self.index_cache) > 8:
                self.index_cache.clear()
            self.index_cache[path] = lv.TreeIndex(lv.tree_module().Tree(Path(path)))
        return self.index_cache[path]


def load_depth(root: Path, kp_root: Path) -> dict:
    out = {}
    for dataset, path in (("KPTracer", kp_root / "downsampled" / "predownsample_depth.csv.gz"),
                          ("Quinn2021", root / "inputs" / "h5ad" / "Quinn2021" / "ds" / "predownsample_depth.csv.gz")):
        if path.exists():
            table = pd.read_csv(path, dtype={"patient_id": str, "sample_id": str, "observation_id": str})
            table = table.drop_duplicates(["patient_id", "sample_id", "observation_id"])
            out[dataset] = table.set_index(["patient_id", "sample_id", "observation_id"])["predownsample_total_counts"]
    return out


def tree_blocks(item, run, lab_s, lab_t, labels: Labels, draws: int, method: str) -> pd.DataFrame:
    rows = []
    dataset, mouse = item["dataset"], item["mouse"]
    ks, kt = lv.known(lab_s, dataset), lv.known(lab_t, dataset)
    shared = sorted(set(lab_s[ks]) & set(lab_t[kt]))
    matchable_total = int((kt & np.isin(lab_t, list(set(lab_s[ks])))).sum())
    for lineage in shared:
        info = labels.trees.get((dataset, mouse, lineage))
        src_all, tgt_all = lab_s == lineage, lab_t == lineage
        row = {"lineage": lineage, "source_cells": int(src_all.sum()), "target_cells": int(tgt_all.sum())}
        if info is None or info.get("status") != "tree":
            rows.append({**row, "status": "no tree" if info is None else info.get("status")}); continue
        leaf = labels.leaves.get((dataset, mouse, lineage), pd.Series(dtype=str))
        s_leaf = leaf.reindex(run["source"]).to_numpy(object)
        t_leaf = leaf.reindex(run["target"]).to_numpy(object)
        s_map = src_all & pd.notna(s_leaf)
        t_map = tgt_all & pd.notna(t_leaf)
        rows_i = np.flatnonzero(s_map & run["rho"])
        cols_j = np.flatnonzero(t_map & run["tau"])
        row.update(tree=info["tree"], source_mapped=float(s_map.sum() / src_all.sum()), target_mapped=float(t_map.sum() / tgt_all.sum()),
                   source_block=len(rows_i), target_block=len(cols_j))
        if len(rows_i) < 2 or len(cols_j) < 2:
            rows.append({**row, "status": "fewer than 2 retained mapped cells on a side"}); continue
        block = run["P"][np.ix_(rows_i, cols_j)]
        if block.sum() <= 0:
            rows.append({**row, "status": "no coupling mass in the block"}); continue
        index = labels.tree_index(info["tree"])
        leaf_of = index.tree.leaf_of
        values = lv.tree_block_scores(index, np.array([leaf_of[x] for x in s_leaf[rows_i]]),
                                      np.array([leaf_of[x] for x in t_leaf[cols_j]]), block, draws=draws,
                                      seed_key=f"{item['pair']}/{item['strategy']}/{method}/tree/{lineage}")
        rows.append({**row, "status": "scored" if values["scored"] else "not scored (Var L = 0 or dS_max = 0)", **values})
    table = pd.DataFrame(rows)
    if len(table):
        table["run_matchable_targets"] = matchable_total
    return table


def score_one(item: dict, method: str, args, labels: Labels, depth: dict) -> None:
    out = args.root / "scores" / method / item["strategy"] / item["block"] / item["pair"]
    if (out / "SUCCESS").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("FAILED", "MISSING"):
        (out / stale).unlink(missing_ok=True)
    started = time.perf_counter()
    try:
        run = lv.load_run(args.root / "runs" / f"acc{args.acceptance}", item["strategy"], item["block"], item["pair"], method)
    except lv.RunMissing as error:
        (out / "MISSING").write_text(f"{error}\n", encoding="utf-8")
        return
    dataset = item["dataset"]
    lab_s = labels.of(dataset, item["mouse"], run["source"])
    lab_t = labels.of(dataset, item["mouse"], run["target"])
    ks = (1, 3) if dataset == "Quinn2021" else (1,)
    shuffles = args.shuffles
    cell = lv.score_cells(run["P"], run["rho"], run["tau"], run["source_margin"], run["target_margin"], lab_s, lab_t,
                          dataset, shuffles=shuffles, seed_key=f"{item['pair']}/{item['strategy']}/{method}", ks=ks)
    m = run["P"].sum()
    mass_rows = (run["P"][:, run["tau"]].sum(axis=1) * run["rho"]) / m if m > 0 else np.zeros(len(run["rho"]))
    sites, site_summary = lv.site_scores(run, lab_s, lab_t, dataset, lv.DESIGN[item["block"]], mass_rows,
                                         cell["target_classes"])
    if dataset == "KPTracer":
        keys_s = (np.repeat(item["mouse"], len(run["source"])), run["source_site"], run["source"])
        keys_t = (np.repeat(item["mouse"], len(run["target"])), run["target_site"], run["target"])
    else:
        keys_s = (np.repeat(item["mouse"], len(run["source"])), np.repeat(item["source_sample"], len(run["source"])), run["source"])
        keys_t = (np.repeat(item["mouse"], len(run["target"])), np.repeat(item["target_sample"], len(run["target"])), run["target"])
    depth_table = lv.depth_control(run, depth.get(dataset), keys_s, keys_t)
    # Group-level coupling (§13 figures): asserted mass from each source lineage to each target site.
    group_names, gs = np.unique(lab_s, return_inverse=True)
    site_names, ts = np.unique(run["target_site"], return_inverse=True)
    by_group = np.asarray(lv.onehot(gs, len(group_names), run["rho"].astype(float)) @ run["P"]
                          @ lv.onehot(ts, len(site_names), run["tau"].astype(float)).T.toarray())
    group_site = pd.DataFrame(by_group / m if m > 0 else by_group, index=pd.Index(group_names, name="source_lineage"),
                              columns=site_names).reset_index().melt(id_vars="source_lineage", var_name="target_site",
                                                                    value_name="asserted_mass")
    group_site["source_cells"] = group_site["source_lineage"].map(pd.Series(lab_s).value_counts())
    trees = tree_blocks(item, run, lab_s, lab_t, labels, args.tree_draws, method)
    summary = {**{k: item[k] for k in ("block", "dataset", "strategy", "pair", "mouse")}, "method": method,
               "design": lv.DESIGN[item["block"]], **cell["summary"], **site_summary,
               "directory": run["directory"], "step1_directory": run["step1_directory"],
               "step2_source": run["step2_source"], "step2_target": run["step2_target"]}
    if len(trees) and "scored" in trees:
        scored = trees[trees["status"].eq("scored")]
        if len(scored):
            weight = scored["target_block"].to_numpy(float)
            summary["tree_dS_norm"] = float(np.average(scored["dS_norm"].to_numpy(float), weights=weight))
            summary["tree_dR"] = float(np.average(scored["dR"].to_numpy(float), weights=weight))
            summary["tree_blocks_scored"] = int(len(scored))
            summary["tree_low_information_blocks"] = int(scored["low_information"].astype(bool).sum())
            summary["tree_target_share"] = float(scored["target_block"].sum() / max(1, cell["summary"]["matchable"]))
    checks = dict(cell["checks"])
    checks["shuffle_draws"] = int(len(cell["shuffle"]))
    if not checks["group_weight_sum_ok"]:
        raise ValueError("sum over groups of W_j(g) is not in {0, 1}")
    summary["checks"] = checks
    summary["seconds"] = time.perf_counter() - started
    write = dict(sep="\t", index=False, lineterminator="\n")
    sites.to_csv(out / "sites.tsv", **write)
    cell["groups"].to_csv(out / "groups.tsv", **write)
    cell["bins"].to_csv(out / "bins.tsv", **write)
    cell["curve"].to_csv(out / "curve.tsv", **write)
    cell["rejection"].to_csv(out / "rejection.tsv", **write)
    cell["shuffle"].to_csv(out / "shuffle.tsv.gz", compression="gzip", **write)
    depth_table.to_csv(out / "depth.tsv", **write)
    trees.to_csv(out / "tree_blocks.tsv", **write)
    group_site.to_csv(out / "group_site_mass.tsv", **write)
    cells = cell["cells"].assign(cell=run["target"], site=run["target_site"])
    cells.to_csv(out / "cells.tsv.gz", compression="gzip", **write)
    lv.write_json(out / "summary.json", summary)
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kp-root", type=Path, default=KP_ROOT)
    parser.add_argument("--acceptance", default="0.99")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--shuffles", type=int, default=lv.SHUFFLES)
    parser.add_argument("--tree-draws", type=int, default=lv.SHUFFLES)
    parser.add_argument("--methods", nargs="+", default=list(lv.METHODS))
    args = parser.parse_args()
    lv.check_frozen()
    if not (args.root / "labels" / "SUCCESS").exists():
        sys.exit("labels/SUCCESS missing; run 02_labels.py first")
    items = run_items(args.root)[args.task_index::args.task_count]
    labels = Labels(args.root)
    depth = load_depth(args.root, args.kp_root)
    failed = 0
    for item in items:
        for method in args.methods:
            out = args.root / "scores" / method / item["strategy"] / item["block"] / item["pair"]
            try:
                score_one(item, method, args, labels, depth)
                print(f"{method} {item['strategy']} {item['pair']}: {'done' if (out / 'SUCCESS').exists() else 'missing run'}", flush=True)
            except Exception as error:  # noqa: BLE001 - listed, the task goes on
                failed += 1
                out.mkdir(parents=True, exist_ok=True)
                (out / "FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
                print(f"FAILED {method} {item['strategy']} {item['pair']}: {error}", flush=True)
    print(f"task {args.task_index}/{args.task_count}: {len(items)} runs, {failed} failures", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
