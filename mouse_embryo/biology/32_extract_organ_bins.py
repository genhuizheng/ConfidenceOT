"""Per-bin tables of one organ case from a finished MOSTA run (read-only on the run).

    python mouse_embryo/biology/32_extract_organ_bins.py --run-name ds --run-root <root> --out <analysis root> \
        --organ-set heart [--organ-set neural ...] [--workers 32] [--expression] [--m4r-coupling --threads 144]

For every pair of the run (adjacent and same-stage) and each side, the bins
whose annotation belongs to the organ set: their gates, decision costs and
margins under M4-E and M4-R, the rejection cost, the M4-E coupling summaries
(retained x retained support) and the balanced-OT summaries, all copied from
pairs/<pair>/bins_<side>.csv.gz; their coordinates, annotation and depths
from slices/<sample>_bins.csv.gz; whether the partner slice holds the same
annotation.  Per bin, the conditional distribution over partner annotations
of the M4-E coupling and of balanced OT is copied from
pairs/<pair>/transport_<side>.npz (exact per-bin quantities saved by 23_; no
coupling is rebuilt for them).

--m4r-coupling (runs with potentials only): the same conditional distribution
for M4-R on its own retained x retained support, rebuilt block by block from
potentials.npz (mosta_coupling.PairCoupling; never a dense N x M matrix), and
the M4-E one rebuilt too and compared with the saved one as a check.

--expression: marker genes (markers.tsv) per analysed bin of every slice the
cases touch, from the run's equalised slices (equalised/<sample>.h5ad, counts
equalised to one depth): log1p(1e4 * count / bin total) and the raw count.
Written once per slice under expression/, shared by every run (same bins).

Outputs, under <out>/organ_bins/<run>/<organ_set>/: bins.csv.gz,
distributions/<pair>__<side>.npz (bin_row, partner_annotations, co, bot[, m4r,
m4e_rebuilt]), pairs.tsv (per pair and side: bins, partner presence, which
inputs existed).  Missing inputs are recorded, never filled in.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bio_common import Outputs, analysed_bins, markers, organ_sets, stage_of  # noqa: E402

BIN_COLUMNS = ["m4e_retained", "m4e_raw_retained", "m4e_decision_cost", "m4e_margin", "m4e_budget_overridden",
               "m4r_retained", "m4r_raw_retained", "m4r_decision_cost", "m4r_margin", "m4r_budget_overridden",
               "co_partner_mass", "co_total_mass", "co_partner_x", "co_partner_y", "co_partner_spread",
               "bot_partner_mass", "bot_total_mass", "bot_partner_x", "bot_partner_y", "bot_partner_spread"]
SLICE_COLUMNS = ["bin_id", "x", "y", "annotation", "total_counts_original", "detected_genes_original",
                 "total_counts_equalised", "detected_genes_equalised", "untouched_by_equalisation"]

ROOT: Path | None = None
SETS: dict[str, list[str]] = {}


@lru_cache(maxsize=8)
def slice_table(sample: str) -> pd.DataFrame:
    return analysed_bins(ROOT, sample)[SLICE_COLUMNS]


@lru_cache(maxsize=1)
def collected_metrics() -> pd.DataFrame | None:
    """audit/pair_metrics.csv of the collect step: used only for a pair whose own files are absent (a partial copy)."""
    path = ROOT / "audit" / "pair_metrics.csv"
    return pd.read_csv(path) if path.is_file() else None


def init(root: str, sets: dict) -> None:
    global ROOT, SETS
    ROOT, SETS = Path(root), sets


def pair_rows(pair: dict) -> dict:
    """Every organ set's rows and distributions for one pair."""
    pid = pair["pair_id"]
    directory = ROOT / "pairs" / pid
    out = {name: {"rows": [], "dists": {}, "status": []} for name in SETS}
    metrics = pd.read_csv(directory / "pair_metrics.csv") if (directory / "pair_metrics.csv").is_file() else None
    if metrics is None and collected_metrics() is not None:
        metrics = collected_metrics()[collected_metrics()["pair_id"] == pid]
        metrics = metrics if len(metrics) else None
    complete = (directory / "SUCCESS").is_file() or (
        metrics is not None and (directory / "bins_source.csv.gz").is_file() and (directory / "bins_target.csv.gz").is_file())
    if not complete:
        for name in SETS:
            out[name]["status"].append({"pair_id": pid, "side": "both", "pair_completed": False})
        return out
    run = {}
    if metrics is not None:
        run["rejection_cost"] = float(metrics["rejection_cost"].iloc[0])
    for side, own, partner in (("source", pair["source_sample"], pair["target_sample"]),
                               ("target", pair["target_sample"], pair["source_sample"])):
        own_bins = slice_table(own)
        partner_counts = slice_table(partner)["annotation"].value_counts()
        table_path = directory / f"bins_{side}.csv.gz"
        npz_path = directory / f"transport_{side}.npz"
        bins = pd.read_csv(table_path)
        if len(bins) != len(own_bins) or not np.array_equal(bins["bin_row"].to_numpy(), np.arange(len(own_bins))):
            raise RuntimeError(f"{pid} {side}: per-bin table does not match the slice table")
        transport = None
        if npz_path.is_file():
            with np.load(npz_path) as z:
                transport = {k: z[k] for k in ("partner_annotations", "co_partner_distribution", "bot_partner_distribution")}
        for name, annotations in SETS.items():
            mask = own_bins["annotation"].isin(annotations).to_numpy()
            present = {a: int(partner_counts.get(a, 0)) for a in annotations}
            out[name]["status"].append({
                "pair_id": pid, "side": side, "pair_completed": True, "own_sample": own, "partner_sample": partner,
                "organ_bins": int(mask.sum()), "own_bins": len(own_bins),
                "partner_bins_same_annotations": int(sum(present.values())),
                "transport_npz": transport is not None, "rejection_cost": run.get("rejection_cost", np.nan)})
            if not mask.any():
                continue
            rows = own_bins.loc[mask].copy()
            rows.insert(0, "bin_row", np.flatnonzero(mask))
            for column in BIN_COLUMNS:
                rows[column] = bins.loc[mask, column].to_numpy()
            rows["partner_annotation_bins"] = rows["annotation"].map(lambda a: int(partner_counts.get(a, 0))).to_numpy()
            rows["partner_has_annotation"] = rows["partner_annotation_bins"] > 0
            for key, value in (("pair_id", pid), ("kind", pair["kind"]), ("group", pair["group"]), ("side", side),
                               ("own_sample", own), ("partner_sample", partner), ("own_stage", stage_of(own)),
                               ("partner_stage", stage_of(partner)),
                               ("rejection_cost", run.get("rejection_cost", np.nan))):
                rows.insert(0, key, value)
            if transport is not None:
                partners = list(transport["partner_annotations"])
                co = transport["co_partner_distribution"][mask]
                bot = transport["bot_partner_distribution"][mask]
                same = np.array([partners.index(a) if a in partners else -1 for a in rows["annotation"]])
                take = lambda d: np.where(same >= 0, d[np.arange(len(d)), np.clip(same, 0, None)], 0.0)  # noqa: E731
                rows["co_p_same"] = np.where(np.isfinite(co).all(axis=1), take(np.nan_to_num(co)), np.nan)
                rows["bot_p_same"] = np.where(np.isfinite(bot).all(axis=1), take(np.nan_to_num(bot)), np.nan)
                for prefix, d in (("co", co), ("bot", bot)):
                    finite = np.isfinite(d).all(axis=1)
                    top = np.argmax(np.nan_to_num(d, nan=-1.0), axis=1)
                    rows[f"{prefix}_top_annotation"] = np.where(finite, np.asarray(partners, dtype=object)[top], None)
                    rows[f"{prefix}_top_p"] = np.where(finite, d[np.arange(len(d)), top], np.nan)
                out[name]["dists"][f"{pid}__{side}"] = {
                    "bin_row": np.flatnonzero(mask).astype(np.int32),
                    "partner_annotations": np.asarray(partners, dtype=str), "co": co, "bot": bot}
            else:
                for column in ("co_p_same", "bot_p_same", "co_top_annotation", "co_top_p", "bot_top_annotation", "bot_top_p"):
                    rows[column] = np.nan
            out[name]["rows"].append(rows)
    return out


def m4r_distributions(pid: str, side: str, bin_rows: np.ndarray, threads: int) -> dict:
    """M4-R (and M4-E, as a check) conditional distribution over partner annotations, active support, rebuilt."""
    import torch
    import mosta_coupling

    torch.set_num_threads(threads)
    coupling = mosta_coupling.PairCoupling(ROOT / "pairs" / pid, root=ROOT, device="cpu")
    pairs = pd.read_csv(ROOT / "manifest" / "pairs.csv").set_index("pair_id")
    source_ann = slice_table(pairs.loc[pid, "source_sample"])["annotation"].to_numpy(dtype=str)
    target_ann = slice_table(pairs.loc[pid, "target_sample"])["annotation"].to_numpy(dtype=str)
    if len(source_ann) != coupling.shape[0] or len(target_ann) != coupling.shape[1]:
        raise RuntimeError(f"{pid}: slice tables and potentials disagree on bin counts")
    partner_ann = target_ann if side == "source" else source_ann
    partners = sorted(set(partner_ann))
    codes = pd.Categorical(partner_ann, categories=partners).codes
    hot = np.zeros((len(partner_ann), len(partners)))
    hot[np.arange(len(partner_ann)), codes] = 1.0
    out = {"partner_annotations": np.asarray(partners, dtype=str)}
    for method, key in (("M4-R", "m4r"), ("M4-E", "m4e_rebuilt")):
        groups = np.zeros((len(bin_rows), len(partners)))
        if side == "source":
            for start, stop, values in coupling.iter_blocks(method, source=bin_rows, support="active"):
                groups[start:stop] = values @ hot
        else:
            for start, stop, values in coupling.iter_blocks(method, target=bin_rows, support="active"):
                groups += values.T @ hot[start:stop]
        total = groups.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[key] = np.where(total[:, None] > 0, groups / total[:, None], np.nan).astype(np.float32)
        out[f"{key}_mass"] = total
    return out


def presence_known(out: Outputs, sample: str, genes: list[str], have: set[str]) -> bool:
    """True if every gene missing from the file was already found absent from this slice."""
    path = out.root / "expression" / "MARKER_PRESENCE.tsv"
    if not path.is_file():
        return False
    known = pd.read_csv(path, sep="	")
    absent = set(known.loc[(known["sample"] == sample) & ~known["present"], "gene"])
    return all(f"{gene}__count" in have or gene in absent for gene in genes)


def expression(out: Outputs, root: Path, samples: list[str]) -> None:
    """Marker genes of every analysed bin of each slice, from the equalised counts."""
    import anndata as ad
    from scipy import sparse

    genes = sorted(set(markers()["gene"]))
    presence = []
    for sample in samples:
        relative = f"expression/{sample}.csv.gz"
        if relative in out.mine:
            have = set(pd.read_csv(out.root / relative, nrows=0).columns)
            if all(f"{gene}__count" in have for gene in genes) or presence_known(out, sample, genes, have):
                continue
        data = ad.read_h5ad(root / "equalised" / f"{sample}.h5ad")
        counts = sparse.csr_matrix(data.X)
        names = pd.Index(data.var_names.astype(str))
        total = np.asarray(counts.sum(axis=1)).ravel().astype(np.float64)
        frame = pd.DataFrame({"bin_id": data.obs_names.astype(str), "total_equalised": total})
        for gene in genes:
            where = names.get_indexer([gene])[0]
            presence.append({"sample": sample, "gene": gene, "present": where >= 0})
            if where < 0:
                continue
            column = np.asarray(counts[:, where].todense()).ravel().astype(np.float64)
            frame[f"{gene}__count"] = column
            with np.errstate(divide="ignore", invalid="ignore"):
                frame[f"{gene}__log1p_cp10k"] = np.log1p(np.where(total > 0, 1e4 * column / total, 0.0))
        out.table(frame, relative, "marker genes, equalised counts")
        print(f"expression {sample}: {len(frame)} bins", flush=True)
        del data, counts
    if presence:
        table = pd.DataFrame(presence)
        name = "expression/MARKER_PRESENCE.tsv"
        if name in out.mine:
            table = pd.concat([pd.read_csv(out.root / name, sep="	"), table]).drop_duplicates(["sample", "gene"], keep="last")
        out.table(table, name, "marker presence per slice")
        summary = table.groupby("gene").agg(slices_present=("present", "sum"), slices_checked=("present", "size")).reset_index()
        summary = summary.merge(markers(), on="gene", how="left")
        name = "expression/MARKER_GENES.tsv"
        if name in out.mine:
            old = pd.read_csv(out.root / name, sep="\t")
            summary = pd.concat([old, summary]).drop_duplicates(["gene", "organ_set"], keep="last")
        out.table(summary, name, "marker presence per gene")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--organ-set", action="append", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--pairs", nargs="*", help="only these pair ids (code checks)")
    parser.add_argument("--expression", action="store_true")
    parser.add_argument("--m4r-coupling", action="store_true")
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    all_sets = organ_sets()
    sets = {name: all_sets[name] for name in args.organ_set}
    out = Outputs(args.out)
    pairs = pd.read_csv(args.run_root / "manifest" / "pairs.csv")
    if args.pairs:
        pairs = pairs[pairs["pair_id"].isin(args.pairs)]
    init(str(args.run_root), sets)
    records = pairs.to_dict("records")
    collected = {name: {"rows": [], "status": []} for name in sets}
    base = {name: f"organ_bins/{args.run_name}/{name}" for name in sets}

    def take(result: dict) -> None:
        for name, part in result.items():
            collected[name]["rows"] += part["rows"]
            collected[name]["status"] += part["status"]
            for key, arrays in part["dists"].items():
                relative = f"{base[name]}/distributions/{key}.npz"
                np.savez_compressed(out.path(relative), **arrays)
                out.record(relative)

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers, initializer=init, initargs=(str(args.run_root), sets)) as pool:
            for i, result in enumerate(pool.map(pair_rows, records, chunksize=2)):
                take(result)
                if i % 50 == 0:
                    print(f"{i + 1}/{len(records)} pairs, {time.time() - started:.0f}s", flush=True)
    else:
        for result in map(pair_rows, records):
            take(result)
    for name in sets:
        rows = pd.concat(collected[name]["rows"], ignore_index=True) if collected[name]["rows"] else pd.DataFrame()
        rows.insert(0, "run", args.run_name)
        out.table(rows, f"{base[name]}/bins.csv.gz", f"{len(rows)} bin rows")
        out.table(pd.DataFrame(collected[name]["status"]), f"{base[name]}/pairs.tsv")
        print(f"{name}: {len(rows)} bin rows from {rows['pair_id'].nunique() if len(rows) else 0} pairs", flush=True)

    if args.m4r_coupling:
        checks = []
        for name in sets:
            table = pd.read_csv(out.root / f"{base[name]}/bins.csv.gz", usecols=["pair_id", "side", "bin_row"])
            for (pid, side), group in table.groupby(["pair_id", "side"], sort=False):
                relative = f"{base[name]}/distributions_m4r/{pid}__{side}.npz"
                if relative in out.mine:
                    continue
                bin_rows = group["bin_row"].to_numpy()
                arrays = m4r_distributions(pid, side, bin_rows, args.threads)
                arrays["bin_row"] = bin_rows.astype(np.int32)
                saved = out.root / f"{base[name]}/distributions/{pid}__{side}.npz"
                diff = np.nan
                if saved.is_file():
                    with np.load(saved) as z:
                        same_order = list(z["partner_annotations"]) == list(arrays["partner_annotations"])
                        if same_order:
                            a, b = z["co"].astype(np.float64), arrays["m4e_rebuilt"].astype(np.float64)
                            both = np.isfinite(a) & np.isfinite(b)
                            diff = float(np.max(np.abs(a[both] - b[both]))) if both.any() else np.nan
                np.savez_compressed(out.path(relative), **arrays)
                out.record(relative)
                checks.append({"organ_set": name, "pair_id": pid, "side": side, "bins": len(bin_rows),
                               "m4e_rebuilt_vs_saved_max_abs_diff": diff})
                print(f"m4r {name} {pid} {side}: {len(bin_rows)} bins, check {diff:.2e}", flush=True)
        if checks:
            out.table(pd.DataFrame(checks), f"organ_bins/{args.run_name}/m4r_coupling_checks.tsv")

    if args.expression:
        samples = sorted(set(pairs["source_sample"]) | set(pairs["target_sample"]))
        expression(out, args.run_root, samples)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
