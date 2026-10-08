"""Audit the all-stage MOSTA run, gather its tables, build the DE inputs and the download list.

**Status.**  Every expected pair in manifest/pairs.csv gets exactly one:

* ``completed``: pairs/<id>/SUCCESS is present, and every file it lists is
  present, readable and of the expected length;
* ``corrupted``: SUCCESS is present but a file is missing, unreadable or the
  wrong length;
* ``failed``: a traceback was recorded (pairs/<id>.FAILED, or the
  representation's);
* ``missing``: no output and no recorded failure; the pair never ran, or its
  job ended mid-pair.

No other label is given.  Calibration flags, M4-R convergence, rejection
fractions, depth and annotation presence are reported in the tables and decide
nothing.

**Built from every completed pair** (there is nothing else to read):

* audit/pair_metrics.csv, audit/annotation_fractions.csv.gz,
  audit/annotation_transitions.csv.gz and audit/calibration.csv;
* audit/depth_diagnostics.csv: per pair, method and side, the Spearman
  correlation of the decision cost with the original depth and with the
  original detected genes, the median original depth of retained and of
  rejected bins, and the rank AUC of original depth (rejected against
  retained).  Reporting only;
* pseudobulk/<pair_id>/<side>.npz: the equalised counts (the matrix the OT
  read) summed per annotation x M4-E status, with bin counts.  Gene names are
  in slices/<sample>_genes.txt.gz;
* audit/deg_completeness.csv: per pair, side and annotation, the retained and
  rejected bin counts and whether the retained-vs-rejected contrast is defined
  (both non-empty).  An undefined contrast is recorded there and nothing is
  dropped.

**Finally** package/PACKAGE_FILES.txt, package/README.md and
package/download_commands.txt.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import Layout, read_equalised_slice, read_pairs, write_json  # noqa: E402

REMOTE_HOST = "ghzheng@vista.tacc.utexas.edu"
LOCAL_ROOT = "/d/Xia_lab/CellOT/mouse_embryo_results/mosta_all_stages"
RAW_DATA = "/scratch/10119/ghzheng/OT_project/data"
EXPECTED_FILES = ("bins_source.csv.gz", "bins_target.csv.gz", "transport_source.npz", "transport_target.npz",
                  "annotation_fractions.csv", "annotation_transitions.csv.gz", "pair_metrics.csv",
                  "calibration.json", "run.json")


def check_pair(layout: Layout, row: pd.Series) -> dict:
    pair_id = row["pair_id"]
    directory = layout.pair(pair_id)
    record = {"pair_id": pair_id, "index": int(row["index"]), "kind": row["kind"], "group": row["group"],
              "source_sample": row["source_sample"], "target_sample": row["target_sample"],
              "output": str(directory), "status": "missing", "reason": ""}
    representation_failed = layout.representations / f"{pair_id}.FAILED"
    ot_failed = layout.pairs / f"{pair_id}.FAILED"
    if not (directory / "SUCCESS").is_file():
        if ot_failed.is_file():
            record.update(status="failed", reason="OT: " + ot_failed.read_text(encoding="utf-8").strip().splitlines()[-1][:300])
        elif representation_failed.is_file():
            record.update(status="failed", reason="representation: " + representation_failed.read_text(encoding="utf-8").strip().splitlines()[-1][:300])
        elif not (layout.representation(pair_id) / "REPRESENTATION_SUCCESS").is_file():
            record["reason"] = "no representation and no recorded failure"
        else:
            record["reason"] = "no OT output and no recorded failure"
        return record
    problems = []
    try:
        listed = {item["file"]: item["bytes"] for item in json.loads((directory / "SUCCESS").read_text(encoding="utf-8"))["files"]}
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        n, m = int(run["source_bins"]), int(run["target_bins"])
        for name in EXPECTED_FILES:
            path = directory / name
            if not path.is_file():
                problems.append(f"{name} missing")
            elif name in listed and path.stat().st_size != listed[name]:
                problems.append(f"{name} size changed")
        for side, count, partner_count in (("source", n, len(run["target_categories"])),
                                           ("target", m, len(run["source_categories"]))):
            bins = pd.read_csv(directory / f"bins_{side}.csv.gz")
            if len(bins) != count:
                problems.append(f"bins_{side} has {len(bins)} rows, expected {count}")
            with np.load(directory / f"transport_{side}.npz") as arrays:
                for key in arrays.files:
                    if key == "partner_annotations":
                        continue
                    if arrays[key].shape != (count, partner_count):
                        problems.append(f"transport_{side}:{key} shape {arrays[key].shape}")
        for name in ("annotation_fractions.csv", "annotation_transitions.csv.gz", "pair_metrics.csv"):
            pd.read_csv(directory / name)
        json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        record.update(source_bins=n, target_bins=m)
    except Exception as error:  # an unreadable file is exactly what this records
        problems.append(f"{type(error).__name__}: {error}")
    if problems:
        record.update(status="corrupted", reason="; ".join(problems)[:500])
    else:
        record["status"] = "completed"
    return record


def rank_auc(values: np.ndarray, positive: np.ndarray) -> float:
    if positive.all() or not positive.any():
        return float("nan")
    ranks = stats.rankdata(values)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def collect_tables(layout: Layout, completed: pd.DataFrame) -> None:
    metrics, fractions, transitions, calibration, depth = [], [], [], [], []

    @lru_cache(maxsize=16)
    def analysed(sample: str) -> pd.DataFrame:
        table = pd.read_csv(layout.slice_bins(sample))
        return table[table["equalised_row"] >= 0].sort_values("equalised_row").reset_index(drop=True)

    for _, row in completed.iterrows():
        directory = layout.pair(row["pair_id"])
        keys = {"pair_id": row["pair_id"], "kind": row["kind"], "group": row["group"]}
        metrics.append(pd.read_csv(directory / "pair_metrics.csv"))
        fractions.append(pd.read_csv(directory / "annotation_fractions.csv").assign(**keys))
        transitions.append(pd.read_csv(directory / "annotation_transitions.csv.gz").assign(**keys))
        payload = json.loads((directory / "calibration.json").read_text(encoding="utf-8"))
        calibration.append({**keys, **{key: payload.get(key) for key in (
            "rejection_cost", "selection_status", "calibration_valid", "feasible_cost_found",
            "m4e_calibration_clean", "m4r_validation_clean", "m4e_inference_valid", "source_monotone",
            "target_monotone", "refinement_method", "validation_source_raw_acceptance",
            "validation_target_raw_acceptance", "validation_aggregate_valid", "calibration_bins_per_side",
            "calibration_seconds")}, "warnings": " | ".join(payload.get("warning_messages") or [])})
        for side in ("source", "target"):
            bins = pd.read_csv(directory / f"bins_{side}.csv.gz")
            meta = analysed(row[f"{side}_sample"])
            depth_values = meta["total_counts_original"].to_numpy(dtype=float)
            genes_values = meta["detected_genes_original"].to_numpy(dtype=float)
            for method in ("m4e", "m4r"):
                retained = bins[f"{method}_retained"].to_numpy(dtype=bool)
                cost = bins[f"{method}_decision_cost"].to_numpy(dtype=float)
                depth.append({**keys, "method": method.upper().replace("M4", "M4-"), "side": side,
                              "bins": len(bins), "rejected": int((~retained).sum()),
                              "spearman_decision_cost_original_depth": float(stats.spearmanr(cost, depth_values)[0]),
                              "spearman_decision_cost_original_detected_genes": float(stats.spearmanr(cost, genes_values)[0]),
                              "median_original_depth_retained": float(np.median(depth_values[retained])) if retained.any() else float("nan"),
                              "median_original_depth_rejected": float(np.median(depth_values[~retained])) if (~retained).any() else float("nan"),
                              "auc_original_depth_rejected_vs_retained": rank_auc(depth_values, ~retained)})
    layout.audit.mkdir(parents=True, exist_ok=True)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(layout.audit / "pair_metrics.csv", index=False)
        pd.concat(fractions, ignore_index=True).to_csv(layout.audit / "annotation_fractions.csv.gz", index=False, compression="gzip")
        pd.concat(transitions, ignore_index=True).to_csv(layout.audit / "annotation_transitions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(calibration).to_csv(layout.audit / "calibration.csv", index=False)
    pd.DataFrame(depth).to_csv(layout.audit / "depth_diagnostics.csv", index=False)


def build_pseudobulks(layout: Layout, completed: pd.DataFrame) -> None:
    """Per slice, load the equalised counts once and sum them for every pair that uses it."""
    users: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for _, row in completed.iterrows():
        for side in ("source", "target"):
            users[row[f"{side}_sample"]].append((row["pair_id"], side))
    completeness = []
    for sample, jobs in sorted(users.items()):
        todo = [(pair_id, side) for pair_id, side in jobs
                if not (layout.pseudobulk / pair_id / f"{side}.npz").is_file()]
        data = read_equalised_slice(layout.equalised_h5ad(sample)) if todo else None
        counts64 = data["counts"].astype(np.int64) if data is not None else None
        for pair_id, side in jobs:
            bins = pd.read_csv(layout.pair(pair_id) / f"bins_{side}.csv.gz", usecols=["bin_row", "m4e_retained"])
            path = layout.pseudobulk / pair_id / f"{side}.npz"
            labels = None
            if data is not None and (pair_id, side) in todo:
                labels = data["annotation"][bins["bin_row"].to_numpy()]
                retained = bins["m4e_retained"].to_numpy(dtype=bool)
                status = np.where(retained, "retained", "rejected")
                groups = pd.MultiIndex.from_arrays([labels, status]).unique().sort_values()
                codes = groups.get_indexer(pd.MultiIndex.from_arrays([labels, status]))
                indicator = sparse.csr_matrix((np.ones(len(codes)), (codes, bins["bin_row"].to_numpy())),
                                              shape=(len(groups), data["counts"].shape[0]))
                summed = (indicator @ counts64).toarray().astype(np.int64)
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_name(f".{side}.partial.npz")
                np.savez_compressed(partial, counts=summed,
                                    annotation=np.asarray(groups.get_level_values(0), dtype=str),
                                    status=np.asarray(groups.get_level_values(1), dtype=str),
                                    n_bins=np.bincount(codes, minlength=len(groups)),
                                    sample=np.asarray(sample), status_gate=np.asarray("M4-E"))
                partial.replace(path)
            if labels is None:
                with np.load(path) as stored:
                    annotation, status, n_bins = stored["annotation"], stored["status"], stored["n_bins"]
            else:
                annotation, status = np.asarray(groups.get_level_values(0), dtype=str), np.asarray(groups.get_level_values(1), dtype=str)
                n_bins = np.bincount(codes, minlength=len(groups))
            counts = pd.DataFrame({"annotation": annotation, "status": status, "n": n_bins}).pivot_table(
                index="annotation", columns="status", values="n", aggfunc="sum", fill_value=0)
            for annotation_name, values in counts.iterrows():
                n_ret, n_rej = int(values.get("retained", 0)), int(values.get("rejected", 0))
                completeness.append({"pair_id": pair_id, "side": side, "sample": sample, "annotation": annotation_name,
                                     "retained_bins": n_ret, "rejected_bins": n_rej,
                                     "contrast_defined": bool(n_ret > 0 and n_rej > 0),
                                     "undefined_because": "" if (n_ret and n_rej) else ("all retained" if n_ret else "all rejected")})
        del data, counts64
    table = pd.DataFrame(completeness)
    if len(table):
        table = table.merge(completed[["pair_id", "kind", "group"]], on="pair_id", how="left")
    table.to_csv(layout.audit / "deg_completeness.csv", index=False)


def package(layout: Layout) -> dict:
    root = layout.root
    include = []
    for pattern in ("manifest/*.csv", "equalisation/*", "slices/*", "representations/*/representation.json",
                    "representations/*.FAILED", "pairs/*/*", "pairs/*.FAILED", "pseudobulk/*/*.npz",
                    "equivalence/*", "audit/*", "logs/*"):
        include += [path for path in root.glob(pattern) if path.is_file() and not path.name.startswith(".")]
    layout.package.mkdir(parents=True, exist_ok=True)
    readme = layout.package / "README.md"
    files_list = layout.package / "PACKAGE_FILES.txt"
    commands = layout.package / "download_commands.txt"
    relative = sorted({str(path.relative_to(root)).replace("\\", "/") for path in include})
    core_bytes = sum((root / name).stat().st_size for name in relative)
    equalised_bytes = sum(path.stat().st_size for path in layout.equalised.glob("*.h5ad"))
    coordinates_bytes = sum(path.stat().st_size for path in layout.representations.glob("*/representation.npz"))
    raw_bytes = sum(path.stat().st_size for path in Path(RAW_DATA).glob("*.MOSTA.h5ad")) if Path(RAW_DATA).is_dir() else 0
    gb = lambda value: f"{value / 1e9:.1f} GB"  # noqa: E731
    readme.write_text(f"""# MOSTA all-stage ConfidenceOT results

Built {time.strftime('%Y-%m-%d %H:%M')} by mouse_embryo/25_collect_mosta.py.

## What is in the package ({len(relative) + 3} files, {gb(core_bytes)})

- `manifest/`: slices.csv (every slice) and pairs.csv (every expected pair, with its seed index).
- `equalisation/`: the common-depth step. report.json has the shared target depth.
- `slices/<sample>_bins.csv.gz`: every bin of every slice, Cavity included.
  - Columns: bin_id, x, y, annotation, analysed, equalised_row, original and equalised depth and detected genes.
  - `equalised_row` is the key the per-pair tables use (`bin_row`).
  - `<sample>_genes.txt.gz` holds the gene order of the pseudobulks.
- `pairs/<pair_id>/`: one completed pair (source__target).
  - `bins_source.csv.gz` and `bins_target.csv.gz`: per bin, joined to the slice table on `bin_row`.
    - M4-E and M4-R calls, raw calls, decision cost, margin, coefficient, budget override.
    - ConfidenceOT (`co_`, M4-E coupling on retained x retained bins) and balanced OT (`bot_`): partner mass, total row mass, partner centroid x/y in the partner slice, partner spread.
  - `transport_source.npz` (rows: source bins; columns: `partner_annotations`, i.e. target annotations):
    - `co_partner_distribution` and `bot_partner_distribution`, each row summing to 1 (NaN for a bin without supported mass);
    - `co_pullback`, `bot_pullback` and `bot_pullback_co_rejected`: the nominal mass of each target annotation pulled back onto each source bin, by M4-E retained targets, by every target under balanced OT, and by the targets M4-E rejected, under balanced OT.
  - `transport_target.npz`: the same for target bins over source annotations, with `co_pushforward`, `bot_pushforward` and `bot_pushforward_co_rejected`.
  - `annotation_fractions.csv`: per side and annotation:
    - M4-E and M4-R retained and rejected counts and fractions;
    - partner presence (reported, not used to include or exclude);
    - the nominal-mass split same / other / retained-without-support;
    - balanced OT's same/other split, overall and for M4-E-rejected bins.
  - `annotation_transitions.csv.gz`: annotation-to-annotation nominal mass (and raw coupling mass) forward and backward, for co, bot and bot_co_rejected.
  - `pair_metrics.csv`: one row per method, with rates, convergence, iterations, cycles, timings and calibration flags (reported only).
  - `calibration.json`: the full calibration record, including the 2,000-bin subsample rows.
  - `run.json`: provenance: preprocessing label, seeds, scale, block size, GPU, code commit.
- `pseudobulk/<pair_id>/<side>.npz`: equalised counts summed per annotation x M4-E status (`counts`, `annotation`, `status`, `n_bins`, `sample`). Genes follow slices/<sample>_genes.txt.gz.
- `audit/`:
  - completeness_manifest.csv: every expected pair and its status, one of completed / corrupted / failed / missing;
  - summary.json;
  - pair_metrics.csv, annotation_fractions.csv.gz, annotation_transitions.csv.gz, calibration.csv and depth_diagnostics.csv, over every completed pair;
  - deg_completeness.csv: which retained-vs-rejected contrasts are defined.
- `equivalence/`: the dense-vs-blockwise check run before the full run.
- `representations/<pair_id>/representation.json`: the preprocessing record and the 2,000 selected genes of each pair.
- `logs/`: job logs.

## Not in the package (separate commands in download_commands.txt)

- `equalised/*.h5ad`, the equalised counts of every analysed bin: {gb(equalised_bytes)}.
- `representations/*/representation.npz`, the 30-dimensional coordinates of every pair: {gb(coordinates_bytes)}.
- the original MOSTA files, {RAW_DATA}: {gb(raw_bytes)}.

Nothing in the package has been selected or filtered. Pairs that are not completed are listed with their status in audit/completeness_manifest.csv.
""", encoding="utf-8")
    relative += [str(path.relative_to(root)).replace("\\", "/") for path in (readme, files_list, commands)]
    files_list.write_text("\n".join(sorted(set(relative))) + "\n", encoding="utf-8")
    remote = str(root).replace("\\", "/")
    commands.write_text(f"""# Run these on the local machine (Git Bash has ssh and tar). Each asks for the TACC password and MFA once.

# 1. The results package ({gb(core_bytes)}), streamed with tar, keeping the directory tree.
mkdir -p {LOCAL_ROOT}
ssh {REMOTE_HOST} "tar -C {remote} -cf - -T {remote}/package/PACKAGE_FILES.txt" | tar -xf - -C {LOCAL_ROOT}

# 1b. The same with rsync, where rsync is installed (it can resume an interrupted copy).
rsync -av --files-from=:{remote}/package/PACKAGE_FILES.txt {REMOTE_HOST}:{remote}/ {LOCAL_ROOT}/

# 2. Optional: the equalised counts ({gb(equalised_bytes)}), for expression plots on the analysed bins.
ssh {REMOTE_HOST} "tar -C {remote} -cf - equalised" | tar -xf - -C {LOCAL_ROOT}

# 3. Optional: the per-pair 30-dimensional coordinates ({gb(coordinates_bytes)}).
ssh {REMOTE_HOST} "cd {remote} && tar -cf - representations/*/representation.npz" | tar -xf - -C {LOCAL_ROOT}

# 4. Optional: the original MOSTA files ({gb(raw_bytes)}).
mkdir -p /d/Xia_lab/data/MOSTA
ssh {REMOTE_HOST} "tar -C {RAW_DATA} -cf - ." | tar -xf - -C /d/Xia_lab/data/MOSTA
""", encoding="utf-8")
    return {"package_files": len(relative), "package_bytes": core_bytes, "equalised_bytes": equalised_bytes,
            "coordinates_bytes": coordinates_bytes, "raw_bytes": raw_bytes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--skip-pseudobulk", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    layout = Layout(args.root)
    pairs = read_pairs(layout)
    status = pd.DataFrame([check_pair(layout, row) for _, row in pairs.iterrows()])
    layout.audit.mkdir(parents=True, exist_ok=True)
    status.to_csv(layout.audit / "completeness_manifest.csv", index=False)
    completed = pairs[pairs["pair_id"].isin(status.loc[status["status"] == "completed", "pair_id"])]
    collect_tables(layout, completed)
    if not args.skip_pseudobulk:
        build_pseudobulks(layout, completed)
    sizes = package(layout)
    counts = status.groupby(["kind", "status"]).size().unstack(fill_value=0)
    summary = {
        "expected_adjacent": int((status["kind"] == "adjacent").sum()),
        "completed_adjacent": int(((status["kind"] == "adjacent") & (status["status"] == "completed")).sum()),
        "expected_same_stage": int((status["kind"] == "same_stage").sum()),
        "completed_same_stage": int(((status["kind"] == "same_stage") & (status["status"] == "completed")).sum()),
        "not_completed": status.loc[status["status"] != "completed", ["pair_id", "kind", "status", "reason"]].to_dict("records"),
        "status_counts": counts.to_dict(),
        "output_root": str(layout.root),
        "completeness_manifest": str(layout.audit / "completeness_manifest.csv"),
        "package_list": str(layout.package / "PACKAGE_FILES.txt"),
        "download_commands": str(layout.package / "download_commands.txt"),
        **sizes,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(layout.audit / "summary.json", summary)
    print(counts.to_string())
    print(json.dumps({key: value for key, value in summary.items() if key != "not_completed"}, indent=2))
    for record in summary["not_completed"]:
        print(f"{record['status']:>9}  {record['pair_id']}  {record['reason']}")


if __name__ == "__main__":
    main()
