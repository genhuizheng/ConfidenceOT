"""Prove on real MOSTA pairs that the saved files rebuild every bin-to-bin coupling.

23_ saves, per pair, the dual potentials, gates, weights and settings of M4-E,
M4-R and balanced OT (mosta_coupling.py), and no dense coupling.  This script
shows that those files, with the pair's representation, give back the
couplings themselves, entry by entry, and everything the run summarised from
them.  It runs before a run's OT workers are submitted.

**Test pairs**, fixed by rule: the two adjacent and the two same-stage pairs
with the fewest cost entries (ties by manifest index), small enough that every
dense coupling and the dense solver fit in memory.

**Per test pair**, 23_'s own ``run_pair`` runs the pair on ``--device`` (the
production path of that device, the same code the workers run) into
coupling_verification/pairs/<pair_id>/, a complete pair output including its
coupling_check.json, never into pairs/.  The in-memory fits it used are kept,
and then, from the saved files only (``mosta_coupling.PairCoupling``, SHA-256
of the representation and of potentials.npz checked):

1. ``readback``: coupling_check.json of the pair (saved arrays bit-identical
   to the fits; rebuilt masses against the summaries).
2. ``coupling``: the whole coupling of each method, rebuilt on the CPU and on
   ``--device``, against ``materialize_coupling`` of the in-memory fit.
3. ``dense_solver``: against the dense torch solver (float64 cost, same
   settings) fitted from scratch on the same device: gates identical, and the
   coupling entry by entry.  An independent solve, not a reread.
4. ``subset``: blocks of random source x target subsets, selected by row
   position, by bin id and by boolean mask, and rebuilt with 97-row blocks,
   against the same entries of the whole rebuilt coupling; conditional
   probabilities P_ij / sum_j P_ij and P_ij / sum_i P_ij (both supports)
   against the whole coupling's, and their rows / columns summing to 1.
5. ``masses``: row and column masses of the rebuilt couplings against the
   in-memory summaries (float64) and against bins_source / bins_target
   (written with 7 significant digits).
6. ``transitions``: annotation x annotation coupling mass (raw_coupling_mass,
   9 digits) and nominal mass (from float32 distributions) of every row of
   annotation_transitions.csv.gz, recomputed from the rebuilt couplings.
7. ``distributions``: the per-bin partner-annotation distributions of
   transport_source.npz / transport_target.npz (float32).
8. ``gates``: the saved gates against bins_*.csv's retained columns.

**Tolerances**, fixed before any run (an error is entrywise relative,
|found - expected| / |expected| over entries above 1e-250, unless it says
"of max", which divides by the largest expected value):

* readback, gates: exact;
* coupling (same fit, rebuilt): 1e-10 entrywise; subset against whole: 1e-12
  entrywise; selection by id or mask against by position: identical;
* dense solver: gates identical, 1e-8 entrywise;
* masses against the in-memory summaries 1e-12 of max; against the 7-digit
  tables 1e-6 entrywise;
* raw_coupling_mass 1e-8 entrywise (9 digits); nominal mass 1e-6 entrywise
  (float32 distributions); distributions 1e-6 absolute.

Writes coupling_verification/report.csv (one row per pair, method and check),
summary.json, REPORT.md, and COUPLING_PASS or COUPLING_FAIL.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mosta_coupling  # noqa: E402
from mosta_common import Layout, ordered_categories, read_pairs, write_json  # noqa: E402

TINY = 1e-250
TOLERANCE = {
    "readback": 0.0, "gates": 0.0,
    "coupling": 1e-10, "subset": 1e-12, "selection": 0.0, "dense_solver": 1e-8,
    "masses_memory": 1e-12, "masses_table": 1e-6,
    "raw_coupling_mass": 1e-8, "nominal_mass": 1e-6, "distributions": 1e-6,
}


def runner_module():
    spec = importlib.util.spec_from_file_location("runner23", Path(__file__).resolve().parent / "23_run_mosta_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def entrywise(found, expected) -> float:
    """max |found - expected| / |expected| over entries where either exceeds TINY; NaN pairs must coincide."""
    found = np.asarray(found, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    nan_found, nan_expected = np.isnan(found), np.isnan(expected)
    if (nan_found != nan_expected).any():
        return float("inf")
    keep = ~nan_found & ((np.abs(found) > TINY) | (np.abs(expected) > TINY))
    if not keep.any():
        return 0.0
    return float(np.max(np.abs(found[keep] - expected[keep]) / np.maximum(np.abs(expected[keep]), TINY)))


def of_max(found, expected) -> float:
    found = np.asarray(found, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    difference = float(np.max(np.abs(found - expected))) if expected.size else 0.0
    return difference / scale if scale > 0 else difference


def test_pairs(layout: Layout) -> pd.DataFrame:
    pairs = read_pairs(layout)
    sizes = pd.read_csv(layout.equalisation / "per_slice.csv").set_index("sample")["bins_analysed"]
    pairs["elements"] = pairs["source_sample"].map(sizes) * pairs["target_sample"].map(sizes)
    pairs = pairs.sort_values(["elements", "index"])
    return pd.concat([pairs[pairs["kind"] == kind].head(2) for kind in ("adjacent", "same_stage")])


def codes(labels: pd.Series) -> tuple[np.ndarray, list[str]]:
    categories = ordered_categories(labels)
    return pd.Categorical(labels, categories=categories).codes.astype(np.int64), categories


def one_hot(values: np.ndarray, count: int) -> np.ndarray:
    out = np.zeros((len(values), count))
    out[np.arange(len(values)), values] = 1.0
    return out


def verify_pair(layout: Layout, row: pd.Series, output: Path, device: str, seed: int) -> list[dict]:
    from confidenceot.blockwise import materialize_coupling
    from confidenceot.cuda import fit_cuda
    from confidenceot.preprocessing import squared_euclidean

    runner = runner_module()
    pair_id = str(row["pair_id"])
    keep: dict = {}
    started = time.perf_counter()
    staging = runner.run_pair(layout, row, device=device, block_rows=None, keep=keep)
    pair_dir = output / "pairs" / pair_id
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    pair_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(staging), str(pair_dir))
    run_seconds = time.perf_counter() - started
    fits, models, cost = keep["fits"], keep["models"], keep["cost"]
    source_bins, target_bins = keep["source_bins"], keep["target_bins"]
    n, m = cost.shape
    results: list[dict] = []

    def add(method, check, quantity, error, tolerance, note=""):
        results.append({"pair_id": pair_id, "kind": row["kind"], "source_bins": n, "target_bins": m,
                        "method": method, "check": check, "quantity": quantity, "error": float(error),
                        "tolerance": tolerance, "pass": bool(error <= tolerance), "note": note})

    # 1. readback, as 23_ recorded it.
    readback = json.loads((pair_dir / "coupling_check.json").read_text(encoding="utf-8"))
    for method, flags in readback["arrays_identical"].items():
        add(method, "readback", "saved arrays bit-identical to the fit", 0.0 if all(flags.values()) else 1.0,
            TOLERANCE["readback"], ", ".join(name for name, same in flags.items() if not same))
    for item in readback["masses"]:
        add(item["method"], "readback", f"{item['mass']} mass ({item['support']}) vs {item['summary']}, of max",
            item["relative_error"], readback["tolerance_relative_to_max"])

    # Everything below reads only the saved files (and the representation they name).
    rebuilt = {name: mosta_coupling.PairCoupling(pair_dir, root=layout.root, device=name)
               for name in sorted({"cpu", device})}
    source64 = np.asarray(cost.source, dtype=np.float64)
    target64 = np.asarray(cost.target, dtype=np.float64)
    dense_cost = squared_euclidean(source64, target64) / cost.scale
    rng = np.random.default_rng(seed)
    source_codes, source_categories = codes(source_bins["annotation"])
    target_codes, target_categories = codes(target_bins["annotation"])
    source_hot = one_hot(source_codes, len(source_categories))
    target_hot = one_hot(target_codes, len(target_categories))
    a = fits["M4-E"].source_weights
    b = fits["M4-E"].target_weights
    full = {}
    for method, fit in fits.items():
        memory = materialize_coupling(cost, fit)
        for name, coupling in rebuilt.items():
            whole = coupling.mass(method, max_elements=n * m)
            add(method, "coupling", f"whole coupling rebuilt on {name} vs the in-memory fit",
                entrywise(whole, memory), TOLERANCE["coupling"])
            if name == "cpu":
                full[method] = whole
        # 3. an independent dense solve.
        dense = fit_cuda(dense_cost, dtype="float64", _torch_device=device, **models[method]._solver_kwargs())
        gate_difference = int(np.sum(dense.source_gate != fit.source_gate) + np.sum(dense.target_gate != fit.target_gate))
        add(method, "dense_solver", "final gates, bins differing", gate_difference, 0)
        if gate_difference == 0:
            add(method, "dense_solver", "whole coupling vs the dense solver's", entrywise(full[method], dense.coupling),
                TOLERANCE["dense_solver"])
        del dense, memory

        # 4. subsets, selections, blocks, conditionals.
        coupling = rebuilt["cpu"]
        rows = np.sort(rng.choice(n, size=min(n, 300), replace=False))
        columns = np.sort(rng.choice(m, size=min(m, 400), replace=False))
        by_position = coupling.mass(method, rows, columns)
        add(method, "subset", "random block vs the whole coupling",
            entrywise(by_position, full[method][np.ix_(rows, columns)]), TOLERANCE["subset"],
            f"{len(rows)} x {len(columns)}")
        by_id = coupling.mass(method, coupling.source_bin_ids[rows], coupling.target_bin_ids[columns])
        mask_rows = np.zeros(n, dtype=bool)
        mask_rows[rows] = True
        mask_columns = np.zeros(m, dtype=bool)
        mask_columns[columns] = True
        by_mask = coupling.mass(method, mask_rows, mask_columns)
        add(method, "subset", "selected by bin id vs by position", entrywise(by_id, by_position), TOLERANCE["selection"])
        add(method, "subset", "selected by mask vs by position", entrywise(by_mask, by_position), TOLERANCE["selection"])
        add(method, "subset", "whole coupling in 97-row blocks vs default blocks",
            entrywise(coupling.mass(method, block_rows=97, max_elements=n * m), full[method]), TOLERANCE["subset"])
        for support in mosta_coupling.SUPPORTS:
            whole = full[method]
            if support == "active":
                whole = whole * (fit.source_gate[:, None] & fit.target_gate[None, :])
            with np.errstate(divide="ignore", invalid="ignore"):
                row_total = whole.sum(axis=1)
                column_total = whole.sum(axis=0)
                expected_source = np.where(row_total[:, None] > 0, whole / row_total[:, None], np.nan)
                expected_target = np.where(column_total[None, :] > 0, whole / column_total[None, :], np.nan)
            found = coupling.mass(method, rows, columns, support=support, normalise="source")
            add(method, "subset", f"P_ij / sum_j P_ij ({support}) on the block vs the whole coupling",
                entrywise(found, expected_source[np.ix_(rows, columns)]), TOLERANCE["subset"])
            found = coupling.mass(method, rows, columns, support=support, normalise="target")
            add(method, "subset", f"P_ij / sum_i P_ij ({support}) on the block vs the whole coupling",
                entrywise(found, expected_target[np.ix_(rows, columns)]), TOLERANCE["subset"])
            rows_sum = coupling.mass(method, rows, None, support=support, normalise="source", max_elements=n * m).sum(axis=1)
            defined = row_total[rows] > 0
            add(method, "subset", f"P_ij / sum_j P_ij ({support}) rows sum to 1, |sum - 1|",
                float(np.max(np.abs(rows_sum[defined] - 1.0), initial=0.0)), TOLERANCE["subset"],
                f"{int(defined.sum())} of {len(rows)} rows have mass")
            columns_sum = coupling.mass(method, None, columns, support=support, normalise="target", max_elements=n * m).sum(axis=0)
            defined = column_total[columns] > 0
            add(method, "subset", f"P_ij / sum_i P_ij ({support}) columns sum to 1, |sum - 1|",
                float(np.max(np.abs(columns_sum[defined] - 1.0), initial=0.0)), TOLERANCE["subset"],
                f"{int(defined.sum())} of {len(columns)} columns have mass")

        # 8. gates against the tables.
        if method in ("M4-E", "M4-R"):
            p = mosta_coupling.prefix(method)
            bins_source = pd.read_csv(pair_dir / "bins_source.csv.gz")
            bins_target = pd.read_csv(pair_dir / "bins_target.csv.gz")
            saved = coupling.method(method)
            differing = int(np.sum(saved["source_gate"] != bins_source[f"{p}_retained"].to_numpy().astype(bool))
                            + np.sum(saved["target_gate"] != bins_target[f"{p}_retained"].to_numpy().astype(bool)))
            add(method, "gates", "saved gates vs bins tables, bins differing", differing, TOLERANCE["gates"])

    # 5. masses.
    m4e = fits["M4-E"]
    active = full["M4-E"] * (m4e.source_gate[:, None] & m4e.target_gate[None, :])
    bins_source = pd.read_csv(pair_dir / "bins_source.csv.gz")
    bins_target = pd.read_csv(pair_dir / "bins_target.csv.gz")
    for method, matrix, summary, keys, columns in (
        ("M4-E", active, keep["co"], ("source_mass", "target_mass"), "co_partner_mass"),
        ("M4-E", full["M4-E"], keep["co"], ("source_total_mass", "target_total_mass"), "co_total_mass"),
        ("balanced", full["balanced"], keep["bot"], ("source_mass", "target_mass"), "bot_partner_mass"),
        ("balanced", full["balanced"], keep["bot"], ("source_total_mass", "target_total_mass"), "bot_total_mass"),
    ):
        for side, found, key, table in (("row", matrix.sum(axis=1), keys[0], bins_source),
                                        ("column", matrix.sum(axis=0), keys[1], bins_target)):
            add(method, "masses", f"{side} masses vs in-memory {key}, of max", of_max(found, summary[key]),
                TOLERANCE["masses_memory"])
            add(method, "masses", f"{side} masses vs bins table {columns}", entrywise(found, table[columns]),
                TOLERANCE["masses_table"])

    # 6. annotation transitions and 7. distributions.
    transitions = pd.read_csv(pair_dir / "annotation_transitions.csv.gz")
    balanced = full["balanced"]
    for direction, name, matrix, rows_mask in (
        ("forward", "co", active, m4e.source_gate), ("forward", "bot", balanced, np.ones(n, dtype=bool)),
        ("forward", "bot_co_rejected", balanced, ~m4e.source_gate),
        ("backward", "co", active, m4e.target_gate), ("backward", "bot", balanced, np.ones(m, dtype=bool)),
        ("backward", "bot_co_rejected", balanced, ~m4e.target_gate),
    ):
        if direction == "forward":
            mass, by_group, own_hot, own_categories, partner_categories, weights = (
                matrix.sum(axis=1), matrix @ target_hot, source_hot, source_categories, target_categories, a)
        else:
            mass, by_group, own_hot, own_categories, partner_categories, weights = (
                matrix.sum(axis=0), matrix.T @ source_hot, target_hot, target_categories, source_categories, b)
        usable = rows_mask & (mass > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            distribution = np.where(mass[:, None] > 0, by_group / mass[:, None], 0.0)
        nominal = own_hot[usable].T @ (weights[usable, None] * distribution[usable])
        raw = own_hot.T @ by_group
        saved = transitions[(transitions["direction"] == direction) & (transitions["method"] == name)]
        own_index = {label: position for position, label in enumerate(own_categories)}
        partner_index = {label: position for position, label in enumerate(partner_categories)}
        ai = saved["annotation"].map(own_index).to_numpy()
        bi = saved["partner_annotation"].map(partner_index).to_numpy()
        add("M4-E" if name == "co" else "balanced", "transitions", f"{direction} {name} nominal_mass",
            entrywise(nominal[ai, bi], saved["nominal_mass"]), TOLERANCE["nominal_mass"], f"{len(saved)} rows")
        if saved["raw_coupling_mass"].notna().any():
            add("M4-E" if name == "co" else "balanced", "transitions", f"{direction} {name} raw_coupling_mass",
                entrywise(raw[ai, bi], saved["raw_coupling_mass"]), TOLERANCE["raw_coupling_mass"], f"{len(saved)} rows")
        if name in ("co", "bot"):
            side = "source" if direction == "forward" else "target"
            with np.load(pair_dir / f"transport_{side}.npz") as stored:
                saved_distribution = stored[f"{name}_partner_distribution"].astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                mine = np.where(mass[:, None] > 0, by_group / mass[:, None], np.nan)
            both = np.isfinite(mine) & np.isfinite(saved_distribution)
            mismatch = int((np.isfinite(mine) != np.isfinite(saved_distribution)).sum())
            error = float(np.max(np.abs(mine[both] - saved_distribution[both]), initial=0.0)) if mismatch == 0 else float("inf")
            add("M4-E" if name == "co" else "balanced", "distributions",
                f"transport_{side}.npz {name}_partner_distribution, absolute", error, TOLERANCE["distributions"])
    for record in results:
        record["pair_run_seconds"] = run_seconds
    return results


def report_markdown(report: pd.DataFrame, summary: dict) -> str:
    lines = [
        "# Coupling reconstruction check", "",
        f"Root: `{summary['root']}`  ",
        f"Preprocessing: `{summary['preprocessing']}`, device `{summary['device']}`, commit `{summary['commit']}`  ",
        f"Outcome: **{'PASS' if summary['pass'] else 'FAIL'}**, {summary['checks']} checks, {summary['failed']} failed", "",
        "Test pairs (fixed rule: the two smallest adjacent and the two smallest same-stage pairs):", "",
        "| pair | kind | source bins | target bins |", "|---|---|---|---|",
    ]
    for pair in summary["pairs"]:
        lines.append(f"| {pair['pair_id']} | {pair['kind']} | {pair['source_bins']} | {pair['target_bins']} |")
    lines += ["", "Largest error per check and quantity, over every test pair:", "",
              "| check | method | quantity | largest error | tolerance | pass |", "|---|---|---|---|---|---|"]
    grouped = report.groupby(["check", "method", "quantity"], sort=False).agg(
        error=("error", "max"), tolerance=("tolerance", "first"), passed=("pass", "all")).reset_index()
    for _, item in grouped.iterrows():
        lines.append(f"| {item['check']} | {item['method']} | {item['quantity']} | {item['error']:.3g} | "
                     f"{item['tolerance']:.0e} | {'yes' if item['passed'] else '**no**'} |")
    lines += ["", "Every row of report.csv is one pair, method and quantity; definitions and tolerances are in "
              "the docstring of mouse_embryo/26_verify_coupling_reconstruction.py."]
    return "\n".join(lines) + "\n"


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261028, help="Seed of the random subsets")
    args = parser.parse_args()
    from mosta_common import PREPROCESSING_LABEL

    layout = Layout(args.root)
    output = layout.root / "coupling_verification"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("COUPLING_PASS", "COUPLING_FAIL"):
        (output / name).unlink(missing_ok=True)
    tests = test_pairs(layout)
    started = time.perf_counter()
    rows = []
    for _, row in tests.iterrows():
        print(f"=== {row['pair_id']} ({row['kind']}, {int(row['elements'])} elements)", flush=True)
        rows += verify_pair(layout, row, output, args.device, args.seed + int(row["index"]))
        failed = [item for item in rows if item["pair_id"] == row["pair_id"] and not item["pass"]]
        print(f"=== {row['pair_id']}: {len(failed)} failed checks", flush=True)
    report = pd.DataFrame(rows)
    report.to_csv(output / "report.csv", index=False)
    summary = {
        "root": str(layout.root), "preprocessing": PREPROCESSING_LABEL, "device": args.device,
        "commit": os.environ.get("CONFIDENCEOT_COMMIT", "unknown"),
        "pairs": report.drop_duplicates("pair_id")[["pair_id", "kind", "source_bins", "target_bins"]].to_dict("records"),
        "checks": int(len(report)), "failed": int((~report["pass"]).sum()),
        "largest_error_by_check": report.groupby("check")["error"].max().to_dict(),
        "tolerances": TOLERANCE, "pass": bool(report["pass"].all()),
        "seconds": time.perf_counter() - started,
    }
    write_json(output / "summary.json", summary)
    (output / "REPORT.md").write_text(report_markdown(report, summary), encoding="utf-8")
    (output / ("COUPLING_PASS" if summary["pass"] else "COUPLING_FAIL")).write_text(
        time.strftime("%Y-%m-%dT%H:%M:%S") + "\n", encoding="utf-8")
    print((output / "REPORT.md").read_text(encoding="utf-8"))
    sys.exit(0 if summary["pass"] else 1)


if __name__ == "__main__":
    main()
