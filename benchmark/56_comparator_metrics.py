"""Three separate evaluations of the comparators, from the scores already saved.

``50_solver_comparison.py`` saves, for every pair, each solver's cell-level
unmatchedness on both sides (``score_<solver>_<side>.npy``),

    u_i = clip(1 - transported mass of cell i / nominal mass of cell i, 0, 1),

with the nominal mass 1/n on a side of n cells. That and the condition's
truth table are all this file reads; no solver is rerun. The three
evaluations are written to three tables and never pooled.

**1. Fixed operating point** (``set1_fixed_operating_point.csv``). No ground
truth is used to decide anything. Vanilla UOT and Partial OT (m = 0.85)
reject a cell with u > 0.5 -- more than half its own mass untransported.
ConfidenceOT keeps its native M4-E gate. Balanced OT transports every unit of
mass, so it rejects nothing by construction; no threshold is applied to it,
and the largest u it saved is recorded only as a check that its plan really
did transport everything.

**2. Score discrimination, oracle / upper bound** (``set2_oracle_threshold.csv``).
The continuous u of Vanilla UOT and Partial OT (m = 0.85): ROC-AUC, PR-AUC,
and precision, recall and F1 at the max-F1 threshold. That threshold is
chosen per pair with the simulation's ground truth, so it is an upper bound
on the score and not a rule anyone could apply to real data. One rule only,
max F1. Defined where a side holds both classes: the source in
population_lost and the target in population_emerged.

**3. Partial OT, fixed budget against oracle budget** (``set3_partial_ot_budget.csv``).
Fixed budget is m = 0.85, chosen without the truth. Oracle budget is m set
from the truth -- 1 in all_shared, 1 minus the source's unmatched mass
fraction in population_lost, 1 minus the target's in population_emerged, 0
in populations_disjoint -- and is read when ``50_solver_comparison.py
--oracle-budget`` has been run. It is not a method anyone could use, and the
table names it so. For each budget, the mass that was not transported is
split by population:

    precision-like = untransported mass of the true unmatched population
                     / total untransported mass
    recall-like    = untransported mass of the true unmatched population
                     / nominal mass of the true unmatched population

on the source in population_lost and on the target in population_emerged;
both are 1 at best. Partial OT never transports more than a cell's own mass,
so u times that mass is exactly the cell's untransported mass and the
coupling is not needed. In populations_disjoint the readout is the total
transported mass, which should be 0; all_shared has no unmatched population.

    python benchmark/56_comparator_metrics.py BENCH_ROOT OUT_DIR
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comparator_rejection import FIXED_CUTOFF, areas, oracle_cutoffs  # noqa: E402

ARMS = ("ranknm256_noscale_ds_cos", "ranknm256_noscale_cos", "raw")
CASES = ("all_shared", "population_lost", "population_emerged", "populations_disjoint")
PAIR = re.compile(r"^N(\d+)_rep(\d+)_(L\d_[a-z_]+?)_(" + "|".join(CASES) + r")$")
FIXED_PARTIAL = "Partial OT m=0.85"
ORACLE_PARTIAL = "Oracle-budget Partial OT"
# The side that holds the unmatched population, where it is one side.
UNMATCHED_SIDE = {"population_lost": "source", "population_emerged": "target"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench_root", type=Path,
                        help="Holds conditions/ (the truth tables) and the solver outputs")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--arm", action="append", default=None,
                        help=f"Repeat; default {', '.join(ARMS)}")
    parser.add_argument("--solvers-dir", default="solvers",
                        help="Under bench_root: <arm>/<pair_id>/ from solvers.slurm")
    parser.add_argument("--oracle-dir", default="solvers_oracle_budget",
                        help="Under bench_root: the --oracle-budget runs, if any")
    return parser.parse_args()


def score_file(directory: Path, solver: str, side: str) -> Path:
    # The name 50_solver_comparison.py saves it under.
    return directory / f"score_{solver.replace(' ', '_').replace('=', '')}_{side}.npy"


def counts(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    return {"n_cells": int(truth.size), "n_should_reject": int(truth.sum()),
            "rejected_n": int(predicted.sum()),
            "rejected_fraction": float(predicted.mean()),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            # 0 when nothing true is rejected, undefined only with nothing to reject.
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else np.nan,
            "false_rejection_rate": fp / (fp + tn) if fp + tn else np.nan}


def composition(u: np.ndarray, truth: np.ndarray) -> dict:
    mass = np.full(u.size, 1.0 / u.size)  # nominal mass, 1/n per cell
    untransported = u * mass
    return {"transported_mass": float(1.0 - untransported.sum()),
            "precision_like": float(untransported[truth].sum() / untransported.sum())
            if untransported.sum() > 0 else np.nan,
            "recall_like": float(untransported[truth].sum() / mass[truth].sum())
            if truth.any() else np.nan}


def global_max_f1(scores: list, truths: list) -> dict:
    """The one cutoff, u > tau, with the largest F1 over a whole set of pair-sides.

    Every pair-side weighs the same, whatever its number of cells, so the
    N = 10,000 pairs do not set the cutoff for the N = 1,000 ones. Every
    distinct score is a candidate, plus rejecting everything; ties go to the
    larger cutoff, which rejects less, as in comparator_rejection.py. It is
    chosen with the truth: an oracle, an upper bound, not a usable rule.
    """
    u = np.concatenate(scores)
    t = np.concatenate(truths).astype(bool)
    w = np.concatenate([np.full(s.size, 1.0 / s.size) for s in scores])
    order = np.argsort(-u, kind="stable")
    u, t, w = u[order], t[order], w[order]
    ends = np.append(np.flatnonzero(np.diff(u)), u.size - 1)
    tp = np.cumsum(w * t)[ends]
    called = np.cumsum(w)[ends]
    fp = called - tp
    fn = float((w * t).sum()) - tp
    f1 = 2 * tp / (2 * tp + fp + fn)
    best = int(np.argmax(f1))  # the first maximum rejects the fewest cells
    tau = float(u[ends[best] + 1]) if ends[best] + 1 < u.size else float("-inf")
    return {"threshold": tau, "pooled_f1": float(f1[best]),
            "pooled_precision": float(tp[best] / called[best]),
            "pooled_recall": float(tp[best] / (tp[best] + fn[best]))}


def two_strategies(set1: list, optimal: list) -> str:
    """Rejected fraction per case and side, and directional F1, by strategy."""
    rows = pd.DataFrame(set1 + optimal)
    rows["strategy"] = np.where(rows["decision"].str.contains("oracle"),
                                "optimal threshold (oracle)",
                                np.where(rows["solver"].isin(["ConfidenceOT", "Balanced OT"]),
                                         "own rule", "fixed u > 0.5"))
    table = (rows.groupby(["arm", "solver", "strategy", "biological_case", "side"])
             ["rejected_fraction"].mean().unstack(["biological_case", "side"]).round(2))
    # Directional F1: the side holding the cells to reject, as in 55_.
    directional = rows[[(case, side) in (("population_lost", "source"),
                                         ("population_emerged", "target"),
                                         ("populations_disjoint", "source"))
                        for case, side in zip(rows["biological_case"], rows["side"])]]
    table[("directional", "F1")] = (directional.groupby(["arm", "solver", "strategy"])["f1"]
                                    .mean().round(2))
    return table.to_string()


def main() -> None:
    args = parse_args()
    arms = args.arm or list(ARMS)
    set1, set2, set3 = [], [], []
    pooled = {}  # (arm, solver) -> [(row, u, truth)] for the one-threshold sweep
    for arm in arms:
        pairs = sorted((args.bench_root / args.solvers_dir / arm).glob("*/solver_comparison.csv"))
        if not pairs:
            print(f"{arm}: no solver outputs under {args.bench_root / args.solvers_dir / arm}")
        for table_path in pairs:
            directory = table_path.parent
            pair_id = directory.name
            match = PAIR.match(pair_id)
            if match is None:
                raise SystemExit(f"cannot read the condition of {pair_id!r}")
            n_cells, replicate, level, case = match.groups()
            truth_path = (args.bench_root / "conditions" / f"N{n_cells}" / f"rep{replicate}"
                          / level / pair_id / "truth.csv.gz")
            truth = pd.read_csv(truth_path)
            should = {side: truth.loc[truth["side"] == side, "should_reject"]
                      .to_numpy(dtype=bool) for side in ("source", "target")}
            saved = pd.read_csv(table_path)
            base = {"arm": arm, "pair_id": pair_id, "n_cells_nominal": int(n_cells),
                    "replicate": int(replicate), "technical_level": level,
                    "biological_case": case}

            for side in ("source", "target"):
                # ConfidenceOT: its own gate, as 50_ scored it from the stored run.
                gate = saved[(saved["method"] == "ConfidenceOT M4-E") & (saved["side"] == side)]
                if len(gate) == 1:
                    row = gate.iloc[0]
                    tp, fp, fn, tn = (int(row[f"fixed_{c}"]) for c in ("tp", "fp", "fn", "tn"))
                    predicted = np.r_[np.ones(tp + fp, bool), np.zeros(fn + tn, bool)]
                    truth_order = np.r_[np.ones(tp, bool), np.zeros(fp, bool),
                                        np.ones(fn, bool), np.zeros(tn, bool)]
                    set1.append({**base, "solver": "ConfidenceOT", "side": side,
                                 "decision": "native M4-E gate",
                                 **counts(predicted, truth_order)})
                for solver in ("Vanilla UOT", FIXED_PARTIAL):
                    u = np.load(score_file(directory, solver, side))
                    if u.size != should[side].size:
                        raise SystemExit(f"{directory}: {solver} {side} has {u.size} scores "
                                         f"against {should[side].size} truth rows")
                    result = counts(u > FIXED_CUTOFF, should[side])
                    stored = saved[(saved["method"] == solver) & (saved["side"] == side)].iloc[0]
                    if (result["tp"], result["fp"]) != (int(stored["fixed_tp"]), int(stored["fixed_fp"])):
                        raise SystemExit(f"{directory}: {solver} {side} does not reproduce the "
                                         f"stored fixed-cutoff counts")
                    set1.append({**base, "solver": solver, "side": side,
                                 "decision": f"u > {FIXED_CUTOFF}", **result})
                    pooled.setdefault((arm, solver), []).append(
                        ({**base, "solver": solver, "side": side}, u, should[side]))
                    if UNMATCHED_SIDE.get(case) == side:
                        area = areas(u, should[side])
                        oracle = oracle_cutoffs(u, should[side])
                        set2.append({**base, "solver": solver, "side": side,
                                     "analysis": "oracle / upper bound",
                                     "roc_auc": area["roc_auc"], "pr_auc": area["pr_auc"],
                                     "max_f1_threshold": oracle.get("oracle_f1_cutoff", np.nan),
                                     "precision": oracle.get("oracle_f1_precision", np.nan),
                                     "recall": oracle.get("oracle_f1_recall", np.nan),
                                     "f1": oracle.get("oracle_f1_f1", np.nan)})
                # The oracle-budget run, under its own name, where it exists.
                oracle_directory = args.bench_root / args.oracle_dir / arm / pair_id
                if (oracle_directory / "solver_comparison.csv").exists():
                    u = np.load(score_file(oracle_directory, ORACLE_PARTIAL, side))
                    set1.append({**base, "solver": ORACLE_PARTIAL, "side": side,
                                 "decision": f"u > {FIXED_CUTOFF}, m from the truth",
                                 **counts(u > FIXED_CUTOFF, should[side])})
                    pooled.setdefault((arm, ORACLE_PARTIAL), []).append(
                        ({**base, "solver": ORACLE_PARTIAL, "side": side}, u, should[side]))
                    if UNMATCHED_SIDE.get(case) == side:
                        area = areas(u, should[side])
                        oracle = oracle_cutoffs(u, should[side])
                        set2.append({**base, "solver": ORACLE_PARTIAL, "side": side,
                                     "analysis": "oracle / upper bound",
                                     "roc_auc": area["roc_auc"], "pr_auc": area["pr_auc"],
                                     "max_f1_threshold": oracle.get("oracle_f1_cutoff", np.nan),
                                     "precision": oracle.get("oracle_f1_precision", np.nan),
                                     "recall": oracle.get("oracle_f1_recall", np.nan),
                                     "f1": oracle.get("oracle_f1_f1", np.nan)})
                balanced = np.load(score_file(directory, "Traditional OT", side))
                set1.append({**base, "solver": "Balanced OT", "side": side,
                             "decision": "none: every unit of mass is transported",
                             "largest_saved_u": float(balanced.max()),
                             **counts(np.zeros(should[side].size, bool), should[side])})

            budgets = [("fixed budget, m = 0.85", directory, FIXED_PARTIAL, 0.85)]
            oracle_directory = args.bench_root / args.oracle_dir / arm / pair_id
            if (oracle_directory / "solver_comparison.csv").exists():
                m = {"all_shared": 1.0, "populations_disjoint": 0.0}.get(case)
                if m is None:
                    m = 1.0 - should[UNMATCHED_SIDE[case]].mean()
                budgets.append(("oracle budget", oracle_directory, ORACLE_PARTIAL, m))
            for budget, where, solver, m in budgets:
                record = {**base, "budget": budget, "m": float(m)}
                for side in ("source", "target"):
                    u = np.load(score_file(where, solver, side))
                    record[f"{side}_transported_mass"] = composition(u, should[side])["transported_mass"]
                side = UNMATCHED_SIDE.get(case)
                if side is not None:
                    u = np.load(score_file(where, solver, side))
                    part = composition(u, should[side])
                    record.update({"composition_side": side,
                                   "precision_like": part["precision_like"],
                                   "recall_like": part["recall_like"]})
                set3.append(record)

    optimal, thresholds = [], []
    for (arm, solver), entries in pooled.items():
        chosen = global_max_f1([u for _, u, _ in entries], [t for _, _, t in entries])
        thresholds.append({"arm": arm, "solver": solver, **chosen,
                           "pair_sides": len(entries), "analysis": "oracle / upper bound"})
        for row, u, truth in entries:
            optimal.append({**row, "decision": f"u > {chosen['threshold']:.4g}, one "
                            f"max-F1 threshold for the arm (oracle)",
                            "threshold": chosen["threshold"], **counts(u > chosen["threshold"], truth)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("set1_fixed_operating_point", set1),
                       ("set2_oracle_threshold", set2),
                       ("set3_partial_ot_budget", set3),
                       ("set2_one_threshold", optimal),
                       ("set2_one_threshold_values", thresholds)):
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows -> {args.output_dir / (name + '.csv')}")
    if optimal:
        print("\nthe one max-F1 threshold per arm and solver (oracle / upper bound):")
        print(pd.DataFrame(thresholds).round(4).to_string(index=False))
        print("\ntwo strategies, rejected fraction per case and side, mean over pairs:")
        print(two_strategies(set1, optimal))
    if set3:
        frame = pd.DataFrame(set3)
        print("\nset 3, mean over pairs:")
        print(frame.groupby(["arm", "budget", "biological_case"])
              [["m", "source_transported_mass", "target_transported_mass",
                "precision_like", "recall_like"]].mean().round(3).to_string())
    if set2:
        frame = pd.DataFrame(set2)
        print("\nset 2 (oracle / upper bound), mean over pairs:")
        print(frame.groupby(["arm", "solver", "biological_case"])
              [["roc_auc", "pr_auc", "precision", "recall", "f1"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
