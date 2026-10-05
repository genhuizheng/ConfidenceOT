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

**The main comparison** (``main_*.csv``) is four methods at three arms.
Partial OT in it is the truth-derived-m run read from ``--oracle-dir``, shown
as "Partial OT (truth-derived m)"; the fixed m = 0.85 run stays in sets 1-3
for reference and is not part of it. Six strategies, as hard decisions per
pair and side (``main_decisions_per_pair.csv``): ConfidenceOT's native gate,
Vanilla UOT and Partial OT each at the fixed cutoff u > 0.5 and at the
global-optimal threshold, and Balanced OT rejecting nothing.

* **Local-optimal F1**: per pair, the max-F1 threshold chosen with that
  pair's truth, then the pair F1s averaged. Pair-specific, an upper bound.
* **Global-optimal threshold**: one per method and arm, the cutoff that
  maximises the mean over pairs of each pair's own F1, using only the sides
  holding both classes (the source in population_lost, the target in
  population_emerged), so every pair counts once and cells are never pooled.
  It is then frozen and applied unchanged to every size, replicate, level,
  case and side. Chosen with the truth: an upper bound.
* **ROC-AUC, PR-AUC**: UOT and Partial OT only, on those same sides.
* **Hard rejected nominal mass fraction** R_hard = sum_i a_i 1[rejected_i] /
  sum_i a_i, from each strategy's final hard decision, for all four methods
  (``hard_rejected_mass_fraction``, with ``retained_mass_fraction`` = 1 -
  R_hard). With every cell weighing 1/n it equals the rejected-cell fraction,
  and it is the one rejection quantity comparable across methods.
* **Untransported plan mass** 1 - sum(pi), the source mass being 1, read
  from each pair's provenance.json (``untransported_plan_mass``). It is a
  property of the transport plan, not of any decision, and is not a rejection
  rate. For Partial OT it equals 1 - m, the budget the truth supplied, which
  is kept separately as ``partial_m``.

``audit_emerged_source.csv`` lists, for every population_emerged pair, the
source side under ConfidenceOT's gate and under Partial OT (truth-derived m):
source cells, those that should be rejected (none), those rejected, the
rejected-cell fraction, R_hard, the retained mass fraction and the false
rejection rate -- the quantities that must agree before any claim about why
the source is rejected.

ConfidenceOT and Balanced OT get no score, no AUC and no threshold:
``main_scores_per_pair.csv`` and ``main_score_quality.csv`` hold UOT and
Partial OT only.

    python benchmark/56_comparator_metrics.py BENCH_ROOT OUT_DIR
"""

from __future__ import annotations

import argparse
import json
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
# The two scored methods of the main comparison, by the name their outputs are
# saved under and the name they are shown under. The fixed m = 0.85 run stays
# in sets 1-3 for reference and is not part of it.
MAIN_SCORED = {"Vanilla UOT": "Vanilla UOT", ORACLE_PARTIAL: "Partial OT (truth-derived m)"}
STRATEGIES = [
    "ConfidenceOT: native gate",
    "Vanilla UOT: fixed cutoff (u > 0.5)",
    "Vanilla UOT: global-optimal threshold",
    "Partial OT (truth-derived m): fixed cutoff (u > 0.5)",
    "Partial OT (truth-derived m): global-optimal threshold",
    "Balanced OT: no rejection",
]
# Directional F1 reads the side holding the cells to reject; all_shared has none.
DIRECTIONAL = {"population_lost": "source", "population_emerged": "target",
               "populations_disjoint": "source"}
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
    parser.add_argument("--heatmap-metrics", type=Path, default=None,
                        help="The arm heatmap's benchmark_metrics.csv, to set its source "
                             "false rejection rate beside the emerged-source audit; "
                             "default bench_root/benchmark_metrics.csv when it exists")
    return parser.parse_args()


AUDITED = ["ConfidenceOT: native gate",
           "Partial OT (truth-derived m): fixed cutoff (u > 0.5)",
           "Partial OT (truth-derived m): global-optimal threshold"]


def audit_emerged_source(decisions: pd.DataFrame, heatmap: Path | None) -> pd.DataFrame:
    """The source side of every population_emerged pair, quantity by quantity."""
    part = decisions[(decisions["biological_case"] == "population_emerged")
                     & (decisions["side"] == "source")
                     & decisions["strategy"].isin(AUDITED)]
    columns = ["strategy", "arm", "pair_id", "n_cells_nominal", "replicate", "technical_level",
               "n_cells", "n_should_reject", "rejected_n", "rejected_fraction",
               "hard_rejected_mass_fraction", "retained_mass_fraction", "false_rejection_rate"]
    audit = part[columns].rename(columns={"n_cells": "source_cells",
                                          "n_should_reject": "source_should_reject",
                                          "rejected_n": "source_rejected"})
    if heatmap is not None and heatmap.exists():
        reference = pd.read_csv(heatmap)
        reference = reference[reference["method"] == "M4-E"][
            ["preprocessing_label", "pair_id", "source_false_rejection_rate"]].rename(
            columns={"preprocessing_label": "arm",
                     "source_false_rejection_rate": "heatmap_source_false_rejection_rate"})
        audit = audit.merge(reference, on=["arm", "pair_id"], how="left")
        audit.loc[audit["strategy"] != AUDITED[0], "heatmap_source_false_rejection_rate"] = np.nan
    return audit.sort_values(["strategy", "arm", "pair_id"]).reset_index(drop=True)


def score_file(directory: Path, solver: str, side: str) -> Path:
    # The name 50_solver_comparison.py saves it under.
    return directory / f"score_{solver.replace(' ', '_').replace('=', '')}_{side}.npy"


def counts(predicted: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(np.sum(predicted & truth))
    fp = int(np.sum(predicted & ~truth))
    fn = int(np.sum(~predicted & truth))
    tn = int(np.sum(~predicted & ~truth))
    # Each cell's nominal mass is 1/n on its side; R_hard is the share of it
    # that the hard decision rejected.
    mass = np.full(truth.size, 1.0 / truth.size)
    hard = float(mass[predicted].sum() / mass.sum())
    return {"n_cells": int(truth.size), "n_should_reject": int(truth.sum()),
            "rejected_n": int(predicted.sum()),
            "rejected_fraction": float(predicted.mean()),
            "hard_rejected_mass_fraction": hard, "retained_mass_fraction": 1.0 - hard,
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


def global_macro_f1(entries: list) -> dict:
    """One cutoff for a method and arm: the u > tau with the largest mean pair F1.

    Only the sides holding both classes take part -- the source in
    population_lost, the target in population_emerged. For every candidate,
    F1 is computed within each of those pairs and the pairs are averaged, so
    each pair counts once whatever its size and cells are never pooled across
    pairs. Candidates are every distinct u on those sides plus rejecting
    everything; ties go to the larger cutoff, which rejects less. It is chosen
    with the truth -- an upper bound -- and is then frozen and applied
    unchanged to every pair, case and side.
    """
    eligible = [(u, t) for row, u, t in entries
                if UNMATCHED_SIDE.get(row["biological_case"]) == row["side"]]
    if not eligible:
        return {"threshold": np.nan, "macro_f1": np.nan, "pairs": 0}
    # High to low, so the first maximum is the largest cutoff.
    candidates = np.append(np.unique(np.concatenate([u for u, _ in eligible]))[::-1], -np.inf)
    total = np.zeros(candidates.size)
    for u, t in eligible:
        positive, negative = np.sort(u[t]), np.sort(u[~t])
        tp = positive.size - np.searchsorted(positive, candidates, side="right")
        fp = negative.size - np.searchsorted(negative, candidates, side="right")
        fn = positive.size - tp
        total += 2 * tp / (2 * tp + fp + fn)
    mean = total / len(eligible)
    best = int(np.argmax(mean))
    return {"threshold": float(candidates[best]), "macro_f1": float(mean[best]),
            "pairs": len(eligible)}


IDENTITY = ("arm", "pair_id", "n_cells_nominal", "replicate", "technical_level",
            "biological_case", "side")


def main_comparison(pooled: dict, set1: list) -> tuple:
    """Per-pair scores of UOT and Partial OT, and every strategy's decisions.

    Returns the per pair-side score table (UOT and truth-derived-m Partial OT
    only), the hard decisions of the six strategies, and the global
    thresholds.
    """
    scores, decisions, thresholds = [], [], []
    for (arm, solver), entries in pooled.items():
        method = MAIN_SCORED[solver]
        chosen = global_macro_f1(entries)
        thresholds.append({"arm": arm, "method": method, **chosen,
                           "analysis": "global truth-tuned threshold, upper bound"})
        for row, u, t in entries:
            ids = {key: row[key] for key in IDENTITY}
            fixed = counts(u > FIXED_CUTOFF, t)
            tuned = counts(u > chosen["threshold"], t)
            record = {**ids, "method": method,
                      "true_unmatched_fraction": float(t.mean()),
                      "untransported_plan_mass": row["untransported_plan_mass"],
                      "partial_m": row.get("partial_m", np.nan),
                      "both_classes": UNMATCHED_SIDE.get(row["biological_case"]) == row["side"]}
            if record["both_classes"]:
                area, local = areas(u, t), oracle_cutoffs(u, t)
                record.update({"roc_auc": area["roc_auc"], "pr_auc": area["pr_auc"],
                               "local_optimal_threshold": local.get("oracle_f1_cutoff", np.nan),
                               "local_optimal_f1": local.get("oracle_f1_f1", np.nan)})
            else:
                record.update({"roc_auc": np.nan, "pr_auc": np.nan,
                               "local_optimal_threshold": np.nan, "local_optimal_f1": np.nan})
            for prefix, result in (("global_optimal", tuned), ("fixed_cutoff", fixed)):
                for name in ("precision", "recall", "f1", "false_rejection_rate"):
                    record[f"{prefix}_{name}"] = result[name]
            record["global_optimal_threshold"] = chosen["threshold"]
            scores.append(record)
            decisions.append({**ids, "strategy": f"{method}: fixed cutoff (u > 0.5)", **fixed})
            decisions.append({**ids, "strategy": f"{method}: global-optimal threshold", **tuned})
    for row in set1:
        if row["solver"] == "ConfidenceOT":
            decisions.append({**{key: row[key] for key in IDENTITY},
                              "strategy": "ConfidenceOT: native gate",
                              **{key: row[key] for key in row if key not in IDENTITY
                                 and key not in ("solver", "decision")}})
        elif row["solver"] == "Balanced OT":
            decisions.append({**{key: row[key] for key in IDENTITY},
                              "strategy": "Balanced OT: no rejection",
                              **{key: row[key] for key in row if key not in IDENTITY
                                 and key not in ("solver", "decision", "largest_saved_u")}})
    return pd.DataFrame(scores), pd.DataFrame(decisions), pd.DataFrame(thresholds)


def score_quality(scores: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Output C: UOT and Partial OT only, on the sides holding both classes."""
    both = scores[scores["both_classes"]]
    table = both.groupby(["arm", "method"])[
        ["roc_auc", "pr_auc", "local_optimal_f1", "global_optimal_f1", "fixed_cutoff_f1"]].mean()
    table = table.join(thresholds.set_index(["arm", "method"])[["threshold"]]
                       .rename(columns={"threshold": "global_optimal_threshold"}))
    # The plan's untransported mass is one number per pair: read it off the source rows.
    mass = (scores[scores["side"] == "source"]
            .groupby(["arm", "method", "biological_case"])["untransported_plan_mass"].mean()
            .unstack("biological_case").add_prefix("untransported_plan_mass: "))
    return table.join(mass)


def decision_summary(decisions: pd.DataFrame) -> str:
    """Rejected fraction per case and side, and directional F1, per strategy."""
    table = (decisions.groupby(["strategy", "arm", "biological_case", "side"])
             ["rejected_fraction"].mean().unstack(["biological_case", "side"]))
    keep = [(case, side) in DIRECTIONAL.items()
            for case, side in zip(decisions["biological_case"], decisions["side"])]
    table[("directional", "F1")] = decisions[keep].groupby(["strategy", "arm"])["f1"].mean()
    order = [s for s in STRATEGIES if s in table.index.get_level_values(0)]
    return table.reindex(order, level=0).round(2).to_string()


def mass_rejection(pooled: dict, set1: list) -> pd.DataFrame:
    """The share of a set of cells' own mass that was not transported.

    No cutoff. Every cell weighs 1/n on its side, so for UOT and Partial OT
    it is the mean u over the set. ConfidenceOT's gate removes a rejected
    cell's whole mass, so its rejected fraction already is this quantity,
    split into unmatched and matched by its counts. Balanced OT transports
    everything: 0.
    """
    rows = []
    for (arm, solver), entries in pooled.items():
        for row, u, truth in entries:
            rows.append({**row, "all": float(u.mean()),
                         "unmatched": float(u[truth].mean()) if truth.any() else np.nan,
                         "matched": float(u[~truth].mean()) if (~truth).any() else np.nan})
    for row in set1:
        if row["solver"] == "ConfidenceOT":
            tp, fp, fn, tn = row["tp"], row["fp"], row["fn"], row["tn"]
            rows.append({key: row[key] for key in ("arm", "pair_id", "biological_case",
                                                   "technical_level", "side", "solver")}
                        | {"all": (tp + fp) / (tp + fp + fn + tn),
                           "unmatched": tp / (tp + fn) if tp + fn else np.nan,
                           "matched": fp / (fp + tn) if fp + tn else np.nan})
        elif row["solver"] == "Balanced OT":
            rows.append({key: row[key] for key in ("arm", "pair_id", "biological_case",
                                                   "technical_level", "side", "solver")}
                        | {"all": 0.0,
                           "unmatched": 0.0 if row["n_should_reject"] else np.nan,
                           "matched": 0.0 if row["n_should_reject"] < row["n_cells"] else np.nan})
    return pd.DataFrame(rows)


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
                    if solver == "Vanilla UOT":
                        # 1 - sum(pi): the source mass is normalised to 1.
                        provenance = json.loads((directory / "provenance.json")
                                                .read_text(encoding="utf-8"))
                        untransported = 1.0 - float(provenance["vanilla_uot"]["transported_mass"])
                        pooled.setdefault((arm, solver), []).append(
                            ({**base, "solver": solver, "side": side,
                              "untransported_plan_mass": untransported}, u, should[side]))
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
                    provenance = json.loads((oracle_directory / "provenance.json")
                                            .read_text(encoding="utf-8"))["oracle_budget"]
                    pooled.setdefault((arm, ORACLE_PARTIAL), []).append(
                        ({**base, "solver": ORACLE_PARTIAL, "side": side,
                          "untransported_plan_mass": 1.0 - float(provenance["transported_mass"]),
                          "partial_m": float(provenance["m"])}, u, should[side]))
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

    scores, decisions, thresholds = main_comparison(pooled, set1)
    mass = mass_rejection(pooled, set1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mass.to_csv(args.output_dir / "set1_mass_rejection.csv", index=False)
    heatmap = args.heatmap_metrics or args.bench_root / "benchmark_metrics.csv"
    audit = audit_emerged_source(decisions, heatmap) if len(decisions) else pd.DataFrame()
    if len(audit):
        audit.to_csv(args.output_dir / "audit_emerged_source.csv", index=False)
        print("\naudit, population_emerged source side, per strategy and arm "
              "(every pair should have source_should_reject = 0):")
        print(audit.groupby(["strategy", "arm"]).agg(
            pairs=("pair_id", "size"), should_reject=("source_should_reject", "sum"),
            rejected_fraction=("rejected_fraction", "mean"),
            hard_rejected_mass=("hard_rejected_mass_fraction", "mean"),
            false_rejection=("false_rejection_rate", "mean"),
            minimum=("hard_rejected_mass_fraction", "min"),
            maximum=("hard_rejected_mass_fraction", "max")).round(4).to_string())
        gaps = {"rejected fraction vs R_hard": (audit["rejected_fraction"]
                                                - audit["hard_rejected_mass_fraction"]).abs().max(),
                "R_hard vs false rejection rate": (audit["hard_rejected_mass_fraction"]
                                                   - audit["false_rejection_rate"]).abs().max()}
        if "heatmap_source_false_rejection_rate" in audit:
            gaps["ConfidenceOT vs the arm heatmap"] = (
                audit["false_rejection_rate"] - audit["heatmap_source_false_rejection_rate"]).abs().max()
        print("largest disagreement: " + "; ".join(f"{k} {v:.2g}" for k, v in gaps.items()))
    for name, rows in (("set1_fixed_operating_point", set1),
                       ("set2_oracle_threshold", set2),
                       ("set3_partial_ot_budget", set3)):
        frame = pd.DataFrame(rows)
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows -> {args.output_dir / (name + '.csv')}")
    # The main comparison: four methods, Partial OT with the truth-derived m only.
    for name, frame in (("main_scores_per_pair", scores),
                        ("main_decisions_per_pair", decisions),
                        ("main_global_thresholds", thresholds)):
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
        print(f"{name}: {len(frame)} rows -> {args.output_dir / (name + '.csv')}")
    if len(scores):
        quality = score_quality(scores, thresholds)
        quality.round(4).to_csv(args.output_dir / "main_score_quality.csv")
        print("\nscore quality, UOT and Partial OT (truth-derived m) only; AUCs and F1s on "
              "the sides holding both classes; the local and global thresholds are chosen "
              "with the truth (upper bounds):")
        print(quality.round(3).to_string())
        print("\nhard decisions per strategy: rejected fraction per case and side, and "
              "directional F1, mean over pairs:")
        print(decision_summary(decisions))
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
