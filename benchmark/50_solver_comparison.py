"""Four solvers on one cost matrix, scored by one rule.

The comparison this file makes is between **solvers**, so everything else has
to be the same object and not merely the same recipe: one manifest, one
preprocessing label, one call to ``prepare_joint_representation``, one cost
matrix. Three of the four solvers then come from POT and the fourth is ours.

**Why the comparators come from POT.** A method comparison whose competing
methods are the proposer's own code invites the reading that they were
crippled, and writing them carefully does not answer that -- only a reference
implementation does. ConfidenceOT stays ours because it is the thing under
test.

    Traditional OT   ot.emd                              exact, balanced
    Vanilla UOT      ot.unbalanced.sinkhorn_unbalanced   KL-penalised marginals
    Partial OT       ot.partial.partial_wasserstein      fixed transported mass
    ConfidenceOT     read from its own completed run

**ConfidenceOT is read, not recomputed.** Its gate comes from a calibration
that ``02_run_pair.py`` already performs, and reimplementing that here would
put two copies of it in the repository to drift apart. What has to be proved
instead is that this file's cost matrix is the one that run used, and that is
checkable: the run records ``cost_scale``, ``cost_median`` and ``cost_max``
in run.json, and this file rebuilds the cost and refuses to continue unless
all three agree. A silent geometry mismatch would make every comparison here
meaningless while looking entirely normal, so it is an error rather than a
warning. It runs before any solver does, since at N = 10,000 the POT solvers
take the better part of an hour and a wrong matrix would waste all of it.

To pass that check the cost is rebuilt along the runner's own path, not a
copy of it: the sides through ``load_exact_side``, every field of the label
forwarded to ``prepare_joint_representation`` (``equalise_depth`` included,
without which a _ds label rebuilds itself under the name of the arm without
it), the PCA seeded with ``seed + index`` and the scale drawn from
``default_rng(seed + index * 104729)``, with the seed the runs were made with.
The truth tables of the conditions name no cells; they are in the h5ad's
order, as ``40_score.py`` reads them, so the gate is aligned on the files'
own cell names.

**The EMD solvers must reach the optimum.** Traditional OT and Partial OT are
network simplex, and POT stops it at 100,000 iterations unless told
otherwise -- well short of the optimum at N = 10,000. A plan cut off there is
not the method's answer, so the cap is lifted out of the way and a plan POT
does not report as optimal fails the pair.

**POT's partial OT does not give a binary score.** This repository's own
``traditional_ot.partial`` solves an exact cardinality matching, so every
cell receives either its whole marginal or none of it and the unmatchedness
is {0, 1}. POT's ``partial_wasserstein`` goes through an EMD with dummy
points, which can split a cell's mass, so its unmatchedness is continuous
and the cutoff at 0.5 is a real choice for it too. The ``score_is_binary``
column records which of the two happened rather than assuming either.

**Partial OT's transported mass is swept, not chosen.** It is the one
parameter that decides that method's entire result -- at m = 0.85 it rejects
exactly 15% of cells by construction -- and handing it a value near the true
unmatched fraction is handing it the answer. Every value in the sweep is its
own row, so the reader sees the curve rather than a point somebody picked.

The rejection rule is ``benchmark/comparator_rejection.py``: one cell-level
unmatchedness score for the POT solvers, a fixed cutoff at 0.5 as the
deployable result, and oracle cutoffs reported separately as upper bounds.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "cancer_metastasis"))
sys.path.insert(0, str(REPO / "benchmark"))

from comparator_rejection import (  # noqa: E402
    score_side, summary_frame, summary_table, traditional_ot_score,
    unmatchedness,
)

# The scale the run must have used. Tighter than float32 would be unfair to a
# median over sampled pairs, looser would let a different geometry through.
GEOMETRY_TOLERANCE = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest_csv", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--preprocessing", required=True,
                        help="The locked configuration, by label")
    parser.add_argument(
        "--confidenceot-root", type=Path, default=None,
        help="Where 02_run_pair.py wrote this label's runs. Given, "
             "ConfidenceOT's gate is read from it and its geometry is "
             "checked against the cost rebuilt here; omitted, the three POT "
             "solvers are compared without it.")
    parser.add_argument("--scope", default="all")
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument(
        "--seed", type=int, default=20260925,
        help="The seed the ConfidenceOT runs were made with: ot_array.slurm's "
             "CONFIDENCEOT_BENCH_SEED, whose default this is. The PCA and the "
             "cost scale are both drawn from it, so any other seed builds a "
             "different matrix and the geometry check refuses it.")
    parser.add_argument("--epsilon", type=float, default=0.10)
    parser.add_argument("--reg-m", type=float, default=1.0,
                        help="Vanilla UOT's marginal penalty, POT's reg_m")
    parser.add_argument(
        "--uot-method", default="sinkhorn_stabilized",
        choices=("sinkhorn", "sinkhorn_stabilized", "sinkhorn_reg_scaling"),
        help="POT's unbalanced solver. Stabilised by default, and not as a "
             "precaution: the cost is median-normalised so its median is 1, "
             "but its maximum runs into the hundreds, and at epsilon = 0.1 "
             "that is exp(-3000) in the kernel. Plain sinkhorn underflows to "
             "a zero kernel, transports 2.7% of the mass and rejects every "
             "cell -- which would be reported as Vanilla UOT failing, when "
             "it is the solver failing. Stabilised, the same problem "
             "transports 58% and recovers the planted cells exactly.")
    parser.add_argument(
        "--partial-mass", type=float, action="append", default=None,
        help="Transported mass for Partial OT; repeat to sweep. Default "
             "0.99 0.95 0.90 0.85 0.80 0.70.")
    parser.add_argument(
        "--oracle-budget", action="store_true",
        help="Run only Partial OT, at the budget read off the truth: m = 1 in "
             "all_shared, 1 minus the unmatched mass fraction of the side that "
             "has it in population_lost and population_emerged, 0 in "
             "populations_disjoint. An upper-bound analysis, written as its own "
             "solver; give it its own output directory.")
    parser.add_argument("--max-iterations", type=int, default=20_000,
                        help="Sinkhorn iterations for Vanilla UOT")
    parser.add_argument(
        "--emd-iterations", type=int, default=1_000_000_000,
        help="Network-simplex iteration cap for Traditional OT and Partial OT. "
             "It only has to be large enough never to bind: the solver stops "
             "at the optimum. POT's own default of 100,000 stops well short "
             "of it at N = 10,000, and a Partial OT plan cut off there puts "
             "mass on the dummy point that the optimum would not, which "
             "changes which cells it rejects.")
    return parser.parse_args()


def side_paths(row: pd.Series, side: str) -> list[str]:
    """The files of one side, read the way 02_run_pair.py reads them."""
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(value) for value in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def geometry(row: pd.Series, args: argparse.Namespace):
    """The representation and cost, built the way 02_run_pair.py builds them.

    Every step is the runner's own call with the runner's own arguments: the
    sides through ``load_exact_side``, every field of the configuration
    forwarded, the PCA seeded with ``seed + index`` and the scale drawn from
    ``default_rng(seed + index * 104729)``. Rebuilt any other way the matrix
    comes out close but not equal, and the geometry check below is there to
    refuse exactly that. Returns the cost with both sides' cell names, in the
    order of the files, which is the order of the truth table.
    """
    from common import load_exact_side, prepare_joint_representation
    from confidenceot import squared_euclidean
    from confidenceot.preprocessing import Preprocessing, median_pair_scale

    configuration = Preprocessing.from_label(args.preprocessing)
    source = load_exact_side(side_paths(row, "source"), str(row["source_sample"]))
    target = load_exact_side(side_paths(row, "target"), str(row["target_sample"]))
    names = {"source": source.obs_names.astype(str).tolist(),
             "target": target.obs_names.astype(str).tolist()}
    source_pca, target_pca, _, recorded = prepare_joint_representation(
        source, target, n_hvg=args.n_hvg, n_pcs=args.n_pcs,
        seed=args.seed + args.index,
        representation=configuration.normalisation,
        rank_top_n=configuration.rank_top_n,
        cost=configuration.cost,
        # Recorded rather than applied: a _ds label reads counts that
        # 27_downsample_counts.py already equalised, and the flag is what
        # lets the representation carry the label's real name.
        equalise_depth=configuration.equalise_depth,
        regress_out=configuration.regress_out,
        regress_on_ranks=configuration.regress_on_ranks,
        scale_genes=configuration.scale_genes,
        label_stem=configuration.label_stem,
        allow_precomputed=args.preprocessing.startswith("raw"),
    )
    built = recorded.get("label")
    if built != args.preprocessing:
        raise SystemExit(
            f"asked for {args.preprocessing} and the representation rebuilt "
            f"itself as {built}")
    # cost='cosine' has already L2-normalised the rows inside
    # prepare_joint_representation, before the scale, as in the runner.
    rng = np.random.default_rng(args.seed + args.index * 104729)
    scale = median_pair_scale(source_pca, target_pca, rng=rng)
    cost = squared_euclidean(source_pca, target_pca) / scale
    return np.asarray(cost, dtype=np.float64), float(scale), names


def check_geometry(cost: np.ndarray, scale: float, run_directory: Path) -> dict:
    """Refuse a comparison whose cost is not the one ConfidenceOT saw."""
    record = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))
    stored = {key: record.get(key) for key in
              ("cost_scale", "cost_median", "cost_max")}
    if any(value is None for value in stored.values()):
        raise SystemExit(
            f"{run_directory / 'run.json'} does not record the cost geometry; "
            f"it predates the commit that added cost_scale, and the two cost "
            f"matrices cannot be shown to be the same one")
    here = {"cost_scale": scale, "cost_median": float(np.median(cost)),
            "cost_max": float(np.max(cost))}
    for key, value in here.items():
        reference = float(stored[key])
        if not np.isclose(value, reference, rtol=GEOMETRY_TOLERANCE, atol=0.0):
            raise SystemExit(
                f"{key} is {value:.9g} here and {reference:.9g} in the stored "
                f"run: the two are not the same cost matrix, so comparing the "
                f"solvers on them would compare the geometries instead")
    return here


def confidenceot_from_disk(run_directory: Path, truth_index: dict) -> dict:
    """The gate and the decision cost of a completed ConfidenceOT run."""
    gate = pd.read_csv(run_directory / "cell_confidence.csv")
    out = {}
    for side in ("source", "target"):
        rows = gate[gate["side"].eq(side) & gate["method"].eq("M4-E")]
        if rows.empty:
            raise SystemExit(f"no M4-E rows for {side} in {run_directory}")
        ordered = rows.set_index(rows["observation_id"].astype(str))
        names = truth_index[side]
        missing = [name for name in names if name not in ordered.index]
        if missing:
            raise SystemExit(
                f"{side}: {len(missing)} cells in the truth table are not in "
                f"the stored gate, first {missing[:4]}")
        ordered = ordered.loc[names]
        out[side] = {
            # The gate names the cells it keeps.
            "gate": ordered["retained"].to_numpy(dtype=bool),
            "score": pd.to_numeric(ordered["decision_cost"],
                                   errors="coerce").to_numpy(dtype=float),
        }
    return out


def main() -> None:
    args = parse_args()
    import ot
    import ot.partial
    import ot.unbalanced

    manifest = pd.read_csv(args.manifest_csv)
    row = manifest.iloc[args.index]
    pair_id = str(row["pair_id"])
    truth = pd.read_csv(row["truth_csv"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cost, scale, names = geometry(row, args)
    n, m = cost.shape
    a, b = np.full(n, 1.0 / n), np.full(m, 1.0 / m)
    sides, ids = {}, {}
    for side, expected in (("source", n), ("target", m)):
        block = truth[truth["side"].eq(side)]
        if len(block) != expected:
            raise SystemExit(
                f"{side}: {len(block)} truth rows against {expected} cells")
        sides[side] = block["should_reject"].to_numpy(dtype=bool)
        # The conditions' truth tables carry no cell names: they are written
        # in the h5ad's order, as 40_score.py reads them. Named or not, the
        # names used below are the files' own.
        if ("observation_id" in block
                and block["observation_id"].astype(str).tolist() != names[side]):
            raise SystemExit(
                f"{side}: the truth table names its cells in another order "
                f"than the h5ad")
        ids[side] = names[side]

    masses = args.partial_mass or [0.99, 0.95, 0.90, 0.85, 0.80, 0.70]
    scored = []
    provenance = {"pair_id": pair_id, "preprocessing": args.preprocessing,
                  "seed": args.seed, "cost_scale": scale, "n_source": int(n),
                  "n_target": int(m), "pot_version": ot.__version__}

    # Before any solver runs: the POT solvers take hours at N = 10,000, and a
    # cost that is not the run's would make all of it worthless.
    run_directory = None
    if args.confidenceot_root is not None:
        found = sorted((args.confidenceot_root / pair_id /
                        f"scope_{args.scope}").glob("budget_*"))
        ran = [path for path in found
               if (path / "cell_confidence.csv").exists()]
        if not ran:
            raise SystemExit(
                f"no completed ConfidenceOT run under "
                f"{args.confidenceot_root / pair_id}")
        run_directory = ran[-1]
        provenance["geometry_check"] = check_geometry(cost, scale, run_directory)
        provenance["confidenceot_run"] = str(run_directory)

    def add(method: str, plan=None, gate=None, score=None,
            kind="mass_deficit") -> None:
        for side, marginal in (("source", a), ("target", b)):
            if score is not None:
                value = score[side]
            elif plan is not None:
                value = unmatchedness(plan, marginal, side=side)
            else:
                value = traditional_ot_score(len(marginal))
            result = score_side(
                method=method, side=side, truth=sides[side], score=value,
                score_kind=kind,
                native_gate=None if gate is None else gate[side])
            record = dict(result.summary)
            record.update({"pair_id": pair_id,
                           "preprocessing": args.preprocessing})
            scored.append(record)
            np.save(args.output_dir /
                    f"score_{method.replace(' ', '_').replace('=', '')}"
                    f"_{side}.npy", result.score)

    if args.oracle_budget:
        # The budget read off the truth: the mass Partial OT would have to be
        # told to leave behind. Only the simulation knows it, so this is an
        # upper-bound analysis and never a method; nothing else is run with it.
        case = str(row["biological_case"])
        m = {"all_shared": 1.0, "populations_disjoint": 0.0,
             "population_lost": 1.0 - float(sides["source"].mean()),
             "population_emerged": 1.0 - float(sides["target"].mean())}[case]
        provenance["oracle_budget"] = {"biological_case": case, "m": m}
        try:
            plan = ot.partial.partial_wasserstein(
                a, b, cost, m=m, numItermax=args.emd_iterations)
        except ValueError as error:
            raise SystemExit(f"Oracle-budget Partial OT m={m:.4f}: {error}") from error
        provenance["oracle_budget"]["transported_mass"] = float(plan.sum())
        add("Oracle-budget Partial OT", plan=plan)
        del plan
    else:
        plan, record = ot.emd(a, b, cost, numItermax=args.emd_iterations, log=True)
        # result_code 1 is POT's "optimal". Anything else is a plan the network
        # simplex stopped on, and it is not the method's answer.
        provenance["traditional_ot"] = {"result_code": int(record["result_code"]),
                                        "warning": record["warning"]}
        if int(record["result_code"]) != 1:
            raise SystemExit(f"Traditional OT did not reach the optimum: "
                             f"{record['warning']}")
        add("Traditional OT", plan=plan)
        del plan
        vanilla, record = ot.unbalanced.sinkhorn_unbalanced(
            a, b, cost, reg=args.epsilon, reg_m=args.reg_m,
            method=args.uot_method, numItermax=args.max_iterations, log=True)
        errors = record.get("err", [])
        # A solver that underflowed returns a plan carrying almost no mass, and
        # every cell then reads as unmatched. That is a failure to report, not a
        # rejection rate to publish, so it is recorded and flagged rather than
        # scored in silence.
        provenance["vanilla_uot"] = {
            "method": args.uot_method,
            "transported_mass": float(vanilla.sum()),
            "finite": bool(np.isfinite(vanilla).all()),
            "error_checks": len(errors),
            "last_error": float(errors[-1]) if len(errors) else None,
        }
        if not np.isfinite(vanilla).all() or vanilla.sum() < 1e-3:
            provenance["vanilla_uot"]["degenerate"] = True
            print(f"WARNING: the unbalanced solver returned "
                  f"{vanilla.sum():.3g} total mass; its rejection rate is a "
                  f"solver failure, not a result")
        add("Vanilla UOT", plan=vanilla)
        del vanilla
        for mass in masses:
            try:
                plan = ot.partial.partial_wasserstein(
                    a, b, cost, m=float(mass), numItermax=args.emd_iterations)
            except ValueError as error:
                # POT raises this for any warning from the EMD inside, including
                # the iteration cap, whatever the message says about dummies.
                raise SystemExit(f"Partial OT m={mass:.2f}: {error}") from error
            add(f"Partial OT m={mass:.2f}", plan=plan)
            del plan

        if run_directory is not None:
            stored = confidenceot_from_disk(run_directory, ids)
            add("ConfidenceOT M4-E",
                gate={side: stored[side]["gate"] for side in stored},
                score={side: stored[side]["score"] for side in stored},
                kind="decision_cost")

    # Both helpers read .summary; the records carry the pair and the label
    # besides, so they are wrapped rather than rebuilt from the SideScores.
    wrapped = [SimpleNamespace(summary=record) for record in scored]
    frame = summary_frame(wrapped)
    frame.to_csv(args.output_dir / "solver_comparison.csv", index=False)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")
    print(summary_table(wrapped))
    print()
    print(f"wrote {args.output_dir / 'solver_comparison.csv'}")


if __name__ == "__main__":
    main()
