"""How the five strategies behave without rejection, checked numerically on the project's solvers (local, synthetic).

    python lineage_validation/level2_pairwise_ot/strategy_check/check_strategy_identities.py --out DIR

Two representations of the same cells, "RNA" (x) and "CNV" (z), each giving a cost scaled by its median pair
cost (C_rna, C_cnv), as 02_run_pair.py does. epsilon 0.1. Solvers: Standard = traditional_ot balanced_ot
(BOT); Ours = ConfidenceOT(backbone="uot", variant="exact", rejection budget 0) (UOT).

1. Two-step as the pipeline defines it (step 1 gates the cells, step 2 runs on the retained cells):
   step-1 retention under budget 0 / BOT, and the step-2 plan against the single-step plan.
2. "Soft" two-step without rejection (step 2 regularised toward the step-1 plan P1 instead of the product of
   the marginals: min <C2, P> + eps KL(P | P1)), solved as OT on cost C2 - eps log P1:
   against single-step OT on C_rna + C_cnv, and against the shipped rna_cnv cost (C_rna + C_cnv) / s, s its
   median, which is the same plan at epsilon * s.
Writes DIR/strategy_identities.tsv.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.preprocessing import median_pair_scale, squared_euclidean  # noqa: E402
from traditional_ot.balanced import balanced_ot  # noqa: E402

EPS = 0.1


def bot(C, eps=EPS):
    r = balanced_ot(C, epsilon=eps, threshold=1e-12, max_iterations=100_000)
    return r.coupling, np.ones(C.shape[0], bool), np.ones(C.shape[1], bool)


def uot(C, eps=EPS):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=1.0, epsilon=eps, source_rejection_budget=0.0,
                         target_rejection_budget=0.0, tolerance=1e-12, max_iterations=100_000, device="cpu").fit(C)
    return r.coupling, r.source_gate, r.target_gate


def scaled_cost(a, b, seed):
    return squared_euclidean(a, b) / median_pair_scale(a, b, rng=np.random.default_rng(seed))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(5)
    n, m = 240, 160
    lab_s, lab_t = rng.integers(0, 3, n), rng.integers(0, 3, m)
    cx, cz = rng.normal(0, 2, (3, 8)), rng.normal(0, 2, (3, 6))
    xs, xt = cx[lab_s] + rng.normal(0, 1, (n, 8)), cx[lab_t] + rng.normal(0, 1, (m, 8)) + 0.5
    zs, zt = cz[lab_s] + rng.normal(0, 1.5, (n, 6)), cz[lab_t] + rng.normal(0, 1.5, (m, 6))
    C_rna, C_cnv = scaled_cost(xs, xt, 1), scaled_cost(zs, zt, 2)
    rows = []
    for name, solve in (("Standard (BOT)", bot), ("Ours (UOT, budget 0)", uot)):
        # 1. pipeline two-step: gate from step 1, step 2 on the retained cells
        for first, second, label in ((C_cnv, C_rna, "CNV->RNA vs RNA-only"), (C_rna, C_cnv, "RNA->CNV vs CNV-only")):
            _, gs, gt = solve(first)
            P2, _, _ = solve(second[np.ix_(gs, gt)])
            P_single, _, _ = solve(second)
            same = gs.all() and gt.all()
            rows.append({"solver": name, "check": f"two-step as defined (gate, then step 2): {label}",
                         "step1_source_retained": gs.mean(), "step1_target_retained": gt.mean(),
                         "max_abs_diff": float(np.abs(P2 - P_single).max()) if same else np.nan,
                         "max_abs_plan": float(P_single.max()), "note": "step 1 removes no cell, so step 2 sees the same cost"})
        # 2. soft two-step: step 2 regularised toward the step-1 plan
        for first, second, label in ((C_cnv, C_rna, "CNV->RNA"), (C_rna, C_cnv, "RNA->CNV")):
            P1, _, _ = solve(first)
            if (P1 <= 0).any():
                raise RuntimeError("step-1 plan has zero entries; log undefined")
            P_soft, _, _ = solve(second - EPS * np.log(P1))
            P_joint, _, _ = solve(C_rna + C_cnv)
            joint_shipped = C_rna + C_cnv
            s = float(np.median(joint_shipped))
            P_shipped_eps_s, _, _ = solve(joint_shipped / s, eps=EPS / s)
            rows.append({"solver": name, "check": f"soft two-step {label} vs single-step OT on C_rna + C_cnv",
                         "max_abs_diff": float(np.abs(P_soft - P_joint).max()), "max_abs_plan": float(P_joint.max()),
                         "total_mass_soft": float(P_soft.sum()), "total_mass_joint": float(P_joint.sum()),
                         "note": "BOT: the step-1 potentials are separable and cancel; UOT: they shift the marginals"})
            rows.append({"solver": name, "check": f"single-step on C_rna + C_cnv vs rna_cnv cost (C_rna + C_cnv)/s at eps/s (s = {s:.3f})",
                         "max_abs_diff": float(np.abs(P_joint - P_shipped_eps_s).max()), "max_abs_plan": float(P_joint.max()),
                         "note": "rna_cnv at epsilon 0.1 is the same family at a different epsilon"})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "strategy_identities.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(table.drop(columns="note").to_string(index=False))


if __name__ == "__main__":
    main()
