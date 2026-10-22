"""Does zero rejection give Balanced OT? Checked on the existing solvers, numerically (synthetic cost, local).

    python lineage_validation/level2_pairwise_ot/verify_zero_rejection.py

Runs, on one synthetic squared-Euclidean cost divided by its median pair distance (the 02_run_pair.py rule),
epsilon 0.1:
  1. ConfidenceOT(backbone="uot") with the project's default gate: rejection happens (control)
  2. ConfidenceOT(backbone="uot", source/target_rejection_bounds=(0, 0)): gates all retained?  marginals?
  3. the same with rejection_budget=0 (legacy argument)
  4. ConfidenceOT(backbone="balanced", rejection_bounds=(0, 0)), both variants
  5. traditional_ot.balanced.balanced_ot(cost, epsilon) directly
and reports, for each: retained fractions, total mass, max |P 1 - a| and max |P^T 1 - b| (a = 1/n, b = 1/m), the
distance to (5), and whether the result moves with the rejection cost (it must not when nothing is gated).
Writes verify_zero_rejection.tsv next to this script.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.preprocessing import median_pair_scale, squared_euclidean  # noqa: E402
from traditional_ot.balanced import balanced_ot  # noqa: E402


def cost_matrix(seed=0, n=300, m=200):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, 3, (4, 10))
    xs = centres[rng.integers(0, 4, n)] + rng.normal(0, 1, (n, 10))
    xt = centres[rng.integers(0, 3, m)] + rng.normal(0, 1, (m, 10)) + np.r_[2.0, np.zeros(9)]
    scale = median_pair_scale(xs, xt, rng=np.random.default_rng(seed + 1))
    return squared_euclidean(xs, xt) / scale


def describe(name, P, gate_s, gate_t, reference):
    n, m = P.shape
    return {"run": name, "source_retained": float(np.mean(gate_s)), "target_retained": float(np.mean(gate_t)),
            "total_mass": float(P.sum()), "max_row_residual": float(np.abs(P.sum(axis=1) - 1 / n).max()),
            "max_col_residual": float(np.abs(P.sum(axis=0) - 1 / m).max()),
            "max_abs_diff_to_balanced_ot": float(np.abs(P - reference).max())}


def main():
    C = cost_matrix()
    eps = 0.1
    bot = balanced_ot(C, epsilon=eps, threshold=1e-12, max_iterations=100_000)
    ref = bot.coupling
    rows = [describe("5 balanced_ot (traditional_ot.balanced)", ref, np.ones(C.shape[0]), np.ones(C.shape[1]), ref)]
    rows[-1]["converged"] = bot.converged
    common = dict(epsilon=eps, tolerance=1e-12, max_iterations=100_000, device="cpu", warn_on_terminal=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for variant in ("exact", "reversible"):
            r = ConfidenceOT(backbone="uot", variant=variant, rejection_cost=0.5, **common).fit(C)
            rows.append(describe(f"1 uot {variant}, default gate (budget 0.15 not enforced)", r.coupling, r.source_gate, r.target_gate, ref))
            for c in (0.5, 2.0):
                r = ConfidenceOT(backbone="uot", variant=variant, rejection_cost=c, source_rejection_bounds=(0.0, 0.0),
                                 target_rejection_bounds=(0.0, 0.0), **common).fit(C)
                rows.append(describe(f"2 uot {variant}, bounds (0,0), c={c}", r.coupling, r.source_gate, r.target_gate, ref))
            r = ConfidenceOT(backbone="uot", variant=variant, rejection_cost=0.5, source_rejection_budget=0.0,
                             target_rejection_budget=0.0, **common).fit(C)
            rows.append(describe(f"3 uot {variant}, rejection_budget=0", r.coupling, r.source_gate, r.target_gate, ref))
            for c in (0.5, 2.0):
                r = ConfidenceOT(backbone="balanced", variant=variant, rejection_cost=c, source_rejection_bounds=(0.0, 0.0),
                                 target_rejection_bounds=(0.0, 0.0), **common).fit(C)
                rows.append(describe(f"4 balanced {variant}, bounds (0,0), c={c}", r.coupling, r.source_gate, r.target_gate, ref))
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    table.to_csv(HERE / "verify_zero_rejection.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
