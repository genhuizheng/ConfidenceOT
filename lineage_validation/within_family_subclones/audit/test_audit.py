"""Minimal tests of the audit's own effective-coupling code and its agreement with classify_subclones.py.

    python lineage_validation/within_family_subclones/audit/test_audit.py

Cases: rejected source cell, rejected target cell, two-stage excluded cell, zero-mass target column, shuffled
cell ids, overlapping subclone assignment, incomplete CRISPR annotation, sparse vs dense coupling, and the
audit vs classifier agreement. Prints PASS / FAIL per check; exit 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import audit_core as ac  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ra = load("run_audit", HERE / "run_audit.py")
cs = load("classify_subclones", HERE.parent / "classify_subclones.py")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def write_gate(path, src, tgt, ks, kt):
    rows = [{"method": m, "side": side, "observation_id": c, "retained": bool(k), "signed_rejection_margin": -1.0 if k else 1.0, "site": "T1" if side == "source" else "M"}
            for m in ("M4-E", "M4-R") for side, ids, keep in (("source", src, ks), ("target", tgt, kt)) for c, k in zip(ids, keep)]
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "cell_confidence.csv", index=False)


def main():
    rng = np.random.default_rng(1)
    # rejected source / target, zero-mass column
    P = np.array([[1.0, 2.0, 0.0, 5.0], [40.0, 40.0, 0.0, 40.0], [1.0, 1.0, 0.0, 1.0]])
    rho = np.array([1, 0, 1], bool); tau = np.array([1, 1, 1, 0], bool)
    Pt = ac.effective(P, rho, tau)
    q, ok = ac.matching(Pt, tau)
    check("rejected source row carries 0 effective mass", Pt[1].sum() == 0)
    check("rejected target column: q = 0", q[:, 3].sum() == 0 and not ok[3])
    check("zero-mass retained target: q = 0, not NaN, flagged", q[:, 2].sum() == 0 and not ok[2] and np.isfinite(q).all())
    check("retained target with mass: q sums to 1", np.allclose(q[:, ok].sum(0), 1))
    # sparse vs dense
    Ps = sparse.csr_matrix(P)
    check("sparse and dense coupling give the same effective coupling and q",
          np.allclose(ac.effective(Ps, rho, tau), Pt) and np.allclose(ac.matching(ac.effective(Ps, rho, tau), tau)[0], q))
    # overlapping / outside labels, duplicated ids
    try:
        ac.group_sum(Pt, ["A", "B", "C"], ["A", "B"]); raised = False
    except ValueError:
        raised = True
    check("a source cell with a label outside the exclusive group list is refused", raised)
    try:
        ac.check_unique(["x", "y", "x"], "test"); raised = False
    except ValueError:
        raised = True
    check("a duplicated cell id (one cell in two subclones) is refused", raised)
    # two-stage exclusion + shuffled ids through run_audit.load
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp)
        src = [f"s{i}" for i in range(6)]; tgt = [f"t{j}" for j in range(5)]
        write_gate(export / "ot" / "gates" / "b" / "p" / "cnv", src, tgt, [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0])
        perm_s, perm_t = rng.permutation(5), rng.permutation(4)
        s2, t2 = [src[i] for i in perm_s], [tgt[j] for j in perm_t]
        write_gate(export / "ot" / "gates" / "b" / "p" / "cnv_then_rna", s2, t2, [c != "s4" for c in s2], [c != "t3" for c in t2])
        base = rng.random((5, 4)) + 0.1
        (export / "couplings" / "b" / "p" / "cnv_then_rna").mkdir(parents=True)
        np.savez_compressed(export / "couplings" / "b" / "p" / "cnv_then_rna" / "coupling_m4e.npz", coupling=base[np.ix_(perm_s, perm_t)])
        run = ra.load(export, "b", "p", "cnv_then_rna", "M4-E")
        expect = np.zeros((6, 5)); expect[:5, :4] = base
        check("shuffled step-2 file order: coupling re-indexed by id equals the id-ordered matrix", np.allclose(run["P"], expect))
        check("two-stage: step-1-excluded cells flagged excluded, zero mass, gate 0",
              run["ex_s"][5] and run["ex_t"][4] and run["P"][5].sum() == 0 and run["P"][:, 4].sum() == 0 and not run["rho"][5] and not run["tau"][4])
        check("two-stage: step-2 rejected cells gate 0 (not excluded)", not run["rho"][4] and not run["ex_s"][4] and not run["tau"][3] and not run["ex_t"][3])
        mrow = ra.mass_row(run, {})
        check("mass rows: excluded rows and columns hold 0; four gate blocks sum to the total",
              mrow["mass_rows_excluded"] == 0 and mrow["mass_cols_excluded"] == 0 and abs(mrow["blocks_sum_minus_total"]) < 1e-12)
        # incomplete annotation: s3 has no subclone label (unlabelled), t2 has none (ambiguous)
        labels = {"s0": "A", "s1": "A", "s2": "B", "s4": "B", "s5": "A"}
        src_g = [labels.get(c, ra.UNLAB) for c in run["source"]]
        tgt_t = ["A", "B", "ambiguous", "B", "A"]
        rej, dec, vs, cmp_ = ra.group_tables(run, {}, "subclone L1", src_g, tgt_t, ["A", "B", ra.UNLAB, ra.OTHER], {"A", "B", ra.UNLAB}, {}, "")
        n_unlab = [r for r in rej if r["group"] == ra.UNLAB][0]["eligible_cells"]
        check("incomplete annotation: the unannotated source cell is counted as family unlabelled", n_unlab == 1)
        check("incomplete annotation: the unannotated target is not scored", cmp_[0]["scored_target_cells"] == 2,
              f"{cmp_[0]['scored_target_cells']} scored (t0 A, t1 B retained with mass; t2 unannotated, t3 rejected, t4 excluded)")
        # agreement with the classifier
        out, summary, _, _ = cs.classify({**run, "source_margin": np.zeros(6), "target_margin": np.where(run["ex_t"], np.inf, 0.0),
                                          "source_site": np.repeat("T1", 6), "target_site": np.repeat("M", 5)},
                                         [x if x in ("A", "B") else cs.UNLABELLED for x in src_g], tgt_t, ["A", "B"], 1, None)
        Pt = ac.effective(run["P"], run["rho"], run["tau"]); q, okk = ac.matching(Pt, run["tau"])
        S = ac.group_sum(q, [x if x in ("A", "B") else ra.UNLAB for x in src_g], ["A", "B", ra.UNLAB])
        check("audit and classifier give the same within-family weights",
              np.allclose(out[["S__A", "S__B", f"S__{cs.UNLABELLED}"]].to_numpy(float).T, S))
        dec_mass = {r["group"]: r["value"] for r in dec if r["composition"] == "effective transport mass"}
        import json
        check("audit and classifier give the same retained effective mass per group",
              all(abs(dec_mass[g] - json.loads(summary["source_effective_mass"])[g if g != ra.UNLAB else cs.UNLABELLED]) < 1e-12 for g in ("A", "B")))
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
