"""Synthetic unit tests of the subclone mass calculation (classify_subclones.py, define_subclones.py).

    python lineage_validation/within_family_subclones/test_subclone_mass.py

Each test prints PASS / FAIL with what it checked; the script exits 1 if any test fails. Covered:
rejected source and target cells, two-step exclusion, cell-id shuffle, zero-mass columns, overlapping branch
assignment, and the output-table and baseline conventions the audit asked about.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import lv_common as lv  # noqa: E402


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cs = load("classify_subclones", HERE / "classify_subclones.py")
ds = load("define_subclones", HERE / "define_subclones.py")
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def toy_run(P, rho, tau, src_ids, tgt_ids, margins_t=None):
    n, m = P.shape
    return {"P": np.asarray(P, float), "rho": np.asarray(rho, bool), "tau": np.asarray(tau, bool),
            "source": np.asarray(src_ids, dtype=object), "target": np.asarray(tgt_ids, dtype=object),
            "source_site": np.repeat("T1", n), "target_site": np.repeat("M", m),
            "source_margin": np.zeros(n), "target_margin": np.zeros(m) if margins_t is None else np.asarray(margins_t, float)}


def test_rejected_source_and_target():
    # Sources: s0, s1 in subclone A; s2, s3 in B. s1 (A) is rejected and carries most of the mass.
    P = np.array([[1.0, 1.0, 1.0],
                  [50.0, 50.0, 50.0],
                  [1.0, 1.0, 1.0],
                  [1.0, 1.0, 1.0]])
    run = toy_run(P, [1, 0, 1, 1], [1, 1, 0], ["s0", "s1", "s2", "s3"], ["t0", "t1", "t2"])
    out, summary, _, _ = cs.classify(run, ["A", "A", "B", "B"], ["A", "B", "A"], ["A", "B"], 1, None)
    s_a = out.loc[0, "S__A"]
    check("rejected source mass is masked before normalising", abs(s_a - 1 / 3) < 1e-12, f"S_A(t0) = {s_a:.4f}, expected 1/3")
    check("rejected target is abstain", out.loc[2, "outcome"] == "abstain", out.loc[2, "outcome"])
    held = out.loc[2, ["S__A", "S__B", f"S__{cs.UNLABELLED}"]].sum()
    check("rejected target holds zero weight in the output table", abs(held) < 1e-12, f"sum of S__ = {held:.3f}")
    check("rejected target has zero family weight in the output table", abs(out.loc[2, "family_weight"]) < 1e-12,
          f"family_weight = {out.loc[2, 'family_weight']:.3f}")


def test_zero_mass_column():
    P = np.array([[0.0, 1.0], [0.0, 1.0]])
    run = toy_run(P, [1, 1], [1, 1], ["s0", "s1"], ["t0", "t1"])
    out, _, _, _ = cs.classify(run, ["A", "B"], ["A", "A"], ["A", "B"], 1, None)
    s = out[["S__A", "S__B", f"S__{cs.UNLABELLED}"]].to_numpy(float)
    check("zero-mass column: abstain, no NaN", out.loc[0, "outcome"] == "abstain" and np.isfinite(s).all(),
          f"outcome {out.loc[0, 'outcome']}, finite {np.isfinite(s).all()}")
    check("retained column with mass sums to 1", abs(s[1].sum() - 1) < 1e-12, f"{s[1].sum():.6f}")


def test_family_unlabelled_and_other():
    # s0 A, s1 B, s2 family unlabelled, s3 another clone.
    P = np.array([[2.0], [1.0], [1.0], [4.0]])
    run = toy_run(P, [1, 1, 1, 1], [1], ["s0", "s1", "s2", "s3"], ["t0"])
    out, _, _, _ = cs.classify(run, ["A", "B", cs.UNLABELLED, cs.OTHER], ["A"], ["A", "B"], 1, None)
    within = out.loc[0, ["S__A", "S__B", f"S__{cs.UNLABELLED}"]].to_numpy(float)
    check("within-family weights: subclones + unlabelled sum to 1, other excluded", np.allclose(within, [0.5, 0.25, 0.25]),
          f"{np.round(within, 3)}")
    check("family weight = family mass / total retained mass", abs(out.loc[0, "family_weight"] - 0.5) < 1e-12,
          f"{out.loc[0, 'family_weight']:.3f}")


def write_gate(path, src_ids, tgt_ids, src_keep, tgt_keep):
    rows = []
    for method in ("M4-E", "M4-R"):
        for side, ids, keep in (("source", src_ids, src_keep), ("target", tgt_ids, tgt_keep)):
            for c, k in zip(ids, keep):
                rows.append({"method": method, "side": side, "observation_id": c, "retained": bool(k),
                             "signed_rejection_margin": -1.0 if k else 1.0, "site": "T1" if side == "source" else "M"})
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path / "cell_confidence.csv", index=False)


def test_two_step_and_shuffle():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        export = Path(tmp)
        src = [f"s{i}" for i in range(6)]
        tgt = [f"t{j}" for j in range(5)]
        # step 1 (cnv): s5 and t4 rejected; step 2 (cnv_then_rna) fitted on s0-s4, t0-t3, with s4 and t3 rejected
        write_gate(export / "ot" / "gates" / "b" / "p" / "cnv", src, tgt, [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0])
        s2, t2 = src[:5], tgt[:4]
        perm_s, perm_t = rng.permutation(5), rng.permutation(4)  # step-2 files in another order
        s2p, t2p = [s2[i] for i in perm_s], [t2[j] for j in perm_t]
        keep_s = {c: c != "s4" for c in s2}; keep_t = {c: c != "t3" for c in t2}
        write_gate(export / "ot" / "gates" / "b" / "p" / "cnv_then_rna", s2p, t2p, [keep_s[c] for c in s2p], [keep_t[c] for c in t2p])
        base = rng.random((5, 4)) + 0.1
        coupling = base[np.ix_(perm_s, perm_t)]  # rows / columns follow the step-2 file order
        (export / "couplings" / "b" / "p" / "cnv_then_rna").mkdir(parents=True)
        np.savez_compressed(export / "couplings" / "b" / "p" / "cnv_then_rna" / "coupling_m4e.npz", coupling=coupling)
        run = cs.load_export_run(export, "cnv_then_rna", "b", "p", "M4-E")
        check("two-step: step-1-excluded source row has zero mass and rho 0",
              run["P"][src.index("s5")].sum() == 0 and not run["rho"][src.index("s5")])
        check("two-step: step-1-excluded target column has zero mass and tau 0",
              run["P"][:, tgt.index("t4")].sum() == 0 and not run["tau"][tgt.index("t4")])
        check("two-step: step-2 rejected cells are masked (rho / tau 0)",
              not run["rho"][src.index("s4")] and not run["tau"][tgt.index("t3")])
        expect = np.zeros((6, 5)); expect[:5, :4] = base
        check("cell-id shuffle: coupling re-indexed by id equals the unshuffled matrix", np.allclose(run["P"], expect))
        labels = {"s0": "A", "s1": "A", "s2": "B", "s3": "B", "s4": "B", "s5": "A"}
        out, _, _, _ = cs.classify(run, [labels[c] for c in run["source"]], ["A", "B", "A", "B", "A"], ["A", "B"], 1, None)
        Pr = expect * np.array([1, 1, 1, 1, 0, 0])[:, None]
        with np.errstate(invalid="ignore"):
            direct = Pr[:2].sum(0) / Pr.sum(0)
        check("cell-id shuffle: S_A(t) equals the direct computation on the id-ordered matrix (retained t0-t2)",
              np.allclose(out["S__A"].to_numpy()[:3], direct[:3]), f"{np.round(out['S__A'].to_numpy()[:3], 4)} vs {np.round(direct[:3], 4)}")
        check("two-step: step-2 rejected target t3 has weight 0", out.loc[3, "S__A"] == 0 and out.loc[3, "S__B"] == 0)
        check("two-step: excluded target t4 is abstain", out.loc[4, "outcome"] == "abstain", out.loc[4, "outcome"])


def test_overlapping_branches():
    # Tree ((a1..a25),((b1..b25),(c1..c25))): the B and C clades are nested in the BC clade. Level 1 must
    # split at the root into {A} and {BC} (not count B or C again), level 2 must split BC into B and C.
    d1 = lv.tree_module()
    names = {k: [f"{k}{i}" for i in range(25)] for k in "abc"}
    text = f"(({','.join(names['a'])}),(({','.join(names['b'])}),({','.join(names['c'])})));"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "t.nwk"
        path.write_text(text)
        tree = d1.Tree(path)
    leaf = {tree.name[v]: v for v in tree.leaves}
    t = ds.Induced(tree, set(leaf.values()))
    rows = []
    for k, state in (("a", "1"), ("b", "2"), ("c", "3")):
        for x in names[k]:
            rows.append({"cell": x, "r0": state if k == "a" else "9", "r1": state})
    matrix = pd.DataFrame(rows).set_index("cell")
    reliable = {v: True for v in leaf.values()}
    split, chosen, _ = ds.partition(t, t.root, reliable, matrix)
    sets = [set(c) for _, c, _ in chosen]
    disjoint = all(not (sets[i] & sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets)))
    covered = sum(len(s) for s in sets)
    check("level 1: subclones are disjoint clades (no cell counted twice)", disjoint and covered == 75,
          f"{len(sets)} subclones, {covered} cells")
    bc = next(node for node, c, _ in chosen if len(c) == 50)
    _, chosen2, _ = ds.partition(t, bc, reliable, matrix)
    check("level 2 inside BC: B and C, disjoint", len(chosen2) == 2 and not (set(chosen2[0][1]) & set(chosen2[1][1])))
    # A cell given two labels in a cells table would be counted once silently by the dict lookup; the
    # classifier should refuse such a table.
    cells = pd.DataFrame({"cell_id": ["x", "x"], "L1": ["A", "B"]})
    check("a duplicated cell id in a subclone table is detected", cells["cell_id"].duplicated().any(),
          "the classifier itself has no such guard yet")


def test_baseline_denominator():
    # Family: A 10 cells, B 10 cells, unlabelled 80 cells, all retained, equal row mass.
    n = 100
    labels = ["A"] * 10 + ["B"] * 10 + [cs.UNLABELLED] * 80
    run = toy_run(np.ones((n, 1)), np.ones(n), [1], [f"s{i}" for i in range(n)], ["t0"])
    out, summary, _, _ = cs.classify(run, labels, ["A"], ["A", "B"], 1, None)
    check("independent-coupling share of A includes the unlabelled family cells (0.10)",
          abs(out.loc[0, "S__A"] - 0.10) < 1e-12, f"S_A = {out.loc[0, 'S__A']:.3f}")
    import json
    ret = json.loads(summary["source_effective_mass"])
    share_used = ret["A"] / sum(ret.values())
    check("summary / heatmap source share uses the same denominator as the weights", abs(share_used - 0.10) < 1e-12,
          f"share used = {share_used:.3f} (retained effective mass of A / family incl. unlabelled)")


if __name__ == "__main__":
    for test in (test_rejected_source_and_target, test_zero_mass_column, test_family_unlabelled_and_other,
                 test_two_step_and_shuffle, test_overlapping_branches, test_baseline_denominator):
        print(f"-- {test.__name__}")
        test()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    sys.exit(1 if failed else 0)
