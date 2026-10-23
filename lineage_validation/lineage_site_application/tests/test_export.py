"""Test of tools/export_level1_gates.py on a fake run tree in the TACC layout (local).

    python lineage_validation/lineage_site_application/tests/test_export.py

One D_A pair with rna, cnv and cnv_then_rna runs (step 1 = cnv, which rejects a source and a target cell, so step 2
analyses a subset), M4-E and M4-R. Checks: states retained / rejected / excluded; a rejected cell keeps raw mass but
has effective mass 0; excluded cells carry no mass and margin inf; effective mass = rho P tau; lineage and site
blocks sum to the effective total; missing strategies are listed in status.tsv, never filled.
Prints PASS / FAIL; exit 1 on any failure.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "tools" / "export_level1_gates.py"
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def write_run(base: Path, src, tgt, src_site, tgt_site, keep_s, keep_t, rng):
    base.mkdir(parents=True)
    rows = []
    for method in ("M4-E", "M4-R"):
        for side, ids, sites, keep in (("source", src, src_site, keep_s), ("target", tgt, tgt_site, keep_t)):
            for c, s, k in zip(ids, sites, keep):
                rows.append({"method": method, "side": side, "observation_id": c, "retained": bool(k),
                             "signed_rejection_margin": -0.5 if k else 0.7, "site": s})
    pd.DataFrame(rows).to_csv(base / "cell_confidence.csv", index=False)
    P = rng.uniform(0.01, 1, (len(src), len(tgt))); P /= P.sum() * 1.3
    for m in ("m4e", "m4r"):
        np.savez_compressed(base / f"coupling_{m}.npz", coupling=P)
    pd.DataFrame([{"method": m, "transported_mass": P.sum()} for m in ("M4-E", "M4-R")]).to_csv(base / "pair_metrics.csv", index=False)
    (base / "SUCCESS").write_text("done")
    return P


def main() -> None:
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "lv"
        (root / "inputs" / "manifests").mkdir(parents=True); (root / "labels").mkdir()
        pair = "9999__D_A__primaries__9999_X_N1"
        pd.DataFrame([{"pair_id": pair, "dataset_id": "KPTracer", "patient_id": "9999"}]).to_csv(root / "inputs" / "manifests" / "kp_D_A_raw.csv", index=False)
        for b in ("kp_D_B", "kp_D_C3", "quinn_D_B"):
            pd.DataFrame(columns=["pair_id", "dataset_id", "patient_id"]).to_csv(root / "inputs" / "manifests" / f"{b}_raw.csv", index=False)
        src = [f"s{i}" for i in range(6)]; tgt = [f"t{j}" for j in range(5)]
        src_site = ["9999_X_T1"] * 4 + ["9999_X_T2"] * 2; tgt_site = ["9999_X_N1"] * 5
        pd.DataFrame({"dataset": "KPTracer", "mouse": "9999", "cell": src + tgt, "site": src_site + tgt_site,
                      "lineage": ["9999_X_T1"] * 4 + ["9999_X_T2"] * 2 + ["9999_X_T1"] * 5}).to_csv(root / "labels" / "cells.tsv.gz", sep="\t", index=False)
        pd.DataFrame([{"side": "x"}]).to_csv(root / "inputs" / "sides.tsv", sep="\t", index=False)
        runs = root / "runs" / "acc0.99"
        stem = lambda s: runs / s / "arm_A" / "kp_D_A" / "ranknm256_noscale_ds_cos" / pair / "scope_malignant" / "budget_0.95"
        P_rna = write_run(stem("rna"), src, tgt, src_site, tgt_site, [1, 1, 1, 0, 1, 1], [1, 1, 0, 1, 1], rng)
        write_run(stem("cnv"), src, tgt, src_site, tgt_site, [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0], rng)
        s2, t2 = src[:5], tgt[:4]  # step 2 of cnv_then_rna analyses what cnv retained
        P2 = write_run(stem("cnv_then_rna"), s2, t2, src_site[:5], tgt_site[:4], [1, 0, 1, 1, 1], [1, 1, 1, 1], rng)
        out = Path(tmp) / "out"
        r = subprocess.run([sys.executable, str(TOOL), "--root", str(root), "--out", str(out)], capture_output=True, text=True)
        check("export exits 0", r.returncode == 0, r.stderr[-600:] if r.returncode else "")
        st = pd.read_csv(out / "status.tsv", sep="\t")
        check("status: rna, cnv, cnv_then_rna ok; rna_cnv and rna_then_cnv missing (listed, not filled)",
              set(st.loc[st.status.eq("ok"), "strategy"]) == {"rna", "cnv", "cnv_then_rna"}
              and set(st.loc[st.status.eq("missing"), "strategy"]) == {"rna_cnv", "rna_then_cnv"})
        c = pd.read_csv(out / "cells_kp_D_A.tsv.gz", sep="\t")
        e = c[(c.strategy == "rna") & (c.method == "M4-E")].set_index(["side", "cell"])
        check("rna: rejected source s3 keeps raw mass but has effective mass 0",
              e.loc[("source", "s3"), "state"] == "rejected" and e.loc[("source", "s3"), "raw_mass"] > 0 and e.loc[("source", "s3"), "effective_mass"] == 0)
        rho = np.array([1, 1, 1, 0, 1, 1], bool); tau = np.array([1, 1, 0, 1, 1], bool)
        eff = P_rna * rho[:, None] * tau[None, :]
        check("rna: effective masses equal the row / column sums of rho P tau",
              np.allclose(e.xs("source").loc[src, "effective_mass"], eff.sum(1)) and np.allclose(e.xs("target").loc[tgt, "effective_mass"], eff.sum(0)))
        t = c[(c.strategy == "cnv_then_rna") & (c.method == "M4-E")].set_index(["side", "cell"])
        check("cnv_then_rna: cells removed at step 1 are excluded (margin inf, no mass); s1 rejected at step 2",
              t.loc[("source", "s5"), "state"] == "excluded" and t.loc[("target", "t4"), "state"] == "excluded"
              and np.isinf(t.loc[("source", "s5"), "signed_margin"]) and t.loc[("source", "s5"), "raw_mass"] == 0
              and t.loc[("source", "s1"), "state"] == "rejected")
        check("cnv_then_rna: S0 x T0 is step 1's analysed cells (6 + 5), step 2's plan placed inside",
              len(t) == 11 and np.isclose(t.xs("source")["raw_mass"].sum(), P2.sum()))
        b = pd.read_csv(out / "blocks_kp_D_A.tsv.gz", sep="\t")
        bb = b[(b.strategy == "rna") & (b.method == "M4-E")]
        tot = st[(st.strategy == "rna") & (st.method == "M4-E")]["effective_total"].iloc[0]
        check("blocks: lineage and site blocks each sum to the effective total; T2 -> T1-family block is a cross-lineage block",
              np.isclose(bb[bb.kind == "lineage"]["effective_mass"].sum(), tot) and np.isclose(bb[bb.kind == "site"]["effective_mass"].sum(), tot)
              and ((bb.kind == "lineage") & (bb.source_group == "9999_X_T2") & (bb.target_group == "9999_X_T1")).any())
        check("SUCCESS written and sides.tsv copied", (out / "SUCCESS").exists() and (out / "sides.tsv").exists())
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
