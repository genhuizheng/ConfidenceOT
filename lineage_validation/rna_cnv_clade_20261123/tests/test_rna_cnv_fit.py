"""The v3-grid RNA + CNV fit path on synthetic data (local): precomputed RNA coordinates + an exported CNV profile through
realdata_benchmark/v3_grid (v3_common.fit with worker.fit_args), as the TACC rows run.

    python lineage_validation/rna_cnv_clade_20261123/tests/test_rna_cnv_fit.py

Checks: (1) rna_cnv with V3_CNV_NPCS gives a CNV block with PCs (the 2026-11-23 bug: a precomputed label set n_pcs 0
and the CNV PCA kept 0 components); (2) without --cnv-n-pcs the run stops instead of fitting RNA alone; (3) beta 4
scales the CNV term: the recorded weight, and a different coupling from beta 1; (4) beta 1 equals the run without
--cnv-weight (gates and coupling identical). Prints PASS / FAIL; exit 1 on failure.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(REPO / "realdata_benchmark" / "v3_grid"), str(REPO / "realdata_benchmark"), str(REPO / "src"),
                str(REPO / "cancer_metastasis"), str(REPO)]
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    import v3_common as v3
    import worker
    rng = np.random.default_rng(3)
    ns, nt, d, g = 60, 50, 10, 120
    X = rng.normal(size=(ns, d)); Y = rng.normal(size=(nt, d)) + 0.3
    X /= np.linalg.norm(X, axis=1, keepdims=True); Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    sid = [f"s{i}" for i in range(ns)]; tid = [f"t{i}" for i in range(nt)]
    obs = {"source": pd.DataFrame({"malignant": "malignant", "sample_id": "U__source"}, index=sid),
           "target": pd.DataFrame({"malignant": "malignant", "sample_id": "U__target"}, index=tid)}
    man = pd.DataFrame([{"pair_id": "U", "dataset_id": "KPTracer", "patient_id": "9999", "source_sample": "U__source",
                         "target_sample": "U__target", "source_h5ad": "", "target_h5ad": "", "eligible": True}])
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); unit = tmp / "cnv" / "KPTracer" / "9999"; unit.mkdir(parents=True)
        prof = np.vstack([rng.normal(1.0, 0.05, (ns, g)), rng.normal(1.0, 0.05, (nt, g))]).astype(np.float32)
        prof[ns:, :40] += 0.3                                          # the target carries a gain
        prof.tofile(unit / "cnv.f32")
        pd.DataFrame({"cell": sid + tid, "barcode": sid + tid}).to_csv(unit / "cells.tsv", sep="\t", index=False)
        pd.DataFrame({"feature": [f"w{i}" for i in range(g)]}).to_csv(unit / "genes.tsv", sep="\t", index=False)

        def fit(env: dict, tag: str):
            for k in ("V3_REPRESENTATION_SOURCE", "V3_CNV_ROOT", "V3_CNV_WEIGHT", "V3_CNV_NPCS"):
                os.environ.pop(k, None)
            os.environ.update(env)
            args = worker.fit_args("kp", "cpu", 1)
            code, log, run = v3.fit(X, Y, True, obs, man, 0, args, tmp / f"work_{tag}")
            if run is None:
                return code, log, None, None, None
            rj = json.loads((run / "run.json").read_text())
            cc = pd.read_csv(run / "cell_confidence.csv")
            P = np.load(run / "coupling_m4e.npz")["coupling"]
            return code, log, rj, cc, P

        base = {"V3_REPRESENTATION_SOURCE": "rna_cnv", "V3_CNV_ROOT": str(tmp / "cnv")}
        c1, log1, rj1, cc1, P1 = fit(base, "b1")
        check("rna_cnv fits", c1 == 0 and rj1 is not None, log1[-600:] if c1 else "")
        pcs = (rj1 or {}).get("cnv", {}) or {}
        check("CNV block has PCs (default V3_CNV_NPCS 30, capped by the data)", pcs.get("pcs", 0) >= 1, f"pcs {pcs.get('pcs')}")
        cE, logE, _, _, _ = fit({**base, "V3_CNV_NPCS": "0"}, "zero")
        check("a CNV block of 0 PCs stops the run", cE != 0 and "set --cnv-n-pcs" in logE, f"exit {cE}")
        c4, log4, rj4, cc4, P4 = fit({**base, "V3_CNV_WEIGHT": "4"}, "b4")
        check("beta 4 recorded", c4 == 0 and (rj4 or {}).get("cnv", {}).get("cnv_weight") == 4.0, log4[-400:] if c4 else "")
        check("beta 4 changes the coupling", c4 == 0 and c1 == 0 and not np.allclose(P1, P4))
        cb, logb, rjb, ccb, Pb = fit({**base, "V3_CNV_WEIGHT": "1"}, "b1explicit")
        check("beta 1 explicit = default (gates, coupling)", cb == 0 and c1 == 0 and np.array_equal(P1, Pb)
              and (cc1.retained.to_numpy() == ccb.retained.to_numpy()).all())
        cr, logr, rjr, ccr, Pr = fit({}, "rna")
        check("rna alone differs from rna_cnv", cr == 0 and c1 == 0 and not np.allclose(Pr, P1))
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
