"""End-to-end local test of one N = 5,000-benchmark row and one condition row through 10_tacc_worker.py, on the
synthetic generated directory of test_conditions_truth.py (L4, population_lost pair).

    python realdata_benchmark/splatter_n5000/tests/test_n5000_rows.py [--keep DIR]

Checks: the row fits (02_run_pair.py) and scores (40_score.py); IC-POT and Vanilla UOT run on the same cost
(50_solver_comparison.py, geometry check included); 23_row_extras.py writes coupling metrics for M4-E, M4-R, balanced
full and balanced on the retained block, and DEG-B per side with the truth comparison on the source side (the side
with the unmatched population); the condition row writes DEG-A / C / D; a rerun skips finished rows. PASS / FAIL.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N5 = HERE.parent
RB = N5.parent
REPO = RB.parent
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--keep", type=Path, default=None); a = p.parse_args()
    spec = importlib.util.spec_from_file_location("tct", HERE / "test_conditions_truth.py"); tct = importlib.util.module_from_spec(spec); spec.loader.exec_module(tct)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", RB, REPO]))}
    with tempfile.TemporaryDirectory() as tmp:
        root = a.keep or Path(tmp)
        gen, cond = root / "generated", root / "conditions"
        tct.fake_generated(gen, "L4_batch_perturb", np.random.default_rng(3))
        subprocess.run([sys.executable, str(N5 / "21_build_conditions_n5000.py"), "--generated", str(gen), "--out", str(cond), "--n-cells", "120",
                        "--replicate", "1", "--level", "L4_batch_perturb"], check=True, env=env, capture_output=True)
        subprocess.run([sys.executable, str(RB / "01_config_registry.py"), "--out", str(root / "reg")], check=True, env=env, capture_output=True)
        m = pd.read_csv(cond / "manifest.csv"); i = int(np.flatnonzero(m.biological_case == "population_lost")[0]); pid = m.pair_id[i]
        truth = str(cond / pid / "truth")
        base = {"pair_index": i, "pair_id": pid, "manifest": str(cond / "manifest.csv"), "manifest_ds": "", "run_root": "", "converted": "",
                "acceptance": 0.99, "size": 120, "truth_dir": truth, "fit_out": ""}
        work = pd.DataFrame([{**base, "dataset": "splatter_n5000", "config_id": "ranknm1024-expected|none|noscale|cos", "out": str(root / "out" / "row")},
                             {**base, "dataset": "splatter_n5000_condition", "config_id": "", "out": str(root / "out" / "condition")}])
        work.to_csv(root / "worklist.tsv", sep="\t", index=False)
        cmd = [sys.executable, str(RB / "10_tacc_worker.py"), "--worklist", str(root / "worklist.tsv"), "--registry", str(root / "reg" / "PREPROCESSING_CONFIG_REGISTRY.tsv"), "--worker", "0", "--workers", "1"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
        print(r.stdout[-800:], r.stderr[-800:])
        row, con = root / "out" / "row", root / "out" / "condition"
        for f in ("error.txt", "comparator_icpot.log", "comparator_uot.log", "extras.log"):
            if (row / f).exists() and not (row / "SUCCESS").exists():
                print(f"--- {f}\n", (row / f).read_text()[-1500:])
        check("row SUCCESS", (row / "SUCCESS").exists())
        sc = pd.read_csv(row / "score.csv") if (row / "score.csv").exists() else pd.DataFrame()
        check("row scored by 40_score (M4-E, M4-R)", {"M4-E", "M4-R"} <= set(sc.get("method", [])))
        ic = pd.read_csv(row / "comparators" / "icpot" / "solver_comparison.csv") if (row / "comparators" / "icpot" / "solver_comparison.csv").exists() else pd.DataFrame()
        uo = pd.read_csv(row / "comparators" / "uot" / "solver_comparison.csv") if (row / "comparators" / "uot" / "solver_comparison.csv").exists() else pd.DataFrame()
        check("IC-POT and Vanilla UOT on the same cost", len(ic) > 0 and len(uo) > 0, f"{ic.columns[:6].tolist() if len(ic) else ''}")
        cp = pd.read_csv(row / "extras" / "coupling.tsv", sep="\t") if (row / "extras" / "coupling.tsv").exists() else pd.DataFrame()
        check("coupling metrics for M4-E, M4-R, balanced full and balanced on the retained block",
              {"M4-E", "M4-R", "balanced_full", "balanced_on_M4E_retained_block"} <= set(cp.get("plan", [])))
        db = pd.read_csv(row / "extras" / "deg_b.tsv", sep="\t") if (row / "extras" / "deg_b.tsv").exists() else pd.DataFrame()
        src = db[db.side == "source"] if len(db) else db
        check("DEG-B per side, truth comparison on the source side",
              len(db) == 2 and (len(src) and ("deg_auroc" in src or str(src.status.iloc[0]).startswith("not_estimable"))), db.to_string()[:600] if len(db) else "")
        dc = pd.read_csv(con / "deg_condition.tsv", sep="\t") if (con / "deg_condition.tsv").exists() else pd.DataFrame()
        check("condition row: DEG-A, DEG-C and DEG-D", len(dc) and {c.split()[0] for c in dc.contrast} >= {"DEG-A", "DEG-C", "DEG-D"},
              dc[["contrast", "group"] + [c for c in dc if c.endswith("auroc")]].to_string()[:800] if len(dc) else "")
        r2 = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=REPO)
        check("a rerun skips finished rows", " ok" not in r2.stdout and "FAILED" not in r2.stdout, r2.stdout[-200:])
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
