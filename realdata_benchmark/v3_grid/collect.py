"""Collect the frozen-calibration-v3 grid of one root (worker.py rows) into tables. Read only on the row directories.

    python realdata_benchmark/v3_grid/collect.py --root ROOT [--tag v3tacc]

Writes, for every dataset:
  STATUS_<tag>.tsv          one row per worklist row x configuration: done / failed (with the log tail) / missing
  FIT_METRICS_<tag>.tsv.gz  pair_metrics.csv of every done fit (rejection cost, rejection rates, convergence, calibration
                            flags) with unit / config / gate statistics of the correction (diag.json)
splatter:  RESULTS_grid_<tag>.tsv (run_local_benchmark columns), so that
           python realdata_benchmark/splatter_local/summarize_local.py --root ROOT --stage grid --tag <tag> \
               --frozen FROZEN_CALIBRATION_V3.json
           gives the per-level Pareto fronts and the overall view exactly as the local grid (effective readout)
ltl331 / kp: CELL_GATES_<tag>/<unit>.tsv.gz, one file per unit (config, method, side, observation_id, rejected, retained
           and the other cell_confidence columns); ltl331 also COUPLING_CLUSTERS_<tag>.tsv.gz
Nothing is filtered: failed and missing fits are listed, never dropped silently or replaced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import v3_common as v3  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tag", default="v3tacc")
    a = p.parse_args()
    work = pd.read_csv(a.root / "WORKLIST.tsv", sep="\t", dtype={"mouse": str, "pair_id": str})
    status, metrics, scores, couplings = [], [], [], []
    gates: dict[str, list] = {}
    gate_dir = a.root / f"CELL_GATES_{a.tag}"

    def flush(keep: str | None) -> None:  # a unit's rows are contiguous in the worklist: write it once its last row is read
        for unit in [u for u in gates if u != keep]:
            gate_dir.mkdir(exist_ok=True)
            pd.concat(gates.pop(unit), ignore_index=True).to_csv(gate_dir / f"{unit.replace('|', '__')}.tsv.gz", sep="\t",
                                                                 index=False, compression="gzip")

    for _, row in work.iterrows():
        flush(row.unit_id)
        spec = json.loads(row.spec)
        for cid in v3.config_ids(spec):
            d = Path(row.out) / cid.replace("|", "__")
            head = {"dataset": row.dataset, "unit_id": row.unit_id, "pair_id": row.pair_id, "config_id": cid}
            if not (d / "DONE").exists():
                log = (d / "fit.log").read_text()[-300:] if (d / "fit.log").exists() else ""
                status.append(head | {"state": "failed" if log else "missing", "log_tail": log.replace("\n", " | ")}); continue
            status.append(head | {"state": "done", "log_tail": ""})
            diag = json.loads((d / "diag.json").read_text())
            pm = pd.read_csv(d / "pair_metrics.csv")
            for k, v in (head | diag).items():
                pm[k] = v
            metrics.append(pm)
            if row.dataset == "splatter":
                scores.append(pd.read_csv(d / "score.tsv", sep="\t"))
            else:
                c = pd.read_csv(d / "cell_confidence.csv.gz")
                c.insert(0, "config_id", cid)
                gates.setdefault(row.unit_id, []).append(c)
                if row.dataset == "ltl331":
                    cc = pd.read_csv(d / "coupling_clusters.tsv.gz", sep="\t")
                    cc.insert(0, "config_id", cid); cc.insert(0, "unit_id", row.unit_id)
                    couplings.append(cc)
    st = pd.DataFrame(status)
    st.to_csv(a.root / f"STATUS_{a.tag}.tsv", sep="\t", index=False)
    if metrics:
        pd.concat(metrics, ignore_index=True).to_csv(a.root / f"FIT_METRICS_{a.tag}.tsv.gz", sep="\t", index=False, compression="gzip")
    if scores:
        pd.concat(scores, ignore_index=True).reindex(columns=v3.lb.COLUMNS).to_csv(a.root / f"RESULTS_grid_{a.tag}.tsv", sep="\t", index=False)
    flush(None)
    if couplings:
        pd.concat(couplings, ignore_index=True).to_csv(a.root / f"COUPLING_CLUSTERS_{a.tag}.tsv.gz", sep="\t", index=False, compression="gzip")
    print(st.groupby(["dataset", "state"]).size().to_string())
    if (st.state != "done").any():
        print(st[st.state != "done"].groupby(["dataset", "unit_id", "state"]).size().to_string())


if __name__ == "__main__":
    main()
