"""Exploratory baselines of the lineage validation (VALIDATION_PROTOCOL.md §10).

    python lineage_validation/04_baselines.py --root ROOT --task-index I --task-count N

For the single strategies (rna, cnv, rna_cnv) of every pair:
1. Rebuild the cost from the run's joint_pca_coordinates.csv.gz and run.json:
   C = squared_euclidean(source PCs, target PCs) / cost_scale, the call 02_run_pair.py makes.
   Accept it only if log P + C~/eps is additively separable (f_i + g_j) to 1e-6 on every entry,
   where P is the saved M4-E coupling, C~ = C on retained x retained and the rejection cost
   elsewhere, eps = 0.1 (Level 3 §4 check). A run that fails gets no baseline and is listed.
2. Balanced entropic OT on C (eps 0.1, uniform marginals) and kNN (k = 15: each source cell
   spreads 1/n equally over its 15 nearest target cells under C).
3. Score both by §8 with every cell retained (rho = tau = 1); no shuffle.
Writes ROOT/scores/baselines/<strategy>/<block>/<pair>/: check.json, <baseline>/summary.json,
groups.tsv, bins.tsv, curve.tsv; SUCCESS. A run without a completed OT run gets MISSING.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lv_common as lv  # noqa: E402

EPSILON = 0.1
K = 15
SEPARABLE = 1e-6

def rebuild_cost(directory: Path, run: dict) -> tuple[np.ndarray, dict]:
    from confidenceot.preprocessing import squared_euclidean
    meta = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    coords = pd.read_csv(directory / "joint_pca_coordinates.csv.gz", dtype={"observation_id": str})
    pcs = [c for c in coords.columns if c.startswith("PC")]
    src = coords[coords["side"].eq("primary")]
    tgt = coords[coords["side"].eq("metastasis")]
    if not (np.array_equal(src["observation_id"].to_numpy(), run["source"]) and
            np.array_equal(tgt["observation_id"].to_numpy(), run["target"])):
        raise ValueError("coordinate rows are not in cell_confidence order")
    xs, xt = src[pcs].to_numpy(np.float32), tgt[pcs].to_numpy(np.float32)
    cost = squared_euclidean(xs, xt) / float(meta["cost_scale"])
    gated = np.where(np.outer(run["rho"], run["tau"]), cost, float(meta["rejection_cost"]))
    P = run["P"]
    check = {"cost_scale": meta["cost_scale"], "rejection_cost": meta["rejection_cost"], "epsilon": EPSILON,
             "zero_entries": int((P <= 0).sum())}
    if check["zero_entries"]:
        check.update(separable=False, max_residual=None, reason="coupling has zero entries; the check needs every entry")
        return cost, check
    X = np.log(P) + gated / EPSILON
    residual = X - X.mean(axis=1, keepdims=True) - X.mean(axis=0, keepdims=True) + X.mean()
    check["max_residual"] = float(np.abs(residual).max())
    check["separable"] = bool(check["max_residual"] <= SEPARABLE)
    return cost, check


def baselines(cost: np.ndarray) -> dict:
    from traditional_ot.balanced import balanced_ot
    n, m = cost.shape
    result = balanced_ot(cost, epsilon=EPSILON)
    knn = np.zeros_like(cost)
    k = min(K, m)
    nearest = np.argpartition(cost, k - 1, axis=1)[:, :k]
    np.put_along_axis(knn, nearest, 1.0 / (n * k), axis=1)
    return {"balanced": (result.coupling, {"converged": bool(result.converged), "marginal_error": float(result.marginal_error),
                                           "iterations": int(result.n_iterations)}),
            "knn": (knn, {"k": k})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acceptance", default="0.99")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    args = parser.parse_args()
    lv.check_frozen()
    score03 = lv.script("score03", "03_score_runs.py")
    labels = score03.Labels(args.root)
    items = [i for i in score03.run_items(args.root) if i["strategy"] in lv.SINGLE][args.task_index::args.task_count]
    failed = 0
    for item in items:
        out = args.root / "scores" / "baselines" / item["strategy"] / item["block"] / item["pair"]
        if (out / "SUCCESS").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        for stale in ("FAILED", "MISSING"):
            (out / stale).unlink(missing_ok=True)
        try:
            run = lv.load_run(args.root / "runs" / f"acc{args.acceptance}", item["strategy"], item["block"], item["pair"], "M4-E")
        except lv.RunMissing as error:
            (out / "MISSING").write_text(f"{error}\n", encoding="utf-8")
            continue
        try:
            cost, check = rebuild_cost(Path(run["directory"]), run)
            lv.write_json(out / "check.json", check)
            if not check["separable"]:
                (out / "FAILED").write_text(f"cost rebuild not accepted: {check}\n", encoding="utf-8")
                print(f"no baseline {item['strategy']} {item['pair']}: rebuild not accepted", flush=True)
                continue
            lab_s = labels.of(item["dataset"], item["mouse"], run["source"])
            lab_t = labels.of(item["dataset"], item["mouse"], run["target"])
            ones_s, ones_t = np.ones(len(lab_s), dtype=bool), np.ones(len(lab_t), dtype=bool)
            zeros_s, zeros_t = np.zeros(len(lab_s)), np.zeros(len(lab_t))
            ks = (1, 3) if item["dataset"] == "Quinn2021" else (1,)
            for name, (coupling, info) in baselines(cost).items():
                scored = lv.score_cells(coupling, ones_s, ones_t, zeros_s, zeros_t, lab_s, lab_t, item["dataset"],
                                        shuffles=0, seed_key="", ks=ks)
                where = out / name
                where.mkdir(exist_ok=True)
                write = dict(sep="\t", index=False, lineterminator="\n")
                scored["groups"].to_csv(where / "groups.tsv", **write)
                scored["bins"].to_csv(where / "bins.tsv", **write)
                scored["curve"].to_csv(where / "curve.tsv", **write)
                lv.write_json(where / "summary.json", {**{k: item[k] for k in ("block", "dataset", "strategy", "pair", "mouse")},
                                                       "baseline": name, **info, **scored["summary"]})
            (out / "SUCCESS").write_text("done\n", encoding="utf-8")
            print(f"baselines {item['strategy']} {item['pair']}: done", flush=True)
        except Exception as error:  # noqa: BLE001 - listed
            failed += 1
            (out / "FAILED").write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
            print(f"FAILED {item['strategy']} {item['pair']}: {error}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
