"""Mouse-level aggregation and the pre-specified tests of the lineage validation (VALIDATION_PROTOCOL.md §11).

    python lineage_validation/05_aggregate.py --root ROOT [--replace]

Reads ROOT/scores (03_, 04_) and writes ROOT/summary/:
  run_status.tsv       every expected (method, strategy, block, pair): scored / MISSING / FAILED / not scored, with the reason
  runs.tsv             one row per scored run: every summary.json field
  baselines.tsv        one row per baseline run; baseline_status.tsv
  mouse_values.tsv     per (method, strategy, block, metric, mouse): mean over the mouse's runs and the run count
  tests.tsv            T1 and T2 (KP-Tracer, rna, M4-E, D_A runs with site truth): n mice, mean and median of mouse
                       means, 95% bootstrap interval, mice > 0, one-sided Wilcoxon p, Holm p, verdict
  lomo.tsv             leave one mouse out for T1 and T2
  companions.tsv       the quantities §11 reports beside T1 and T2, same runs, with intervals
  exploratory.tsv      every other (method, strategy, block, metric): mean of mouse means, interval, n (no tests)
  paired_vs_rna.tsv    per (method, block, metric, strategy): mouse-level paired differences from rna, interval
  negative_controls.tsv KP-Tracer D_B by B case: false asserted mass F and rejection rates
  REPORT.md            the tests in words, the run counts and every missing or failed run
With --replace an existing summary/ is moved to summary_<time>/ first; otherwise it refuses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lv_common as lv  # noqa: E402

BOOTSTRAP = 10_000
MIN_MICE = 5
METRICS = ["e_M", "site_top1", "site_auroc", "target_site_retained", "e_s", "mean_t_related", "spearman_t_f",
           "E2E", "CH", "E2E_minus_CH", "E2E_ind", "COND", "COND_ind", "COND_minus_ind",
           "top1_E2E", "top1_COND", "top1_chance", "top1_ind_COND", "top3_E2E", "top3_COND", "top3_chance", "top3_ind_COND",
           "asserted_A", "asserted_C", "asserted_F", "asserted_U", "asserted_precision", "asserted_precision_ind",
           "cov_T", "cov_S", "target_retained", "target_rejection_auroc", "source_rejection_auroc",
           "E2E_bal", "E2E_bal_raw", "CH_bal", "COND_bal", "curve_area",
           "E2E_ge5", "CH_ge5", "COND_ge5", "tree_dS_norm", "tree_dR", "tree_target_share",
           "shuffle_E2E_p", "shuffle_COND_p", "shuffle_top1_COND_p", "shuffle_precision_p"]
COMPANIONS = {"T1": ["site_top1", "site_auroc", "target_site_retained"],
              "T2": ["COND_minus_ind", "top1_COND", "cov_T", "E2E_bal", "E2E", "CH"]}


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(values) < 2:
        return np.nan, np.nan
    draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def test(values: pd.Series, seed_key: str) -> dict:
    values = values.dropna()
    x = values.to_numpy(float)
    lo, hi = bootstrap(x, lv.rng_for(seed_key))
    row = {"mice": len(x), "mean": float(x.mean()) if len(x) else np.nan, "median": float(np.median(x)) if len(x) else np.nan,
           "ci_low": lo, "ci_high": hi, "mice_above_0": int((x > 0).sum()), "values": json.dumps({k: float(v) for k, v in values.items()})}
    if len(x) >= MIN_MICE and np.any(x != 0):
        row["p"] = float(wilcoxon(x, alternative="greater").pvalue)
    else:
        row["p"] = np.nan
        row["note"] = f"not tested: {len(x)} mice (needs >= {MIN_MICE})" if len(x) < MIN_MICE else "all mouse values 0"
    return row


def holm(pvalues: dict) -> dict:
    keys = [k for k, p in pvalues.items() if np.isfinite(p)]
    ordered = sorted(keys, key=lambda k: pvalues[k])
    out, running = {k: np.nan for k in pvalues}, 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * pvalues[key]))
        out[key] = running
    return out


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    module = lv.script("score03", "03_score_runs.py")
    status, rows = [], []
    for item in module.run_items(root):
        for method in lv.METHODS:
            out = root / "scores" / method / item["strategy"] / item["block"] / item["pair"]
            base = {"method": method, **{k: item[k] for k in ("block", "dataset", "strategy", "pair", "mouse")}}
            if (out / "SUCCESS").exists():
                status.append({**base, "status": "scored", "reason": ""})
                summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
                summary.pop("checks", None)
                rows.append(summary)
            elif (out / "MISSING").exists():
                status.append({**base, "status": "MISSING", "reason": (out / "MISSING").read_text(encoding="utf-8").strip()})
            elif (out / "FAILED").exists():
                status.append({**base, "status": "FAILED", "reason": (out / "FAILED").read_text(encoding="utf-8").splitlines()[0]})
            else:
                status.append({**base, "status": "not scored", "reason": "03_score_runs.py has not run on it"})
    return pd.DataFrame(status), pd.DataFrame(rows)


def mouse_values(runs: pd.DataFrame) -> pd.DataFrame:
    present = [m for m in METRICS if m in runs]
    long = runs.melt(id_vars=["method", "strategy", "block", "mouse", "pair"], value_vars=present, var_name="metric")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return (long.groupby(["method", "strategy", "block", "metric", "mouse"])["value"]
            .agg(value="mean", runs="size").reset_index())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    lv.check_frozen()
    out = args.root / "summary"
    if out.exists():
        if not args.replace:
            sys.exit(f"{out} exists; pass --replace to move it to summary_<time>/ first")
        out.rename(args.root / f"summary_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True)
    write = dict(sep="\t", index=False, lineterminator="\n")
    status, runs = collect(args.root)
    status.to_csv(out / "run_status.tsv", **write)
    runs.to_csv(out / "runs.tsv", **write)
    if runs.empty:
        (out / "REPORT.md").write_text("No scored run yet.\n", encoding="utf-8")
        return

    values = mouse_values(runs)
    values.to_csv(out / "mouse_values.tsv", **write)

    # §11 pre-specified tests: KP-Tracer, rna, M4-E, D_A runs with site truth.
    primary = runs[runs["method"].eq("M4-E") & runs["strategy"].eq("rna") & runs["block"].eq("kp_D_A")]
    if "site_truth" in primary:
        primary = primary[primary["site_truth"].astype(str).str.lower().eq("true")]
    tests, lomo, companions = [], [], []
    for name, metric in (("T1", "e_M"), ("T2", "E2E_minus_CH")):
        per_mouse = pd.to_numeric(primary.get(metric), errors="coerce").groupby(primary["mouse"]).mean() if len(primary) else pd.Series(dtype=float)
        tests.append({"test": name, "metric": metric, "runs": int(primary[metric].notna().sum()) if metric in primary else 0,
                      **test(per_mouse, f"bootstrap/{name}")})
        for mouse in per_mouse.index:
            rest = per_mouse.drop(mouse)
            lomo.append({"test": name, "left_out": mouse, "left_out_value": float(per_mouse[mouse]),
                         **{k: v for k, v in test(rest, f"bootstrap/{name}/without/{mouse}").items() if k != "values"}})
        for metric_c in COMPANIONS[name]:
            if metric_c in primary:
                pm = pd.to_numeric(primary[metric_c], errors="coerce").groupby(primary["mouse"]).mean()
                row = test(pm, f"bootstrap/{name}/{metric_c}")
                row.pop("p"); row.pop("note", None)
                companions.append({"test": name, "metric": metric_c, **row})
    adjusted = holm({t["test"]: t["p"] for t in tests})
    for t in tests:
        t["p_holm"] = adjusted[t["test"]]
        t["verdict"] = ("supported" if np.isfinite(t["p_holm"]) and t["p_holm"] < 0.05 and t["mean"] > 0 else "inconclusive")
    pd.DataFrame(tests).to_csv(out / "tests.tsv", **write)
    pd.DataFrame(lomo).to_csv(out / "lomo.tsv", **write)
    pd.DataFrame(companions).to_csv(out / "companions.tsv", **write)

    # Exploratory: every configuration, estimates and intervals only.
    explore = []
    for (method, strategy, block, metric), part in values.groupby(["method", "strategy", "block", "metric"]):
        series = part.set_index("mouse")["value"]
        row = test(series, f"bootstrap/{method}/{strategy}/{block}/{metric}")
        row.pop("p"); row.pop("note", None)
        explore.append({"method": method, "strategy": strategy, "block": block, "metric": metric, **row})
    pd.DataFrame(explore).to_csv(out / "exploratory.tsv", **write)
    paired = []
    wide = values.pivot_table(index=["method", "block", "metric", "mouse"], columns="strategy", values="value")
    for strategy in lv.STRATEGIES:
        if strategy == "rna" or strategy not in wide or "rna" not in wide:
            continue
        diff = (wide[strategy] - wide["rna"]).dropna()
        for (method, block, metric), part in diff.groupby(level=["method", "block", "metric"]):
            series = part.droplevel(["method", "block", "metric"])
            row = test(series, f"bootstrap/paired/{method}/{block}/{metric}/{strategy}")
            row.pop("p"); row.pop("note", None)
            paired.append({"method": method, "block": block, "metric": metric, "strategy": strategy,
                           "difference": f"{strategy} - rna", **row})
    pd.DataFrame(paired).to_csv(out / "paired_vs_rna.tsv", **write)

    controls = runs[runs["block"].eq("kp_D_B")]
    if "B_case" in controls:
        keep = [c for c in ("asserted_F", "asserted_A", "asserted_precision", "cov_S", "target_retained") if c in controls]
        (controls.groupby(["method", "strategy", "B_case"])[keep].agg(["mean", "count"])
         .reset_index().to_csv(out / "negative_controls.tsv", sep="\t", index=False, lineterminator="\n"))

    base_status, base_rows = [], []
    for path in sorted((args.root / "scores" / "baselines").glob("*/*/*")):
        state = "scored" if (path / "SUCCESS").exists() else "MISSING" if (path / "MISSING").exists() else \
            "FAILED" if (path / "FAILED").exists() else "not scored"
        base_status.append({"strategy": path.parent.parent.name, "block": path.parent.name, "pair": path.name, "status": state})
        for name in ("balanced", "knn"):
            if (path / name / "summary.json").exists():
                base_rows.append(json.loads((path / name / "summary.json").read_text(encoding="utf-8")))
    pd.DataFrame(base_status).to_csv(out / "baseline_status.tsv", **write)
    pd.DataFrame(base_rows).to_csv(out / "baselines.tsv", **write)

    lines = ["# Lineage validation summary", "", f"Protocol: {lv.FROZEN}. Code commit {lv.commit()}.", "",
             "## Pre-specified tests (KP-Tracer, rna, M4-E, D_A runs with site truth)", ""]
    for t in tests:
        lines.append(f"- **{t['test']}** ({t['metric']}): {t['mice']} mice, mean of mouse means {t['mean']:.4g} "
                     f"(95% bootstrap {t['ci_low']:.4g} to {t['ci_high']:.4g}), median {t['median']:.4g}, "
                     f"{t['mice_above_0']} mice > 0, one-sided Wilcoxon p {t['p']:.4g}, Holm p {t['p_holm']:.4g}: "
                     f"**{t['verdict']}**." + (f" {t.get('note')}" if isinstance(t.get("note"), str) else ""))
    lines += ["", "## Runs", ""]
    counts = status.groupby(["method", "status"]).size().unstack(fill_value=0)
    lines += ["```", counts.to_string(), "```"]
    bad = status[~status["status"].eq("scored")]
    lines += ["", f"## Not scored ({len(bad)})", ""]
    lines += [f"- {r.method} {r.strategy} {r.block} {r.pair}: {r.status} - {r.reason}" for r in bad.itertuples()]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lv.write_json(out / "provenance.json", {"commit": lv.commit(), "protocol": lv.FROZEN, "bootstrap": BOOTSTRAP, "seed": lv.SEED})
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")
    print("\n".join(lines[:12]), flush=True)


if __name__ == "__main__":
    main()
