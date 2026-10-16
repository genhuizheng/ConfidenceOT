"""KP-Tracer tumour level: candidate-primary ranking and representation comparison (exploratory).

    python kp_lineage/tumour_level/01_ranking_and_representations.py \\
        --export results/kp_tracer_lineage_validation/export \\
        --out results/kp_tracer_lineage_validation/tumour_level_exploratory

Reads only the verified, downloaded export (readout_tumour_level/, tables/cells.csv.gz,
config/ot_runs/). The prespecified tumour-level readout (d_M, 02_kptracer_readout.py)
is not recomputed or changed; its numbers are re-read and checked.

**Runs.** Every KP-Tracer OT label present in the export. As of 2026-10-15 these are
ranknm256_noscale_ds_cos and raw only; the manifest says so rather than assuming more.

**Candidate ranking, per metastatic lesion.** The candidates are the lesion's completed
pairs (eligible primaries of the same mouse), scored by the source retained fraction r.
Positives are the same-family primaries (authors' MetFamily). A lesion is informative
when it has at least one positive and one negative candidate, the d_M entry rule.
Ties in r are handled exactly, as the expectation over a uniformly random order within
tied candidates:

* top1: probability that the first-ranked candidate is a positive.
* top2: probability that a positive is among the first two (lesions with >= 3 candidates).
* rr: expected reciprocal rank of the best-ranked positive (one value per lesion; its mean
  over lesions is the MRR); several positives are allowed.
* auroc: P(r_pos > r_neg) + 0.5 P(r_pos = r_neg) over all positive-negative candidate pairs
  (pairwise ranking accuracy).

Each has its exact expectation under a uniformly random ranking given the lesion's numbers
of candidates n and positives p: top1 p/n; top2 1 - C(n-p,2)/C(n,2); rr sum_j P(first
positive at j)/j with P = C(n-j, p-1)/C(n, p); auroc 1/2. Per lesion the excess over that
expectation is reported.

**Aggregation.** Lesion -> mouse (mean over the mouse's informative lesions) -> mean of
mouse means, with a 95% percentile bootstrap over mice (10,000 resamples, seed 20261015)
and the number of mice above their random expectation. Exploratory; no test replaces d_M.

**Representation comparison.** Paired on the lesions informative in both labels and on
the same mice. Per pair: r in each label and, from tables/cells.csv.gz, the share of
source cells whose retained/rejected label differs between the two labels (cells scored
in both).

Outputs: run_manifest.tsv, preprocessing_configurations.tsv, candidates.tsv,
lesion_ranking.tsv, mouse_ranking.tsv, ranking_summary.json, representation_pairs.tsv,
representation_mice.tsv, readout_check.json.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261015
BOOTSTRAP = 10_000
METRICS = ("top1", "top2", "rr", "auroc")


def first_positive_distribution(scores: np.ndarray, positive: np.ndarray) -> dict[int, float]:
    """P(first positive is at position k), ties broken uniformly at random."""
    order = sorted(set(scores.tolist()), reverse=True)
    before = 0
    for value in order:
        group = scores == value
        size, k = int(group.sum()), int((group & positive).sum())
        if k:
            return {before + j: comb(size - j, k - 1) / comb(size, k) for j in range(1, size - k + 2)}
        before += size
    return {}


def lesion_metrics(scores: np.ndarray, positive: np.ndarray) -> dict:
    n, p = len(scores), int(positive.sum())
    dist = first_positive_distribution(scores, positive)
    pos, neg = scores[positive], scores[~positive]
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    random_first = {j: comb(n - j, p - 1) / comb(n, p) for j in range(1, n - p + 2)}
    row = {
        "candidates": n, "positives": p,
        "top1": dist.get(1, 0.0), "top1_random": p / n,
        "rr": sum(prob / j for j, prob in dist.items()),
        "rr_random": sum(prob / j for j, prob in random_first.items()),
        "auroc": float((greater + 0.5 * ties) / (len(pos) * len(neg))), "auroc_random": 0.5,
        "tied_pos_neg_pairs": int(ties),
    }
    if n >= 3:
        row["top2"] = dist.get(1, 0.0) + dist.get(2, 0.0)
        row["top2_random"] = 1 - comb(n - p, 2) / comb(n, 2)
    else:
        row["top2"], row["top2_random"] = np.nan, np.nan
    for metric in METRICS:
        row[f"{metric}_excess"] = row[metric] - row[f"{metric}_random"]
    return row


def bootstrap(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    draws = values[rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))].mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    readout = args.export / "readout_tumour_level"
    summary = json.loads((readout / "readout_summary.json").read_text())
    labels = sorted(p.name for p in (args.export / "config" / "ot_runs").iterdir() if p.is_dir())

    # Run manifest and configurations, from each label's own run.json files.
    manifest_rows, config_rows = [], []
    for label in labels:
        runs = sorted((args.export / "config" / "ot_runs" / label).glob("*/run.json"))
        pairs = pd.read_csv(readout / f"pairs_{label}.csv")
        done = pairs[pairs["status"].eq("completed")]
        first = json.loads(runs[0].read_text())
        pre = first["preprocessing"]
        manifest_rows.append({
            "label": label, "completed_pairs": len(done), "eligible_pairs": int(pairs["eligible"].sum()),
            "mice_with_pairs": int(done["patient_id"].nunique()), "run_json_files": len(runs),
            "output_root": f"kp_tracer_confidenceot/ot/kptracer/{label}",
            "readout": f"kp_tracer_confidenceot/readout/pairs_{label}.csv",
            "gate": "M4-E", "within_side_acceptance": first.get("within_side_acceptance_minimum"),
            "calibration_null": first.get("calibration_null"), "representation_source": first.get("representation_source"),
            "cnv": first.get("cnv"),
        })
        qc = first.get("cell_qc", {})
        config_rows.append({
            "label": label, "input_counts": "equalised (27_downsample_counts.py, pooled 10th percentile)"
            if pre.get("equalise_depth") else "original counts",
            "cell_qc_where": "27_ before equalisation (1000 counts, 500 genes, 20% mito)" if pre.get("equalise_depth")
            else f"02_run_pair.py ({qc.get('minimum_total_counts')} counts, {qc.get('minimum_detected_genes')} genes, "
                 f"{qc.get('maximum_mitochondrial_percent')}% mito)",
            "transform": pre.get("transform"), "rank_top_n": pre.get("rank_top_n"), "n_hvg": pre.get("n_hvg"),
            "hvg": pre.get("joint_hvg"), "gene_scaling": "z-score before PCA" if pre.get("scale_genes") else "none",
            "n_pcs": pre.get("n_pcs"), "cost": pre.get("cost"), "cost_scaling": pre.get("scale"),
        })
    pd.DataFrame(manifest_rows).to_csv(args.out / "run_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(config_rows).to_csv(args.out / "preprocessing_configurations.tsv", sep="\t", index=False)

    # Candidates and per-lesion ranking.
    candidates, lesion_rows, check = [], [], {}
    for label in labels:
        pairs = pd.read_csv(readout / f"pairs_{label}.csv")
        pairs["same_met_family"] = pairs["same_met_family"].fillna(False).astype(bool)
        done = pairs[pairs["status"].eq("completed")].copy()
        done["label"] = label
        candidates.append(done[["label", "patient_id", "target_sample", "source_sample", "same_met_family",
                                "source_retained_fraction", "target_retained_fraction", "source_analyzed_n",
                                "target_analyzed_n"]])
        for (mouse, lesion), group in done.groupby(["patient_id", "target_sample"], sort=True):
            positive = group["same_met_family"].to_numpy(bool)
            base = {"label": label, "patient_id": mouse, "metastasis": lesion, "candidates": len(group),
                    "positives": int(positive.sum())}
            if positive.any() and (~positive).any():
                lesion_rows.append({**base, "informative": True,
                                    **lesion_metrics(group["source_retained_fraction"].to_numpy(float), positive)})
            else:
                lesion_rows.append({**base, "informative": False})
        # Re-read d_M and check it against the prespecified readout.
        contrast = pd.read_csv(readout / f"metastasis_contrast_source_{label}.csv")
        entering = contrast[contrast["enters_test"].astype(bool)]
        reported = summary["labels"][label]["primary_source"]
        check[label] = {"lesions": int(len(entering)), "reported_lesions": reported["metastases"],
                        "median_d": float(entering["d"].median()), "reported_median_d": reported["median_d"],
                        "match": bool(len(entering) == reported["metastases"]
                                      and abs(entering["d"].median() - reported["median_d"]) < 1e-12)}
    candidates = pd.concat(candidates, ignore_index=True)
    candidates.to_csv(args.out / "candidates.tsv", sep="\t", index=False)
    lesions = pd.DataFrame(lesion_rows)
    lesions.to_csv(args.out / "lesion_ranking.tsv", sep="\t", index=False)
    (args.out / "readout_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")

    informative = lesions[lesions["informative"]]
    mouse = informative.groupby(["label", "patient_id"]).agg(
        lesions=("metastasis", "size"), **{m: (m, "mean") for m in METRICS},
        **{f"{m}_random": (f"{m}_random", "mean") for m in METRICS},
        **{f"{m}_excess": (f"{m}_excess", "mean") for m in METRICS}).reset_index()
    mouse.to_csv(args.out / "mouse_ranking.tsv", sep="\t", index=False)
    result = {"note": "exploratory; the prespecified tumour-level endpoint is d_M (readout_summary.json)",
              "bootstrap": BOOTSTRAP, "seed": SEED, "labels": {}}
    for label in labels:
        m = mouse[mouse["label"].eq(label)]
        entry = {"informative_lesions": int((informative["label"] == label).sum()), "mice": int(len(m))}
        for metric in METRICS:
            values = m[metric].dropna().to_numpy(float)
            excess = m[f"{metric}_excess"].dropna().to_numpy(float)
            lesion_values = informative.loc[informative["label"].eq(label), metric].dropna()
            entry[metric] = {
                "lesions": int(len(lesion_values)), "mice": int(len(values)),
                "lesion_mean": float(lesion_values.mean()),
                "random_lesion_mean": float(informative.loc[informative["label"].eq(label), f"{metric}_random"].dropna().mean()),
                "mean_of_mouse_means": float(values.mean()), "bootstrap_95": bootstrap(values),
                "random_mean_of_mouse_means": float(m[f"{metric}_random"].dropna().mean()),
                "excess_mean_of_mouse_means": float(excess.mean()), "excess_bootstrap_95": bootstrap(excess),
                "mice_above_random": int((excess > 0).sum()), "mice_below_random": int((excess < 0).sum()),
            }
        result["labels"][label] = entry
    (args.out / "ranking_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Representation comparison, paired.
    if len(labels) >= 2:
        main, other = "ranknm256_noscale_ds_cos", "raw"
        wide = candidates.pivot_table(index=["patient_id", "target_sample", "source_sample", "same_met_family"],
                                      columns="label", values="source_retained_fraction").reset_index()
        cells = pd.read_csv(args.export / "tables" / "cells.csv.gz",
                            usecols=["label", "pair_id", "cell_id", "side", "retained"])
        cells = cells[cells["side"].eq("source")]
        gate = cells.pivot_table(index=["pair_id", "cell_id"], columns="label", values="retained", aggfunc="first").dropna()
        change = gate.groupby(level=0).apply(lambda g: pd.Series({
            "source_cells_in_both": len(g),
            "label_changed_share": float((g[main].astype(bool) != g[other].astype(bool)).mean()),
            "retained_main_rejected_other": float((g[main].astype(bool) & ~g[other].astype(bool)).mean()),
            "rejected_main_retained_other": float((~g[main].astype(bool) & g[other].astype(bool)).mean())}))
        wide["pair_id"] = [f"{m}__{s}__{t}" for m, s, t in zip(wide["patient_id"], wide["source_sample"], wide["target_sample"])]
        wide = wide.merge(change, left_on="pair_id", right_index=True, how="left")
        wide.to_csv(args.out / "representation_pairs.tsv", sep="\t", index=False)
        contrast = {label: pd.read_csv(readout / f"metastasis_contrast_source_{label}.csv").set_index(["patient_id", "metastasis"])
                    for label in (main, other)}
        both = contrast[main][contrast[main]["enters_test"].astype(bool)].index.intersection(
            contrast[other][contrast[other]["enters_test"].astype(bool)].index)
        paired = pd.DataFrame({main: contrast[main].loc[both, "d"], other: contrast[other].loc[both, "d"]}).reset_index()
        per_mouse = paired.groupby("patient_id").agg(lesions=("metastasis", "size"),
                                                     **{f"median_d_{k}": (k, "median") for k in (main, other)},
                                                     **{f"positive_share_{k}": (k, lambda v: float((v > 0).mean()))
                                                        for k in (main, other)}).reset_index()
        per_mouse.to_csv(args.out / "representation_mice.tsv", sep="\t", index=False)
        result["representation_comparison"] = {
            "paired_lesions": int(len(both)), "paired_mice": int(per_mouse["patient_id"].nunique()),
            "median_d": {k: float(paired[k].median()) for k in (main, other)},
            "positive_lesions": {k: int((paired[k] > 0).sum()) for k in (main, other)},
            "mice_median_d_above_0": {k: int((per_mouse[f"median_d_{k}"] > 0).sum()) for k in (main, other)},
            "mice_where_main_median_exceeds_raw": int((per_mouse[f"median_d_{main}"] > per_mouse[f"median_d_{other}"]).sum()),
            "source_retained_fraction_median": {k: float(candidates.loc[candidates["label"].eq(k), "source_retained_fraction"].median())
                                                for k in (main, other)},
            "label_changed_share_median_over_pairs": float(wide["label_changed_share"].median()),
        }
        (args.out / "ranking_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"check": check, **result}, indent=2))


if __name__ == "__main__":
    main()
