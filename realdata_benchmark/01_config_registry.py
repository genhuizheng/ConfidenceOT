"""PREPROCESSING_CONFIG_REGISTRY.tsv: every configuration of the joint benchmark (JOINT_BENCHMARK_PROTOCOL.md), each
described from the code that builds it, never from its name.

    python realdata_benchmark/01_config_registry.py --out OUT_DIR

Two kinds of rows:
  historical  the labels this project executed before (the 32-arm grid, log1p_noscale_pcc, the scripted log1p
              dimension arms), described from confidenceot.Preprocessing.from_label + Preprocessing.representation
              + common.prepare_joint_representation + 27_downsample_counts.py + 02_run_pair.py defaults, with where
              they ran. raw* labels are a record only (in_new_grid False; user 2026-10-31: raw arms removed, log1p =
              the logcpm arm kept); so are the code-label log1p arms (no library size), which are not in the matrix.
  registered  the pre-specified contrasts R1-R6 of the protocol (config_id = encoding|ds|scale|cost), with role
              main or sensitivity and the contrasts each belongs to.
Every row carries the exact factor values (input depth and seed, library-size normalisation, rank transformation,
gene-median division, Top-K boundary, tie rule, HVG selection, gene scaling, PCA dimension, cost geometry, median cost
scaling, entropic regularisation, calibration). Nothing is removed for its result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from confidenceot import Preprocessing

GRID = [f"{stem}{scale}{ds}{cos}" for stem in ("raw", "logcpm", "rank256", "ranknm256")
        for scale in ("", "_noscale") for ds in ("", "_ds") for cos in ("", "_cos")]
EXTRA = ["log1p_noscale_pcc", "log1p_noscale", "log1p_allgenes_noscale", "log1p_nopca_noscale", "log1p_allgenes_nopca_noscale"]
EXECUTED = {label: ["synthetic native benchmark, acceptance 0.90 (benchmark/ARCHIVE_MANIFEST.md, 32 arms)",
                    "real-data pan-cancer preprocessing factorial (cancer_metastasis/submit_preprocessing_factorial.sh default grid)"]
            for label in GRID}
for label in ("ranknm256_noscale_ds_cos", "ranknm256_noscale_cos", "raw"):
    EXECUTED[label] = EXECUTED[label] + ["synthetic benchmark, acceptance 0.95 and 0.99 (ARCHIVE_MANIFEST.md)"]
EXECUTED["ranknm256_noscale_ds_cos"] += [
    "CNV/RNA strategy comparison (submit_cnv_rna_strategies.sh)", "KP-Tracer runs and lineage Level 1-3 (method B)",
    "Quinn A549 lineage runs", "MOSTA all stages", "LTL331 GSE297328 (frozen)"]
EXECUTED["raw"] += ["KP-Tracer second arm (kp_lineage/submit_kptracer_chain.sh)", "cycling x CNV factorial"]
EXECUTED["logcpm"] += ["MOSTA logcpm arm", "LTL331 GSE297328 (frozen)"]
EXECUTED["logcpm_noscale"] = EXECUTED["logcpm_noscale"] + ["KP-Tracer / Quinn lineage Level 2-3 (method A, balanced OT)"]
EXECUTED["log1p_noscale_pcc"] = ["synthetic benchmark, acceptance 0.99 / 0.95 / 0.90 (benchmark/EXPERIMENT_2026-10-19_LOG1P_PEARSON.md)"]
for label in EXTRA[1:]:
    EXECUTED[label] = ["scripted (benchmark/tacc/submit_log1p_dimensions.sh); no run record in the repository"]

LIB = {"precomputed": "none: counts used as supplied", "log1p": "none: log1p(counts), no library-size division",
       "log_cpm": "each cell scaled to 1e4 total, then log1p",
       "rank_value": "each cell divided by its total (does not change the within-cell order)",
       "rank_no_median": "each cell divided by its total (does not change the within-cell order)"}
TIE = {"production": "stable argsort over the stored row order of the joint matrix as Preprocessing.representation builds it "
                     "(implementation-defined permutation; rb_common.joint_counts)",
       "expected": "average over random tie-breaking (tie groups share their slots' mean value; a group straddling the cut shares the slots)",
       "alphabetical": "ascending gene-key order", "random": "uniform random, seeded",
       "midrank": "tied genes share their average rank", "order": "ascending gene-key order"}
DS_TARGET = {"none": "none", "P10": "2,817 (pooled 10th percentile)", "P25": "5,023 (pooled 25th percentile)", "P50": "9,360 (pooled 50th percentile)"}
COMMON = {"pca_dimension": "30, sklearn PCA on both sides stacked, random_state = run seed (20260831 + pair index)",
          "median_cost_scaling": "cost / median of up to 1,000,000 sampled cross-side squared distances, zeros excluded",
          "entropic_regularisation": "epsilon 0.1, lambda_a = lambda_b = 1, tolerance 1e-4",
          "calibration": "within-side split null, 5 + 5 replicates, grid 5; acceptance 0.99 (main), 0.90 (Arm A comparability)"}


def historical(label: str) -> dict:
    c = Preprocessing.from_label(label)
    rank = c.normalisation in ("rank_value", "rank_no_median")
    return {"config_id": f"label:{label}", "kind": "historical", "label": label, "role": "historical",
            "in_new_grid": not label.startswith("raw") and c.normalisation != "log1p", "contrasts": "H" if not label.startswith("raw") and c.normalisation != "log1p" else "",
            "encoding": c.normalisation, "K": c.rank_top_n if rank else None, "H": None, "tie": "production" if rank else None,
            "ds": "P10" if c.equalise_depth else "none", "ds_seed": "frozen (27_)" if c.equalise_depth else None,
            "scale": c.scale_genes, "cost": {"cosine": "cos", "squared_euclidean": "sqeuc", "pearson": "pearson"}[c.cost],
            "input_counts": "depth-equalised counts (27_)" if c.equalise_depth else "original counts after QC",
            "downsampling_target": DS_TARGET["P10" if c.equalise_depth else "none"],
            "library_size_normalisation": LIB[c.normalisation],
            "rank_transformation": f"top {c.rank_top_n} genes per cell, value 1 - r/{c.rank_top_n}, others 0" if rank else "none",
            "gene_median_normalisation": "each gene / its nonzero median over both sides" if c.normalisation == "rank_value" else ("none" if rank else "not applicable"),
            "tie_breaking": TIE["production"] if rank else "not applicable (HVG variance ties keep column order)",
            "hvg_selection": "every common gene" if c.n_hvg == 0 else f"top {c.n_hvg} by variance of the transformed matrix over both sides",
            "gene_scaling": "centred and divided by sd" if c.scale_genes else "none (PCA centres)",
            "cost_geometry": {"cosine": "unit rows, 2 - 2 cos", "squared_euclidean": "squared Euclidean", "pearson": "2 - 2 r over genes"}[c.cost],
            **COMMON, "pca_dimension": "none" if c.n_pcs == 0 else COMMON["pca_dimension"],
            "executed_in": " | ".join(EXECUTED.get(label, ["no run record"]))}


def registered(encoding, K=None, H=None, tie=None, ds="none", ds_seed=None, scale=False, cost="cos", role="main", contrasts=""):
    rank = encoding in ("ranknm", "rankhvg")
    cid = (f"ranknm{K}-{tie}" if encoding == "ranknm" else f"rankhvg{H}-{tie}" if encoding == "rankhvg" else "logcpm") + \
          f"|{ds}{'' if ds == 'none' else 's' + str(ds_seed)}|{'scale' if scale else 'noscale'}|{cost}"
    return {"config_id": cid, "kind": "registered", "label": "", "role": role, "in_new_grid": True, "contrasts": contrasts,
            "encoding": encoding, "K": K, "H": H, "tie": tie, "ds": ds, "ds_seed": ds_seed, "scale": scale, "cost": cost,
            "input_counts": "original counts after QC" if ds == "none" else ("frozen 27_ equalised counts" if ds_seed == "frozen" else f"original counts equalised here with confidenceot.equalise_depth, seed {ds_seed}"),
            "downsampling_target": DS_TARGET[ds],
            "library_size_normalisation": "each cell scaled to 1e4 total, then log1p" if encoding == "logcpm" else
                                          ("each cell divided by its total (order unchanged)" if encoding == "ranknm" else "HVG selection on log1p(1e4 x count / total); ranks on counts"),
            "rank_transformation": (f"top {K} genes per cell over all common genes, value 1 - r/{K}, others 0" if encoding == "ranknm" else
                                    f"expressed genes within the {H} HVGs, value 1 - (r - 1)/{H}, zeros stay 0, no Top-K boundary" if encoding == "rankhvg" else "none"),
            "gene_median_normalisation": "none" if rank else "not applicable",
            "tie_breaking": TIE.get(tie, "not applicable") if rank else "not applicable",
            "hvg_selection": ("top 2,000 by variance of the rank encoding over both sides" if encoding == "ranknm" else
                              f"top {H} by variance of log1p(1e4 x count / total) over both sides, before encoding" if encoding == "rankhvg" else
                              "top 2,000 by variance of log1p(1e4 x count / total) over both sides"),
            "gene_scaling": "centred and divided by sd" if scale else "none (PCA centres)",
            "cost_geometry": "unit rows, 2 - 2 cos" if cost == "cos" else "squared Euclidean",
            **COMMON, "executed_in": "new (this benchmark)"}


def matrix() -> list[dict]:
    rows = {}

    def add(r, contrast):
        k = r["config_id"]
        if k in rows:
            rows[k]["contrasts"] = ",".join(sorted((set(rows[k]["contrasts"].split(",")) | {contrast}) - {""}))
            if r["role"] == "main":
                rows[k]["role"] = "main"
        else:
            r["contrasts"] = contrast; rows[k] = r
    for K in (128, 256, 512, 1024, 2048):                                     # R1 Top-K truncation
        add(registered("ranknm", K=K, tie="expected"), "R1")
    for K in (256, 1024):                                                     # R2 tie handling
        for ds, seed in (("none", None), ("P10", "frozen")):
            for tie in ("production", "expected", "random1", "random2", "random3"):
                add(registered("ranknm", K=K, tie=tie, ds=ds, ds_seed=seed, role="main" if tie in ("production", "expected") else "sensitivity"), "R2")
    for H in (2000, 3000):                                                    # R3 HVG-restricted ranking
        add(registered("rankhvg", H=H, tie="midrank"), "R3")
        add(registered("rankhvg", H=H, tie="order", role="sensitivity"), "R3")
    for K in (1024, 2048):
        add(registered("ranknm", K=K, tie="expected"), "R3")
    add(registered("logcpm"), "R3")
    reps = [dict(encoding="logcpm"), dict(encoding="ranknm", K=256, tie="production"),
            dict(encoding="ranknm", K=1024, tie="expected"), dict(encoding="rankhvg", H=2000, tie="midrank")]
    for rep in reps:                                                          # R4 downsampling, R5 cost
        add(registered(**rep), "R4")
        add(registered(**rep, ds="P10", ds_seed="frozen"), "R4")
        for ds in ("P10", "P25", "P50"):
            for seed in (1, 2, 3):
                add(registered(**rep, ds=ds, ds_seed=seed, role="main" if seed == 1 else "sensitivity"), "R4")
        for ds, seed in (("none", None), ("P10", "frozen")):
            add(registered(**rep, ds=ds, ds_seed=seed, cost="sqeuc"), "R5")
            add(registered(**rep, ds=ds, ds_seed=seed), "R5")
    for rep in reps[:2]:                                                      # R6 gene scaling
        for ds, seed in (("none", None), ("P10", "frozen")):
            add(registered(**rep, ds=ds, ds_seed=seed, scale=True), "R6")
            add(registered(**rep, ds=ds, ds_seed=seed), "R6")
    return list(rows.values())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    target = a.out / "PREPROCESSING_CONFIG_REGISTRY.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    a.out.mkdir(parents=True, exist_ok=True)
    rows = [historical(label) for label in GRID + EXTRA] + matrix()
    t = pd.DataFrame(rows); t.to_csv(target, sep="\t", index=False)
    print(t.groupby(["kind", "role", "in_new_grid"]).size().to_string()); print(f"{len(t)} rows -> {target}")


if __name__ == "__main__":
    main()
