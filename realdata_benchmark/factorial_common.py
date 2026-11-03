"""Shared pieces of the 592-configuration preprocessing factorial (lineage_validation/preprocessing_factorial_20261103/
PLAN.md section 3; the Splatter factorial uses the same grid): the grid, depth handling, the encoders and the
02_run_pair.py call.

Every grid configuration goes through one path: counts (original, or equalised to a pooled quantile with
confidenceot.equalise_depth) -> the encoding computed here on the joint matrix (rb_common.joint_counts, so rank ties
follow the production stored order where the rule says so) -> written as a precomputed matrix -> 02_run_pair.py with
label bmenc[_allgenes][_noscale][_cos] and --n-pcs 50, which selects 2,000 top-variance encoded genes (all H genes
for rankhvg: _allgenes), scales genes if asked, runs PCA 50, builds the cost and divides it by the median pair cost.

  logcpm      log1p(1e4 x count / total)
  ranknm      rank_encode(K, gene_median=False, tie rule): Top-K per cell, value 1 - r / K
  rank_value  rank_encode(K, gene_median=True, tie rule): the same after dividing each gene by its nonzero median
  rankhvg     rankhvg_encode(H, tie rule): ranks within the H top-variance log-CPM genes, no Top-K cut
Tie rules: production / expected / midrank (rb_common). For rankhvg, expected equals midrank (no cut); both are run.
"""

from __future__ import annotations

import itertools

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import rb_common as rb

N_PCS = 50
N_HVG = 2000
DS_LEVELS = ("none", "P10", "P25", "P50")
DS_SEED = 1
TIES = ("production", "expected", "midrank")
KS = (128, 256, 512, 1024, 2048)
HS = (2000, 3000)
SIDE_SEED = {"source": 1000 * DS_SEED + 1, "target": 1000 * DS_SEED + 2}


def config_id(s: dict) -> str:
    enc = s["encoding"]
    head = "logcpm" if enc == "logcpm" else f"{enc}{int(s['K'] if enc != 'rankhvg' else s['H'])}-{s['tie']}"
    ds = s["ds"] if s["ds"] == "none" else f"{s['ds']}s{DS_SEED}"
    return f"{head}|{ds}|{'scale' if s['scale'] else 'noscale'}|{s['cost']}|pca{N_PCS}"


def grid() -> list[dict]:
    """The 592 configurations, in a fixed order."""
    rows = []
    tail = list(itertools.product(DS_LEVELS, (False, True), ("cos", "sqeuc")))
    for ds, scale, cost in tail:
        rows.append(dict(encoding="logcpm", K=np.nan, H=np.nan, tie="", ds=ds, scale=scale, cost=cost))
    for enc in ("ranknm", "rank_value"):
        for K, tie in itertools.product(KS, TIES):
            for ds, scale, cost in tail:
                rows.append(dict(encoding=enc, K=K, H=np.nan, tie=tie, ds=ds, scale=scale, cost=cost))
    for H, tie in itertools.product(HS, TIES):
        for ds, scale, cost in tail:
            rows.append(dict(encoding="rankhvg", K=np.nan, H=H, tie=tie, ds=ds, scale=scale, cost=cost))
    for r in rows:
        r.update(config_id=config_id(r), role="grid", n_pcs=N_PCS, n_hvg=N_HVG, ds_seed=DS_SEED if r["ds"] != "none" else np.nan, label="")
    return rows


def registry(reproduction: list[dict] | None = None) -> pd.DataFrame:
    """The grid plus reproduction rows (a production label fitted exactly as an earlier run, e.g.
    {'config_id': 'label:ranknm256_noscale_ds_cos|pca30', 'label': 'ranknm256_noscale_ds_cos', 'n_pcs': 30})."""
    rows = grid() + [dict(role="reproduction", encoding="label", K=np.nan, H=np.nan, tie="", ds="frozen", scale=np.nan, cost="",
                          n_hvg=N_HVG, ds_seed=np.nan, **r) for r in (reproduction or [])]
    t = pd.DataFrame(rows)
    if t.config_id.duplicated().any():
        raise ValueError("duplicate config ids")
    cols = ["config_id", "role", "encoding", "K", "H", "tie", "ds", "ds_seed", "scale", "cost", "n_hvg", "n_pcs", "label"]
    return t[cols]


def equalise_sides(source: ad.AnnData, target: ad.AnnData, ds: str, targets: dict) -> tuple[ad.AnnData, ad.AnnData]:
    """Original counts for ds 'none'; otherwise each side equalised to the pooled quantile targets[ds] (counts per cell)
    with confidenceot.equalise_depth, seeds 1001 / 1002 (rb_common.equalised; deterministic per side and data)."""
    if ds == "none":
        return source, target
    t = int(targets[ds])
    return rb.equalised(source, t, SIDE_SEED["source"]), rb.equalised(target, t, SIDE_SEED["target"])


def encode_sides(spec: dict, source: ad.AnnData, target: ad.AnnData) -> tuple[ad.AnnData, ad.AnnData, str, dict]:
    """The encoded source / target matrices (obs kept) and the 02_run_pair label of the configuration."""
    enc = spec["encoding"]
    joint = rb.joint_counts(source, target)
    genes = pd.Index(rb.common_genes(source, target)[0], dtype=str)
    if enc == "logcpm":
        totals = np.asarray(joint.sum(axis=1)).ravel()
        e = sp.csr_matrix(sp.diags(1e4 / np.maximum(totals, 1.0)) @ joint); e.data = np.log1p(e.data)
        stem = "bmenc"
    elif enc in ("ranknm", "rank_value"):
        e, _ = rb.rank_encode(joint, int(spec["K"]), enc == "rank_value", spec["tie"])
        stem = "bmenc"
    elif enc == "rankhvg":
        e, hvg = rb.rankhvg_encode(joint, int(spec["H"]), spec["tie"]); genes = genes[hvg]
        stem = "bmenc_allgenes"
    else:
        raise ValueError(enc)
    label = stem + ("" if bool(spec["scale"]) else "_noscale") + ("_cos" if spec["cost"] == "cos" else "")
    e = sp.csr_matrix(e, dtype=np.float32); n = source.n_obs
    # no uns metadata, as the Splatter worker's precomputed files: 02_run_pair records the kind as unknown
    s = ad.AnnData(X=e[:n], obs=source.obs.copy(), var=pd.DataFrame(index=genes))
    t = ad.AnnData(X=e[n:], obs=target.obs.copy(), var=pd.DataFrame(index=genes))
    return s, t, label, {"encoded_genes": int(len(genes)), "label": label}


def fit_command(python: str, run_pair: str, manifest: str, out: str, index: int, label: str, n_pcs: int, workers: int,
                precomputed: bool, seed: int | None = None) -> list[str]:
    """02_run_pair.py exactly as the KP-Tracer runs (cancer_metastasis/tacc_preprocessing_factorial.slurm via
    kp_lineage/submit_kptracer_chain.sh): malignant scope, QC recorded with thresholds that filter nothing (the cells
    are the frozen equalised files' cells), within-side split null 5 + 5, grid 5, acceptance 0.99, no cell cap;
    plus --n-pcs and --save-coupling."""
    cmd = [python, run_pair, manifest, out, "--index", str(index), "--preprocessing", label,
           "--analysis-scope", "malignant", "--malignant-column", "malignant", "--malignant-value", "malignant",
           "--calibration-null", "within_side_split", "--max-observed-cells-per-side", "0", "--minimum-scope-cells", "4",
           "--cell-qc", "--minimum-total-counts", "0", "--minimum-detected-genes", "0", "--maximum-mitochondrial-percent", "100",
           "--null-calibration-replicates", "5", "--null-validation-replicates", "5", "--calibration-grid-size", "5",
           "--tolerance", "1e-4", "--workers", str(workers), "--device", "cpu", "--within-side-acceptance-minimum", "0.99",
           "--n-pcs", str(n_pcs), "--save-coupling"]
    if precomputed:
        cmd.append("--allow-precomputed")
    if seed is not None:
        cmd += ["--seed", str(seed)]
    return cmd
