"""LTL331 OT benchmark on one pair (pilot: 16 -> 20 weeks, index 5): the configurations the protocol fits with OT,
with exactly the factorial run's arguments, plus balanced OT on the same cost.

    python realdata_benchmark/05_ltl331_ot_pilot.py --run-root RUN_ROOT --converted CONVERTED --registry REGISTRY \
        --out OUT_DIR [--pair-index 5] [--only CONFIG_ID ...]

Which configurations (JOINT_BENCHMARK_PROTOCOL.md section 3): the historical non-raw labels (contrast H) and every
registered configuration with role main (R1-R6: production and expected ties, DS none / frozen P10 / seed 1 of P10,
P25, P50, cos and sqeuc, scale and noscale). Sensitivity rows (random ties, DS seeds 2-3) are representation-only.
Plus one mechanism check: the production ranknm256 encoding of the frozen equalised cells supplied as a precomputed
matrix must give the frozen gates (check_precomputed_path).

How each is fitted (OUT/fits/pair<i>/<config>/): cancer_metastasis/02_run_pair.py on a manifest equal to the frozen
one except the pair's two files, at --index PAIR_INDEX, so every seed equals the frozen run's; LTL331 factorial
arguments; --save-coupling.
  production-equivalent (logcpm any DS, ranknm-K production, historical labels): the production label on count files
  (frozen ds / original / equalised here with rb_common.equalised, seed 1000 * s + 1 / + 2 as in 04_);
  other encodings: the encoded matrix written as h5ad and fitted through the precomputed path with label
  bmenc[_allgenes][_noscale][_cos] (--allow-precomputed): rank-K -> 2,000 top-variance encoded genes, rank-HVG ->
  every encoded HVG (allgenes), then the configuration's scaling, PCA 30 and cost.
Rows of LTL331_OT_BENCHMARK_pair<i>.tsv: see the columns; a non-converged M4-R row has formal_result False.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import roc_auc_score

import rb_common as rb
from common import prepare_joint_representation
from confidenceot import Preprocessing
from confidenceot.preprocessing import median_pair_scale, squared_euclidean
from traditional_ot.balanced import balanced_ot

REPO = Path(__file__).resolve().parent.parent
TYPES = ["PRAD", "Basal", "Other", "NE"]
FACTORIAL_ARGS = ["--analysis-scope", "malignant", "--malignant-column", "malignant", "--malignant-value", "malignant",
                  "--calibration-null", "within_side_split", "--max-observed-cells-per-side", "0", "--minimum-scope-cells", "4",
                  "--cell-qc", "--minimum-total-counts", "0", "--minimum-detected-genes", "0", "--maximum-mitochondrial-percent", "100",
                  "--null-calibration-replicates", "5", "--null-validation-replicates", "5", "--calibration-grid-size", "5",
                  "--tolerance", "1e-4", "--workers", "8", "--device", "cpu", "--within-side-acceptance-minimum", "0.99", "--save-coupling"]


def type_mass(P, st, tt):
    return np.array([[P[np.ix_(st == x, tt == y)].sum() for y in TYPES] for x in TYPES])


def correspondence(Ma, Mb):
    ok = (Ma.sum(1) > 0) & (Mb.sum(1) > 0)
    js = [jensenshannon(Ma[i] / Ma[i].sum(), Mb[i] / Mb[i].sum()) for i in np.flatnonzero(ok)]
    a, b = (Ma / Ma.sum()).ravel(), (Mb / Mb.sum()).ravel()
    return (float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan), (float(np.mean(js)) if js else np.nan)


def fit(manifest: Path, out: Path, index: int, label: str, precomputed: bool) -> int:
    sep = ";" if os.name == "nt" else ":"
    env = {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO]))}
    cmd = [sys.executable, str(REPO / "cancer_metastasis" / "02_run_pair.py"), str(manifest), str(out / "out"),
           "--index", str(index), "--preprocessing", label, *FACTORIAL_ARGS] + (["--allow-precomputed"] if precomputed else [])
    with open(out / "run.log", "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode


def manifest_with(run_root: Path, equalised: bool, pair_id: str, files: tuple[Path, Path] | None, path: Path) -> Path:
    sub, name = rb.LTL_LABEL_INPUT[equalised]
    m = pd.read_csv(run_root / "inputs" / sub / name)
    for side in ("source", "target"):
        m[f"{side}_h5ad"] = [str(rb.local(v, run_root).resolve()) for v in m[f"{side}_h5ad"]]
    if files:
        k = m.index[m.pair_id == pair_id][0]
        m.loc[k, "source_h5ad"], m.loc[k, "target_h5ad"] = str(files[0].resolve()), str(files[1].resolve())
    for side in ("source", "target"):
        m[f"{side}_h5ads_json"] = [json.dumps([v]) for v in m[f"{side}_h5ad"]]
    m.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair-index", type=int, default=5)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--no-check", action="store_true", help="skip the precomputed-path mechanism check (TACC workers)")
    a = p.parse_args()
    target = a.out / f"LTL331_OT_BENCHMARK_pair{a.pair_index}.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    fits = a.out / "fits" / f"pair{a.pair_index}"; fits.mkdir(parents=True, exist_ok=True)
    ctype = rb.cell_types(a.converted)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    targets = rb.ds_targets(depth)
    reg = pd.read_csv(a.registry, sep="\t")
    reg = reg[(reg.kind == "historical") & reg.in_new_grid.astype(bool) | (reg.kind == "registered") & (reg.role == "main")]
    if a.only:
        reg = reg[reg.config_id.isin(a.only)]
    orig = rb.ltl_pairs(a.run_root, False).iloc[a.pair_index]; froz = rb.ltl_pairs(a.run_root, True).iloc[a.pair_index]
    pair_id = orig.pair_id
    base = {"none": (ad.read_h5ad(rb.local(orig.source_h5ad, a.run_root)), ad.read_h5ad(rb.local(orig.target_h5ad, a.run_root))),
            "P10sfrozen": (ad.read_h5ad(rb.local(froz.source_h5ad, a.run_root)), ad.read_h5ad(rb.local(froz.target_h5ad, a.run_root)))}
    frozen_dir = lambda label: next((a.run_root / "runs" / "acc0.99" / "ltl331" / label / pair_id / "scope_malignant").glob("budget_*"))
    jobs = []
    for _, r in reg.iterrows():
        r = r.to_dict()
        if r["kind"] == "historical":
            c = Preprocessing.from_label(r["label"])
            jobs.append(dict(config=r["config_id"], ds_key="P10sfrozen" if c.equalise_depth else "none", label=r["label"], encode=None))
            continue
        ds_key = "none" if r["ds"] == "none" else f"{r['ds']}s{r['ds_seed']}"
        lab = rb.production_label(r["encoding"], int(r["K"]) if r["K"] == r["K"] else 0, bool(r["scale"]), r["ds"] != "none", r["cost"]) \
            if (r["encoding"] == "logcpm" or r["tie"] == "production") else None
        jobs.append(dict(config=r["config_id"], ds_key=ds_key, label=lab, encode=None if lab else r))
    if not a.no_check:
        jobs.append(dict(config="check_precomputed_path", ds_key="P10sfrozen", label=None,
                         encode={"encoding": "ranknm", "K": 256, "tie": "production", "scale": False, "cost": "cos", "ds": "P10", "H": None}))
    rows = []
    for job in jobs:
        safe = job["config"].replace("|", "__").replace(":", "_")
        out = fits / safe; (out / "input").mkdir(parents=True, exist_ok=True)
        if job["ds_key"] in base:
            s, t = base[job["ds_key"]]
        else:
            ds, seed = job["ds_key"][:3], int(job["ds_key"][4:])
            s, t = rb.equalised(base["none"][0], targets[ds], 1000 * seed + 1), rb.equalised(base["none"][1], targets[ds], 1000 * seed + 2)
        files, precomputed = None, False
        if job["encode"] is not None:
            e = job["encode"]; joint = rb.joint_counts(s, t); genes = pd.Index(rb.common_genes(s, t)[0])
            if e["encoding"] == "ranknm":
                rule, tseed = ("random", int(e["tie"][6:])) if str(e["tie"]).startswith("random") else (e["tie"], 0)
                enc, _ = rb.rank_encode(joint, int(e["K"]), False, rule, tseed); stem = "bmenc"
            else:
                enc, hvg = rb.rankhvg_encode(joint, int(e["H"]), e["tie"]); genes = genes[hvg]; stem = "bmenc_allgenes"
            label = stem + ("" if e["scale"] else "_noscale") + ("_cos" if e["cost"] == "cos" else "")
            enc = sp.csr_matrix(enc, dtype=np.float32); n = s.n_obs
            files = (out / "input" / "source.h5ad", out / "input" / "target.h5ad")
            ad.AnnData(X=enc[:n], obs=s.obs.copy(), var=pd.DataFrame(index=genes)).write_h5ad(files[0])
            ad.AnnData(X=enc[n:], obs=t.obs.copy(), var=pd.DataFrame(index=genes)).write_h5ad(files[1])
            precomputed = True
        else:
            label = job["label"]
            if job["ds_key"] not in base:
                files = (out / "input" / "source.h5ad", out / "input" / "target.h5ad"); s.write_h5ad(files[0]); t.write_h5ad(files[1])
        manifest = manifest_with(a.run_root, job["ds_key"] == "P10sfrozen", pair_id, files, out / "manifest.csv")
        code = fit(manifest, out, a.pair_index, label, precomputed)
        base_row = {"pair": pair_id.replace("LTL331__", ""), "config_id": job["config"], "fit_label": label, "ds": job["ds_key"], "precomputed": precomputed, "exit": code}
        if code != 0:
            rows.append({**base_row, "status": "fit failed; see run.log"}); print(job["config"], "FAILED", flush=True); continue
        d = next((out / "out" / pair_id / "scope_malignant").glob("budget_*"))
        run = json.loads((d / "run.json").read_text()); metrics = pd.read_csv(d / "pair_metrics.csv").set_index("method")
        cc = pd.read_csv(d / "cell_confidence.csv")
        sm = pd.read_csv(manifest).iloc[a.pair_index]
        S, T = ad.read_h5ad(sm.source_h5ad), ad.read_h5ad(sm.target_h5ad)
        c = Preprocessing.from_label(label)
        X, Y, _, _ = prepare_joint_representation(
            S, T, n_hvg=c.n_hvg, n_pcs=c.n_pcs, seed=rb.SEED + a.pair_index, representation=c.normalisation, rank_top_n=c.rank_top_n,
            cost=c.cost, equalise_depth=c.equalise_depth, regress_out=c.regress_out, regress_on_ranks=c.regress_on_ranks,
            scale_genes=c.scale_genes, label_stem=c.label_stem, allow_precomputed=precomputed or c.normalisation == "precomputed")
        scale = median_pair_scale(X, Y, rng=np.random.default_rng(rb.SEED + a.pair_index * 104729))
        C = squared_euclidean(X, Y) / scale
        bot = balanced_ot(C, epsilon=0.1)
        st = ctype.loc[S.obs_names.astype(str)].to_numpy(); tt = ctype.loc[T.obs_names.astype(str)].to_numpy()
        sd, td = depth.loc[S.obs_names.astype(str)].to_numpy(), depth.loc[T.obs_names.astype(str)].to_numpy()
        M_bot = type_mass(bot.coupling, st, tt)
        for method in ("M4-E", "M4-R"):
            m = metrics.loc[method]; g = cc[cc.method == method]
            src = g[g.side == "source"].set_index("observation_id").loc[S.obs_names.astype(str)]
            tgt = g[g.side == "target"].set_index("observation_id").loc[T.obs_names.astype(str)]
            rs, rt = ~src.rejected.to_numpy(bool), ~tgt.rejected.to_numpy(bool)
            P = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
            R = P / P.sum(1, keepdims=True); H = -(R * np.log(np.where(R > 0, R, 1))).sum(1) / np.log(P.shape[1])
            M = type_mass(P, st, tt); ind = np.outer(M.sum(1), M.sum(0)) / M.sum()
            Q = P[np.ix_(rs, rt)]
            valid = bool(m.calibration_m4e_inference_valid) if method == "M4-E" else bool(m.calibration_valid_for_m4r)
            r = {**base_row, "status": "fitted", "method": method,
                 "cost_reproduced": abs(scale - run["cost_scale"]) <= 1e-4 * max(1, abs(run["cost_scale"])),
                 "inner_converged": bool(m.inner_converged), "outer_converged": bool(m.outer_converged), "cycle_detected": bool(m.cycle_detected),
                 "m4e_inference_valid": bool(m.calibration_m4e_inference_valid), "m4r_calibration_valid": bool(m.calibration_valid_for_m4r),
                 "formal_result": bool(m.outer_converged) and valid, "rejection_cost": float(m.rejection_cost),
                 "source_rejected": float(m.source_final_rejection_rate), "target_rejected": float(m.target_final_rejection_rate),
                 "asymmetry_source_minus_target": float(m.source_final_rejection_rate - m.target_final_rejection_rate),
                 "coupling_row_entropy_median": float(np.median(H)), "transported_mass": float(P.sum()),
                 "retained_block_mass_fraction": float(Q.sum() / P.sum()) if P.sum() > 0 else np.nan,
                 "source_ne_minus_nonne_retained": float(rs[st == "NE"].mean() - rs[st != "NE"].mean()) if (st == "NE").sum() >= 20 else np.nan,
                 "source_auc_depth_rejected": float(roc_auc_score(~rs, sd)) if 0 < rs.mean() < 1 else np.nan,
                 "target_auc_depth_rejected": float(roc_auc_score(~rt, td)) if 0 < rt.mean() < 1 else np.nan,
                 "balanced_converged": bool(bot.converged)}
            for i, x in enumerate(TYPES):
                for j, y in enumerate(TYPES):
                    if ind[i, j] > 0:
                        r[f"enrich_{x}_to_{y}"] = float(M[i, j] / ind[i, j])
            r["full_vs_balanced_type_corr"], r["full_vs_balanced_js"] = correspondence(M, M_bot)
            if rs.sum() >= 4 and rt.sum() >= 4:
                Mq = type_mass(Q, st[rs], tt[rt])
                sub = balanced_ot(C[np.ix_(rs, rt)], epsilon=0.1)
                r["matched_subset_vs_balanced_type_corr"], r["matched_subset_vs_balanced_js"] = correspondence(Mq, type_mass(sub.coupling, st[rs], tt[rt]))
                share = Mq.sum(0) / Mq.sum() if Mq.sum() > 0 else np.zeros(4)
                r["target_types_covered"] = int((share >= 0.01).sum())
                r["target_cells_covered"] = float(np.mean(Q.sum(0) / max(Q.sum(), 1e-300) > 1.0 / max(rt.sum(), 1)))
            for ref_label in ("ranknm256_noscale_ds_cos", "logcpm"):
                if job["config"] in (f"label:{ref_label}", "check_precomputed_path" if ref_label.startswith("ranknm") else ""):
                    fz = pd.read_csv(frozen_dir(ref_label) / "cell_confidence.csv"); fz = fz[fz.method == method].set_index(["side", "observation_id"])["rejected"]
                    mine = g.set_index(["side", "observation_id"])["rejected"]
                    r["reproduces_frozen_gates"] = bool((fz.loc[mine.index] == mine).all())
            rows.append(r)
        print(job["config"], "done", flush=True)
        pd.DataFrame(rows).to_csv(target.with_suffix(".partial.tsv"), sep="\t", index=False)
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    target.with_suffix(".partial.tsv").unlink(missing_ok=True)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
