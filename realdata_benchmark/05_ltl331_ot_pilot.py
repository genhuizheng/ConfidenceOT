"""LTL331 OT benchmark, pilot on 16 -> 20 weeks (or any pair index): every registered configuration and the tie-rule
variants of the production one, fitted with exactly the factorial run's arguments, plus balanced OT on the same cost.

    python realdata_benchmark/05_ltl331_ot_pilot.py --run-root RUN_ROOT --converted CONVERTED --registry REGISTRY \
        --out OUT_DIR [--pair-index 5] [--labels ...] [--tie-variants]

Fits (OUT/fits/<config>/): cancer_metastasis/02_run_pair.py on the frozen manifest (ds or original counts as the label
reads) at --index PAIR_INDEX, so seeds equal the frozen run's; the LTL331 factorial arguments; --save-coupling.
Tie variants (--tie-variants): ranknm256 encodings of the same equalised cells with the tie rules of
rb_common.rank_encode (production, alphabetical, random seed 1, expected) written as h5ad and run through the
precomputed path with labels tie<rule>_noscale_cos (--allow-precomputed): the rank-arm downstream (2,000 top-variance
genes, no gene scaling, PCA 30, cosine) applies unchanged. The production variant must reproduce the frozen gates;
that is checked and reported (production_reproduced).

Per fit and method (M4-E, M4-R) LTL331_OT_BENCHMARK rows hold: convergence (inner, outer, cycle), M4-E inference
validity and the M4-R calibration flags, source / target rejected fractions and their difference (asymmetry), median
normalised coupling row entropy, annotation-level transport (observed type-to-type mass over the independent
coupling with the same marginals), coverage (fraction of target types receiving >= 1% of the retained mass, fraction
of target cells with a retained partner mass above 1 / n_target of the retained block), NE minus non-NE source
retention, AUC of pre-equalisation depth for rejected cells, and the correspondence with balanced OT on the same cost
(full) and on the M4-E retained block (matched subset): Pearson correlation of the type-to-type mass matrices and the
Jensen-Shannon divergence of each source type's target-type distribution. A non-converged M4-R row is marked
formal_result = False and must not enter an average.
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
TIE_RULES = [("production", 0), ("alphabetical", 0), ("random", 1), ("expected", 0)]


def type_mass(P, st, tt):
    return np.array([[P[np.ix_(st == x, tt == y)].sum() for y in TYPES] for x in TYPES])


def correspondence(Ma, Mb):
    ok = (Ma.sum(1) > 0) & (Mb.sum(1) > 0)
    js = [jensenshannon(Ma[i] / Ma[i].sum(), Mb[i] / Mb[i].sum()) for i in np.flatnonzero(ok)]
    a, b = (Ma / Ma.sum()).ravel(), (Mb / Mb.sum()).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan, float(np.mean(js)) if js else np.nan


def run_fit(manifest: Path, out: Path, index: int, label: str, precomputed: bool) -> int:
    sep = ";" if os.name == "nt" else ":"
    env = {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO]))}
    cmd = [sys.executable, str(REPO / "cancer_metastasis" / "02_run_pair.py"), str(manifest), str(out / "out"),
           "--index", str(index), "--preprocessing", label, *FACTORIAL_ARGS] + (["--allow-precomputed"] if precomputed else [])
    with open(out / "run.log", "w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode


def localised_manifest(run_root: Path, equalised: bool, replace: dict | None, path: Path) -> Path:
    sub, name = rb.LTL_LABEL_INPUT[equalised]
    m = pd.read_csv(run_root / "inputs" / sub / name)
    for side in ("source", "target"):
        m[f"{side}_h5ad"] = [str(rb.local(v, run_root).resolve()) for v in m[f"{side}_h5ad"]]
        m[f"{side}_h5ads_json"] = [json.dumps([v]) for v in m[f"{side}_h5ad"]]
    if replace:
        for (pair_id, side), v in replace.items():
            k = m.index[m.pair_id == pair_id][0]
            m.loc[k, f"{side}_h5ad"] = str(v); m.loc[k, f"{side}_h5ads_json"] = json.dumps([str(v)])
    m.to_csv(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pair-index", type=int, default=5)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--tie-variants", action="store_true")
    a = p.parse_args()
    target = a.out / f"LTL331_OT_BENCHMARK_pair{a.pair_index}.tsv"
    if target.exists():
        sys.exit(f"{target} exists; not overwritten")
    fits = a.out / "fits" / f"pair{a.pair_index}"; fits.mkdir(parents=True, exist_ok=True)
    ctype = rb.cell_types(a.converted)
    depth = pd.read_csv(a.run_root / "inputs" / "ds" / "predownsample_depth.csv.gz").set_index("observation_id")["predownsample_total_counts"]
    labels = a.labels if a.labels is not None else pd.read_csv(a.registry, sep="\t")["label"].tolist()
    jobs = [(label, Preprocessing.from_label(label).equalise_depth, None, False) for label in labels]
    pair_id = rb.ltl_pairs(a.run_root, True).iloc[a.pair_index].pair_id
    if a.tie_variants:
        row = rb.ltl_pairs(a.run_root, True).iloc[a.pair_index]
        s = ad.read_h5ad(rb.local(row.source_h5ad, a.run_root)); t = ad.read_h5ad(rb.local(row.target_h5ad, a.run_root))
        _, si, ti = rb.common_genes(s, t)
        genes = s.var_names[si]
        joint = sp.vstack([sp.csr_matrix(s.X)[:, si], sp.csr_matrix(t.X)[:, ti]], format="csr").astype(np.float64)
        for rule, seed in TIE_RULES:
            name = f"tie{rule}{seed if rule == 'random' else ''}"
            d = fits / f"{name}_noscale_cos" / "input"; d.mkdir(parents=True, exist_ok=True)
            enc, _ = rb.rank_encode(joint, 256, False, rule, seed)
            enc = sp.csr_matrix(enc, dtype=np.float32)
            for side, sl, obs in (("source", slice(0, s.n_obs), s.obs), ("target", slice(s.n_obs, None), t.obs)):
                ad.AnnData(X=enc[sl], obs=obs.copy(), var=pd.DataFrame(index=genes)).write_h5ad(d / f"{side}.h5ad")
            jobs.append((f"{name}_noscale_cos", True, {(pair_id, "source"): (d / "source.h5ad").resolve(),
                                                        (pair_id, "target"): (d / "target.h5ad").resolve()}, True))
    frozen = next((a.run_root / "runs" / "acc0.99" / "ltl331" / "ranknm256_noscale_ds_cos" / pair_id / "scope_malignant").glob("budget_*"))
    frozen_cc = pd.read_csv(frozen / "cell_confidence.csv")
    rows = []
    for label, eq, replace, pre in jobs:
        fit = fits / label; fit.mkdir(parents=True, exist_ok=True)
        manifest = localised_manifest(a.run_root, eq, replace, fit / "manifest.csv")
        code = run_fit(manifest, fit, a.pair_index, label, pre)
        base = {"pair": pair_id.replace("LTL331__", ""), "config": label, "input": "ds" if eq else "original",
                "tie_variant": pre, "exit": code}
        if code != 0:
            rows.append({**base, "status": "fit failed; see run.log"}); print(label, "FAILED", flush=True); continue
        d = next((fit / "out" / pair_id / "scope_malignant").glob("budget_*"))
        run = json.loads((d / "run.json").read_text())
        metrics = pd.read_csv(d / "pair_metrics.csv").set_index("method")
        cc = pd.read_csv(d / "cell_confidence.csv")
        # balanced OT on the same cost, rebuilt from the run's own representation
        sm = pd.read_csv(manifest).iloc[a.pair_index]
        s = ad.read_h5ad(sm.source_h5ad); t = ad.read_h5ad(sm.target_h5ad)
        c = Preprocessing.from_label(label)
        X, Y, _, _ = prepare_joint_representation(
            s, t, n_hvg=c.n_hvg, n_pcs=c.n_pcs, seed=rb.SEED + a.pair_index, representation=c.normalisation,
            rank_top_n=c.rank_top_n, cost=c.cost, equalise_depth=c.equalise_depth, regress_out=c.regress_out,
            regress_on_ranks=c.regress_on_ranks, scale_genes=c.scale_genes, label_stem=c.label_stem, allow_precomputed=pre or c.normalisation == "precomputed")
        scale = median_pair_scale(X, Y, rng=np.random.default_rng(rb.SEED + a.pair_index * 104729))
        cost_ok = abs(scale - run["cost_scale"]) <= 1e-4 * max(1, abs(run["cost_scale"]))
        C = squared_euclidean(X, Y) / scale
        bot = balanced_ot(C, epsilon=0.1)
        st = ctype.loc[s.obs_names.astype(str)].to_numpy(); tt = ctype.loc[t.obs_names.astype(str)].to_numpy()
        M_bot = type_mass(bot.coupling, st, tt)
        for method in ("M4-E", "M4-R"):
            m = metrics.loc[method]
            g = cc[cc.method == method]
            src = g[g.side == "source"].set_index("observation_id").loc[s.obs_names.astype(str)]
            tgt = g[g.side == "target"].set_index("observation_id").loc[t.obs_names.astype(str)]
            rs, rt = ~src.rejected.to_numpy(bool), ~tgt.rejected.to_numpy(bool)
            P = np.load(d / f"coupling_{method.lower().replace('-', '')}.npz")["coupling"]
            R = P / P.sum(1, keepdims=True); H = -(R * np.log(np.where(R > 0, R, 1))).sum(1) / np.log(P.shape[1])
            M = type_mass(P, st, tt); ind = np.outer(M.sum(1), M.sum(0)) / M.sum()
            Q = P[np.ix_(rs, rt)]
            r = {**base, "status": "fitted", "method": method, "cost_reproduced": cost_ok,
                 "inner_converged": bool(m.inner_converged), "outer_converged": bool(m.outer_converged), "cycle_detected": bool(m.cycle_detected),
                 "m4e_inference_valid": bool(m.calibration_m4e_inference_valid), "m4r_calibration_valid": bool(m.calibration_valid_for_m4r),
                 "formal_result": bool(m.outer_converged) and (bool(m.calibration_m4e_inference_valid) if method == "M4-E" else bool(m.calibration_valid_for_m4r)),
                 "rejection_cost": float(m.rejection_cost), "source_rejected": float(m.source_final_rejection_rate),
                 "target_rejected": float(m.target_final_rejection_rate),
                 "asymmetry_source_minus_target": float(m.source_final_rejection_rate - m.target_final_rejection_rate),
                 "coupling_row_entropy_median": float(np.median(H)), "transported_mass": float(P.sum()),
                 "retained_block_mass_fraction": float(Q.sum() / P.sum()) if P.sum() > 0 else np.nan,
                 "source_ne_minus_nonne_retained": float(rs[st == "NE"].mean() - rs[st != "NE"].mean()) if (st == "NE").sum() >= 20 else np.nan,
                 "source_auc_depth_rejected": float(roc_auc_score(~rs, depth.loc[s.obs_names.astype(str)])) if 0 < rs.mean() < 1 else np.nan,
                 "target_auc_depth_rejected": float(roc_auc_score(~rt, depth.loc[t.obs_names.astype(str)])) if 0 < rt.mean() < 1 else np.nan,
                 "balanced_converged": bool(bot.converged)}
            for i, x in enumerate(TYPES):
                for j, y in enumerate(TYPES):
                    if ind[i, j] > 0 and x in ("NE", "PRAD") and y in ("NE", "PRAD"):
                        r[f"enrich_{x}_to_{y}"] = float(M[i, j] / ind[i, j])
            r["full_vs_balanced_type_corr"], r["full_vs_balanced_js"] = correspondence(M, M_bot)
            if rs.sum() >= 4 and rt.sum() >= 4:
                Mq = type_mass(Q, st[rs], tt[rt])
                sub = balanced_ot(C[np.ix_(rs, rt)], epsilon=0.1)
                r["matched_subset_vs_balanced_type_corr"], r["matched_subset_vs_balanced_js"] = correspondence(Mq, type_mass(sub.coupling, st[rs], tt[rt]))
                share = Mq.sum(0) / Mq.sum() if Mq.sum() > 0 else np.zeros(4)
                r["target_types_covered"] = int((share >= 0.01).sum())
                r["target_cells_covered"] = float(np.mean(Q.sum(0) / max(Q.sum(), 1e-300) > 1.0 / max(rt.sum(), 1)))
            if label == "tieproduction_noscale_cos" or label == "ranknm256_noscale_ds_cos":
                fz = frozen_cc[frozen_cc.method == method].set_index(["side", "observation_id"])["rejected"]
                mine = g.set_index(["side", "observation_id"])["rejected"]
                r["production_reproduced"] = bool((fz.loc[mine.index] == mine).all())
            rows.append(r)
        print(label, "done", flush=True)
        pd.DataFrame(rows).to_csv(target.with_suffix(".partial.tsv"), sep="\t", index=False)
    pd.DataFrame(rows).to_csv(target, sep="\t", index=False)
    target.with_suffix(".partial.tsv").unlink(missing_ok=True)
    print(f"-> {target}")


if __name__ == "__main__":
    main()
