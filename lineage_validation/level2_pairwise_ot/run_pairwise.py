"""Level 2 OT without rejection: Method A and Method B from one primary to its family's metastases (PLAN.md).

    python lineage_validation/level2_pairwise_ot/run_pairwise.py --index I --out OUT_ROOT [--strategy rna|cnv|rna_cnv]

Strategy (methods.py; the cost is built as 02_run_pair.py --representation-source builds it, from the existing
infercnvpy exports, nothing recomputed):
  rna      the method's RNA coordinates (below)
  cnv      02_run_pair.py:cnv_coordinates (CNV profile - 1 -> joint PCA 30, seed 20260831 + I): the same
           coordinates for A and B, which then differ only in the solver
  rna_cnv  [the method's RNA coordinates | the CNV PCA], each block divided by the root of its median pair cost
           (seed 20260831 + I * 104729 + 1)
The two-step strategies are not run: without rejection they equal rna and cnv exactly (strategy_check/README.md).

Row I of inputs/cases.tsv (every eligible family; make_inputs.py). Cells: the malignant cells of the primary and of the target sites in the
depth-equalised files (KP-Tracer: the per-tumour files of the 104-pair runs, the target sites concatenated;
Quinn: 27_'s output for the lineage-validation LL and metastases sides), and the same cell ids in the raw
files. One plan covers every target site; a primary -> site block is a column slice of it (Level 3).
Exactly two OT runs, on the same cells. Representations are built by
cancer_metastasis/common.py:prepare_joint_representation exactly as 02_run_pair.py builds them (seed
20260831 + I, 2,000 HVGs, 30 PCs, cost = squared Euclidean / median_pair_scale with seed 20260831 + I * 104729);
epsilon 0.1 for both:
  A_standard_OT  logcpm_noscale on raw counts -> traditional_ot.balanced.balanced_ot (uniform marginals,
                 threshold 1e-9, <= 10,000 iterations), no rejection
  B_our_OT       ranknm256_noscale_ds_cos on depth-equalised counts -> ConfidenceOT(backbone="uot",
                 variant="exact") with source/target rejection budget 0, lambda 1/1, tolerance 1e-4: KL-relaxed
                 marginals kept, no cell gated (checked)

Writes OUT_ROOT/<case_id>/:
  sites.json                       target sites used (cells) and left out (no equalised file, < 20 cells)
  cells.tsv.gz                     side, cell id, site, total counts raw and equalised (plan row/column order)
  coordinates_<label>.npz          rna: source/target coordinates and cell ids, cost scale, HVGs
  coordinates_cnv.npz              cnv: the CNV PCA (shared by A and B)
  coordinates_rna_cnv__<label>.npz rna_cnv: the block-scaled joint coordinates of each method
  RUNNING                          the SLURM job id while the task runs (removed at the end, success or not)
  <method>/preprocessing.json      strategy, configuration, representation / CNV record, block scales, seeds, cost scale
  <method>/coupling.npz            the full plan (float32), for Level 3
  <method>/source_groups.tsv       per family and level: group, cells, row mass of the plan
  <method>/target_mass_<family>_L<k>.tsv.gz  per target cell: site, column mass, plan mass from each source group
  <method>/diagnostics.json        convergence, total mass, marginal deviations, gate retention, seconds
  SUCCESS
CRISPR labels (inputs/family_cells.tsv.gz) are read only after every coupling is computed, to sum it by
source group. Refuses if SUCCESS exists. On any error it writes FAILED.json (error, traceback, SLURM ids) and
exits non-zero; nothing is skipped or re-parameterised. A retry of the same index removes FAILED.json first.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "cancer_metastasis")]
from common import load_exact_side, prepare_joint_representation  # noqa: E402
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.preprocessing import Preprocessing, median_pair_scale, squared_euclidean  # noqa: E402
from methods import EPSILON, METHODS, STRATEGIES  # noqa: E402
from traditional_ot.balanced import balanced_ot  # noqa: E402

SEED, N_HVG, N_PCS = 20260831, 2000, 30
UNLABELLED, OTHER = "family unlabelled", "other"
KP = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")
LV = Path("/scratch/10119/ghzheng/OT_project/lineage_validation")
# The existing infercnvpy exports (49_infercnvpy_unit.py), read as 02_run_pair.py --cnv-root reads them; not recomputed.
KP_CNV = KP / "cnv_profiles" / "infercnvpy_mouse"
QUINN_CNV = Path("/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot/cnv_profiles/infercnvpy_experiment")


def load_run_pair():
    """02_run_pair.py as a module, for its cnv_coordinates (the shipped CNV representation, used unchanged)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_pair_02", REPO / "cancer_metastasis" / "02_run_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def side_paths(row: pd.Series, side: str) -> list[str]:
    column = f"{side}_h5ads_json"
    if column in row and pd.notna(row[column]):
        return [str(v) for v in json.loads(str(row[column]))]
    return [str(row[f"{side}_h5ad"])]


def files_for(pair: pd.Series) -> dict:
    """Per side, the parts to load: (equalised paths, sample), (raw paths, sample), site (None = whole file)."""
    src, tgt = pair["source_site"], pair["target_sites"]
    if pair["dataset"] == "KPTracer":
        ds = pd.read_csv(KP / "downsampled" / "pair_manifest_downsampled.csv", dtype=str)
        ds_path = {}
        for side in ("source", "target"):
            for _, r in ds.iterrows():
                ds_path.setdefault(r[f"{side}_sample"], side_paths(r, side))
        # One file per tumour (sample_id = the tumour): every cell's site is the tumour itself. The files' own
        # obs 'site' is the tissue category ('primary' / ...), so it is not used to select cells.
        part = lambda t: {"ds": (ds_path[t], t), "raw": ([str(KP / "prepared" / "h5ad" / f"{t}.h5ad")], t), "site": t,
                          "site_from_obs": False}
        # A site without an equalised file did not pass the 104-pair eligibility (>= 20 malignant cells).
        missing = {t: "no equalised file" for t in tgt.split(";") if t not in ds_path}
        return {"source": [part(src)], "target": [part(t) for t in tgt.split(";") if t in ds_path], "left_out": missing}
    raw = pd.read_csv(LV / "inputs" / "manifests" / "quinn_D_B_raw.csv", dtype=str)
    ds = pd.read_csv(LV / "inputs" / "h5ad" / "Quinn2021" / "ds" / "pair_manifest_downsampled.csv", dtype=str)
    rows = {}
    for kind, manifest in (("raw", raw), ("ds", ds)):
        r = manifest[manifest["patient_id"].astype(str).eq(pair["mouse"])]
        if len(r) != 1:
            raise ValueError(f"Quinn {pair['mouse']}: {len(r)} rows in the {kind} manifest")
        rows[kind] = r.iloc[0]
    # Quinn sides are pooled over tissues; obs 'site' of the raw file is the tissue (sampleID), so it labels
    # and, for a named site, selects the cells.
    spec = lambda side, site: [{k: (side_paths(rows[k], side), rows[k][f"{side}_sample"]) for k in ("ds", "raw")}
                               | {"site": site, "site_from_obs": True}]
    return {"source": spec("source", None), "target": spec("target", None if tgt == "ALL" else tgt), "left_out": {}}


def load_part(part: dict, side: str):
    ds, raw = load_exact_side(*part["ds"]), load_exact_side(*part["raw"])
    if "malignant" in ds.obs:
        ds = ds[ds.obs["malignant"].astype(str).eq("malignant").to_numpy()].copy()
    if not ds.obs_names.isin(raw.obs_names).all():
        raise ValueError(f"{side}: {int((~ds.obs_names.isin(raw.obs_names)).sum())} equalised cells missing from the raw file")
    raw = raw[ds.obs_names].copy()
    if not part["site_from_obs"]:  # KP-Tracer: the file is the tumour; its sample_id was checked by load_exact_side
        return raw, ds, np.repeat(part["site"], raw.n_obs).astype(object)
    # Quinn: the site of each cell is read off the raw file (the equalised files need not carry it).
    if "site" not in raw.obs:
        raise ValueError(f"{side}: the raw file has no obs 'site'")
    site = raw.obs["site"].astype(str).to_numpy()
    if part["site"] is not None:
        keep = site == part["site"]
        ds, raw, site = ds[keep].copy(), raw[keep].copy(), site[keep]
    return raw, ds, site


def load_side(parts: list, side: str, left_out: dict):
    loaded = []
    for p in parts:
        raw, ds, site = load_part(p, side)
        if ds.n_obs >= 20:
            loaded.append((raw, ds, site))
        elif side == "target" and p["site"] is not None:
            left_out[p["site"]] = f"{ds.n_obs} malignant cells (< 20)"
        else:
            raise ValueError(f"{side} {p['site']}: {ds.n_obs} cells")
    if not loaded:
        raise ValueError(f"{side}: no site with >= 20 cells")
    if len(loaded) == 1:
        return loaded[0]
    import anndata as ad
    join = lambda xs: ad.concat(xs, join="inner", index_unique=None)
    raw, ds = join([x[0] for x in loaded]), join([x[1] for x in loaded])
    if ds.obs_names.duplicated().any():
        raise ValueError(f"{side}: a cell id repeats across sites")
    return raw, ds, np.concatenate([x[2] for x in loaded])


def solve(cost: np.ndarray, solver: str):
    started = time.perf_counter()
    if solver == "BOT":
        r = balanced_ot(cost, epsilon=EPSILON, threshold=1e-9, max_iterations=10_000)
        P, info = r.coupling, {"converged": bool(r.converged), "iterations": int(r.n_iterations),
                               "solver_marginal_error": float(r.marginal_error)}
    else:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # Budget 0 maps to the bounds (0, 0): no cell can be gated, and the plan does not depend on the
            # rejection cost (verify_zero_rejection.tsv). The backbone stays UOT: marginals KL-relaxed.
            r = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=1.0, epsilon=EPSILON, lambda_a=1.0, lambda_b=1.0,
                             source_rejection_budget=0.0, target_rejection_budget=0.0,
                             tolerance=1e-4, device="cpu").fit(cost)
        P = r.coupling
        info = {"backbone": r.backbone, "source_rejection_bounds": list(r.source_rejection_bounds),
                "target_rejection_bounds": list(r.target_rejection_bounds),
                "converged": bool(r.inner_converged and r.outer_converged), "inner_converged": bool(r.inner_converged),
                "outer_converged": bool(r.outer_converged), "iterations": int(r.total_inner_iterations),
                "source_retained": float(np.mean(r.source_gate)), "target_retained": float(np.mean(r.target_gate)),
                "warnings": [str(w.message)[:300] for w in caught]}
        if not (r.source_gate.all() and r.target_gate.all()):
            raise RuntimeError("UOT with rejection budget 0 gated a cell")
        if r.backbone != "uot":
            raise RuntimeError(f"Method B ran backbone {r.backbone!r}, not uot")
    n, m = P.shape
    rows, cols = P.sum(1), P.sum(0)
    info.update(seconds=time.perf_counter() - started, total_mass=float(P.sum()),
                row_rel_dev_max=float(np.abs(rows * n - 1).max()), row_rel_dev_median=float(np.median(np.abs(rows * n - 1))),
                col_rel_dev_max=float(np.abs(cols * m - 1).max()), col_rel_dev_median=float(np.median(np.abs(cols * m - 1))),
                finite=bool(np.isfinite(P).all()))
    return P, info


def groups_for(pair: pd.Series, source_ids, cells: pd.DataFrame, subs: pd.DataFrame) -> dict:
    """{(family, level): group of each source cell} under classify_subclones.py's rules."""
    out = {}
    for fam in pair["families"].split(";"):
        table = cells[cells["name"].eq(fam)].set_index("cell_id")
        for level in (1, 2):
            allowed = set(subs.loc[subs["name"].eq(fam) & subs["level"].eq(level), "subclone"])
            if not allowed:
                continue
            lab = table[f"L{level}"]
            g = []
            for c in source_ids:
                if c in lab.index:
                    g.append(lab[c] if lab[c] in allowed else UNLABELLED)
                else:  # KP-Tracer: every cell of the primary is the family's; Quinn: outside the clone
                    g.append(UNLABELLED if pair["dataset"] == "KPTracer" else OTHER)
            out[(fam, level)] = (np.array(g, dtype=object), sorted(allowed))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=HERE / "inputs" / "cases.tsv")
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs", help="family_cells.tsv.gz and subclones.tsv")
    parser.add_argument("--strategy", choices=tuple(STRATEGIES), default="rna",
                        help="rna (default, the first runs), cnv or rna_cnv; --out is that strategy's folder")
    parser.add_argument("--kp-root", type=Path, default=KP)
    parser.add_argument("--lv-root", type=Path, default=LV)
    parser.add_argument("--kp-cnv-root", type=Path, default=None, help="default <kp-root>/cnv_profiles/infercnvpy_mouse")
    parser.add_argument("--quinn-cnv-root", type=Path, default=QUINN_CNV)
    args = parser.parse_args()
    globals().update(KP=args.kp_root, LV=args.lv_root, KP_CNV=args.kp_cnv_root or args.kp_root / "cnv_profiles" / "infercnvpy_mouse",
                     QUINN_CNV=args.quinn_cnv_root)
    pair = pd.read_csv(args.cases, sep="\t", dtype=str).iloc[args.index]
    if int(pair["index"]) != args.index:
        raise ValueError("cases.tsv row order does not match its index column")
    out = args.out / pair["case_id"]
    if (out / "SUCCESS").exists():
        sys.exit(f"{out}/SUCCESS exists")
    out.mkdir(parents=True, exist_ok=True)
    (out / "FAILED.json").unlink(missing_ok=True)  # a retry starts clean; the earlier reason stays in the log
    # RUNNING names this SLURM job while the task runs, so submit_full.sh never submits the case twice.
    (out / "RUNNING").write_text(os.environ.get("SLURM_JOB_ID", "local") + "\n", encoding="utf-8")
    started = time.perf_counter()
    try:
        run(args, pair, out)
    except BaseException as error:  # recorded, never swallowed: the task still exits non-zero
        (out / "FAILED.json").write_text(json.dumps({
            "case_id": pair["case_id"], "index": args.index, "strategy": args.strategy, "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(), "seconds": time.perf_counter() - started,
            "job": os.environ.get("SLURM_JOB_ID", ""), "array_job": os.environ.get("SLURM_ARRAY_JOB_ID", ""),
            "array_task": os.environ.get("SLURM_ARRAY_TASK_ID", "")}, indent=1), encoding="utf-8")
        raise
    finally:
        (out / "RUNNING").unlink(missing_ok=True)


def run(args, pair: pd.Series, out: Path) -> None:
    spec = files_for(pair)
    left_out = dict(spec["left_out"])
    (s_raw, s_ds, s_site), (t_raw, t_ds, t_site) = load_side(spec["source"], "source", left_out), load_side(spec["target"], "target", left_out)
    used = {str(k): int(v) for k, v in pd.Series(t_site).value_counts().items()}
    src_sites = {str(k): int(v) for k, v in pd.Series(s_site).value_counts().items()}
    if set(src_sites) != {pair["source_site"]}:
        raise ValueError(f"source side holds sites {src_sites}, expected only {pair['source_site']}")
    (out / "sites.json").write_text(json.dumps({"source": pair["source_site"], "source_cells": int(s_ds.n_obs),
                                                "target_sites_used": used, "target_sites_left_out": left_out}, indent=1), encoding="utf-8")
    print(f"{pair['case_id']}: source {s_ds.n_obs}, target {t_ds.n_obs} cells ({used}); left out {left_out}", flush=True)
    total = lambda a: np.asarray(a.X.sum(1)).ravel()
    pd.concat([pd.DataFrame({"side": side, "cell": d.obs_names, "site": s, "total_counts_raw": total(r), "total_counts_ds": total(d)})
               for side, r, d, s in (("source", s_raw, s_ds, s_site), ("target", t_raw, t_ds, t_site))]).to_csv(
        out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    strategy = args.strategy
    cnv = None
    if strategy != "rna":  # the shipped CNV PCA, once: it does not depend on the method (cell ids are the same)
        cnv_root = KP_CNV if pair["dataset"] == "KPTracer" else QUINN_CNV
        cs, ct, cnv_record = load_run_pair().cnv_coordinates(cnv_root, pair["dataset"], pair["mouse"], s_ds.obs_names.astype(str),
                                                             t_ds.obs_names.astype(str), n_pcs=N_PCS, seed=SEED + args.index)
        cnv = (cs, ct, {**cnv_record, "cnv_seed": SEED + args.index, "cnv_root": str(cnv_root)})
        if strategy == "cnv":
            np.savez_compressed(out / "coordinates_cnv.npz", source=cs, target=ct, source_ids=np.asarray(s_ds.obs_names, dtype=str),
                                target_ids=np.asarray(t_ds.obs_names, dtype=str))
    plans = {}
    for method, spec_m in METHODS.items():
        label, kind, solver = spec_m["preprocessing"], spec_m["counts"], spec_m["solver"]
        src, tgt = (s_raw, t_raw) if kind == "raw" else (s_ds, t_ds)
        meta = {"method": method, "strategy": strategy, "representation": STRATEGIES[strategy]["representation"],
                "source_cells": int(src.n_obs), "target_cells": int(tgt.n_obs), "scale_seed": SEED + args.index * 104729,
                "cost": "squared_euclidean / median_pair_scale"}
        if strategy == "cnv":
            xs, xt = cnv[0], cnv[1]
            meta.update(cnv_record=cnv[2], rna_preprocessing="not used (the CNV path reads no RNA representation)")
        else:
            cfg = Preprocessing.from_label(label)
            xs, xt, hvg, record = prepare_joint_representation(
                src, tgt, n_hvg=N_HVG, n_pcs=N_PCS, seed=SEED + args.index, representation=cfg.normalisation,
                rank_top_n=cfg.rank_top_n, minimum_detection_rate=0.0, cost=cfg.cost, equalise_depth=cfg.equalise_depth,
                regress_out=cfg.regress_out, regress_on_ranks=cfg.regress_on_ranks, scale_genes=cfg.scale_genes,
                label_stem=cfg.label_stem)
            if record.get("label") != label:
                raise RuntimeError(f"{label}: the representation built {record.get('label')!r}")
            meta.update(preprocessing=label, counts=kind, configuration=dataclasses.asdict(cfg), representation_record=record,
                        n_hvg=N_HVG, n_pcs=N_PCS, representation_seed=SEED + args.index, n_hvg_used=int(len(hvg)))
            if strategy == "rna":
                np.savez_compressed(out / f"coordinates_{label}.npz", source=xs, target=xt,
                                    scale=median_pair_scale(xs, xt, rng=np.random.default_rng(SEED + args.index * 104729)),
                                    hvg=np.asarray(hvg, dtype=str), source_ids=np.asarray(src.obs_names, dtype=str),
                                    target_ids=np.asarray(tgt.obs_names, dtype=str))
            else:  # rna_cnv, as 02_run_pair.py: each block divided by the root of its median pair cost
                block_rng = np.random.default_rng(SEED + args.index * 104729 + 1)
                rna_scale = median_pair_scale(xs, xt, rng=block_rng)
                cnv_scale = median_pair_scale(cnv[0], cnv[1], rng=block_rng)
                xs = np.hstack([xs / np.sqrt(rna_scale), cnv[0] / np.sqrt(cnv_scale)]).astype(np.float32)
                xt = np.hstack([xt / np.sqrt(rna_scale), cnv[1] / np.sqrt(cnv_scale)]).astype(np.float32)
                meta.update(cnv_record=cnv[2], rna_block_scale=float(rna_scale), cnv_block_scale=float(cnv_scale))
                np.savez_compressed(out / f"coordinates_rna_cnv__{label}.npz", source=xs, target=xt, hvg=np.asarray(hvg, dtype=str),
                                    source_ids=np.asarray(src.obs_names, dtype=str), target_ids=np.asarray(tgt.obs_names, dtype=str))
        scale = median_pair_scale(xs, xt, rng=np.random.default_rng(SEED + args.index * 104729))
        meta["cost_scale"] = float(scale)
        (out / method).mkdir(exist_ok=True)
        (out / method / "preprocessing.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        P, info = solve(squared_euclidean(xs, xt) / scale, solver)
        info.update(method=method, strategy=strategy, preprocessing=label if strategy != "cnv" else "cnv (no RNA preprocessing)",
                    counts=kind, solver=solver, epsilon=EPSILON, cost_scale=float(scale),
                    n_source=int(P.shape[0]), n_target=int(P.shape[1]), solver_call=METHODS[method]["solver_call"],
                    rejection=METHODS[method]["rejection"])
        print(json.dumps(info), flush=True)
        np.savez_compressed(out / method / "coupling.npz", coupling=P.astype(np.float32))
        (out / method / "diagnostics.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        plans[method] = P
    # CRISPR labels enter only here, after every coupling exists.
    cells = pd.read_csv(args.inputs / "family_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    subs = pd.read_csv(args.inputs / "subclones.tsv", sep="\t", dtype={"level": int})
    groups = groups_for(pair, list(s_ds.obs_names), cells, subs)
    for method, P in plans.items():
        rows, cols = P.sum(1), P.sum(0)
        summary = []
        for (fam, level), (g, allowed) in groups.items():
            names = allowed + [UNLABELLED, OTHER]
            onehot = np.stack([(g == x) for x in names], axis=1).astype(float)
            mass = P.T @ onehot
            table = pd.DataFrame(mass, columns=[f"mass__{x}" for x in names])
            table.insert(0, "column_mass", cols)
            table.insert(0, "site", t_site)
            table.insert(0, "cell", list(t_ds.obs_names))
            table.to_csv(out / method / f"target_mass_{fam}_L{level}.tsv.gz", sep="\t", index=False, compression="gzip")
            for k, x in enumerate(names):
                summary.append({"family": fam, "level": level, "group": x, "cells": int(onehot[:, k].sum()),
                                "row_mass": float(rows @ onehot[:, k])})
        pd.DataFrame(summary).to_csv(out / method / "source_groups.tsv", sep="\t", index=False)
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
