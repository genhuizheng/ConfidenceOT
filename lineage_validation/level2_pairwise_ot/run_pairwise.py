"""Level 2 OT without rejection: four runs from one primary to its family's metastases (PLAN.md).

    python lineage_validation/level2_pairwise_ot/run_pairwise.py --index I --out OUT_ROOT

Row I of inputs/pilot_pairs.tsv. Cells: the malignant cells of the primary and of the target sites in the
depth-equalised files (KP-Tracer: the per-tumour files of the 104-pair runs, the target sites concatenated;
Quinn: 27_'s output for the lineage-validation LL and metastases sides), and the same cell ids in the raw
files. One plan covers every target site; a primary -> site block is a column slice of it (Level 3).
Two representations, built by cancer_metastasis/common.py:prepare_joint_representation exactly as
02_run_pair.py builds them (seed 20260831 + I, 2,000 HVGs, 30 PCs, cost scale median_pair_scale with
seed 20260831 + I * 104729):
  logcpm_noscale            on raw counts
  ranknm256_noscale_ds_cos  on depth-equalised counts
Two solvers on each cost, epsilon 0.1:
  BOT  traditional_ot.balanced.balanced_ot (uniform marginals, threshold 1e-9, <= 10,000 iterations)
  UOT  ConfidenceOT(backbone="uot", variant="exact", bounds (0, 0) on both sides), lambda 1/1, tolerance 1e-4
Arms: A = logcpm_noscale + BOT, B = ranknm256_noscale_ds_cos + UOT; AxUOT and BxBOT are the decomposition.

Writes OUT_ROOT/<pair_id>/:
  sites.json                       target sites used (cells) and left out (no equalised file, < 20 cells)
  cells.tsv.gz                     side, cell id, site, total counts raw and equalised (plan row/column order)
  coordinates_<label>.npz          source/target coordinates, cost scale, HVGs
  <arm>/coupling.npz               the full plan (float32), for Level 3
  <arm>/source_groups.tsv          per family and level: group, cells, row mass of the plan
  <arm>/target_mass_<family>_L<k>.tsv.gz  per target cell: site, column mass, plan mass from each source group
  <arm>/diagnostics.json           convergence, total mass, marginal deviations, seconds
  SUCCESS
CRISPR labels (inputs/family_cells.tsv.gz) are read only after every coupling is computed, to sum it by
source group. Refuses if SUCCESS exists.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path[:0] = [str(REPO / "src"), str(REPO / "cancer_metastasis")]
from common import load_exact_side, prepare_joint_representation  # noqa: E402
from confidenceot import ConfidenceOT  # noqa: E402
from confidenceot.preprocessing import Preprocessing, median_pair_scale, squared_euclidean  # noqa: E402
from traditional_ot.balanced import balanced_ot  # noqa: E402

SEED, EPSILON, N_HVG, N_PCS = 20260831, 0.1, 2000, 30
LABELS = {"logcpm_noscale": "raw", "ranknm256_noscale_ds_cos": "ds"}
ARMS = {"A_logcpm_BOT": ("logcpm_noscale", "BOT"), "B_ranknm256dscos_UOT": ("ranknm256_noscale_ds_cos", "UOT"),
        "AxUOT_logcpm_UOT": ("logcpm_noscale", "UOT"), "BxBOT_ranknm256dscos_BOT": ("ranknm256_noscale_ds_cos", "BOT")}
UNLABELLED, OTHER = "family unlabelled", "other"
KP = Path("/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot")
LV = Path("/scratch/10119/ghzheng/OT_project/lineage_validation")


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
        part = lambda t: {"ds": (ds_path[t], t), "raw": ([str(KP / "prepared" / "h5ad" / f"{t}.h5ad")], t), "site": t}
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
    spec = lambda side, site: [{k: (side_paths(rows[k], side), rows[k][f"{side}_sample"]) for k in ("ds", "raw")} | {"site": site}]
    return {"source": spec("source", src), "target": spec("target", None if tgt == "ALL" else tgt), "left_out": {}}


def load_part(part: dict, side: str):
    ds, raw = load_exact_side(*part["ds"]), load_exact_side(*part["raw"])
    if "malignant" in ds.obs:
        ds = ds[ds.obs["malignant"].astype(str).eq("malignant").to_numpy()].copy()
    if not ds.obs_names.isin(raw.obs_names).all():
        raise ValueError(f"{side}: {int((~ds.obs_names.isin(raw.obs_names)).sum())} equalised cells missing from the raw file")
    raw = raw[ds.obs_names].copy()
    # The site of each cell is read off the raw file (Quinn's equalised files need not carry it).
    site = raw.obs["site"].astype(str).to_numpy() if "site" in raw.obs else np.repeat(part["site"], raw.n_obs)
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
            r = ConfidenceOT(backbone="uot", variant="exact", rejection_cost=1.0, epsilon=EPSILON, lambda_a=1.0, lambda_b=1.0,
                             source_rejection_bounds=(0.0, 0.0), target_rejection_bounds=(0.0, 0.0),
                             tolerance=1e-4, device="cpu").fit(cost)
        P = r.coupling
        info = {"converged": bool(r.inner_converged and r.outer_converged), "inner_converged": bool(r.inner_converged),
                "outer_converged": bool(r.outer_converged), "iterations": int(r.total_inner_iterations),
                "source_retained": float(np.mean(r.source_gate)), "target_retained": float(np.mean(r.target_gate)),
                "warnings": [str(w.message)[:300] for w in caught]}
        if not (r.source_gate.all() and r.target_gate.all()):
            raise RuntimeError("UOT with bounds (0, 0) gated a cell")
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
    parser.add_argument("--pairs", type=Path, default=HERE / "inputs" / "pilot_pairs.tsv")
    parser.add_argument("--inputs", type=Path, default=HERE / "inputs", help="family_cells.tsv.gz and subclones.tsv")
    parser.add_argument("--kp-root", type=Path, default=KP)
    parser.add_argument("--lv-root", type=Path, default=LV)
    args = parser.parse_args()
    globals().update(KP=args.kp_root, LV=args.lv_root)
    pair = pd.read_csv(args.pairs, sep="\t", dtype=str).iloc[args.index]
    if int(pair["index"]) != args.index:
        raise ValueError("pilot_pairs.tsv row order does not match its index column")
    out = args.out / pair["pair_id"]
    if (out / "SUCCESS").exists():
        sys.exit(f"{out}/SUCCESS exists")
    out.mkdir(parents=True, exist_ok=True)
    spec = files_for(pair)
    left_out = dict(spec["left_out"])
    (s_raw, s_ds, s_site), (t_raw, t_ds, t_site) = load_side(spec["source"], "source", left_out), load_side(spec["target"], "target", left_out)
    used = {str(k): int(v) for k, v in pd.Series(t_site).value_counts().items()}
    (out / "sites.json").write_text(json.dumps({"source": pair["source_site"], "source_cells": int(s_ds.n_obs),
                                                "target_sites_used": used, "target_sites_left_out": left_out}, indent=1), encoding="utf-8")
    print(f"{pair['pair_id']}: source {s_ds.n_obs}, target {t_ds.n_obs} cells ({used}); left out {left_out}", flush=True)
    total = lambda a: np.asarray(a.X.sum(1)).ravel()
    pd.concat([pd.DataFrame({"side": side, "cell": d.obs_names, "site": s, "total_counts_raw": total(r), "total_counts_ds": total(d)})
               for side, r, d, s in (("source", s_raw, s_ds, s_site), ("target", t_raw, t_ds, t_site))]).to_csv(
        out / "cells.tsv.gz", sep="\t", index=False, compression="gzip")
    costs = {}
    for label, kind in LABELS.items():
        cfg = Preprocessing.from_label(label)
        src, tgt = (s_raw, t_raw) if kind == "raw" else (s_ds, t_ds)
        xs, xt, hvg, record = prepare_joint_representation(
            src, tgt, n_hvg=N_HVG, n_pcs=N_PCS, seed=SEED + args.index, representation=cfg.normalisation,
            rank_top_n=cfg.rank_top_n, minimum_detection_rate=0.0, cost=cfg.cost, equalise_depth=cfg.equalise_depth,
            regress_out=cfg.regress_out, regress_on_ranks=cfg.regress_on_ranks, scale_genes=cfg.scale_genes,
            label_stem=cfg.label_stem)
        if record.get("label") != label:
            raise SystemExit(f"{label}: the representation built {record.get('label')!r}")
        scale = median_pair_scale(xs, xt, rng=np.random.default_rng(SEED + args.index * 104729))
        costs[label] = squared_euclidean(xs, xt) / scale
        np.savez_compressed(out / f"coordinates_{label}.npz", source=xs, target=xt, scale=scale, hvg=np.asarray(hvg, dtype=str),
                            source_ids=np.asarray(src.obs_names, dtype=str), target_ids=np.asarray(tgt.obs_names, dtype=str))
    plans = {}
    for arm, (label, solver) in ARMS.items():
        P, info = solve(costs[label], solver)
        info.update(arm=arm, preprocessing=label, solver=solver, epsilon=EPSILON, n_source=int(P.shape[0]), n_target=int(P.shape[1]))
        print(json.dumps(info), flush=True)
        (out / arm).mkdir(exist_ok=True)
        np.savez_compressed(out / arm / "coupling.npz", coupling=P.astype(np.float32))
        (out / arm / "diagnostics.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        plans[arm] = P
    # CRISPR labels enter only here, after every coupling exists.
    cells = pd.read_csv(args.inputs / "family_cells.tsv.gz", sep="\t", dtype=str, keep_default_na=False)
    subs = pd.read_csv(args.inputs / "subclones.tsv", sep="\t", dtype={"level": int})
    groups = groups_for(pair, list(s_ds.obs_names), cells, subs)
    for arm, P in plans.items():
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
            table.to_csv(out / arm / f"target_mass_{fam}_L{level}.tsv.gz", sep="\t", index=False, compression="gzip")
            for k, x in enumerate(names):
                summary.append({"family": fam, "level": level, "group": x, "cells": int(onehot[:, k].sum()),
                                "row_mass": float(rows @ onehot[:, k])})
        pd.DataFrame(summary).to_csv(out / arm / "source_groups.tsv", sep="\t", index=False)
    (out / "SUCCESS").write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    main()
