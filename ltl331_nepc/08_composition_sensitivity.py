"""Target-composition sensitivity of the 16 -> 20 week gate (local; new OT fits in their own directory, the frozen
run is only read).

    python ltl331_nepc/08_composition_sensitivity.py --run-root RUN_ROOT --converted CONVERTED --out OUT

The 20-week target is 1,198 NE of 1,328 cells, so NE source cells may be retained only because almost every target
cell is NE. Here the target's NE cells are subsampled while every non-NE target cell (130) is kept, and the pair is
refitted with exactly the factorial run's arguments (cancer_metastasis/tacc_preprocessing_factorial.slurm with the
LTL331 settings: scope malignant, within-side split null, acceptance 0.99, 5 + 5 null replicates, grid 5,
tolerance 1e-4, no cell cap, cell QC recorded only, cpu). Levels: NE kept 130 (target 50% NE), 33 (20%) and 0
(no NE), seeds 1-3 for the first two, one fit for 0; both labels. Source cells are never changed.

--size-control (added 2026-10-31 after the first fits showed that every subsampled target also changed the target
size, and that the ds_cos calibration was invalid in all of them): the target is instead subsampled uniformly at
random, whatever the cell type, to the same totals (260, 163, 130 cells; seeds 1-3), so its composition stays as
observed (about 90% NE). Composition and size can then be told apart. Output goes to its own OUT.

Writes OUT/fits/<label>/<level>_s<seed>/ (02_run_pair.py output and log), OUT/composition_sensitivity.tsv (per fit:
target composition, M4-E calibration validity, source retained fraction per type, NE minus non-NE) and the same for
the OT-free kNN baseline of 07_validity_audit.py computed on each subsampled pair.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PAIR = "LTL331__w16_16wkCX_1__w20_20wkCX_1_R"
LEVELS = [(130, [1, 2, 3]), (33, [1, 2, 3]), (0, [1])]
FACTORIAL_ARGS = ["--analysis-scope", "malignant", "--malignant-column", "malignant", "--malignant-value", "malignant",
                  "--calibration-null", "within_side_split", "--max-observed-cells-per-side", "0", "--minimum-scope-cells", "4",
                  "--cell-qc", "--minimum-total-counts", "0", "--minimum-detected-genes", "0", "--maximum-mitochondrial-percent", "100",
                  "--null-calibration-replicates", "5", "--null-validation-replicates", "5", "--calibration-grid-size", "5",
                  "--tolerance", "1e-4", "--workers", "8", "--device", "cpu", "--within-side-acceptance-minimum", "0.99"]


def audit_module():
    spec = importlib.util.spec_from_file_location("audit07", HERE / "07_validity_audit.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--size-control", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        sys.exit(f"{a.out} exists; not overwritten")
    a.out.mkdir(parents=True)
    au = audit_module()
    ctype = pd.concat([ad.read_h5ad(f, backed="r").obs["coarse_celltype"].astype(str) for f in sorted((a.converted / "h5ad").glob("*.h5ad"))])
    sep = ";" if os.name == "nt" else ":"
    env = {**os.environ, "PYTHONPATH": sep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO]))}
    rows = []
    for label in au.LABELS:
        man = pd.read_csv(a.run_root / "inputs" / au.MANIFESTS[label])
        row = man[man.pair_id == PAIR].iloc[0].copy()
        source = ad.read_h5ad(au.local(row.source_h5ad, a.run_root)); target = ad.read_h5ad(au.local(row.target_h5ad, a.run_root))
        tt = ctype.loc[target.obs_names.astype(str)].to_numpy(); st = ctype.loc[source.obs_names.astype(str)].to_numpy()
        ne_idx, other_idx = np.flatnonzero(tt == "NE"), np.flatnonzero(tt != "NE")
        cfg = au.Preprocessing.from_label(label)
        levels = [(n, [1, 2, 3]) for n in (260, 163, 130)] if a.size_control else LEVELS
        for keep, seeds in levels:
            for seed in seeds:
                name = f"random{keep}_s{seed}" if a.size_control else f"ne{keep}_s{seed}"
                fit = a.out / "fits" / label / name; (fit / "input").mkdir(parents=True)
                if a.size_control:
                    chosen = np.sort(np.random.default_rng(seed).choice(len(tt), keep, replace=False))
                else:
                    chosen = np.sort(np.concatenate([other_idx, np.random.default_rng(seed).choice(ne_idx, keep, replace=False)]))
                sub = target[chosen].copy(); h5 = fit / "input" / "target.h5ad"; sub.write_h5ad(h5)
                r = row.copy(); r["target_h5ad"] = str(h5.resolve()); r["target_h5ads_json"] = json.dumps([str(h5.resolve())])
                r["source_h5ad"] = str(au.local(row.source_h5ad, a.run_root).resolve()); r["source_h5ads_json"] = json.dumps([r["source_h5ad"]])
                pd.DataFrame([r]).to_csv(fit / "input" / "manifest.csv", index=False)
                cmd = [sys.executable, str(REPO / "cancer_metastasis" / "02_run_pair.py"), str(fit / "input" / "manifest.csv"), str(fit / "out"),
                       "--index", "0", "--preprocessing", label, *FACTORIAL_ARGS]
                with open(fit / "run.log", "w") as log:
                    code = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
                ne_n = int((tt[chosen] == "NE").sum())
                rec = {"label": label, "design": "random_size" if a.size_control else "ne_subsample", "level": name.split("_")[0],
                       "seed": seed, "target_n": len(chosen), "target_ne_n": ne_n, "target_non_ne_n": len(chosen) - ne_n,
                       "target_ne_fraction": ne_n / len(chosen), "exit": code}
                if code != 0:
                    rows.append({**rec, "status": "fit failed; see run.log"}); print(label, name, "FAILED", flush=True); continue
                d = next((fit / "out" / PAIR / "scope_malignant").glob("budget_*"))
                m = pd.read_csv(d / "pair_metrics.csv").set_index("method").loc["M4-E"]
                cc = pd.read_csv(d / "cell_confidence.csv"); cc = cc[(cc.method == "M4-E") & (cc.side == "source")].set_index("observation_id")
                ret = ~cc.loc[source.obs_names.astype(str), "rejected"].to_numpy(bool)
                # OT-free kNN baseline on the same subsampled pair, in the representation the fit used
                sp, tp, _, _ = au.prepare_joint_representation(
                    source, sub, n_hvg=cfg.n_hvg or 2000, n_pcs=cfg.n_pcs or 30, seed=au.SEED, representation=cfg.normalisation,
                    rank_top_n=cfg.rank_top_n, cost=cfg.cost, equalise_depth=cfg.equalise_depth, regress_out=cfg.regress_out,
                    regress_on_ranks=cfg.regress_on_ranks, scale_genes=cfg.scale_genes, label_stem=cfg.label_stem)
                scale = au.median_pair_scale(sp, tp, rng=np.random.default_rng(au.SEED))
                kb = au.side_costs(sp, tp, scale, 10, np.random.default_rng(seed))["knn_retained"]
                rec.update(status="fitted", m4e_inference_valid=bool(m.calibration_m4e_inference_valid),
                           source_rejected_fraction=float(m.source_final_rejection_rate), target_rejected_fraction=float(m.target_final_rejection_rate))
                for t in ["PRAD", "Basal", "Other", "NE"]:
                    rec[f"ot_retained_{t}"] = ret[st == t].mean(); rec[f"knn_retained_{t}"] = kb[st == t].mean()
                rec["ot_ne_minus_nonne"] = ret[st == "NE"].mean() - ret[st != "NE"].mean()
                rec["knn_ne_minus_nonne"] = kb[st == "NE"].mean() - kb[st != "NE"].mean()
                rows.append(rec)
                print(label, name, f"OT NE {rec['ot_retained_NE']:.2f} vs non-NE {ret[st != 'NE'].mean():.2f}; valid {rec['m4e_inference_valid']}", flush=True)
    pd.DataFrame(rows).to_csv(a.out / "composition_sensitivity.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
