"""Conditions and ground truth of one generated level (N = 5,000 benchmark, L0-L5 and the ablations).

    python realdata_benchmark/splatter_n5000/21_build_conditions_n5000.py --generated GEN --out OUT --n-cells 5000 \
        --replicate R --level LEVEL

The frozen benchmark/20_build_conditions.py builds the four case pairs (all_shared, population_lost,
population_emerged, populations_disjoint; Case 1 / 2 / 2 / 3) unchanged. It accepts only the native level names, so
L3-L5 and the ablations, whose observation setting is L2's (depth x 0.5 and target dropout; their own factors are
already in the counts), are built as L2_depth_dropout with the native mechanism (theta 1.0, depth from Splatter), and
the manifest then records the real level: technical_level, benchmark_level, level_inherits, level_adds,
benchmark_pair_id (N<n>_rep<r>_<level>_<case>; pair_id keeps the builder's name, which the h5ad sample ids carry).
Ground truth per pair (OUT/<pair_id>/truth/):
  population_truth.tsv   per cell in h5ad order: side, cell_id, group, shared, should_reject, perturbed
  deg_truth.tsv          DEG-A reference, per side that holds an unmatched population: per gene the true log2 fold
                         change of the unmatched group against the side's shared cells,
                         log2(DEFac_U x Pert_U / sum_g w_g DEFac_g x Pert_g), w_g the side's shared-group cell
                         fractions (GeneMean and the side's batch factor cancel within a side); true_de = |log2FC| >= 0.25
  shared_shift_truth.tsv DEG-D reference, per shared group: biological log2FC target vs source (log2 PerturbationFactor
                         for the perturbed group in the target, else 0) and technical log2FC (log2 BatchFacBatch2 /
                         BatchFacBatch1), kept apart
  batch_factors.tsv, perturbation_truth.tsv  copies of the generator's gene truth, for the deliverables
Refuses if OUT/manifest.csv exists.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
NATIVE = {"L0_matched", "L1_depth", "L2_depth_dropout"}
LEVELS = {
    "L0_matched": ("", "matched depth, no dropout"),
    "L1_depth": ("L0", "target library size x 0.5"),
    "L2_depth_dropout": ("L1", "target dropout"),
    "L3_batch": ("L2", "source / target batch effect (strength 0.5 D)"),
    "L4_batch_perturb": ("L3", "target perturbation of Group1 (strength 0.5 D)"),
    "L5_batch_perturb_imbalance": ("L4", "target population proportions 0.40 / 0.25 / 0.15 / 0.12 / 0.08"),
    "A4_perturb_only": ("L2", "target perturbation of Group1 (ablation: no batch)"),
    "A5_imbalance_only": ("L2", "target population proportions (ablation: no batch, no perturbation)"),
}
CASE = {"all_shared": "Case1_all_shared", "population_lost": "Case2_source_unmatched",
        "population_emerged": "Case2_target_unmatched", "populations_disjoint": "Case3_none_shared"}
DE_THRESHOLD = 0.25


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-cells", type=int, required=True)
    p.add_argument("--replicate", type=int, required=True)
    p.add_argument("--level", required=True, choices=sorted(LEVELS))
    a = p.parse_args()
    if (a.out / "manifest.csv").exists():
        sys.exit(f"{a.out}/manifest.csv exists; not overwritten")
    gen = json.loads((a.generated / "generation.json").read_text())
    if gen.get("technical_level") != a.level:
        sys.exit(f"generated level {gen.get('technical_level')} is not {a.level}")
    native = a.level if a.level in NATIVE else "L2_depth_dropout"
    builder_input = a.generated
    if native != a.level:
        # the frozen builder checks that the generation record names the level it is asked for; it is given a staging
        # directory whose record names L2 (the observation setting) and keeps the real level, the files hard-linked
        builder_input = a.out.parent / f".builder_input_{a.out.name}"
        builder_input.mkdir(parents=True, exist_ok=True)
        for f in a.generated.iterdir():
            if f.is_file() and f.name != "generation.json":
                dst = builder_input / f.name
                if not dst.exists():
                    try:
                        os.link(f, dst)
                    except OSError:
                        shutil.copy2(f, dst)
        (builder_input / "generation.json").write_text(json.dumps({**gen, "technical_level": native, "benchmark_level": a.level}))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO]))}
    cmd = [sys.executable, str(REPO / "benchmark" / "20_build_conditions.py"), "--generated", str(builder_input), "--out", str(a.out),
           "--n-cells", str(a.n_cells), "--replicate", str(a.replicate), "--level", native, "--theta", "1.0",
           "--depth-mechanism", "splatter", "--rank-top-n", "256", "--seed", "20260925"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"20_build_conditions.py failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    m = pd.read_csv(a.out / "manifest.csv")
    inherits, adds = LEVELS[a.level]
    m["native_builder_level"] = native
    m["technical_level"] = a.level
    m["benchmark_level"] = a.level
    m["level_inherits"] = inherits
    m["level_adds"] = adds
    m["case"] = m.biological_case.map(CASE)
    m["benchmark_pair_id"] = [f"N{a.n_cells}_rep{a.replicate}_{a.level}_{c}" for c in m.biological_case]
    for k in ("batch_strength", "batch_fac_scale", "perturb_strength", "perturb_group", "perturbed_genes", "perturbed_target_cells",
              "group_de_distance_D", "seed", "splatter_version", "n_semantics"):
        m[f"gen_{k}"] = gen.get(k)
    m["gen_target_group_prob"] = "/".join(f"{v:g}" for v in gen.get("target_group_prob", []))
    m.to_csv(a.out / "manifest.csv", index=False)

    genes = pd.read_csv(a.generated / "genes_truth.csv")
    cells_truth = pd.read_csv(a.generated / "cells_truth.csv").set_index("cell_id")
    de_cols = sorted([c for c in genes if c.startswith("DEFacGroup")])
    pert = genes.PerturbationFactor.to_numpy(float)
    pert_group = gen.get("perturb_group", "Group1")
    batch_cols = [c for c in genes if c.startswith("BatchFacBatch")]
    for _, row in m.iterrows():
        d = a.out / str(row.pair_id) / "truth"; d.mkdir(parents=True, exist_ok=True)
        truth = pd.read_csv(row.truth_csv)
        cell = []
        for s in ("source", "target"):
            obs = ad.read_h5ad(row[f"{s}_h5ad"], backed="r").obs
            t = truth[truth.side == s].reset_index(drop=True)
            if len(t) != len(obs):
                sys.exit(f"{row.pair_id}: truth rows {len(t)} vs {len(obs)} {s} cells")
            # the builder names cells <sample>:<generator cell_id> and keeps the generator's id in obs['cell_id']
            t["observation_id"] = list(obs.index.astype(str)); t["cell_id"] = list(obs["cell_id"].astype(str))
            ct = cells_truth.reindex(t.cell_id)
            if ct.group.isna().any() or (ct.group.to_numpy() != t.group.to_numpy()).any():
                sys.exit(f"{row.pair_id}: {s} cells do not match the generator's cells / groups")
            t["perturbed"] = ct["perturbed"].astype(bool).to_numpy()
            cell.append(t)
        cell = pd.concat(cell, ignore_index=True)
        groups_on = {s: set(cell.loc[cell.side == s, "group"]) for s in ("source", "target")}
        shared = groups_on["source"] & groups_on["target"]
        cell["shared"] = cell.group.isin(shared)
        cell[["side", "observation_id", "cell_id", "group", "shared", "should_reject", "perturbed"]].to_csv(d / "population_truth.tsv", sep="\t", index=False)
        deg = []
        for s in ("source", "target"):
            side = cell[cell.side == s]
            unmatched = sorted(set(side.group) - shared)
            if not unmatched or not shared:
                continue
            w = side[side.group.isin(shared)].group.value_counts(normalize=True)
            fac = lambda g: genes[f"DEFac{g}"].to_numpy(float) * (pert if (s == "target" and g == pert_group) else 1.0)
            mix = sum(w[g] * fac(g) for g in w.index)
            for u in unmatched:
                lfc = np.log2(fac(u) / mix)
                deg.append(pd.DataFrame({"side": s, "unmatched_group": u, "gene_id": genes.gene_id, "true_log2fc": lfc,
                                         "true_de": np.abs(lfc) >= DE_THRESHOLD, "direction": np.sign(lfc).astype(int),
                                         "shared_weights": ";".join(f"{g}:{w[g]:.4f}" for g in w.index)}))
        pd.concat(deg).to_csv(d / "deg_truth.tsv", sep="\t", index=False) if deg else \
            pd.DataFrame(columns=["side", "unmatched_group", "gene_id", "true_log2fc", "true_de", "direction", "shared_weights"]).to_csv(d / "deg_truth.tsv", sep="\t", index=False)
        tech = np.log2(genes[batch_cols[1]].to_numpy(float) / genes[batch_cols[0]].to_numpy(float)) if len(batch_cols) == 2 else np.zeros(len(genes))
        shift = [pd.DataFrame({"group": g, "gene_id": genes.gene_id,
                               "biological_log2fc": np.log2(pert) if g == pert_group else 0.0,
                               "technical_log2fc": tech}) for g in sorted(shared)]
        (pd.concat(shift) if shift else pd.DataFrame(columns=["group", "gene_id", "biological_log2fc", "technical_log2fc"])).to_csv(
            d / "shared_shift_truth.tsv", sep="\t", index=False)
        genes[["gene_id"] + batch_cols].to_csv(d / "batch_factors.tsv", sep="\t", index=False)
        genes[["gene_id", "PerturbationFactor"]].assign(perturbed=pert != 1, direction=np.sign(np.log(pert)).astype(int),
                                                         log2_effect=np.log2(pert), group=pert_group).to_csv(d / "perturbation_truth.tsv", sep="\t", index=False)
    shutil.copy2(a.generated / "generation.json", a.out / "generation.json")
    print(f"{a.level} rep {a.replicate}: {len(m)} pairs -> {a.out}")


if __name__ == "__main__":
    main()
