"""Local test of 21_build_conditions_n5000.py on a generated directory shaped like 20_generate_n5000.R's output
(R and Splatter are on TACC only). Synthetic counts: 5 groups x 120 cells per side, 400 genes, group DE factors,
batch factors and a Group1 target perturbation, Poisson counts from the means.

    python realdata_benchmark/splatter_n5000/tests/test_conditions_truth.py [--keep DIR]

Checks: the frozen builder runs for a non-native level through the L2 mapping; the manifest records the real level,
its inheritance and the four cases; population truth aligns with the h5ad cells and marks Group5 unmatched in the
partial cases and everything in the disjoint case; DEG-A truth equals log2(DEFac_U / weighted shared mix) computed
independently; perturbed cells are the target Group1 cells; shared-shift truth separates the biological (Group1
target) and technical (batch) parts. PASS / FAIL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok)); print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def fake_generated(out: Path, level: str, rng) -> None:
    n, genes, K = 120, 400, 5
    base = rng.gamma(0.6, 3.0, genes) + 0.05
    defac = np.ones((K, genes))
    for k in range(K):
        hit = rng.random(genes) < 0.1
        defac[k, hit] = np.exp(rng.choice([-1, 1], hit.sum()) * rng.normal(0.4, 0.3, hit.sum()))
    b1, b2 = np.exp(rng.normal(0, 0.1, genes)), np.exp(rng.normal(0, 0.1, genes))
    pert = np.ones(genes); hit = rng.random(genes) < 0.1; pert[hit] = np.exp(rng.choice([-1, 1], hit.sum()) * 0.5)
    rows, cells = [], []
    for side, batch, bf in (("source", "Batch1", b1), ("target", "Batch2", b2)):
        for k in range(K):
            for i in range(n):
                mu = base * defac[k] * bf * (pert if (side == "target" and k == 0) else 1.0) * rng.uniform(5, 15)
                rows.append(rng.poisson(mu)); cells.append((f"{batch}_Cell{side[0]}{k}_{i}", batch, f"Group{k + 1}", side == "target" and k == 0))
    X = np.array(rows)
    out.mkdir(parents=True)
    scipy.io.mmwrite(str(out / "counts.mtx"), sp.csr_matrix(X.T))
    c = pd.DataFrame(cells, columns=["cell_id", "batch", "group", "perturbed"])
    c[["cell_id", "batch", "group"]].to_csv(out / "cells.csv", index=False)
    gid = [f"Gene{j + 1}" for j in range(genes)]
    pd.DataFrame({"gene_id": gid, "gene_mean": base}).to_csv(out / "genes.csv", index=False)
    gt = pd.DataFrame({"gene_id": gid, "GeneMean": base, "BaseGeneMean": base, "OutlierFactor": 1.0,
                       **{f"DEFacGroup{k + 1}": defac[k] for k in range(K)}, "BatchFacBatch1": b1, "BatchFacBatch2": b2,
                       "PerturbationFactor": pert})
    gt.to_csv(out / "genes_truth.csv", index=False)
    c.assign(side=np.where(c.batch == "Batch1", "source", "target"), ExpLibSize=1.0)[["cell_id", "side", "batch", "group", "ExpLibSize", "perturbed"]].to_csv(out / "cells_truth.csv", index=False)
    (out / "generation.json").write_text(json.dumps({"technical_level": level, "perturb_group": "Group1", "batch_strength": 0.5,
                                                     "perturb_strength": 0.5, "target_group_prob": [0.2] * 5, "seed": 1,
                                                     "n_semantics": "cells per side"}))
    (out / "GENERATION_DONE").write_text("SUCCESS")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--keep", type=Path, default=None); a = p.parse_args()
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        root = a.keep or Path(tmp)
        gen, out = root / "generated", root / "conditions"
        fake_generated(gen, "L4_batch_perturb", rng)
        r = subprocess.run([sys.executable, str(HERE.parent / "21_build_conditions_n5000.py"), "--generated", str(gen), "--out", str(out),
                            "--n-cells", "120", "--replicate", "1", "--level", "L4_batch_perturb"], capture_output=True, text=True)
        check("21_ exits 0", r.returncode == 0, (r.stdout + r.stderr)[-1500:] if r.returncode else "")
        if r.returncode:
            sys.exit(1)
        m = pd.read_csv(out / "manifest.csv")
        check("four cases, level recorded as L4 with its inheritance", len(m) == 4 and set(m.technical_level) == {"L4_batch_perturb"}
              and set(m.level_inherits) == {"L3"} and set(m.native_builder_level) == {"L2_depth_dropout"}, str(m.case.tolist()))
        genes = pd.read_csv(gen / "genes_truth.csv")
        for _, row in m.iterrows():
            d = out / row.pair_id / "truth"
            pt = pd.read_csv(d / "population_truth.tsv", sep="\t")
            case = row.biological_case
            if case == "population_lost":
                check("lost: source Group5 should_reject, nothing else", set(pt[pt.should_reject].group) == {"Group5"}
                      and set(pt[pt.should_reject].side) == {"source"})
                dt = pd.read_csv(d / "deg_truth.tsv", sep="\t")
                side = pt[(pt.side == "source") & pt.shared]
                w = side.group.value_counts(normalize=True)
                mix = sum(w[g] * genes[f"DEFac{g}"].to_numpy() for g in w.index)
                expect = np.log2(genes["DEFacGroup5"].to_numpy() / mix)
                check("DEG-A truth equals the independent derivation (source side, Group5)",
                      np.allclose(dt[dt.side == "source"].true_log2fc.to_numpy(), expect), f"{dt.true_de.sum()} true DE genes")
            if case == "population_emerged":
                dt = pd.read_csv(d / "deg_truth.tsv", sep="\t")
                side = pt[(pt.side == "target") & pt.shared]
                w = side.group.value_counts(normalize=True)
                fac = lambda g: genes[f"DEFac{g}"].to_numpy() * (genes.PerturbationFactor.to_numpy() if g == "Group1" else 1)
                expect = np.log2(fac("Group5") / sum(w[g] * fac(g) for g in w.index))
                check("DEG-A truth on the target includes the Group1 perturbation in the shared mix",
                      np.allclose(dt[dt.side == "target"].true_log2fc.to_numpy(), expect))
            if case == "populations_disjoint":
                check("disjoint: every cell should_reject, no shared group", pt.should_reject.all() and not pt.shared.any())
            if case == "all_shared":
                check("all_shared: nothing to reject; perturbed = target Group1 cells only",
                      not pt.should_reject.any() and set(pt[pt.perturbed].group) == {"Group1"} and set(pt[pt.perturbed].side) == {"target"}
                      and pt[(pt.side == "target") & (pt.group == "Group1")].perturbed.all())
                sh = pd.read_csv(d / "shared_shift_truth.tsv", sep="\t")
                g1, g2 = sh[sh.group == "Group1"], sh[sh.group == "Group2"]
                check("shared shift: biological part only in Group1, technical part = log2 batch ratio in every group",
                      (g2.biological_log2fc == 0).all() and np.allclose(g1.biological_log2fc, np.log2(genes.PerturbationFactor))
                      and np.allclose(g2.technical_log2fc, np.log2(genes.BatchFacBatch2 / genes.BatchFacBatch1)))
        r2 = subprocess.run([sys.executable, str(HERE.parent / "21_build_conditions_n5000.py"), "--generated", str(gen), "--out", str(out),
                             "--n-cells", "120", "--replicate", "1", "--level", "L4_batch_perturb"], capture_output=True, text=True)
        check("refuses to overwrite", r2.returncode != 0)
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
