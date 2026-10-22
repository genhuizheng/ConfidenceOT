"""End-to-end smoke test of run_pairwise.py + score_pairwise.py on synthetic counts (local, no TACC data).

    python lineage_validation/level2_pairwise_ot/test_pipeline.py

Builds a KP-Tracer-shaped pair (per-tumour raw + equalised files) and a Quinn-shaped pair (pooled sides,
target restricted to one site) in a temporary folder. Source subclones A (80%) and B (20%) carry different
expression programmes; the KP target is 100 B cells of the family and 20 of another family, the Quinn
target 40 A + 40 B. Checks: BOT marginals exact, UOT keeps every
cell, group masses sum to the column mass, the planted rare subclone B is recovered by the enrichment rule,
the Quinn cells of another clone go to "other", and the cell ids line up between raw and equalised files.
Prints PASS / FAIL; exit 1 on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def counts(rng, programme, n, genes=600, depth=(800, 4000)):
    base = rng.gamma(0.6, 1.0, genes)
    X = []
    for k in range(n):
        p = base * programme[k]
        p /= p.sum()
        X.append(rng.multinomial(int(rng.integers(*depth)), p))
    return np.array(X)


def adata(X, ids, sample, site, extra=None):
    obs = pd.DataFrame({"sample_id": sample, "site": site, "malignant": "malignant", **(extra or {})}, index=ids)
    a = ad.AnnData(sparse.csr_matrix(X.astype(np.float32)), obs=obs)
    a.var_names = [f"G{g}" for g in range(X.shape[1])]
    return a


def main():
    rng = np.random.default_rng(3)
    genes = 600
    prog = {"A": np.ones(genes), "B": np.ones(genes)}
    prog["A"][:60] *= 6
    prog["B"][60:120] *= 6
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        kp, lv, inputs, out = tmp / "kp", tmp / "lv", tmp / "inputs", tmp / "runs"
        for d in (kp / "prepared" / "h5ad", kp / "downsampled", lv / "inputs" / "manifests", lv / "inputs" / "h5ad" / "Quinn2021" / "ds", inputs):
            d.mkdir(parents=True)
        # KP-shaped: T1 = 240 A + 60 B + 30 no lineage; K1 = 100 B (family) + 20 A-like of another family
        src_lab = ["A"] * 240 + ["B"] * 60 + ["none"] * 30
        tgt_lab = ["B"] * 100 + ["foreign"] * 20
        Xs = counts(rng, [prog["A"] if l in ("A", "none") else prog["B"] for l in src_lab], len(src_lab))
        Xt = counts(rng, [prog["B"] if l == "B" else prog["A"] for l in tgt_lab], len(tgt_lab))
        sid, tid = [f"s{i}" for i in range(len(src_lab))], [f"t{j}" for j in range(len(tgt_lab))]
        # equalised = binomial thinning; the raw file has 10 extra non-malignant cells the equalised file lacks
        thin = lambda X: rng.binomial(X, 0.5)
        adata(Xs, sid, "M_T1", "M_T1").write_h5ad(kp / "downsampled" / "M_T1.h5ad")
        adata(thin(Xt), tid, "M_K1", "M_K1").write_h5ad(kp / "downsampled" / "M_K1.h5ad")
        adata(np.vstack([Xs, counts(rng, [prog["A"]] * 10, 10)]), sid + [f"x{i}" for i in range(10)], "M_T1", "M_T1").write_h5ad(kp / "prepared" / "h5ad" / "M_T1.h5ad")
        adata(Xt, tid, "M_K1", "M_K1").write_h5ad(kp / "prepared" / "h5ad" / "M_K1.h5ad")
        pd.DataFrame([{"source_sample": "M_T1", "target_sample": "M_K1", "source_h5ad": str(kp / "downsampled" / "M_T1.h5ad"),
                       "target_h5ad": str(kp / "downsampled" / "M_K1.h5ad")}]).to_csv(kp / "downsampled" / "pair_manifest_downsampled.csv", index=False)
        # Quinn-shaped: LL (clone LG1: 160 A + 40 B; clone LG9: 50) and a pooled metastases side with sites M1
        # (LG1, 40 A + 40 B: a mixed metastasis) and M2
        q_src = ["A"] * 160 + ["B"] * 40 + ["LG9"] * 50
        q_tgt = ["A"] * 40 + ["B"] * 40 + ["M2"] * 40
        Qs = counts(rng, [prog["A"] if l in ("A", "LG9") else prog["B"] for l in q_src], len(q_src))
        Qt = counts(rng, [prog["A"] if l == "A" else prog["B"] for l in q_tgt], len(q_tgt))
        qs, qt = [f"q{i}" for i in range(len(q_src))], [f"r{j}" for j in range(len(q_tgt))]
        sites = ["M1"] * 80 + ["M2"] * 40
        for kind, f in (("raw", lambda X: X), ("ds", thin)):
            folder = lv / "inputs" / "h5ad" / "Quinn2021" / kind
            folder.mkdir(parents=True, exist_ok=True)
            adata(f(Qs), qs, "Z_LL", "LL").write_h5ad(folder / "Z_LL.h5ad")
            a = adata(f(Qt), qt, "Z_metastases", sites)
            if kind == "ds":
                a.obs = a.obs.drop(columns="site")  # the site is read off the raw file
            a.write_h5ad(folder / "Z_metastases.h5ad")
            manifest = pd.DataFrame([{"patient_id": "Z", "source_sample": "Z_LL", "target_sample": "Z_metastases",
                                      "source_h5ad": str(folder / "Z_LL.h5ad"), "target_h5ad": str(folder / "Z_metastases.h5ad")}])
            manifest.to_csv(lv / "inputs" / "manifests" / "quinn_D_B_raw.csv" if kind == "raw" else folder / "pair_manifest_downsampled.csv", index=False)
        pd.DataFrame([{"index": 0, "pair_id": "KPTracer_M__M_T1__M_K1", "dataset": "KPTracer", "mouse": "M", "source_site": "M_T1",
                       "target_site": "M_K1", "families": "KPTracer_M_M_T1"},
                      {"index": 1, "pair_id": "Quinn2021_Z__LL__M1", "dataset": "Quinn2021", "mouse": "Z", "source_site": "LL",
                       "target_site": "M1", "families": "Quinn2021_Z_LG1"}]).to_csv(inputs / "pilot_pairs.tsv", sep="\t", index=False)
        pd.DataFrame([("KPTracer_M_M_T1", 1, "L1.1"), ("KPTracer_M_M_T1", 1, "L1.2"), ("Quinn2021_Z_LG1", 1, "L1.1"), ("Quinn2021_Z_LG1", 1, "L1.2")],
                     columns=["name", "level", "subclone"]).to_csv(inputs / "subclones.tsv", sep="\t", index=False)
        lab = {"A": "L1.1", "B": "L1.2"}
        fc = [("KPTracer_M_M_T1", c, lab[l]) for c, l in zip(sid, src_lab) if l in lab] + \
             [("KPTracer_M_M_T1", c, "L1.2") for c, l in zip(tid, tgt_lab) if l == "B"] + \
             [("Quinn2021_Z_LG1", c, lab[l]) for c, l in zip(qs, q_src) if l in lab] + \
             [("Quinn2021_Z_LG1", c, lab[l]) for c, l in zip(qt, q_tgt) if l in lab]
        pd.DataFrame([{"name": n, "dataset": n.split("_")[0], "mouse": n.split("_")[1], "cell_id": c, "site": "", "reliable": "True",
                       "L1": l, "L2": ""} for n, c, l in fc]).to_csv(inputs / "family_cells.tsv.gz", sep="\t", index=False, compression="gzip")
        for i in (0, 1):
            r = subprocess.run([sys.executable, str(HERE / "run_pairwise.py"), "--index", str(i), "--out", str(out), "--pairs",
                                str(inputs / "pilot_pairs.tsv"), "--inputs", str(inputs), "--kp-root", str(kp), "--lv-root", str(lv)],
                               capture_output=True, text=True)
            check(f"run_pairwise index {i} exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        kp_dir = out / "KPTracer_M__M_T1__M_K1"
        cells = pd.read_csv(kp_dir / "cells.tsv.gz", sep="\t")
        check("KP: source = the 330 equalised cells (raw-only cells left out), target 120",
              (cells["side"] == "source").sum() == 330 and (cells["side"] == "target").sum() == 120)
        for arm in ("A_logcpm_BOT", "B_ranknm256dscos_UOT", "AxUOT_logcpm_UOT", "BxBOT_ranknm256dscos_BOT"):
            d = json.loads((kp_dir / arm / "diagnostics.json").read_text())
            if d["solver"] == "BOT":
                check(f"{arm}: BOT marginals exact", d["converged"] and d["row_rel_dev_max"] < 1e-6 and d["col_rel_dev_max"] < 1e-6,
                      f"row {d['row_rel_dev_max']:.1e}, col {d['col_rel_dev_max']:.1e}")
            else:
                check(f"{arm}: UOT keeps every cell, mass not fixed", d["source_retained"] == 1 and d["target_retained"] == 1,
                      f"total mass {d['total_mass']:.3f}, row dev max {d['row_rel_dev_max']:.2f}")
            tm = pd.read_csv(kp_dir / arm / "target_mass_KPTracer_M_M_T1_L1.tsv.gz", sep="\t")
            P = np.load(kp_dir / arm / "coupling.npz")["coupling"]
            check(f"{arm}: group masses sum to the column mass and match the saved plan",
                  np.allclose(tm.filter(like="mass__").sum(1), tm["column_mass"], rtol=1e-6) and np.allclose(P.sum(0), tm["column_mass"], rtol=1e-4))
            src = pd.read_csv(kp_dir / arm / "source_groups.tsv", sep="\t").set_index("group")
            check(f"{arm}: source groups 240 / 60 / 30 unlabelled / 0 other",
                  list(src.loc[["L1.1", "L1.2", "family unlabelled", "other"], "cells"]) == [240, 60, 30, 0])
        q_dir = out / "Quinn2021_Z__LL__M1"
        qc = pd.read_csv(q_dir / "cells.tsv.gz", sep="\t")
        check("Quinn: target restricted to site M1 by the raw file's site (80 cells)", (qc["side"] == "target").sum() == 80)
        qsrc = pd.read_csv(q_dir / "A_logcpm_BOT" / "source_groups.tsv", sep="\t").set_index("group")
        check("Quinn: the other clone's 50 LL cells are 'other'", qsrc.loc["other", "cells"] == 50)
        r = subprocess.run([sys.executable, str(HERE / "score_pairwise.py"), "--runs", str(out), "--out", str(tmp / "score"),
                            "--inputs", str(inputs)], capture_output=True, text=True)
        check("score_pairwise exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        s = pd.read_csv(tmp / "score" / "summary.tsv", sep="\t")
        k = s[s["pair_id"].str.startswith("KP")]
        check("KP: 100 family targets scored, 20 foreign not", (k["scored"] == 100).all() and (k["target_family_cells"] == 100).all())
        # The identity behind PLAN.md's amendment: under BOT the mean over target cells of S_g is the source
        # cell share exactly, so a metastasis made of one subclone has mean own enrichment 1 whatever the plan.
        P = np.load(kp_dir / "A_logcpm_BOT" / "coupling.npz")["coupling"].astype(float)
        g = np.array(src_lab)
        S_B = P[g == "B"].sum(0) / P.sum(0)
        check("identity: under BOT the mean of S_B over all target cells is B's source cell share (60/330)",
              abs(S_B.mean() - 60 / 330) < 1e-5, f"{S_B.mean():.6f} vs {60 / 330:.6f}")
        check("planted rare subclone L1.2 (20%) recovered by the enrichment rule in every arm (> 0.8; chance 0.5)",
              (s["balanced_accuracy_enrichment"] > 0.8).all(), s[["arm", "balanced_accuracy_enrichment"]].to_string(index=False))
        check("own-subclone log2 enrichment > 0 in every arm", (s["mean_own_log2E"] > 0).all(), " ".join(f"{v:.2f}" for v in s["mean_own_log2E"]))
        per = pd.read_csv(tmp / "score" / "per_subclone.tsv", sep="\t")
        b = per[(per["subclone"] == "L1.2") & per["pair_id"].str.startswith("KP")]
        check("KP BOT: pi of L1.2 = cell share within the family (60/330)",
              np.allclose(b.loc[b["arm"].str.endswith("BOT"), "pi"], 60 / 330))
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
