"""End-to-end smoke test of run_pairwise.py + score_pairwise.py on synthetic counts (local, no TACC data).

    python lineage_validation/level2_pairwise_ot/test_pipeline.py

Builds a KP-Tracer-shaped case (per-tumour raw + equalised files; target = two metastases K1 and K2 in one
plan) and a Quinn-shaped case (pooled LL and metastases sides, sites read off the raw file) in a temporary
folder. Source subclones A (80%) and B (20%) carry different expression programmes; K1 is 100 B cells of
the family and 20 of another family, K2 60 A, the Quinn M1 40 A + 40 B. Checks: exactly two OT runs per case (A_standard_OT, B_our_OT) on the same
cells; A is Balanced OT with exact marginals; B is our UOT with rejection budget 0 (backbone uot, bounds
(0, 0), no cell gated, KL-relaxed marginals kept, pi = its actual row-mass share); group masses sum to the
column mass; the A-vs-B comparison table; the BOT identity (mean S_g = source cell share);
the planted rare subclone B is recovered by the enrichment rule, the Quinn cells of another clone go to
"other", the per-site table, and the cell ids line up between raw and equalised files.
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
        # a second metastasis of the family, K2 = 60 A cells: the plan covers K1 and K2 together
        Xk2, k2id = counts(rng, [prog["A"]] * 60, 60), [f"k{j}" for j in range(60)]
        adata(thin(Xk2), k2id, "M_K2", "M_K2").write_h5ad(kp / "downsampled" / "M_K2.h5ad")
        adata(Xk2, k2id, "M_K2", "M_K2").write_h5ad(kp / "prepared" / "h5ad" / "M_K2.h5ad")
        pd.DataFrame([{"source_sample": "M_T1", "target_sample": t, "source_h5ad": str(kp / "downsampled" / "M_T1.h5ad"),
                       "target_h5ad": str(kp / "downsampled" / f"{t}.h5ad")} for t in ("M_K1", "M_K2")]).to_csv(
            kp / "downsampled" / "pair_manifest_downsampled.csv", index=False)
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
        # case 2 names a metastasis with no equalised file: it must fail loudly, be recorded, and not stop the rest
        pd.DataFrame([{"index": 0, "case_id": "KPTracer_M__M_T1__family_metastases", "dataset": "KPTracer", "mouse": "M", "source_site": "M_T1",
                       "target_sites": "M_K1;M_K2", "families": "KPTracer_M_M_T1"},
                      {"index": 1, "case_id": "Quinn2021_Z__LL__metastases", "dataset": "Quinn2021", "mouse": "Z", "source_site": "LL",
                       "target_sites": "ALL", "families": "Quinn2021_Z_LG1"},
                      {"index": 2, "case_id": "KPTracer_M__M_T1__broken", "dataset": "KPTracer", "mouse": "M", "source_site": "M_T1",
                       "target_sites": "M_K9", "families": "KPTracer_M_M_T1"}]).to_csv(inputs / "cases.tsv", sep="\t", index=False)
        # Quinn level 2: LG1's B cells split 20 L1.2.1 / 20 L1.2.2 in LL, every labelled M1 cell is L1.2.1 -> not_evaluable
        pd.DataFrame([("KPTracer_M_M_T1", 1, "L1.1"), ("KPTracer_M_M_T1", 1, "L1.2"), ("Quinn2021_Z_LG1", 1, "L1.1"), ("Quinn2021_Z_LG1", 1, "L1.2"),
                      ("Quinn2021_Z_LG1", 2, "L1.2.1"), ("Quinn2021_Z_LG1", 2, "L1.2.2")],
                     columns=["name", "level", "subclone"]).to_csv(inputs / "subclones.tsv", sep="\t", index=False)
        lab = {"A": "L1.1", "B": "L1.2"}
        q_l2 = {c: ("L1.2.1" if k < 20 else "L1.2.2") for k, c in enumerate(c for c, l in zip(qs, q_src) if l == "B")}
        q_l2.update({c: "L1.2.1" for c, l in zip(qt, q_tgt) if l == "B"})
        fc = [("KPTracer_M_M_T1", c, lab[l]) for c, l in zip(sid, src_lab) if l in lab] + \
             [("KPTracer_M_M_T1", c, "L1.2") for c, l in zip(tid, tgt_lab) if l == "B"] + \
             [("KPTracer_M_M_T1", c, "L1.1") for c in k2id] + \
             [("Quinn2021_Z_LG1", c, lab[l]) for c, l in zip(qs, q_src) if l in lab] + \
             [("Quinn2021_Z_LG1", c, lab[l]) for c, l in zip(qt, q_tgt) if l in lab]
        pd.DataFrame([{"name": n, "dataset": n.split("_")[0], "mouse": n.split("_")[1], "cell_id": c, "site": "", "reliable": "True",
                       "L1": l, "L2": q_l2.get(c, "") if n.startswith("Quinn") else ""} for n, c, l in fc]).to_csv(
            inputs / "family_cells.tsv.gz", sep="\t", index=False, compression="gzip")
        codes = {}
        for i in (0, 1, 2):
            r = subprocess.run([sys.executable, str(HERE / "run_pairwise.py"), "--index", str(i), "--out", str(out), "--cases",
                                str(inputs / "cases.tsv"), "--inputs", str(inputs), "--kp-root", str(kp), "--lv-root", str(lv)],
                               capture_output=True, text=True)
            codes[i] = r.returncode
            if i < 2:
                check(f"run_pairwise index {i} exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        broken = out / "KPTracer_M__M_T1__broken"
        failed = json.loads((broken / "FAILED.json").read_text()) if (broken / "FAILED.json").exists() else {}
        check("a failing case exits non-zero, writes FAILED.json with the error and traceback, and no SUCCESS",
              codes[2] != 0 and "no site with >= 20 cells" in failed.get("error", "") and "Traceback" in failed.get("traceback", "")
              and not (broken / "SUCCESS").exists(), failed.get("error", ""))
        kp_dir = out / "KPTracer_M__M_T1__family_metastases"
        cells = pd.read_csv(kp_dir / "cells.tsv.gz", sep="\t")
        check("KP: source = the 330 equalised cells (raw-only cells left out), target 120 K1 + 60 K2",
              (cells["side"] == "source").sum() == 330 and (cells["side"] == "target").sum() == 180
              and (cells["site"] == "M_K2").sum() == 60)
        for case in (kp_dir, out / "Quinn2021_Z__LL__metastases"):
            folders = sorted(p.name for p in case.iterdir() if p.is_dir())
            check(f"{case.name}: exactly two OT runs, A_standard_OT and B_our_OT", folders == ["A_standard_OT", "B_our_OT"], str(folders))
            ids = []
            for x in ("logcpm_noscale", "ranknm256_noscale_ds_cos"):
                with np.load(case / f"coordinates_{x}.npz") as z:
                    ids.append({k: z[k] for k in ("source_ids", "target_ids")})
            check(f"{case.name}: A and B use the same source and target cells in the same order",
                  all((ids[0][k] == ids[1][k]).all() for k in ("source_ids", "target_ids")))
        for method in ("A_standard_OT", "B_our_OT"):
            d = json.loads((kp_dir / method / "diagnostics.json").read_text())
            if method == "A_standard_OT":
                check(f"{method}: logcpm_noscale on raw counts, Balanced OT, marginals exact",
                      d["preprocessing"] == "logcpm_noscale" and d["counts"] == "raw" and d["solver"] == "BOT"
                      and d["converged"] and d["row_rel_dev_max"] < 1e-6 and d["col_rel_dev_max"] < 1e-6,
                      f"row {d['row_rel_dev_max']:.1e}, col {d['col_rel_dev_max']:.1e}")
            else:
                check(f"{method}: ranknm256_noscale_ds_cos on equalised counts, backbone uot, budget 0 -> bounds (0, 0)",
                      d["preprocessing"] == "ranknm256_noscale_ds_cos" and d["counts"] == "ds" and d["backbone"] == "uot"
                      and d["source_rejection_bounds"] == [0, 0] and d["target_rejection_bounds"] == [0, 0])
                check(f"{method}: no cell rejected, KL-relaxed marginals kept (not Balanced OT)",
                      d["source_retained"] == 1 and d["target_retained"] == 1 and d["row_rel_dev_max"] > 1e-2 and abs(d["total_mass"] - 1) > 1e-2,
                      f"total mass {d['total_mass']:.3f}, row dev max {d['row_rel_dev_max']:.2f}")
            tm = pd.read_csv(kp_dir / method / "target_mass_KPTracer_M_M_T1_L1.tsv.gz", sep="\t")
            P = np.load(kp_dir / method / "coupling.npz")["coupling"]
            check(f"{method}: group masses sum to the column mass and match the saved plan",
                  np.allclose(tm.filter(like="mass__").sum(1), tm["column_mass"], rtol=1e-6) and np.allclose(P.sum(0), tm["column_mass"], rtol=1e-4))
            src = pd.read_csv(kp_dir / method / "source_groups.tsv", sep="\t").set_index("group")
            check(f"{method}: source groups 240 / 60 / 30 unlabelled / 0 other",
                  list(src.loc[["L1.1", "L1.2", "family unlabelled", "other"], "cells"]) == [240, 60, 30, 0])
        q_dir = out / "Quinn2021_Z__LL__metastases"
        qc = pd.read_csv(q_dir / "cells.tsv.gz", sep="\t")
        check("Quinn: the whole metastases side, sites read off the raw file (80 M1 + 40 M2)",
              (qc["side"] == "target").sum() == 120 and (qc.loc[qc["side"] == "target", "site"] == "M1").sum() == 80)
        qsrc = pd.read_csv(q_dir / "A_standard_OT" / "source_groups.tsv", sep="\t").query("level == 1").set_index("group")
        check("Quinn: the other clone's 50 LL cells are 'other'", qsrc.loc["other", "cells"] == 50)
        r = subprocess.run([sys.executable, str(HERE / "score_pairwise.py"), "--runs", str(out), "--out", str(tmp / "score"),
                            "--inputs", str(inputs)], capture_output=True, text=True)
        check("score_pairwise exits 0", r.returncode == 0, r.stderr[-800:] if r.returncode else "")
        s = pd.read_csv(tmp / "score" / "summary.tsv", sep="\t")
        k = s[s["case_id"].str.startswith("KP")]
        check("KP: 160 family targets scored, 20 foreign not", (k["scored"] == 160).all() and (k["target_family_cells"] == 160).all())
        ps = pd.read_csv(tmp / "score" / "per_site.tsv", sep="\t")
        check("per-site table: K1 holds the 100 L1.2 cells, K2 the 60 L1.1 cells",
              set(map(tuple, ps.loc[ps["case_id"].str.startswith("KP") & ps["method"].eq("A_standard_OT"), ["site", "true", "cells"]].values))
              == {("M_K1", "L1.2", 100), ("M_K2", "L1.1", 60)})
        cmp_ = pd.read_csv(tmp / "score" / "comparison.tsv", sep="\t", keep_default_na=False, na_values=[""])
        check("comparison.tsv: one row per case and family / level (KP L1, Quinn L1, Quinn L2), A and B side by side with B - A",
              len(cmp_) == 3 and {"balanced_accuracy_enrichment__A_standard_OT", "balanced_accuracy_enrichment__B_our_OT",
                                  "balanced_accuracy_enrichment__B_minus_A", "rare_recall_enrichment__B_minus_A",
                                  "coverage_scored", "mean_family_mass_share__B_minus_A"} <= set(cmp_.columns))
        q2 = cmp_[cmp_["case_id"].str.startswith("Quinn") & cmp_["level"].eq(2)]
        check("a one-subclone family level is kept and marked not_evaluable with its reason",
              len(q2) == 1 and q2["evaluation_status"].iloc[0] == "not_evaluable"
              and q2["not_evaluable_reason"].iloc[0] == "one true subclone among the scored cells")
        check("the two-subclone levels are evaluable", (cmp_.loc[cmp_["level"].eq(1), "evaluation_status"] == "evaluable").all())
        check("coverage: every labelled family target cell is scored here (coverage 1)", np.allclose(cmp_.loc[cmp_["level"].eq(1), "coverage_scored"], 1))
        st = pd.read_csv(tmp / "score" / "run_status.tsv", sep="\t", keep_default_na=False)
        check("run_status.tsv: 2 cases ok for both methods, the broken one failed for both with its recorded error",
              sorted(st.loc[st["status"].eq("ok"), "index"].unique()) == [0, 1] and (st["status"].eq("ok").sum() == 4)
              and set(st.loc[st["index"].eq(2), "status"]) == {"failed"} and st.loc[st["index"].eq(2), "reason"].str.contains("no site").all())
        ov = pd.read_csv(tmp / "score" / "comparison_overview.tsv", sep="\t")
        o1 = ov[ov["level"].eq(1) & ov["metric"].eq("balanced_accuracy_enrichment")]
        check("comparison_overview.tsv: level 1 counts 2 evaluable families", len(o1) == 1 and o1["evaluable_with_value"].iloc[0] == 2)
        s = s[s["evaluation_status"].eq("evaluable")]
        # The identity behind PLAN.md's amendment: under BOT the mean over target cells of S_g is the source
        # cell share exactly, so a metastasis made of one subclone has mean own enrichment 1 whatever the plan.
        P = np.load(kp_dir / "A_standard_OT" / "coupling.npz")["coupling"].astype(float)
        g = np.array(src_lab)
        S_B = P[g == "B"].sum(0) / P.sum(0)
        check("identity: under BOT the mean of S_B over all target cells is B's source cell share (60/330)",
              abs(S_B.mean() - 60 / 330) < 1e-5, f"{S_B.mean():.6f} vs {60 / 330:.6f}")
        check("planted rare subclone L1.2 (20%) recovered by the enrichment rule by both methods (> 0.8; chance 0.5)",
              (s["balanced_accuracy_enrichment"] > 0.8).all(), s[["method", "balanced_accuracy_enrichment"]].to_string(index=False))
        check("own-subclone log2 enrichment > 0 for both methods", (s["mean_own_log2E"] > 0).all(), " ".join(f"{v:.2f}" for v in s["mean_own_log2E"]))
        per = pd.read_csv(tmp / "score" / "per_subclone.tsv", sep="\t")
        b = per[(per["subclone"] == "L1.2") & per["case_id"].str.startswith("KP")].set_index("method")
        check("KP, A (BOT): pi of L1.2 = its cell share within the family (60/330)", np.isclose(b.loc["A_standard_OT", "pi"], 60 / 330))
        check("KP, B (UOT): pi of L1.2 is the plan's actual row-mass share, not the cell share",
              np.isclose(b.loc["B_our_OT", "count_share"], 60 / 330) and not np.isclose(b.loc["B_our_OT", "pi"], 60 / 330, atol=1e-3),
              f"pi {b.loc['B_our_OT', 'pi']:.4f}, cell share {60 / 330:.4f}")
    # The committed full-run inputs: every eligible family in exactly one case, exactly two methods per case.
    manifest = pd.read_csv(HERE / "inputs" / "run_manifest.tsv", sep="\t", dtype=str)
    cases = pd.read_csv(HERE / "inputs" / "cases.tsv", sep="\t", dtype=str)
    feas = pd.read_csv(HERE.parent / "within_family_subclones" / "output" / "feasibility.tsv", sep="\t", dtype=str)
    eligible = {f"{d}_{m}_{f}" for d, m, f, e in feas[["dataset", "mouse", "family_or_clone", "subclone_structure_eligible"]].values
                if str(e).lower() == "true"}
    in_cases = [f for fs in cases["families"] for f in fs.split(";")]
    check("cases.tsv: every eligible family of the systematic audit appears exactly once (none dropped)",
          sorted(in_cases) == sorted(eligible), f"{len(in_cases)} families in {len(cases)} cases, {len(eligible)} eligible")
    fc = pd.read_csv(HERE / "inputs" / "family_cells.tsv.gz", sep="\t", dtype=str, usecols=["name"])
    sb = pd.read_csv(HERE / "inputs" / "subclones.tsv", sep="\t", dtype=str)
    check("family_cells.tsv.gz and subclones.tsv cover exactly the families of cases.tsv",
          set(fc["name"]) == set(in_cases) == set(sb["name"]), f"{fc['name'].nunique()} / {sb['name'].nunique()} / {len(set(in_cases))}")
    check("run_manifest.tsv: every case has exactly A_standard_OT and B_our_OT, as methods.py defines them",
          set(manifest["index"]) == set(cases["index"])
          and all(sorted(g["method"]) == ["A_standard_OT", "B_our_OT"] for _, g in manifest.groupby("index"))
          and set(manifest.loc[manifest["method"].eq("B_our_OT"), "solver"]) == {"UOT"}
          and set(manifest.loc[manifest["method"].eq("A_standard_OT"), "solver"]) == {"BOT"},
          f"{len(manifest)} rows for {len(cases)} cases")
    print(f"\n{sum(RESULTS)} passed, {len(RESULTS) - sum(RESULTS)} failed")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
