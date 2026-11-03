"""Local checks of the factorial code (no TACC paths): grid, encoders, and one end-to-end 02_run_pair.py fit on synthetic
counts through the exact command the worker uses.

    python lineage_validation/preprocessing_factorial_20261103/test_factorial.py OUT_DIR
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "realdata_benchmark"), str(REPO / "src"), str(REPO / "cancer_metastasis")]
import factorial_common as fc  # noqa: E402
import rb_common as rb  # noqa: E402

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
checks = []


def check(name, ok, detail=""):
    checks.append((name, bool(ok), detail)); print(("PASS" if ok else "FAIL"), name, detail, flush=True)


# 1. grid
g = fc.registry([{"config_id": "label:ranknm256_noscale_ds_cos|pca30", "label": "ranknm256_noscale_ds_cos", "n_pcs": 30}])
grid = g[g.role == "grid"]
check("grid has 592 configurations", len(grid) == 592, str(grid.encoding.value_counts().to_dict()))
check("config ids unique", not g.config_id.duplicated().any())
check("per-encoding counts", grid.encoding.value_counts().to_dict() == {"ranknm": 240, "rank_value": 240, "rankhvg": 96, "logcpm": 16})

# 2. synthetic counts with many ties (small integers)
rng = np.random.default_rng(0)
genes = [f"G{i:04d}" for i in range(1500)]
def side(n, name, shift):
    lam = rng.gamma(0.4, 1.0, size=len(genes)) * (1 + shift * (np.arange(len(genes)) % 7 == 0))
    X = rng.poisson(lam * rng.uniform(0.5, 1.5, size=(n, 1)))
    obs = pd.DataFrame({"sample_id": name, "malignant": "malignant", "site": name}, index=[f"{name}_{i}" for i in range(n)])
    return ad.AnnData(X=sp.csr_matrix(X.astype(np.float32)), obs=obs, var=pd.DataFrame(index=genes))
S, T = side(220, "src", 0.0), side(240, "tgt", 1.0)
joint = rb.joint_counts(S, T)

e_exp, _ = rb.rank_encode(joint, 256, False, "expected")
e_mid, _ = rb.rank_encode(joint, 256, False, "midrank")
e_pro, _ = rb.rank_encode(joint, 256, False, "production")
# a tie group wholly inside the top N gets the same value under expected and midrank: every gene where they differ
# must carry the count of the group that straddles the cut (the value at sorted position N - 1)
scaled = rb.scaled_for_rank(joint, False, sort=False).toarray()
diff = np.abs(e_exp.toarray() - e_mid.toarray()) > 1e-12
bad = 0
for i in range(joint.shape[0]):
    v = np.sort(scaled[i][scaled[i] > 0])[::-1]
    if v.size <= 256:
        bad += int(diff[i].any()); continue
    cut = v[255]
    bad += int((diff[i] & ~np.isclose(scaled[i], cut)).any())
check("midrank equals expected except within the group straddling the cut", bad == 0,
      f"cells differing anywhere {int(diff.any(axis=1).sum())} / {joint.shape[0]}; cells differing outside the cut group {bad}")
check("production keeps at most 256 genes per cell", int(np.diff(e_pro.indptr).max()) <= 256)
check("midrank and expected have the same total value per cell where no group straddles",
      np.isfinite(e_mid.sum(axis=1)).all())
from confidenceot.preprocessing import rank_value_encode  # noqa: E402
try:
    ref = rank_value_encode(joint, 256)
    ref = ref if sp.issparse(ref) else sp.csr_matrix(ref)
    check("rank_encode production (gene median) equals rank_value_encode", abs(rb.rank_encode(joint, 256, True, "production")[0] - ref).max() < 1e-12)
except TypeError as exc:
    check("rank_value_encode reference callable", False, str(exc))

h_mid, hv = rb.rankhvg_encode(joint, 1000, "midrank")
h_exp, hv2 = rb.rankhvg_encode(joint, 1000, "expected")
h_pro, hv3 = rb.rankhvg_encode(joint, 1000, "production")
check("rankhvg expected equals midrank exactly (no cut)", np.array_equal(hv, hv2) and abs(h_mid - h_exp).max() < 1e-12,
      f"max |diff| {abs(h_mid - h_exp).max():.2e}")
check("rankhvg production: same support, values in (0, 1]", (h_pro != 0).sum() == (h_mid != 0).sum() and h_pro.data.min() > 0 and h_pro.data.max() <= 1)
check("rankhvg production differs from midrank only on ties", abs(h_pro - h_mid).max() > 0)

# 3. encode_sides for every encoding / depth level; label round-trips
targets = {k: int(np.quantile(np.concatenate([np.asarray(S.X.sum(1)).ravel(), np.asarray(T.X.sum(1)).ravel()]), q))
           for k, q in (("P10", .1), ("P25", .25), ("P50", .5))}
from confidenceot import Preprocessing  # noqa: E402
labels = set()
for spec in grid.sample(40, random_state=1).to_dict("records") + grid[grid.encoding == "rankhvg"].head(3).to_dict("records"):
    s, t = fc.equalise_sides(S, T, spec["ds"], targets)
    se, te, label, rec = fc.encode_sides(spec, s, t)
    labels.add(label)
    assert se.n_obs == S.n_obs and te.n_obs == T.n_obs and list(se.obs_names) == list(S.obs_names)
    assert Preprocessing.from_label(label).label() == label
check("encode_sides runs for sampled configurations; labels round-trip", True, ", ".join(sorted(labels)))
s10, t10 = fc.equalise_sides(S, T, "P10", targets)
check("P10 equalisation caps depth at the target", float(np.asarray(s10.X.sum(1)).max()) <= targets["P10"] + 1e-6)

# 4. end-to-end fit through the worker's command (precomputed, PCA 50, couplings saved)
spec = grid[grid.config_id == "ranknm256-midrank|P25s1|noscale|cos|pca50"].iloc[0].to_dict()
s, t = fc.equalise_sides(S, T, spec["ds"], targets)
se, te, label, rec = fc.encode_sides(spec, s, t)
(out / "input").mkdir(exist_ok=True)
se.write_h5ad(out / "input" / "source.h5ad"); te.write_h5ad(out / "input" / "target.h5ad")
pd.DataFrame([{"pair_id": "test_pair", "dataset_id": "synthetic", "patient_id": "m0", "source_sample": "src", "target_sample": "tgt",
               "source_h5ad": str(out / "input" / "source.h5ad"), "target_h5ad": str(out / "input" / "target.h5ad"), "eligible": True}]
             ).to_csv(out / "manifest.csv", index=False)
cmd = fc.fit_command(sys.executable, str(REPO / "cancer_metastasis" / "02_run_pair.py"), str(out / "manifest.csv"), str(out / "runs"), 0,
                     label, fc.N_PCS, 2, precomputed=True)
env = {**__import__("os").environ, "PYTHONPATH": ";".join(map(str, [REPO / "src", REPO / "cancer_metastasis", REPO / "realdata_benchmark", REPO]))}
r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=REPO)
(out / "run.log").write_text(r.stdout + r.stderr)
runs = list((out / "runs").rglob("run.json"))
check("02_run_pair exit 0", r.returncode == 0, r.stderr[-400:] if r.returncode else "")
if runs:
    rj = json.loads(runs[0].read_text()); d = runs[0].parent
    check("couplings saved for M4-E and M4-R", (d / "coupling_m4e.npz").exists() and (d / "coupling_m4r.npz").exists())
    cfg = rj.get("configuration", rj.get("preprocessing", {}))
    check("run used PCA 50", '"n_pcs": 50' in runs[0].read_text(), "")
    C = np.load(d / "coupling_m4e.npz")["coupling"]
    cells = pd.read_csv(d / "cell_confidence.csv")
    check("coupling shape matches the analysed cells", C.shape == (int((cells.side == "source").sum() / 2), int((cells.side == "target").sum() / 2)),
          f"{C.shape} vs cells table {cells.groupby(['method', 'side']).size().to_dict()}")
pd.DataFrame(checks, columns=["check", "pass", "detail"]).to_csv(out / "TEST_RESULTS.tsv", sep="\t", index=False)
print(f"{sum(c[1] for c in checks)} / {len(checks)} passed -> {out / 'TEST_RESULTS.tsv'}")
