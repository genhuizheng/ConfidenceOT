"""The frozen calibration v3 preprocessing grid, shared by the TACC runs of Splatter N = 5,000, LTL331 (PDX) and KP-Tracer
(CRISPR lineage). Everything that decides a gate is taken from the local benchmark that froze it
(realdata_benchmark/splatter_local/run_local_benchmark.py, FROZEN_CALIBRATION_v3.json, CALIBRATION_V3.md), so the TACC
fits are the local fits at a larger N:

  grid          factorial_common.grid() with DS none only (downsampling removed, user 2026-11-04): 37 encoding keys
                (logcpm; ranknm / rank_value K 128-2048 x production / expected / midrank ties; rankhvg H 2000 / 3000 x
                the three ties) x PCA {30, 50} x gene scaling {no, yes} x cost {cos, sqeuc} = 296 configurations. The
                baseline ranknm256-production|none|noscale|cos is the grid's ranknm256-production|none|noscale|cos|pca30:
                locally the two gave identical gates on all 160 rows (max |rejected difference| 0).
  representation  the encoding of the joint counts (factorial_common.encode_sides) -> rb_common.pca_coordinates (2,000
                top-variance encoded genes, every H gene for rankhvg; scaling if asked; PCA with seed BASE + pair index)
                -> unit rows for cos.
  correction    within_side_split+mnncovlocal (run_local_benchmark.correct, imported, not copied): the credible-coverage
                gate, and when it opens the local kernel shift of the target.
  fit           02_run_pair.py in this process on the corrected coordinates, label coord_allgenes_nopca_noscale[_cos]
                (no HVG selection, no PCA, no scaling: the coordinates enter unchanged; the cost is divided by its median
                pair cost and the within-side split null calibrates it), with the dataset's own fit arguments.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "realdata_benchmark"), str(REPO / "realdata_benchmark" / "splatter_local"), str(REPO / "src"),
                str(REPO / "cancer_metastasis"), str(REPO)]

import factorial_common as fc  # noqa: E402
import run_local_benchmark as lb  # noqa: E402

ARM = "within_side_split+mnncovlocal"
BASELINE = "ranknm256-production|none|noscale|cos|pca30"
PCS = (30, 50)
_RUN_PAIR = None


def grid_keys() -> list[dict]:
    """The 37 encoding keys of the no-DS grid, in factorial_common.grid() order (one spec per key)."""
    keys = {}
    for s in fc.grid():
        if s["ds"] == "none":
            keys.setdefault((s["encoding"], s["K"], s["H"], s["tie"]), s)
    return list(keys.values())


def key_name(spec: dict) -> str:
    """The configuration id without scale / cost / PCA, e.g. ranknm256-production|none."""
    return fc.config_id({**spec, "scale": False, "cost": "cos"}).rsplit("|", 3)[0]


def config_ids(spec: dict) -> list[str]:
    return [fc.config_id({**spec, "scale": scale, "cost": cost}).replace(f"|pca{fc.N_PCS}", f"|pca{n}")
            for n in PCS for scale in (False, True) for cost in ("cos", "sqeuc")]


def representations(spec: dict, source, target, seed: int):
    """(config_id, X, Y, cos, diagnostics) for the 8 configurations of one key, the encoding and each PCA computed once.
    source / target: count AnnData (DS none)."""
    import scipy.sparse as sp
    import rb_common as rb
    from confidenceot.preprocessing import unit_rows
    se, te, _, _ = fc.encode_sides(spec, source, target)
    joint = sp.vstack([sp.csr_matrix(se.X), sp.csr_matrix(te.X)], format="csr"); n = se.n_obs
    n_hvg = int(spec["H"]) if spec["encoding"] == "rankhvg" else fc.N_HVG
    for n_pcs in PCS:
        for scale in (False, True):
            P, _ = rb.pca_coordinates(joint, n_hvg, n_pcs, scale, False, seed)
            for cost in ("cos", "sqeuc"):
                C = unit_rows(P) if cost == "cos" else P
                lb._LAST.clear()
                X, Y = lb.correct(C[:n], C[n:], ARM.split("+")[1])
                diag = {f"diag_{k}": v for k, v in lb._LAST.items() if not k.startswith("_")}
                cid = fc.config_id({**spec, "scale": scale, "cost": cost}).replace(f"|pca{fc.N_PCS}", f"|pca{n_pcs}")
                yield cid, X, Y, cost == "cos", diag


def run_pair(argv: list[str]) -> tuple[int, str]:
    """02_run_pair.py main() in this process (loaded once per worker)."""
    global _RUN_PAIR
    if _RUN_PAIR is None:
        spec = importlib.util.spec_from_file_location("run_pair_02", REPO / "cancer_metastasis" / "02_run_pair.py")
        _RUN_PAIR = importlib.util.module_from_spec(spec); spec.loader.exec_module(_RUN_PAIR)
    old, buf, code = sys.argv, io.StringIO(), 0
    sys.argv = ["02_run_pair.py", *argv]
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            _RUN_PAIR.main()
    except SystemExit as e:
        code = int(e.code or 0) if not isinstance(e.code, str) else 1
    except Exception:
        code = 1; buf.write(traceback.format_exc())
    finally:
        sys.argv = old
    return code, buf.getvalue()


def fit(X: np.ndarray, Y: np.ndarray, cos: bool, obs: dict, manifest: pd.DataFrame, index: int, fit_args: list[str],
        work: Path) -> tuple[int, str, Path | None]:
    """Write the coordinates as the manifest row's two files and fit. Returns (exit code, log, run directory)."""
    import anndata as ad
    shutil.rmtree(work, ignore_errors=True); (work / "input").mkdir(parents=True)
    cols = pd.Index([f"PC{k + 1}" for k in range(X.shape[1])], dtype=str)
    files = {}
    for coords, side in ((X, "source"), (Y, "target")):
        files[side] = work / "input" / f"{side}.h5ad"
        ad.AnnData(X=np.asarray(coords, np.float32), obs=obs[side], var=pd.DataFrame(index=cols)).write_h5ad(files[side])
    m = manifest.copy()
    for side in ("source", "target"):
        m.loc[index, f"{side}_h5ad"] = str(files[side])
        if f"{side}_h5ads_json" in m:
            m.loc[index, f"{side}_h5ads_json"] = json.dumps([str(files[side])])
    m.to_csv(work / "manifest.csv", index=False)
    code, log = run_pair([str(work / "manifest.csv"), str(work / "runs"), "--index", str(index), "--preprocessing",
                          "coord_allgenes_nopca_noscale" + ("_cos" if cos else ""), *fit_args, "--allow-precomputed"])
    for f in files.values():
        f.unlink(missing_ok=True)
    run = next((work / "runs").rglob("cell_confidence.csv"), None) if code == 0 else None
    return code, log, (run.parent if run is not None else None)


def frozen_record() -> dict:
    """What a TACC root records as its calibration (FROZEN_CALIBRATION_V3.json there; summarize_local --frozen reads
    its config as the baseline)."""
    return {"arm": ARM, "config": BASELINE, "frozen": "2026-11-04", "source": "realdata_benchmark/splatter_local/CALIBRATION_V3.md",
            "readout": "effective: when one side retains no cell, the other side's retained cells count as unmatched",
            "downsampling": "none", "rejection_cap": "none", "acceptance": 0.99}
