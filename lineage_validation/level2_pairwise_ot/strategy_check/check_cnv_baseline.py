"""Value range and neutral baseline of the exported CNV profiles (cnv.f32), and whether the "- 1.0" in
02_run_pair.py:cnv_coordinates changes the CNV PCA coordinates or the cost. Read-only; recomputes no CNV.

    python lineage_validation/level2_pairwise_ot/strategy_check/check_cnv_baseline.py --synthetic --out DIR
    python lineage_validation/level2_pairwise_ot/strategy_check/check_cnv_baseline.py --root CNV_ROOT [--root ...] --out DIR

The export (cancer_metastasis/49_infercnvpy_unit.py:224) writes infercnvpy's obsm["X_cnv"] + 1 as float32, cells x
windows; cnv_coordinates subtracts 1 again and runs a joint PCA (30 components, seed). PCA centres each column,
so a constant shift cannot change the coordinates; this script measures it.

Per unit (<root>/<dataset>/<patient>/): shape, value quantiles, share of entries exactly 1 (copy-neutral: the
sparse zeros of X_cnv), share within 1e-3 of 1, of 0, row-mean range (on up to --max-cells rows, fixed seed).
Invariance, on up to --max-cells cells split in two halves as source / target:
  coordinates with the shipped code (cnv_coordinates: values - 1), with values as stored (no shift) and with
  values - 1 + 7 (any constant); max |difference| of coordinates and of cost = squared Euclidean /
  median_pair_scale (the 02_run_pair.py rule and seed), relative to the coordinate / cost scale.
Writes DIR/cnv_values.tsv and DIR/cnv_shift_invariance.tsv.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "cancer_metastasis")]
from confidenceot.preprocessing import median_pair_scale, squared_euclidean  # noqa: E402

SEED, N_PCS = 20260831, 30


def load_run_pair():
    spec = importlib.util.spec_from_file_location("run_pair_02", REPO / "cancer_metastasis" / "02_run_pair.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pca_coords(values: np.ndarray, n_source: int, shift: float):
    """The PCA step of cnv_coordinates with an explicit shift (shift = -1.0 is the shipped code)."""
    from sklearn.decomposition import PCA
    joint = np.asarray(values, dtype=np.float64) + shift
    k = min(N_PCS, joint.shape[0] - 1, joint.shape[1])
    coords = PCA(n_components=k, random_state=SEED).fit_transform(joint).astype(np.float32)
    return coords[:n_source], coords[n_source:]


def cost(xs, xt):
    return squared_euclidean(xs, xt) / median_pair_scale(xs, xt, rng=np.random.default_rng(SEED))


def check_unit(unit: Path, rp, max_cells: int, rng) -> tuple[dict, list]:
    cells = pd.read_csv(unit / "cells.tsv", sep="\t", dtype=str)
    genes = pd.read_csv(unit / "genes.tsv", sep="\t", dtype=str)
    values = np.memmap(unit / "cnv.f32", dtype=np.float32, mode="r", shape=(len(cells), len(genes)))
    pick = np.sort(rng.choice(len(cells), min(max_cells, len(cells)), replace=False))
    sample = np.asarray(values[pick], dtype=np.float64)
    q = np.quantile(sample, [0, 0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999, 1])
    row_mean = sample.mean(1)
    stats = {"unit": str(unit), "cells": len(cells), "windows": len(genes), "cells_sampled": len(pick),
             **{f"q{p}": v for p, v in zip(("min", "0.001", "0.01", "0.25", "0.5", "0.75", "0.99", "0.999", "max"), q)},
             "share_exactly_1": float((sample == 1.0).mean()), "share_within_1e-3_of_1": float((np.abs(sample - 1) < 1e-3).mean()),
             "share_within_1e-3_of_0": float((np.abs(sample) < 1e-3).mean()), "mean": float(sample.mean()),
             "row_mean_min": float(row_mean.min()), "row_mean_max": float(row_mean.max())}
    # invariance: the shipped function vs explicit shifts, on the same cells
    ids = cells["barcode"].to_numpy()[pick]
    half = len(ids) // 2
    s_ship, t_ship, _ = rp.cnv_coordinates(unit.parent.parent, unit.parent.name, unit.name, ids[:half], ids[half:],
                                           n_pcs=N_PCS, seed=SEED)
    rows = []
    ref_c = cost(s_ship, t_ship)
    scale_x = float(np.abs(np.vstack([s_ship, t_ship])).max())
    for name, shift in (("no shift (values as stored, 1 = neutral)", 0.0), ("values - 1 (same as shipped, recomputed)", -1.0),
                        ("values - 1 + 7 (an arbitrary constant)", 6.0)):
        xs, xt = pca_coords(sample, half, shift)
        c = cost(xs, xt)
        rows.append({"unit": str(unit), "variant": name, "cells": len(ids),
                     "max_abs_coord_diff": float(max(np.abs(xs - s_ship).max(), np.abs(xt - t_ship).max())),
                     "max_abs_coord_diff_rel": float(max(np.abs(xs - s_ship).max(), np.abs(xt - t_ship).max()) / scale_x),
                     "max_abs_cost_diff": float(np.abs(c - ref_c).max()), "cost_median": float(np.median(ref_c))})
    return stats, rows


def synthetic_root(tmp: Path) -> Path:
    """A unit shaped like the export: sparse X_cnv (mostly 0, a few gained / lost windows) + 1 as float32."""
    rng = np.random.default_rng(0)
    n, w = 1200, 900
    x = np.zeros((n, w))
    clone = rng.integers(0, 3, n)
    for k, (lo, hi, amp) in enumerate(((100, 260, 0.25), (400, 520, -0.2), (700, 760, 0.3))):
        x[clone == k, lo:hi] = amp
    noise = rng.normal(0, 0.05, (n, w)) * (rng.random((n, w)) < 0.15)
    x = np.where(np.abs(x + noise) < 0.03, 0.0, x + noise)  # infercnvpy's sparse zeros
    unit = tmp / "SYN" / "P1"
    unit.mkdir(parents=True)
    (x + 1.0).astype(np.float32).tofile(unit / "cnv.f32")
    pd.DataFrame({"barcode": [f"c{i}" for i in range(n)]}).to_csv(unit / "cells.tsv", sep="\t", index=False)
    pd.DataFrame({"gene": [f"w{j}" for j in range(w)], "chr": "chr1", "start": range(w), "stop": range(w)}).to_csv(
        unit / "genes.tsv", sep="\t", index=False)
    return tmp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append", default=[])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-cells", type=int, default=4000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rp = load_run_pair()
    rng = np.random.default_rng(SEED)
    stats, inv = [], []
    with tempfile.TemporaryDirectory() as tmp:
        roots = args.root + ([synthetic_root(Path(tmp))] if args.synthetic else [])
        for root in roots:
            for unit in sorted(p for p in root.glob("*/*") if (p / "cnv.f32").exists()):
                s, r = check_unit(unit, rp, args.max_cells, rng)
                stats.append(s)
                inv += r
                print(f"{unit}: {s['cells']} cells x {s['windows']} windows; median {s['q0.5']:.4f}, share exactly 1 "
                      f"{s['share_exactly_1']:.3f}; max coord diff {max(x['max_abs_coord_diff'] for x in r):.2e}, "
                      f"max cost diff {max(x['max_abs_cost_diff'] for x in r):.2e}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats).to_csv(args.out / "cnv_values.tsv", sep="\t", index=False)
    pd.DataFrame(inv).to_csv(args.out / "cnv_shift_invariance.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
