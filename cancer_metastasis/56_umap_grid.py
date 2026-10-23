"""A grid of UMAP settings for showing the strategy results, run on TACC; outputs are small.

    python cancer_metastasis/56_umap_grid.py --cells EXPORT/cells \\
        --manifest PANCANCER_ELIGIBLE.csv --manifest PROSTATE_ELIGIBLE.csv \\
        --ds-manifest UNIFORM_DS.csv --ds-manifest PROSTATE_DS.csv --out OUT \\
        [--datasets ...] --task-index I --task-count N

Visualisation only: no OT is run and no gate result is read. The cells are 53_patient_umap.py's
(the union by cell_uid of every reference cell of a dataset's pairs, both sides).

A task is one dataset x representation x PCA size; it computes the representation once and one
UMAP per (neighbours, min_dist). Representations:
  replication  the old deck UMAP's: original counts, common.prepare_joint_representation at its
               defaults (log CPM, per-gene z-score), 2,000 HVGs
  log1p        scanpy on original counts: normalize_total 1e4, log1p, 2,000 Seurat HVGs,
               scale (max 10), PCA (arpack)
  gate         ranknm256_noscale_ds_cos on the depth-equalised counts, by the gate's own code
               (2,000 HVGs, rows L2-normalised); cells absent from the equalised h5ads are left out
UMAP: umap-learn on the PCs for every representation (Euclidean), seed 20260904.

Writes OUT/<dataset>/<representation>_pc<P>_nn<K>_md<D>/: coords.csv.gz (cell_uid, umap_1, umap_2),
thumb.png (primary purple, metastasis green), metrics.json (the primary-metastasis mixing ratio:
the share of a primary cell's 20 UMAP neighbours from the metastasis, over the share it would be
if the two sides were mixed; 1 = mixed, 0 = apart), DONE. A setting with DONE is skipped.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import prepare_joint_representation  # noqa: E402

_spec = importlib.util.spec_from_file_location("patient53", HERE / "53_patient_umap.py")
patient53 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(patient53)

SEED = 20260904
REPRESENTATIONS = ("replication", "log1p", "gate")
PCS = (30, 50)
NEIGHBOURS = (15, 30)
MIN_DIST = (0.1, 0.25, 0.5)
SIDE_COLOURS = {"primary": "#762A83", "metastasis": "#1B7837"}


def represent(kind: str, n_pcs: int, cells: pd.DataFrame, raw: dict, ds: dict) -> tuple[np.ndarray, list]:
    """PCs for every cell of the dataset, rows in the order of the returned cell_uids."""
    missing: list = []
    sources = ds if kind == "gate" else raw
    primary = patient53.load_side(cells[cells["side"] == "primary"], sources, missing)
    metastasis = patient53.load_side(cells[cells["side"] == "metastasis"], sources, missing)
    if missing and kind != "gate":
        raise KeyError(f"{len(missing)} reference cells are absent from the original h5ads")
    names = list(primary.obs_names) + list(metastasis.obs_names)
    if kind == "log1p":
        import anndata as ad
        import scanpy as sc
        from scipy import sparse

        from common import expression_matrix
        data = ad.concat([ad.AnnData(X=sparse.csr_matrix(expression_matrix(d), dtype=np.float32),
                                     var=pd.DataFrame(index=d.var_names.astype(str)),
                                     obs=pd.DataFrame(index=d.obs_names.astype(str)))
                          for d in (primary, metastasis)], join="inner", index_unique=None)
        sc.pp.normalize_total(data, target_sum=1e4)
        sc.pp.log1p(data)
        sc.pp.highly_variable_genes(data, n_top_genes=2000, flavor="seurat", subset=True)
        sc.pp.scale(data, max_value=10)
        sc.tl.pca(data, n_comps=n_pcs, svd_solver="arpack", random_state=SEED)
        return np.asarray(data.obsm["X_pca"], dtype=np.float32), names
    if kind == "gate":
        from confidenceot.preprocessing import Preprocessing
        configuration = Preprocessing.from_label(patient53.GATE_LABEL)
        source, target, _, provenance = prepare_joint_representation(
            primary, metastasis, n_hvg=2000, n_pcs=n_pcs, seed=SEED,
            representation=configuration.normalisation, rank_top_n=configuration.rank_top_n,
            minimum_detection_rate=configuration.minimum_detection_rate, cost=configuration.cost,
            equalise_depth=configuration.equalise_depth, scale_genes=configuration.scale_genes,
            label_stem=configuration.label_stem)
        if provenance.get("label") != patient53.GATE_LABEL:
            raise SystemExit(f"built {provenance.get('label')!r}")
    else:
        source, target, _, _ = prepare_joint_representation(primary, metastasis, n_hvg=2000, n_pcs=n_pcs, seed=SEED)
    return np.vstack([np.asarray(source), np.asarray(target)]).astype(np.float32), names


def mixing(xy: np.ndarray, metastasis: np.ndarray) -> float:
    from sklearn.neighbors import NearestNeighbors
    index = NearestNeighbors(n_neighbors=21).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    other = (metastasis[index] != metastasis[:, None]).mean(axis=1)
    expected = metastasis.mean()  # a primary cell's metastasis share among neighbours if mixed
    return float(other[~metastasis].mean() / expected)


def thumb(xy: np.ndarray, metastasis: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    order = np.random.default_rng(0).permutation(len(xy))
    fig, ax = plt.subplots(figsize=(4, 4))
    colours = np.where(metastasis[order], SIDE_COLOURS["metastasis"], SIDE_COLOURS["primary"])
    ax.scatter(xy[order, 0], xy[order, 1], s=float(np.clip(20000 / len(xy), 0.1, 4)), c=colours,
               linewidths=0, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--ds-manifest", type=Path, action="append", required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    args = parser.parse_args()
    import umap

    cells = patient53.patient_cells(args.cells, args.datasets)
    raw, ds = patient53.sample_sources(args.manifest), patient53.sample_sources(args.ds_manifest)
    units = list(itertools.product(sorted(cells["dataset_id"].unique()), REPRESENTATIONS, PCS))
    failures = 0
    for index, (dataset, kind, n_pcs) in enumerate(units):
        if index % args.task_count != args.task_index:
            continue
        settings = [(k, d) for k in NEIGHBOURS for d in MIN_DIST]
        folders = {s: args.out / dataset / f"{kind}_pc{n_pcs}_nn{s[0]}_md{s[1]}" for s in settings}
        if all((f / "DONE").exists() for f in folders.values()):
            print(f"skip {dataset} {kind} pc{n_pcs}: DONE", flush=True)
            continue
        try:
            part = cells[cells["dataset_id"] == dataset]
            pcs, names = represent(kind, n_pcs, part, raw, ds)
            metastasis = part.set_index("cell_uid").loc[names, "side"].to_numpy() == "metastasis"
            for (k, d), folder in folders.items():
                if (folder / "DONE").exists():
                    continue
                xy = umap.UMAP(n_neighbors=k, min_dist=d, metric="euclidean", random_state=SEED).fit_transform(pcs)
                folder.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"cell_uid": names, "umap_1": xy[:, 0], "umap_2": xy[:, 1]}).to_csv(
                    folder / "coords.csv.gz", index=False, compression="gzip", float_format="%.4f")
                score = mixing(xy, metastasis)
                (folder / "metrics.json").write_text(json.dumps(
                    {"dataset": dataset, "representation": kind, "n_pcs": n_pcs, "n_neighbors": k,
                     "min_dist": d, "cells": len(names), "metastasis_cells": int(metastasis.sum()),
                     "mixing_ratio": score}, indent=2), encoding="utf-8")
                thumb(xy, metastasis, f"{dataset} {kind} pc{n_pcs} nn{k} md{d}  mixing {score:.2f}",
                      folder / "thumb.png")
                (folder / "DONE").write_text("done\n", encoding="utf-8")
                print(f"{dataset} {kind} pc{n_pcs} nn{k} md{d}: mixing {score:.2f}", flush=True)
        except Exception as error:  # noqa: BLE001 - a failed unit is recorded and reported, never hidden
            failures += 1
            failed = args.out / dataset / f"FAILED_{kind}_pc{n_pcs}.txt"
            failed.parent.mkdir(parents=True, exist_ok=True)
            failed.write_text(f"{type(error).__name__}: {error}\n{traceback.format_exc()}", encoding="utf-8")
            print(f"FAILED {dataset} {kind} pc{n_pcs}: {type(error).__name__}: {error}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
