"""Shared settings and plumbing for the all-stage MOSTA run (scripts 20-25).

Everything that has to be identical across the scripts lives here once: the
fixed analysis settings, the result layout, the manifest readers, the claim
files that let several jobs share one pair list, and the atomic publication of
a finished pair.  No function here decides whether a pair or bin is analysed;
the only exclusion in the run is the predefined Cavity annotation, applied
when the slices are equalised (21_).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import socket
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

STAGES = ("E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5")

# Fixed analysis settings.  The label is written out in full: the shorthand
# rank256_nm_ds_cos parses as a 'precomputed' pass-through, and
# ranknm256_ds_cos is gene-scaled (checked locally, 2026-10-08).
PREPROCESSING_LABEL = "ranknm256_noscale_ds_cos"
EXCLUDED_ANNOTATIONS = ("Cavity",)
# cancer_metastasis/27_downsample_counts.py: one shared target, the 10th
# percentile of every analysed bin's depth pooled over all slices, and its
# default seed.
EQUALISATION_QUANTILE = 0.10
EQUALISATION_SEED = 20260914
# cancer_metastasis/02_run_pair.py defaults: the seed rule, the solver and the
# calibration protocol (2,000 label-blind bins, 5 + 5 within-side replicates,
# grid 5, acceptance 0.99 as in the cycling x CNV run).
SEED = 20260831
EPSILON = 0.1
LAMBDA = 1.0
TOLERANCE = 1e-4
CALIBRATION_MAX_BINS = 2000
NULL_CALIBRATION_REPLICATES = 5
NULL_VALIDATION_REPLICATES = 5
CALIBRATION_GRID_SIZE = 5
WITHIN_SIDE_ACCEPTANCE = 0.99
# 02_run_pair.py's default shared budget.  Not enforced, so the rejection
# interval resolves to (0, 1); recorded because the bounds are derived from it.
REJECTION_BUDGET = 0.95


def check_preprocessing(configuration: Any) -> None:
    """Stop unless the configuration is exactly ranknm256_noscale_ds_cos."""
    expected = {
        "label": PREPROCESSING_LABEL,
        "normalisation": "rank_no_median",
        "rank_top_n": 256,
        "scale_genes": False,
        "equalise_depth": True,
        "cost": "cosine",
        "scale": "median_sampled_pair",
        "n_hvg": 2000,
        "n_pcs": 30,
        "minimum_detection_rate": 0.0,
        "regress_out": [],
    }
    found = configuration.as_dict()
    wrong = {key: (found.get(key), value) for key, value in expected.items() if found.get(key) != value}
    if wrong:
        raise SystemExit(f"preprocessing is not {PREPROCESSING_LABEL}: {wrong}")


class Layout:
    """Where every artefact of the run lives, under one root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def manifest(self) -> Path:
        return self.root / "manifest"

    @property
    def slices_csv(self) -> Path:
        return self.manifest / "slices.csv"

    @property
    def pairs_csv(self) -> Path:
        return self.manifest / "pairs.csv"

    @property
    def equalised(self) -> Path:
        return self.root / "equalised"

    @property
    def equalisation(self) -> Path:
        return self.root / "equalisation"

    @property
    def slices(self) -> Path:
        return self.root / "slices"

    @property
    def representations(self) -> Path:
        return self.root / "representations"

    @property
    def pairs(self) -> Path:
        return self.root / "pairs"

    @property
    def pseudobulk(self) -> Path:
        return self.root / "pseudobulk"

    @property
    def equivalence(self) -> Path:
        return self.root / "equivalence"

    @property
    def audit(self) -> Path:
        return self.root / "audit"

    @property
    def package(self) -> Path:
        return self.root / "package"

    @property
    def claims(self) -> Path:
        return self.root / "claims"

    @property
    def tmp(self) -> Path:
        return self.root / "tmp"

    def equalised_h5ad(self, sample: str) -> Path:
        return self.equalised / f"{sample}.h5ad"

    def slice_bins(self, sample: str) -> Path:
        return self.slices / f"{sample}_bins.csv.gz"

    def slice_genes(self, sample: str) -> Path:
        return self.slices / f"{sample}_genes.txt.gz"

    def representation(self, pair_id: str) -> Path:
        return self.representations / pair_id

    def pair(self, pair_id: str) -> Path:
        return self.pairs / pair_id


def read_pairs(layout: Layout) -> pd.DataFrame:
    pairs = pd.read_csv(layout.pairs_csv)
    if pairs["pair_id"].duplicated().any() or pairs["index"].duplicated().any():
        raise SystemExit(f"{layout.pairs_csv} repeats a pair_id or an index")
    return pairs


def read_slices(layout: Layout) -> pd.DataFrame:
    return pd.read_csv(layout.slices_csv)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return {key: json_ready(getattr(value, key)) for key in value.__dataclass_fields__}
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def claim(directory: Path, key: str) -> bool:
    """Take ``key`` for this process; False if another process already has it.

    ``mkdir`` is atomic on the shared file system, so two jobs walking the same
    pair list never start the same pair.  A claim left by a job that died is
    cleared by the submit helper before the next submission, never here.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / key
    try:
        path.mkdir()
    except FileExistsError:
        return False
    (path / "owner.txt").write_text(
        f"job={os.environ.get('SLURM_JOB_ID', 'local')} host={socket.gethostname()} "
        f"pid={os.getpid()} started={time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
        encoding="utf-8",
    )
    return True


def staging_directory(layout: Layout, pair_id: str, stage: str) -> Path:
    tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    path = layout.tmp / stage / f"{pair_id}.{tag}"
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def publish(staging: Path, final: Path) -> bool:
    """Move a finished staging directory into place in one rename.

    False, and the staging copy removed, if the destination already exists:
    another worker finished the same pair first, and its copy stands.
    """
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        shutil.rmtree(staging, ignore_errors=True)
        return False
    try:
        os.rename(staging, final)
    except OSError:
        if final.exists():
            shutil.rmtree(staging, ignore_errors=True)
            return False
        raise
    return True


def file_inventory(directory: Path) -> list[dict[str, Any]]:
    return [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted(directory.iterdir()) if path.is_file()
    ]


def read_equalised_slice(path: Path):
    """An equalised slice: counts (CSR int32), genes, bin ids, labels, xy."""
    import anndata as ad
    from scipy import sparse

    data = ad.read_h5ad(path)
    counts = sparse.csr_matrix(data.X)
    return {
        "counts": counts,
        "genes": np.asarray(data.var_names.astype(str)),
        "bin_ids": np.asarray(data.obs_names.astype(str)),
        "annotation": np.asarray(data.obs["annotation"].astype(str)),
        "x": data.obs["x"].to_numpy(dtype=np.float64),
        "y": data.obs["y"].to_numpy(dtype=np.float64),
        "predownsample_total_counts": data.obs["predownsample_total_counts"].to_numpy(),
    }


def ordered_categories(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values})
