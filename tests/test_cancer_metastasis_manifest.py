from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

ad = pytest.importorskip("anndata")


def load_common():
    path = Path(__file__).parents[1] / "cancer_metastasis" / "common.py"
    spec = importlib.util.spec_from_file_location("cancer_common", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_file(path, *, patient, samples, sample_ids, source_ids=(), target_ids=(),
               file_kind="site_class", pairing_verified=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = ad.AnnData(
        X=np.ones((len(sample_ids), 3)),
        obs=pd.DataFrame({"sample_id": sample_ids}, index=[f"cell_{i}" for i in range(len(sample_ids))]),
    )
    data.uns["pairing_metadata"] = {
        "patient_id": patient, "samples": list(samples), "pairing_verified": pairing_verified,
        "pair_ids_as_source": list(source_ids), "pair_ids_as_target": list(target_ids),
        "n_cells_per_sample": {sample: sample_ids.count(sample) for sample in samples},
        "file_kind": file_kind,
    }
    data.write_h5ad(path)


def test_manifest_resolves_exact_samples_in_shared_site_files(tmp_path):
    common = load_common()
    pair_id = "P1__primary_A__met_B"
    root = tmp_path / "converted" / "GSETEST" / "P1"
    write_file(root / "primary.h5ad", patient="P1", samples=["primary_A", "primary_C"],
               sample_ids=["primary_A"] * 3 + ["primary_C"] * 2, source_ids=[pair_id])
    write_file(root / "metastasis.h5ad", patient="P1", samples=["met_B"],
               sample_ids=["met_B"] * 4, target_ids=[pair_id])
    manifest = common.build_pair_manifest(tmp_path / "converted", minimum_cells=2)
    row = manifest.iloc[0]
    assert bool(row.eligible)
    assert row.source_sample == "primary_A"
    assert row.target_sample == "met_B"
    assert row.source_n == 3
    assert row.target_n == 4


def test_manifest_combines_library_files_and_does_not_gate_on_provenance_flag(tmp_path):
    common = load_common()
    pair_id = "P2__primary__met"
    root = tmp_path / "converted" / "GSELIB" / "P2"
    for library in ("immune", "nonimmune"):
        write_file(root / f"primary_{library}.h5ad", patient="P2", samples=["primary"],
                   sample_ids=["primary"] * 3, source_ids=[pair_id], file_kind="library",
                   pairing_verified=False)
        write_file(root / f"met_{library}.h5ad", patient="P2", samples=["met"],
                   sample_ids=["met"] * 4, target_ids=[pair_id], file_kind="library",
                   pairing_verified=False)
    row = common.build_pair_manifest(tmp_path / "converted", minimum_cells=2).iloc[0]
    assert bool(row.eligible)
    assert row.source_file_n == 2
    assert row.target_file_n == 2
    assert row.source_n == 6
    assert row.target_n == 8
