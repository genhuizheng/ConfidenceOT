from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "cancer_metastasis"))
SPEC = importlib.util.spec_from_file_location(
    "cancer_malignant_manifest",
    ROOT / "cancer_metastasis" / "07_build_malignant_scope_manifest.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CancerMalignantManifestTest(unittest.TestCase):
    def test_exact_annotation_counts_select_evaluable_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "source.h5ad"
            target_path = root / "target.h5ad"
            var = pd.DataFrame(index=["G1", "G2"])
            source = ad.AnnData(
                X=np.ones((4, 2)),
                obs=pd.DataFrame({
                    "sample_id": ["primary"] * 4,
                    "cell_type": ["Tumor", "Tumor", "Tumor", "T cell"],
                }, index=[f"s{i}" for i in range(4)]),
                var=var.copy(),
            )
            target = ad.AnnData(
                X=np.ones((3, 2)),
                obs=pd.DataFrame({
                    "sample_id": ["metastasis"] * 3,
                    "cell_type": ["Tumor", "Tumor", "T cell"],
                }, index=[f"t{i}" for i in range(3)]),
                var=var.copy(),
            )
            source.write_h5ad(source_path)
            target.write_h5ad(target_path)
            manifest = root / "manifest.csv"
            pd.DataFrame([{
                "pair_id": "pair_a",
                "dataset_id": "dataset_a",
                "patient_id": "patient_a",
                "source_sample": "primary",
                "target_sample": "metastasis",
                "source_h5ad": str(source_path),
                "target_h5ad": str(target_path),
                "source_h5ads_json": json.dumps([str(source_path)]),
                "target_h5ads_json": json.dumps([str(target_path)]),
            }]).to_csv(manifest, index=False)

            eligible = MODULE.build_scope_manifest(
                manifest,
                root / "output",
                dataset_id="dataset_a",
                annotations=["Tumor"],
                minimum_cells=2,
            )
            self.assertEqual(len(eligible), 1)
            self.assertEqual(int(eligible.loc[0, "source_malignant_n"]), 3)
            self.assertEqual(int(eligible.loc[0, "target_malignant_n"]), 2)


if __name__ == "__main__":
    unittest.main()
