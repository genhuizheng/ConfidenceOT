from __future__ import annotations

import unittest

import anndata as ad
import numpy as np
import pandas as pd

from cancer_metastasis.common import gene_keys, prepare_joint_representation


class CancerGeneKeyTest(unittest.TestCase):
    def test_symbol_only_na_gene_ids_fall_back_to_symbols(self):
        data = ad.AnnData(
            X=np.array([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0]]),
            var=pd.DataFrame(
                {
                    "gene_symbol": ["TP53", "EPCAM", "VIM"],
                    "gene_id": ["NA", "NA", "NA"],
                    "gene_id_source": ["symbol", "symbol", "symbol"],
                },
                index=["TP53", "EPCAM", "VIM"],
            ),
        )
        np.testing.assert_array_equal(gene_keys(data), ["TP53", "EPCAM", "VIM"])

    def test_real_gene_ids_remain_preferred(self):
        data = ad.AnnData(
            X=np.ones((2, 2)),
            var=pd.DataFrame(
                {
                    "gene_symbol": ["TP53", "EPCAM"],
                    "gene_id": ["ENSG00000141510", "ENSG00000119888"],
                },
                index=["TP53", "EPCAM"],
            ),
        )
        np.testing.assert_array_equal(
            gene_keys(data), ["ENSG00000141510", "ENSG00000119888"]
        )

    def test_symbol_only_pair_builds_joint_representation(self):
        var = pd.DataFrame(
            {
                "gene_symbol": ["TP53", "EPCAM", "VIM"],
                "gene_id": ["NA", "NA", "NA"],
            },
            index=["TP53", "EPCAM", "VIM"],
        )
        source = ad.AnnData(
            X=np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]]), var=var.copy()
        )
        target = ad.AnnData(
            X=np.array([[2.0, 1.0, 0.0], [1.0, 2.0, 3.0]]), var=var.copy()
        )
        source_pca, target_pca, selected, _ = prepare_joint_representation(
            source, target, n_hvg=3, n_pcs=2, seed=1
        )
        self.assertEqual(source_pca.shape, (2, 2))
        self.assertEqual(target_pca.shape, (2, 2))
        self.assertEqual(set(selected), {"TP53", "EPCAM", "VIM"})


if __name__ == "__main__":
    unittest.main()
