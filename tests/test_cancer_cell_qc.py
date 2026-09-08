import numpy as np
import pandas as pd
from scipy import sparse

from cancer_metastasis.common import cell_qc_table


class MinimalData:
    def __init__(self):
        self.X = sparse.csr_matrix([
            [900, 0, 0],
            [900, 100, 0],
            [700, 100, 200],
        ])
        self.layers = {}
        self.obs_names = pd.Index(["low", "pass", "mito"])
        self.var_names = pd.Index(["GENE1", "GENE2", "MT-X"])
        self.var = pd.DataFrame(index=self.var_names)
        self.n_obs = 3


def test_cell_qc_table_records_transparent_failure_reasons():
    result = cell_qc_table(
        MinimalData(),
        minimum_total_counts=1000,
        minimum_detected_genes=2,
        maximum_mitochondrial_percent=10,
    ).set_index("observation_id")
    assert not bool(result.loc["low", "qc_pass"])
    assert "low_total_counts" in result.loc["low", "qc_failure_reason"]
    assert "low_detected_genes" in result.loc["low", "qc_failure_reason"]
    assert bool(result.loc["pass", "qc_pass"])
    assert not bool(result.loc["mito", "qc_pass"])
    assert result.loc["mito", "qc_failure_reason"] == "high_mitochondrial_percent"
