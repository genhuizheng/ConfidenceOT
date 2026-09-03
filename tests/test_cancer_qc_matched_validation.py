from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]


def load_module():
    path = ROOT / "cancer_metastasis" / "16_run_qc_matched_tumor_validation.py"
    spec = importlib.util.spec_from_file_location("qc_matched_validation", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_propensity_match_returns_balanced_nonoverlapping_sets():
    module = load_module()
    values = np.tile(np.arange(20, dtype=float), 2)
    qc = pd.DataFrame({
        "confidence_status": ["robust_rejected"] * 20 + ["robust_retained"] * 20,
        "total_counts": 1_000 + values,
        "n_genes_by_counts": 500 + values,
        "pct_counts_mitochondrial": 5 + values / 100,
        "pct_counts_ribosomal": 10 + values / 100,
    })
    rejected, retained, summary = module.propensity_match(qc, 7, 0.2)
    assert len(rejected) == len(retained) == summary["matched_pair_n"]
    assert len(rejected) == 20
    assert not set(rejected).intersection(retained)


def test_standardized_mean_difference_is_zero_for_identical_groups():
    module = load_module()
    values = np.tile(np.arange(10, dtype=float), 2)
    labels = np.array(["robust_rejected"] * 10 + ["robust_retained"] * 10)
    assert module.standardized_mean_difference(values, labels) == 0.0
