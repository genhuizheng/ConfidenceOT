from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).parents[1] / "cancer_metastasis" / "17_run_qc_matched_gsea.py"
    spec = importlib.util.spec_from_file_location("qc_matched_gsea", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_pathway_consensus_requires_four_of_five_same_direction_significant_runs():
    module = load_module()
    tables = []
    for replicate in range(5):
        tables.append(pd.DataFrame({
            "pathway": ["STABLE_POSITIVE", "MIXED", "STABLE_NEGATIVE"],
            "collection": ["Hallmark"] * 3,
            "NES": [2.0, 1.0 if replicate < 3 else -1.0, -2.0],
            "fdr": [0.01, 0.01, 0.01 if replicate < 4 else 0.2],
            "replicate": [replicate] * 3,
        }))
    result = module.pathway_consensus(tables, 5, 0.8).set_index("pathway")
    assert bool(result.loc["STABLE_POSITIVE", "stable"])
    assert result.loc["STABLE_POSITIVE", "dominant_direction"] == "robust_rejected_enriched"
    assert bool(result.loc["STABLE_NEGATIVE", "stable"])
    assert result.loc["STABLE_NEGATIVE", "dominant_direction"] == "robust_retained_enriched"
    assert not bool(result.loc["MIXED", "stable"])
