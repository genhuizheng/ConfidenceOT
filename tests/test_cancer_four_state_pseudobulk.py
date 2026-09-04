from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "cancer_metastasis"))
path = ROOT / "cancer_metastasis" / "21_prepare_four_state_malignant_pseudobulk.py"
spec = importlib.util.spec_from_file_location("four_state_pseudobulk", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_source_consensus_requires_complete_pair_coverage_and_agreement():
    records = pd.DataFrame({
        "entity_id": ["S::a", "S::a", "S::b", "S::b", "S::c", "S::d"],
        "sample": ["S"] * 6,
        "observation_id": ["a", "a", "b", "b", "c", "d"],
        "pair_id": ["p1", "p2", "p1", "p2", "p1", "p1"],
        "pair_robust_status": [
            "retained", "retained", "rejected", "rejected",
            "retained", "cap_discordant",
        ],
        "baseline_rejection_score": [0.1, 0.2, 0.8, 0.9, 0.1, 0.5],
    })
    result = module.consensus_classification(records, {"S": 2}).set_index(
        "observation_id"
    )
    assert result.loc["a", "consensus_status"] == "retained"
    assert result.loc["b", "consensus_status"] == "rejected"
    assert result.loc["c", "consensus_status"] == "incomplete_pair_coverage"
    assert result.loc["d", "consensus_status"] == "incomplete_pair_coverage"
