from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "cancer_metastasis"))
path = ROOT / "cancer_metastasis" / "24_visualize_four_state_confidence_umap.py"
spec = importlib.util.spec_from_file_location("confidence_umap", path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_selected_records_preserve_exact_side_and_sample():
    manifest = pd.DataFrame({
        "patient_id": ["P1", "P1"],
        "source_sample": ["primary_a", "primary_a"],
        "target_sample": ["met_a", "met_b"],
        "source_h5ads_json": ['["shared.h5ad"]', '["shared.h5ad"]'],
        "target_h5ads_json": ['["shared.h5ad"]', '["shared.h5ad"]'],
    })
    records = module.selected_h5ad_records(manifest, {"P1"})
    observed = set(zip(records["side"], records["sample"]))
    assert observed == {
        ("primary", "primary_a"),
        ("metastasis", "met_a"),
        ("metastasis", "met_b"),
    }


def test_pair_table_keeps_primary_and_metastasis_separate(monkeypatch):
    coordinates = pd.DataFrame({
        "side": ["primary", "primary", "metastasis", "metastasis"],
        "patient_id": ["P1"] * 4,
        "sample": ["primary_a", "other_primary", "met_a", "other_met"],
        "observation_id": ["s1", "sx", "t1", "tx"],
        "umap_1": [0.0, 1.0, 2.0, 3.0],
        "umap_2": [0.0, 1.0, 2.0, 3.0],
    })

    def fake_gate(_root, _pair_id, side, prefix):
        observation = "s1" if side == "source" else "t1"
        rejected = prefix == "baseline"
        return pd.DataFrame({
            "observation_id": [observation],
            f"{prefix}_rejected": [rejected],
            f"{prefix}_rejection_score": [0.8 if rejected else 0.2],
        })

    monkeypatch.setattr(module, "read_pair_gate", fake_gate)
    row = pd.Series({
        "pair_id": "P1__primary_a__met_a",
        "patient_id": "P1",
        "source_sample": "primary_a",
        "target_sample": "met_a",
    })
    result = module.pair_cell_table(coordinates, row, Path("baseline"), Path("sensitivity"))
    assert set(result["observation_id"]) == {"s1", "t1"}
    assert set(result["side"]) == {"primary", "metastasis"}
    assert set(result["consensus_status"]) == {"site_or_cap_discordant"}

