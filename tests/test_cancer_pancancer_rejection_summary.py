from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def load_module():
    path = Path(__file__).parents[1] / "cancer_metastasis" / "19_summarize_pancancer_all_cell_rejection.py"
    spec = importlib.util.spec_from_file_location("pancancer_rejection_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_explicit_malignant_overrides_annotation_name_but_epithelial_stays_ambiguous():
    module = load_module()
    table = pd.DataFrame({
        "annotation": ["Epithelial", "Ovarian.cancer.cell", "T cell"],
        "malignant": [True, False, False],
    })
    result = module.classify_cells(table).tolist()
    assert result == [
        "explicit_malignant",
        "explicit_non_malignant",
        "explicit_non_malignant",
    ]
