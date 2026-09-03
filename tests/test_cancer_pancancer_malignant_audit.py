from __future__ import annotations

import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).parents[1] / "cancer_metastasis" / "18_audit_pancancer_malignant_labels.py"
    spec = importlib.util.spec_from_file_location("pancancer_malignant_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_annotation_classifier_does_not_promote_generic_epithelial_cells():
    module = load_module()
    assert module.classify_annotation("Ovarian.cancer.cell") == "name_supported_malignant_candidate"
    assert module.classify_annotation("Tumor") == "name_supported_malignant_candidate"
    assert module.classify_annotation("Malignant epithelial") == "name_supported_malignant_candidate"
    assert module.classify_annotation("Epithelial") == "ambiguous_epithelial"
    assert module.classify_annotation("Tumor-associated macrophage") == "named_background"
    assert module.classify_annotation("Endothelial.cell") == "named_background"
    assert module.classify_annotation("T cell") == "named_background"
