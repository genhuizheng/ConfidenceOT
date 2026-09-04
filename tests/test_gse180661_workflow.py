from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reciprocal_dominant_pairs_are_one_to_one():
    module = load("pair_runner", ROOT / "cancer_metastasis" / "02_run_pair.py")
    coupling = np.asarray([
        [0.8, 0.1, 0.0],
        [0.2, 0.7, 0.0],
        [0.0, 0.2, 0.9],
    ])
    result = module.reciprocal_dominant_pairs(
        "M4-E", coupling, np.asarray(["s0", "s1", "s2"]),
        np.asarray(["t0", "t1", "t2"]), np.ones(3, bool), np.ones(3, bool),
    )
    assert len(result) == 3
    assert result["source_observation_id"].is_unique
    assert result["target_observation_id"].is_unique
    assert result["reciprocal_dominant"].all()


def test_all_cell_enrichment_uses_marginal_abundance_null():
    module = load(
        "all_cell_validation",
        ROOT / "cancer_metastasis" / "gse180661" / "all_cell_validation.py",
    )
    table = pd.DataFrame({
        "source_annotation": ["A", "A", "B", "B"],
        "target_annotation": ["A", "B", "A", "B"],
        "transported_mass": [4.0, 1.0, 1.0, 4.0],
        "source_conditional_probability": [0.8, 0.2, 0.2, 0.8],
    })
    enriched, summary = module.enrich_pair(table)
    assert np.isclose(enriched["abundance_null_mass"].sum(), 10.0)
    assert np.isclose(summary["same_annotation_mass_fraction"], 0.8)
    assert np.isclose(summary["abundance_null_same_annotation_fraction"], 0.5)
    assert np.isclose(summary["same_annotation_enrichment_ratio"], 1.6)


def test_leading_gene_table_requires_effect_and_patient_consistency():
    module = load(
        "primary_deg", ROOT / "cancer_metastasis" / "gse180661" / "primary_deg.py"
    )
    table = pd.DataFrame({
        "gene": ["PASS_POS", "LOW_LFC", "INCONSISTENT", "PASS_NEG"],
        "fdr": [0.01, 0.01, 0.01, 0.02],
        "log2_fold_change": [1.5, 0.8, 1.4, -1.2],
        "detected_patient_fraction": [0.8, 0.8, 0.8, 0.9],
        "patient_direction_consistency": [0.8, 0.8, 0.6, 0.75],
        "absolute_wald_statistic": [5.0, 4.0, 6.0, 3.0],
        "absolute_log2_fold_change": [1.5, 0.8, 1.4, 1.2],
    })
    selected = module.leading_table(table, 1.0)
    assert set(selected["gene"]) == {"PASS_POS", "PASS_NEG"}
    assert dict(zip(selected["gene"], selected["direction"])) == {
        "PASS_POS": "metastasis_compatible_enriched",
        "PASS_NEG": "primary_restricted_enriched",
    }


def test_tcga_expression_scale_is_not_double_logged():
    module = load(
        "tcga_survival",
        ROOT / "cancer_metastasis" / "gse180661" / "tcga_survival.py",
    )
    log2_tpm = pd.DataFrame({"P1": [1.0, 3.0]}, index=["A", "B"])
    observed = module.expression_on_log2_tpm_scale(
        log2_tpm, "log2_tpm_plus_one"
    )
    pd.testing.assert_frame_equal(observed, log2_tpm)
    raw_tpm = pd.DataFrame({"P1": [0.0, 3.0]}, index=["A", "B"])
    transformed = module.expression_on_log2_tpm_scale(raw_tpm, "raw_tpm")
    assert transformed.loc["A", "P1"] == 0.0
    assert transformed.loc["B", "P1"] == 2.0
