from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "cancer_metastasis"))


def load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cancer_metastasis" / file)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


META = load("robust_target_meta", "12_meta_analyze_robust_target_deg.py")
DEG = load("robust_target_deg", "11_run_robust_target_deg.py")
PYDESEQ2 = load("paired_pydeseq2", "13_run_paired_pydeseq2.py")


class CancerRobustTargetDegTest(unittest.TestCase):
    def test_pseudobulk_collapses_target_groups_within_patient(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for index, target in enumerate(("met1", "met2")):
                group = root / "groups" / f"{index:03d}_P1__{target}"
                group.mkdir(parents=True)
                (group / "PSEUDOBULK_READY").write_text("ready\n")
                samples = [
                    f"P1__{target}__robust_rejected",
                    f"P1__{target}__robust_retained",
                ]
                pd.DataFrame(
                    [[10, 1], [2, 8]], index=samples, columns=["G1", "G2"]
                ).rename_axis("sample_id").to_csv(
                    group / "pseudobulk_raw_counts.csv.gz", compression="gzip"
                )
                pd.DataFrame({
                    "sample_id": samples,
                    "group_id": f"P1__{target}",
                    "patient_id": "P1",
                    "target_sample": target,
                    "confidence_status": ["robust_rejected", "robust_retained"],
                    "cell_n": [15, 15],
                }).to_csv(group / "pseudobulk_sample_metadata.csv", index=False)
                pd.DataFrame({
                    "gene": ["G1", "G2"], "used_for_ot": [True, False]
                }).to_csv(
                    group / "pseudobulk_gene_metadata.csv.gz",
                    index=False, compression="gzip",
                )
            counts, metadata, genes = PYDESEQ2.load_patient_pseudobulk(root, 20)
            self.assertEqual(counts.shape, (2, 2))
            self.assertEqual(counts.loc["P1__robust_rejected", "G1"], 20)
            self.assertEqual(metadata.loc["P1__robust_retained", "cell_n"], 30)
            self.assertTrue(genes.set_index("gene").loc["G1", "used_for_ot_anywhere"])

    def test_analysis_groups_include_single_and_multi_primary_targets(self):
        manifest = pd.DataFrame({
            "dataset_id": ["D", "D", "D"],
            "patient_id": ["P1", "P2", "P2"],
            "target_sample": ["met1", "met2", "met2"],
            "source_sample": ["only", "left", "right"],
        })
        robustness = pd.DataFrame({
            "dataset_id": ["D"],
            "patient_id": ["P2"],
            "target_sample": ["met2"],
            "baseline_winner": ["left"],
            "source090_winner": ["right"],
            "recommended_range_exact_robust": [False],
            "recommended_range_laterality_robust": [True],
        })
        result = DEG.analysis_groups(manifest, robustness, "source090")
        self.assertEqual(len(result), 2)
        single = result[result["patient_id"].eq("P1")].iloc[0]
        self.assertEqual(single["origin_group_type"], "single_primary_compatibility")
        self.assertEqual(single["baseline_winner"], "only")
        self.assertEqual(single["source090_winner"], "only")
        multi = result[result["patient_id"].eq("P2")].iloc[0]
        self.assertEqual(multi["candidate_primary_n"], 2)
        self.assertEqual(multi["source090_winner"], "right")

    def test_bh_adjust_is_monotone_in_rank(self):
        p = np.array([0.01, 0.04, 0.03, np.nan])
        adjusted = META.bh_adjust(p)
        self.assertTrue(np.isnan(adjusted[3]))
        self.assertTrue(np.all((adjusted[:3] >= p[:3]) & (adjusted[:3] <= 1)))
        order = np.argsort(p[:3])
        self.assertTrue(np.all(np.diff(adjusted[:3][order]) >= -1e-12))

    def test_meta_uses_patient_log2fc_and_effect_weighted_gsea_rank(self):
        table = pd.DataFrame({
            "gene": ["G1"] * 6,
            "log2_fold_change": [1.0, 0.8, 0.7, 0.9, 1.1, 0.6],
        })
        result = META.meta_table(table, minimum_patients=5).iloc[0]
        self.assertGreater(result["median_patient_log2_fold_change"], 0)
        self.assertGreater(result["gsea_rank_score"], 0)
        self.assertEqual(result["patient_n"], 6)

    def test_jaccard_uses_direction_specific_gene_sets(self):
        table = pd.DataFrame({
            "group_id": ["a", "a", "b", "b"],
            "gene": ["G1", "G2", "G1", "G3"],
            "direction": ["rejected_enriched"] * 4,
        })
        result = META.jaccard_table(table, "rejected_enriched")
        value = result[
            result["left_group"].eq("a") & result["right_group"].eq("b")
        ].iloc[0]
        self.assertEqual(value["intersection_gene_n"], 1)
        self.assertEqual(value["union_gene_n"], 3)
        self.assertAlmostEqual(value["jaccard"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
