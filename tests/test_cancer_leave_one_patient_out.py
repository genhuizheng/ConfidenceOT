"""Cover the leave-one-patient-out machinery without needing PyDESeq2.

The fit itself is a subprocess and needs the solver; everything that decides
*which* patients go into it does not, and that is where a leave-one-out design
goes wrong quietly. Three failures this pins:

* an exclusion that does not actually exclude, so every fold reproduces the
  full fit and the design reports maximum stability;
* holding out a patient that was never in the fit, which produces a fold
  identical to the full one and inflates the same count;
* a fold-change tag computed differently here and in the script that writes
  the file, so the result table is looked for under a name nothing wrote.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for extra in (REPO / "src", REPO / "cancer_metastasis"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

GENES = [f"G{index:03d}" for index in range(12)]
CONTRAST = "primary_rejected_vs_primary_retained"


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO / "cancer_metastasis" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_patient(root: Path, index: int, patient: str, cell_n: int,
                   ready: bool = True) -> None:
    directory = root / "patients" / f"{index:03d}_{patient}"
    directory.mkdir(parents=True, exist_ok=True)
    samples = [f"{patient}__{CONTRAST}__{status}"
               for status in ("case", "reference")]
    rng = np.random.default_rng(index)
    counts = pd.DataFrame(
        rng.integers(5, 400, size=(2, len(GENES))),
        index=samples, columns=GENES)
    counts.to_csv(directory / "pseudobulk_raw_counts.csv.gz", compression="gzip")
    pd.DataFrame({
        "sample_id": samples,
        "patient_id": patient,
        "group_id": patient,
        "contrast": CONTRAST,
        "comparison_status": ["case", "reference"],
        "cell_set": ["primary_rejected", "primary_retained"],
        "cell_n": [cell_n, cell_n],
    }).to_csv(directory / "pseudobulk_sample_metadata.csv", index=False)
    pd.DataFrame({"gene": GENES, "used_for_ot": False}).to_csv(
        directory / "pseudobulk_gene_metadata.csv.gz", index=False,
        compression="gzip")
    if ready:
        (directory / "PSEUDOBULK_READY").write_text("ready\n", encoding="utf-8")


class ExcludePatientActuallyExcludes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for index, patient in enumerate(["P0", "P1", "P2", "P3"]):
            _write_patient(self.root, index, patient, cell_n=40)
        self.deg = _load("paired_pydeseq2", "13_run_paired_pydeseq2.py")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _patients_in(self, exclude: frozenset[str]) -> list[str]:
        counts, metadata, _ = self.deg.load_patient_pseudobulk(
            self.root, 10, exclude)
        return sorted(set(metadata["patient_id"].astype(str)))

    def test_without_exclusion_every_patient_is_present(self) -> None:
        self.assertEqual(self._patients_in(frozenset()),
                         ["P0", "P1", "P2", "P3"])

    def test_one_excluded_patient_is_gone_and_only_that_one(self) -> None:
        self.assertEqual(self._patients_in(frozenset({"P1"})),
                         ["P0", "P2", "P3"])

    def test_counts_and_metadata_stay_aligned_after_exclusion(self) -> None:
        counts, metadata, _ = self.deg.load_patient_pseudobulk(
            self.root, 10, frozenset({"P2"}))
        self.assertEqual(list(counts.index), list(metadata.index))
        self.assertNotIn("P2", set(metadata["patient_id"].astype(str)))
        self.assertEqual(len(counts), 6)

    def test_excluding_everyone_raises_rather_than_fitting_nothing(self) -> None:
        with self.assertRaises(RuntimeError):
            self.deg.load_patient_pseudobulk(
                self.root, 10, frozenset({"P0", "P1", "P2", "P3"}))

    def test_the_cell_floor_still_applies_alongside_the_exclusion(self) -> None:
        _write_patient(self.root, 4, "P4", cell_n=3)
        self.assertNotIn("P4", self._patients_in(frozenset()))


class PatientDiscovery(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.loo = _load("leave_one_out", "34_leave_one_patient_out.py")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_patient_without_a_ready_marker_is_not_held_out(self) -> None:
        # 21_ leaves a directory behind for a patient it skipped. Holding that
        # patient out would produce a fold identical to the full fit and count
        # as evidence of stability it is not.
        _write_patient(self.root, 0, "P0", cell_n=40)
        _write_patient(self.root, 1, "P1", cell_n=40, ready=False)
        self.assertEqual(self.loo.patients(self.root), ["P0"])

    def test_ids_come_from_metadata_not_directory_names(self) -> None:
        _write_patient(self.root, 0, "SPECTRUM-OV-003", cell_n=40)
        self.assertEqual(self.loo.patients(self.root), ["SPECTRUM-OV-003"])


class TagAndJaccard(unittest.TestCase):
    def setUp(self) -> None:
        self.loo = _load("leave_one_out_tag", "34_leave_one_patient_out.py")
        self.deg = _load("paired_pydeseq2_tag", "13_run_paired_pydeseq2.py")

    def test_the_fold_change_tag_matches_the_writer(self) -> None:
        # 34_ reads a file 13_ named. If the two spell the threshold
        # differently the table is looked for under a name nothing wrote.
        for value in (0.5, 1.0, 1.5, 2.0):
            with self.subTest(value=value):
                mine = f"{value:g}".replace(".", "p")
                self.assertEqual(mine, self.deg.threshold_tag(value))

    def test_jaccard_edges(self) -> None:
        self.assertEqual(self.loo.jaccard(set(), set()), 1.0)
        self.assertEqual(self.loo.jaccard({"a"}, set()), 0.0)
        self.assertEqual(self.loo.jaccard({"a", "b"}, {"a", "b"}), 1.0)
        self.assertAlmostEqual(self.loo.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)

    def test_a_missing_result_table_names_what_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "contrasts" / CONTRAST).mkdir(parents=True)
            (root / "contrasts" / CONTRAST
             / "pydeseq2_all_gene_fdr_005_abs_log2fc_0p5.csv").write_text(
                "gene\nG000\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as caught:
                self.loo.significant_genes(root, CONTRAST, "1")
            self.assertIn("0p5", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
