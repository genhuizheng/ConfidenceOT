"""Shared pieces of the MOSTA biological application (scripts 31-34 in this folder).

Reading: finished MOSTA runs (mouse_embryo/23_ layout) are only read, never
written.  The three runs:

    ds      mosta_all_stages                         ranknm256_noscale_ds_cos, no potentials
    logcpm  mosta_all_stages_logcpm                  logcpm, potentials saved (couplings rebuildable)
    ds_pot  mosta_all_stages_ds_potentials_20261028  ds rerun with potentials (OT not run as of 2026-10-30)

Writing: every output goes under one versioned analysis root, through
``Outputs``, which refuses to replace a file it did not write in this root.

Definitions used everywhere (checked against the code, 2026-10-30):
* M4-E decision cost of a bin = its support-restricted mean transport cost
  (score / partner mass + rejection cost); retained iff score < 0, i.e.
  decision cost < rejection cost; margin = decision cost - rejection cost
  (src/confidenceot/blockwise.py).  A bin with no partner mass on the support
  has decision cost 0 by construction (recorded as ``*_no_partner``).
* M4-R decision cost = the counterfactual cost (reversible variant), same
  margin rule.
* rejection cost = the calibrated within-side cost of the pair (23_ calibrate).
* co_* = the M4-E coupling on its retained x retained support; bot_* = the
  balanced entropic coupling on every bin (budget 0), same cost and epsilon.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STAGES = ("E9.5", "E10.5", "E11.5", "E12.5", "E13.5", "E14.5", "E15.5", "E16.5")
SCRATCH = Path("/scratch/10119/ghzheng/OT_project/mouse_embryo_results")
RUNS = {
    "ds": SCRATCH / "mosta_all_stages",
    "logcpm": SCRATCH / "mosta_all_stages_logcpm",
    "ds_pot": SCRATCH / "mosta_all_stages_ds_potentials_20261028",
}
METHODS = ("m4e", "m4r")
ORGAN_SETS = HERE / "organ_sets.tsv"
MARKERS = HERE / "markers.tsv"
CURATED = HERE / "literature" / "moscot_curated_transitions.tsv"


def stage_of(sample: str) -> str:
    return sample.split("_")[0]


def stage_index(stage: str) -> int:
    return STAGES.index(stage)


def organ_sets() -> dict[str, list[str]]:
    table = pd.read_csv(ORGAN_SETS, sep="\t", comment="#")
    return {name: list(group["annotation"]) for name, group in table.groupby("organ_set", sort=False)}


def markers() -> pd.DataFrame:
    return pd.read_csv(MARKERS, sep="\t", comment="#")


def curated_transitions() -> pd.DataFrame | None:
    """The moscot curated transitions (source_annotation -> target_annotation), or None if not obtained."""
    if not CURATED.is_file():
        return None
    table = pd.read_csv(CURATED, sep="\t", comment="#")
    return table if len(table) else None


def support_classes(own: str, partners: list[str], curated: pd.DataFrame | None, direction: str) -> np.ndarray:
    """Per partner annotation: 'same', 'curated' (listed successor / predecessor), or 'other'.

    ``direction`` 'forward' (own is the earlier side) reads the table as
    source -> target; 'backward' reads it target <- source.  Without a table
    only 'same' and 'other' exist; 'other' is unclassified, not wrong.
    """
    out = np.array(["other"] * len(partners), dtype=object)
    allowed: set[str] = set()
    if curated is not None:
        if direction == "forward":
            allowed = set(curated.loc[curated["source_annotation"] == own, "target_annotation"])
        else:
            allowed = set(curated.loc[curated["target_annotation"] == own, "source_annotation"])
    for i, partner in enumerate(partners):
        if partner == own:
            out[i] = "same"
        elif partner in allowed:
            out[i] = "curated"
    return out


def read_pairs(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "manifest" / "pairs.csv")


def analysed_bins(root: Path, sample: str) -> pd.DataFrame:
    """The analysed bins of one slice in equalised-row order (the row order of every per-pair table)."""
    table = pd.read_csv(root / "slices" / f"{sample}_bins.csv.gz")
    table = table[table["equalised_row"] >= 0].sort_values("equalised_row").reset_index(drop=True)
    if not np.array_equal(table["equalised_row"].to_numpy(), np.arange(len(table))):
        raise RuntimeError(f"{sample}: equalised rows are not 0..n-1")
    return table


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Outputs:
    """Writes under one root; refuses to replace a file this root's manifest does not list."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = self.root / "manifests" / "OUTPUT_MANIFEST.tsv"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        self.mine: set[str] = set()
        if self.manifest.is_file():
            with open(self.manifest, newline="", encoding="utf-8") as handle:
                self.mine = {row["path"] for row in csv.DictReader(handle, delimiter="\t")}

    def path(self, relative: str) -> Path:
        target = self.root / relative
        if target.exists() and relative not in self.mine:
            raise FileExistsError(f"{target} exists and was not written by this analysis; not replaced")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def record(self, relative: str, note: str = "") -> None:
        new = not self.manifest.is_file()
        with open(self.manifest, "a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            if new:
                writer.writerow(["path", "bytes", "written", "job", "note"])
            writer.writerow([relative, (self.root / relative).stat().st_size, time.strftime("%Y-%m-%dT%H:%M:%S"),
                             os.environ.get("SLURM_JOB_ID", "local"), note])
        self.mine.add(relative)

    def table(self, frame: pd.DataFrame, relative: str, note: str = "") -> Path:
        target = self.path(relative)
        sep = "\t" if relative.endswith((".tsv", ".tsv.gz")) else ","
        frame.to_csv(target, sep=sep, index=False, lineterminator="\n", float_format="%.6g")
        self.record(relative, note)
        return target

    def text(self, content: str, relative: str, note: str = "") -> Path:
        target = self.path(relative)
        target.write_text(content, encoding="utf-8", newline="\n")
        self.record(relative, note)
        return target

    def json(self, payload, relative: str, note: str = "") -> Path:
        return self.text(json.dumps(payload, indent=2, default=str), relative, note)
