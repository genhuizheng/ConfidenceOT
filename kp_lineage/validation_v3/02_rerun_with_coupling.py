"""Rerun KP-Tracer OT pairs with the original calibrated cost, saving the coupling.

    python kp_lineage/validation_v3/02_rerun_with_coupling.py --root ROOT --label LABEL \\
        --pairs PAIR_ID [PAIR_ID ...] --out OUT_ROOT

The original runs (kp_lineage/submit_kptracer_chain.sh -> tacc_preprocessing_factorial.slurm
-> 02_run_pair.py) saved gates but no coupling. This reruns 02_run_pair.py with the
exact original command line -- same manifest, the same row --index (the
representation seed is seed + index), the same QC, scope, preprocessing, tolerance
and one BLAS thread per pair as in the original pair-parallel run -- plus:

* ``--fixed-rejection-cost`` set to the original run.json ``rejection_cost``
  (stored at full double precision), so the gate is fitted at the calibrated cost
  without recalibrating;
* ``--save-coupling`` (dense coupling per method, npz) and ``--save-pairing-edges``
  (which also writes the joint PCA coordinates, needed for the kNN baseline).

Outputs go to OUT_ROOT/<label>/<pair_id>/... in 02_run_pair.py's own layout; the
original outputs are only read. A pair whose new output has SUCCESS is skipped.
03_compare_reproduction.py decides whether a rerun reproduced the original gate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

ARMS = {
    # label: (manifest relative to root, minimum counts, minimum genes, maximum mito %, extra)
    "ranknm256_noscale_ds_cos": ("downsampled/pair_manifest_downsampled.csv", "0", "0", "100", []),
    "raw": ("prepared/pair_manifest_eligible.csv", "1000", "500", "20", ["--allow-precomputed"]),
}


def original_run(root: Path, label: str, pair_id: str) -> Path:
    done = sorted((root / "ot" / "kptracer" / label / pair_id).glob("scope_malignant/*/SUCCESS"))
    if len(done) != 1:
        raise SystemExit(f"{pair_id} {label}: {len(done)} completed original outputs")
    return done[0].parent


def command(repo: Path, root: Path, label: str, pair_id: str, out_root: Path, extra: list[str]) -> list[str]:
    manifest_rel, counts, genes, mito, arm_extra = ARMS[label]
    manifest_path = root / manifest_rel
    manifest = pd.read_csv(manifest_path)
    hits = manifest.index[manifest["pair_id"].astype(str).eq(pair_id)].tolist()
    if len(hits) != 1:
        raise SystemExit(f"{pair_id}: {len(hits)} rows in {manifest_path}")
    run = json.loads((original_run(root, label, pair_id) / "run.json").read_text())
    if run.get("rejection_cost_mode") != "null_calibrated":
        raise SystemExit(f"{pair_id} {label}: original cost was not null-calibrated")
    cost = repr(float(run["rejection_cost"]))
    return [sys.executable, str(repo / "cancer_metastasis" / "02_run_pair.py"), str(manifest_path), str(out_root / label),
            "--index", str(hits[0]),
            "--analysis-scope", "malignant", "--malignant-column", "malignant", "--malignant-value", "malignant",
            *arm_extra, "--skip-completed", "--preprocessing", label,
            "--calibration-null", "within_side_split", "--max-observed-cells-per-side", "0",
            "--minimum-scope-cells", "4", "--cell-qc",
            "--minimum-total-counts", counts, "--minimum-detected-genes", genes, "--maximum-mitochondrial-percent", mito,
            "--null-calibration-replicates", "5", "--null-validation-replicates", "5", "--calibration-grid-size", "5",
            "--tolerance", "1e-4", "--workers", "8", "--device", "cpu",
            "--within-side-acceptance-minimum", "0.99",
            "--fixed-rejection-cost", cost, "--save-coupling", "--save-pairing-edges", *extra]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="kp_tracer_confidenceot")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--label", choices=sorted(ARMS), required=True)
    parser.add_argument("--pairs", nargs="*", default=[])
    parser.add_argument("--pair-list", type=Path, help="file with one pair_id per line")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="appended to every 02_run_pair.py call")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    pairs = list(args.pairs)
    if args.pair_list:
        pairs += [line.strip() for line in args.pair_list.read_text().splitlines() if line.strip()]
    pairs = [p for k, p in enumerate(pairs) if k % args.task_count == args.task_index]
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = os.pathsep.join([str(args.repo / "src"), str(args.repo / "cancer_metastasis"), env.get("PYTHONPATH", "")])
    failures = 0
    for pair_id in pairs:
        cmd = command(args.repo, args.root, args.label, pair_id, args.out, args.extra)
        print("RUN", " ".join(cmd), flush=True)
        if args.dry_run:
            continue
        status = subprocess.run(cmd, env=env, cwd=args.repo).returncode
        print(f"{'DONE' if status == 0 else 'FAILED'} {args.label} {pair_id} exit {status}", flush=True)
        failures += status != 0
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
