"""Can several deposits of one cancer be merged into one paired pseudobulk design?

A cancer-level DEG pools the patients of several deposits into one paired fit,
~ patient + status. That is valid because each patient sits inside one
deposit, so the patient term absorbs every deposit-level difference and the
retained-versus-rejected effect is still estimated within patients. Three
things have to hold first, and this checks them on the manifest and the gates
before anything is submitted:

1. **Patient IDs are unique across the deposits.**
   ``21_prepare_four_state_malignant_pseudobulk.py`` and
   ``13_run_paired_pydeseq2.py`` key a patient on ``patient_id`` alone, so an
   ID shared by two deposits would pool two people into one pseudobulk.
2. **The genes the merged fit can use.** A pseudobulk lists only the genes its
   patient had counts for and ``13_`` fills the rest with 0, so a merged run
   is restricted to the genes every deposit measured
   (``tools/shared_gene_universe.py``, which the job runs itself). This says
   how many that is.
3. **Each patient has both states under each arm's gate.** The paired DEG
   keeps a patient only with at least ``--minimum-cells`` retained and as many
   rejected primary cells. Counted with ``21_``'s own functions -- the same
   largest-lesion choice, the same M4-E gate -- so the count is the one the
   pseudobulk will see, not a second estimate.

Everything is read on the pairs the gates actually cover. The manifest the
expression stages read lists more pairs than the factorial ran -- the
factorial's own manifest was trimmed to the evaluable ones -- and the job
trims to the gate root before anything else, so this does the same.

Usage:

    python cancer_metastasis/tools/check_cancer_merge.py MANIFEST_CSV BLOCK_DIR \\
        --dataset-id GSE225857 --dataset-id GSE178318 --dataset-id GSE315534
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shared_gene_universe import deposit_genes, load_pseudobulk_module  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest_csv", type=Path,
                        help="The manifest the expression stages read")
    parser.add_argument("block_dir", type=Path,
                        help="The factorial block, e.g. preprocessing_factorial/uniform")
    parser.add_argument("--dataset-id", action="append", required=True,
                        dest="dataset_ids", metavar="GSE")
    parser.add_argument("--scope", default="scope_malignant")
    parser.add_argument("--metastasis-size-csv", type=Path, default=None,
                        help="Defaults to downsample_per_sample.csv beside the "
                             "block's PREDOWNSAMPLE_DEPTH record")
    parser.add_argument("--minimum-cells", type=int, default=10)
    return parser.parse_args()


def gated_pairs(gate_root: Path, scope: str) -> set[str]:
    """The pairs 35_trim_manifest_to_gate.py would keep for this gate root."""
    return {path.parents[2].name
            for path in gate_root.glob(f"*/{scope}/*/cell_confidence.csv")}


def main() -> None:
    args = parse_args()
    pseudobulk = load_pseudobulk_module()
    manifest = pd.read_csv(args.manifest_csv)
    known = set(manifest["dataset_id"].astype(str))
    unknown = [value for value in args.dataset_ids if value not in known]
    if unknown:
        raise SystemExit(f"dataset_id {unknown} not in {args.manifest_csv}")
    listed = manifest[manifest["dataset_id"].astype(str).isin(args.dataset_ids)].copy()

    arms = sorted(path.name for path in args.block_dir.iterdir() if path.is_dir())
    gated = {label: gated_pairs(args.block_dir / label, args.scope) for label in arms}
    in_any = set.union(*gated.values())
    in_every = set.intersection(*gated.values())
    pairs = listed[listed["pair_id"].astype(str).isin(in_any)].copy()

    print("== pairs ==")
    print(f"listed in the manifest: {len(listed)}; gated in any arm: {len(pairs)}; "
          f"gated in all {len(arms)} arms: "
          f"{int(listed['pair_id'].astype(str).isin(in_every).sum())}")
    columns = ["pair_id", "dataset_id", "patient_id", "source_sample", "target_sample"]
    print(pairs[columns].to_string(index=False))
    print(f"per deposit: {pairs['dataset_id'].value_counts().to_dict()}")

    print("\n== 1. patient IDs ==")
    deposits = pairs.groupby(pairs["patient_id"].astype(str))["dataset_id"].nunique()
    shared = sorted(deposits.index[deposits > 1])
    per_patient = pairs.groupby(pairs["patient_id"].astype(str)).size()
    print(f"{len(pairs)} pairs, {pairs['patient_id'].nunique()} patients, "
          f"{pairs['dataset_id'].nunique()} deposits")
    print(f"patients with more than one pair: "
          f"{per_patient[per_patient > 1].to_dict() or 'none'}")
    print(f"patient_id shared by two deposits: {shared or 'none'}")

    print("\n== 2. genes ==")
    found = deposit_genes(pairs, pseudobulk)
    for deposit, record in found.items():
        print(f"{deposit}: {len(record['any'])} gene symbols in any of "
              f"{record['files']} file(s), {len(record['every'])} in every one")
    common = set.intersection(*(record["any"] for record in found.values()))
    union = set.union(*(record["any"] for record in found.values()))
    print(f"measured in every deposit: {len(common)} (the merged fit uses these); "
          f"in any: {len(union)}")

    print(f"\n== 3. patients usable per arm (>= {args.minimum_cells} retained and "
          f">= {args.minimum_cells} rejected primary cells, M4-E) ==")
    size_csv = args.metastasis_size_csv
    if size_csv is None:
        record = args.block_dir / "PREDOWNSAMPLE_DEPTH"
        size_csv = Path(record.read_text(encoding="utf-8").strip()).parent / "downsample_per_sample.csv"
    for label in arms:
        gate_root = args.block_dir / label
        trimmed = listed[listed["pair_id"].astype(str).isin(gated[label])].copy()
        if trimmed.empty:
            print(f"{label:28s} no gated pair")
            continue
        groups = pseudobulk.analysis_groups(trimmed, None, "source090")
        sizes, _ = pseudobulk.metastasis_sizes(trimmed, size_csv)
        groups, _ = pseudobulk.designate_metastasis(groups, sizes)
        patients = sorted(groups["patient_id"].astype(str).unique())
        usable, notes = 0, []
        for patient in patients:
            retained = rejected = 0
            for row in groups[groups["patient_id"].astype(str).eq(patient)].itertuples():
                pair = pseudobulk.pair_id(patient, str(row.source_sample), str(row.target_sample))
                flags = pseudobulk.read_gate(gate_root, pair, "source", "baseline")["baseline_rejected"].astype(bool)
                rejected += int(flags.sum())
                retained += int((~flags).sum())
            ok = retained >= args.minimum_cells and rejected >= args.minimum_cells
            usable += ok
            if not ok:
                notes.append(f"{patient}: {retained} retained, {rejected} rejected")
        line = f"{label:28s} {usable} of {len(patients)} patients usable"
        print(line + (f"   [{'; '.join(notes)}]" if notes else ""))


if __name__ == "__main__":
    main()
