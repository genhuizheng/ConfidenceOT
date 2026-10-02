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
2. **The deposits measure the same genes.** ``13_`` takes the union of the
   patients' genes and fills a gene a matrix lacks with 0, so a gene missing
   from one deposit is tested on the other deposits' patients only.
3. **Each patient has both states under each arm's gate.** The paired DEG
   keeps a patient only with at least ``--minimum-cells`` retained and as many
   rejected primary cells. Counted with ``21_``'s own functions -- the same
   largest-lesion choice, the same M4-E gate, the same calibration refusal --
   so the count is the one the pseudobulk will see, not a second estimate.

Usage:

    python cancer_metastasis/tools/check_cancer_merge.py MANIFEST_CSV BLOCK_DIR \\
        --dataset-id GSE225857 --dataset-id GSE178318 --dataset-id GSE315534
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent.parent


def load_pseudobulk_module():
    """21_ itself, so its grouping and gate reading are not copied here."""
    # 21_ imports common, and common imports confidenceot from src/.
    sys.path.insert(0, str(HERE.parent / "src"))
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "pseudobulk", HERE / "21_prepare_four_state_malignant_pseudobulk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest_csv", type=Path,
                        help="The manifest the expression stages read")
    parser.add_argument("block_dir", type=Path,
                        help="The factorial block, e.g. preprocessing_factorial/uniform")
    parser.add_argument("--dataset-id", action="append", required=True,
                        dest="dataset_ids", metavar="GSE")
    parser.add_argument("--metastasis-size-csv", type=Path, default=None,
                        help="Defaults to downsample_per_sample.csv beside the "
                             "block's PREDOWNSAMPLE_DEPTH record")
    parser.add_argument("--minimum-cells", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pseudobulk = load_pseudobulk_module()
    manifest = pd.read_csv(args.manifest_csv)
    known = set(manifest["dataset_id"].astype(str))
    unknown = [value for value in args.dataset_ids if value not in known]
    if unknown:
        raise SystemExit(f"dataset_id {unknown} not in {args.manifest_csv}")
    manifest = manifest[manifest["dataset_id"].astype(str).isin(args.dataset_ids)].copy()

    print("== pairs ==")
    columns = ["pair_id", "dataset_id", "patient_id", "source_sample", "target_sample"]
    print(manifest[columns].to_string(index=False))

    print("\n== 1. patient IDs ==")
    deposits = manifest.groupby(manifest["patient_id"].astype(str))["dataset_id"].nunique()
    shared = sorted(deposits.index[deposits > 1])
    pairs_per_patient = manifest.groupby(manifest["patient_id"].astype(str)).size()
    print(f"{len(manifest)} pairs, {manifest['patient_id'].nunique()} patients, "
          f"{manifest['dataset_id'].nunique()} deposits")
    print(f"patients with more than one pair: "
          f"{pairs_per_patient[pairs_per_patient > 1].to_dict() or 'none'}")
    print(f"patient_id shared by two deposits: {shared or 'none'}")

    print("\n== 2. genes ==")
    import anndata as ad

    genes_by_deposit: dict[str, set[str]] = {}
    for deposit, table in manifest.groupby("dataset_id"):
        paths = set()
        for column in ("source_h5ads_json", "target_h5ads_json"):
            for value in table[column]:
                paths.update(json.loads(str(value)))
        symbols: set[str] = set()
        for path in sorted(paths):
            data = ad.read_h5ad(path, backed="r")
            symbols |= set(pseudobulk.gene_symbols(data))
            data.file.close()
        genes_by_deposit[str(deposit)] = symbols
        print(f"{deposit}: {len(symbols)} gene symbols over {len(paths)} h5ad file(s)")
    shared_genes = set.intersection(*genes_by_deposit.values())
    union_genes = set.union(*genes_by_deposit.values())
    print(f"in every deposit: {len(shared_genes)}; in any: {len(union_genes)}; "
          f"in only some: {len(union_genes - shared_genes)}")

    print(f"\n== 3. patients usable per arm (>= {args.minimum_cells} retained and "
          f">= {args.minimum_cells} rejected primary cells, M4-E) ==")
    size_csv = args.metastasis_size_csv
    if size_csv is None:
        record = args.block_dir / "PREDOWNSAMPLE_DEPTH"
        size_csv = Path(record.read_text(encoding="utf-8").strip()).parent / "downsample_per_sample.csv"
    groups = pseudobulk.analysis_groups(manifest, None, "source090")
    sizes, _ = pseudobulk.metastasis_sizes(manifest, size_csv)
    groups, _ = pseudobulk.designate_metastasis(groups, sizes)
    patients = sorted(groups["patient_id"].astype(str).unique())
    print(f"after the largest-lesion choice: {len(patients)} patients, "
          f"{len(groups)} primary-metastasis groups")
    arms = sorted(path.name for path in args.block_dir.iterdir() if path.is_dir())
    for label in arms:
        gate_root = args.block_dir / label
        usable, notes = 0, []
        for patient in patients:
            retained = rejected = 0
            for row in groups[groups["patient_id"].astype(str).eq(patient)].itertuples():
                pair = pseudobulk.pair_id(patient, str(row.source_sample), str(row.target_sample))
                try:
                    refusal = pseudobulk.calibration_refusal(pseudobulk.read_run(gate_root, pair))
                    gate = pseudobulk.read_gate(gate_root, pair, "source", "baseline")
                except (RuntimeError, FileNotFoundError) as error:
                    notes.append(f"{patient}: {error}")
                    continue
                if refusal is not None:
                    notes.append(f"{patient}: excluded, {refusal}")
                    continue
                flags = gate["baseline_rejected"].astype(bool)
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
