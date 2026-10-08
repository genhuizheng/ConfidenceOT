"""Build the ranknm256_noscale_ds_cos representation of every MOSTA pair.

Each pair takes its two equalised slices whole: every analysed bin, with no
per-side cap.  The counts go to ``confidenceot.Preprocessing`` the way
cancer_metastasis/02_run_pair.py sends them through
prepare_joint_representation:

* joint rank encoding of the top 256 genes per bin, without the gene-median
  division;
* joint selection of 2,000 genes by variance of the encoding;
* no per-gene scaling, then PCA to 30 components with random_state = seed +
  index;
* unit rows for the cosine cost.

The ``ds`` is recorded, not applied, because 21_ equalised the counts at the
file level.

Also counts the bins whose encoding is shorter than 256 genes on the pair's
common genes, the check cancer_metastasis/tools/audit_rank_top_n.py makes.  It
is reported only.

Writes representations/<pair_id>/representation.npz (float32 coordinates),
representation.json (the Preprocessing record, selected genes, the rank-cut
count) and REPRESENTATION_SUCCESS.  Several processes and jobs share the pair
list through claim files, largest pairs first.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mosta_common import (  # noqa: E402
    PREPROCESSING_LABEL, SEED, Layout, check_preprocessing, claim, publish,
    read_equalised_slice, read_pairs, staging_directory, write_json,
)

SUCCESS = "REPRESENTATION_SUCCESS"


def common_columns(source_genes: np.ndarray, target_genes: np.ndarray):
    """Preprocessing.representation's gene intersection: first occurrence, sorted."""
    source_first: dict[str, int] = {}
    target_first: dict[str, int] = {}
    for index, key in enumerate(source_genes):
        source_first.setdefault(str(key), index)
    for index, key in enumerate(target_genes):
        target_first.setdefault(str(key), index)
    common = sorted(set(source_first) & set(target_first))
    return common, [source_first[key] for key in common], [target_first[key] for key in common]


def prepare_pair(layout: Layout, row: pd.Series) -> dict:
    from confidenceot import Preprocessing

    configuration = Preprocessing.from_label(PREPROCESSING_LABEL)
    check_preprocessing(configuration)
    started = time.perf_counter()
    source = read_equalised_slice(layout.equalised_h5ad(row["source_sample"]))
    target = read_equalised_slice(layout.equalised_h5ad(row["target_sample"]))
    loaded = time.perf_counter() - started
    common, source_columns, target_columns = common_columns(source["genes"], target["genes"])
    rank_short = {
        side: int(np.sum(np.diff(data["counts"][:, columns].indptr) < configuration.rank_top_n))
        for side, data, columns in (("source", source, source_columns), ("target", target, target_columns))
    }
    seed = SEED + int(row["index"])
    representation = configuration.representation(
        source["counts"], target["counts"], seed=seed,
        source_genes=source["genes"], target_genes=target["genes"],
        extra_provenance={
            "equalise_applied_here": False,
            "equalise_note": "counts equalised at the file level by mouse_embryo/21_equalise_mosta_depth.py",
        },
    )
    if representation.provenance["label"] != PREPROCESSING_LABEL:
        raise SystemExit(f"built {representation.provenance['label']!r}, asked for {PREPROCESSING_LABEL!r}")
    return {
        "source": np.asarray(representation.source, dtype=np.float32),
        "target": np.asarray(representation.target, dtype=np.float32),
        "record": {
            "pair_id": row["pair_id"], "index": int(row["index"]), "kind": row["kind"],
            "source_sample": row["source_sample"], "target_sample": row["target_sample"],
            "source_bins": int(source["counts"].shape[0]), "target_bins": int(target["counts"].shape[0]),
            "seed": seed,
            "preprocessing": representation.provenance,
            "selected_genes": [str(gene) for gene in representation.selected_genes],
            "common_genes": len(common),
            "rank_top_n": configuration.rank_top_n,
            "source_bins_encoded_under_rank_top_n": rank_short["source"],
            "target_bins_encoded_under_rank_top_n": rank_short["target"],
            "load_seconds": loaded,
            "seconds": time.perf_counter() - started,
        },
    }


def run_worker(task: tuple[str, int]) -> list[dict]:
    root, worker = task
    layout = Layout(Path(root))
    pairs = read_pairs(layout)
    sizes = pd.read_csv(layout.equalisation / "per_slice.csv").set_index("sample")["bins_analysed"]
    pairs["elements"] = pairs["source_sample"].map(sizes) * pairs["target_sample"].map(sizes)
    pairs = pairs.sort_values(["elements", "index"], ascending=[False, True])
    done = []
    for _, row in pairs.iterrows():
        final = layout.representation(row["pair_id"])
        if (final / SUCCESS).is_file():
            continue
        if not claim(layout.claims / "representation", row["pair_id"]):
            continue
        staging = staging_directory(layout, row["pair_id"], "representation")
        try:
            built = prepare_pair(layout, row)
            np.savez(staging / "representation.npz", source=built["source"], target=built["target"])
            write_json(staging / "representation.json", built["record"])
            (staging / SUCCESS).write_text("complete\n", encoding="utf-8")
            published = publish(staging, final)
            done.append({"pair_id": row["pair_id"], "status": "complete" if published else "already_present",
                         "seconds": built["record"]["seconds"]})
            print(f"[worker {worker}] {row['pair_id']} {built['record']['source_bins']}x"
                  f"{built['record']['target_bins']} {built['record']['seconds']:.0f}s", flush=True)
        except Exception:  # recorded and reported by the audit; the next pair goes on
            message = traceback.format_exc()
            layout.representations.mkdir(parents=True, exist_ok=True)
            (layout.representations / f"{row['pair_id']}.FAILED").write_text(message, encoding="utf-8")
            done.append({"pair_id": row["pair_id"], "status": "failed", "seconds": float("nan")})
            print(f"[worker {worker}] FAILED {row['pair_id']}\n{message}", flush=True)
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = [item for chunk in pool.map(run_worker, [(str(args.root), i) for i in range(args.workers)])
                   for item in chunk]
    table = pd.DataFrame(results, columns=["pair_id", "status", "seconds"])
    print(table["status"].value_counts().to_string() if len(table) else "no pair needed work")
    print(f"wall {time.perf_counter() - started:.0f}s")
    if (table["status"] == "failed").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
