"""Phase 0: copy the existing lineage-validation results, data and code to an archive drive, and verify them.

    python lineage_validation/lineage_site_application/tools/archive_to_f.py --dest F:/ConfidenceOT_Archive --date 20261022

Copies (never moves) these repository paths with their relative structure under DEST/<date>[_vN]/:
  lineage_validation/                   Level 1 local export, Level 2 (54 couplings, tar + unpacked), Level 3
                                        (incl. per-cell tables not in git), within-family subclones, mass audit
  results/kp_tracer_lineage_validation/ the authors' KP-Tracer trees, character matrices, annotations, manifest
  src/ cancer_metastasis/ kp_lineage/   code, configurations, SLURM scripts
  the uncommitted working-tree files named by `git status`
plus git/ConfidenceOT.bundle (every branch and commit) and git/{HEAD,log,status,remotes}.txt.
DEST/<date> is used if it does not exist; otherwise <date>_v2, _v3, ...: nothing on the drive is overwritten.
SHA-256 is computed while copying and again by re-reading the archived file; sizes and counts are compared.
Writes ARCHIVE_MANIFEST.tsv (source, archive path, bytes, sha256 source, sha256 archive, verified) in the archive
and prints the totals. Exit 1 if any file fails verification.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TREES = ["lineage_validation", "results/kp_tracer_lineage_validation", "src", "cancer_metastasis", "kp_lineage"]
CHUNK = 8 << 20


def sha_copy(src: Path, dst: Path) -> str:
    h = hashlib.sha256()
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(src, "rb") as fi, open(dst, "xb") as fo:  # "x": never overwrite
        while True:
            b = fi.read(CHUNK)
            if not b:
                break
            h.update(b)
            fo.write(b)
    shutil.copystat(src, dst)
    return h.hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    root = args.dest / args.date
    v = 1
    while root.exists():
        v += 1
        root = args.dest / f"{args.date}_v{v}"
    root.mkdir(parents=True)
    print(f"archive root {root}", flush=True)
    files = []
    for t in TREES:
        files += sorted(p for p in (REPO / t).rglob("*") if p.is_file())
    dirty = [line[3:].strip() for line in git("status", "--porcelain").splitlines() if line.strip()]
    extra = [REPO / p for p in dirty if (REPO / p).is_file() and not any(p.replace("\\", "/").startswith(t + "/") for t in TREES)]
    files += extra
    total = sum(p.stat().st_size for p in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB to copy ({len(extra)} uncommitted files outside the archived trees)", flush=True)
    rows, done, t0 = [], 0, time.time()
    for k, src in enumerate(files):
        rel = src.relative_to(REPO)
        dst = root / "repo" / rel
        digest = sha_copy(src, dst)
        size = src.stat().st_size
        rows.append([str(src), str(dst), size, digest])
        done += size
        if size > 50e6 or k % 2000 == 0:
            print(f"  {k + 1}/{len(files)}  {done / 1e9:.2f}/{total / 1e9:.2f} GB  {rel}", flush=True)
    # git: a bundle of every ref, and the state of the working copy
    gdir = root / "git"
    gdir.mkdir()
    subprocess.run(["git", "-C", str(REPO), "bundle", "create", str(gdir / "ConfidenceOT.bundle"), "--all"], check=True, capture_output=True)
    (gdir / "HEAD.txt").write_text(git("rev-parse", "HEAD") + git("log", "-1", "--format=%H %ad %s", "--date=iso"), encoding="utf-8")
    (gdir / "log.txt").write_text(git("log", "--date=iso", "--format=%h %ad %s"), encoding="utf-8")
    (gdir / "status.txt").write_text(git("status", "--porcelain"), encoding="utf-8")
    (gdir / "remotes.txt").write_text(git("remote", "-v"), encoding="utf-8")
    for p in sorted(gdir.iterdir()):
        rows.append([f"(git) {p.name}", str(p), p.stat().st_size, sha_file(p)])
    # verification: re-read every archived file
    print("verifying (re-reading every archived file) ...", flush=True)
    bad = 0
    for r in rows:
        dst = Path(r[1])
        ok_size = dst.stat().st_size == r[2]
        d2 = sha_file(dst)
        r += [d2, ok_size and d2 == r[3]]
        bad += not r[-1]
    with open(root / "ARCHIVE_MANIFEST.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("source\tarchive_path\tbytes\tsha256_source\tsha256_archive\tverified\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    n_src = len(files)
    n_dst = sum(1 for p in (root / "repo").rglob("*") if p.is_file())
    print(f"copied {n_src} files ({total / 1e9:.2f} GB) in {time.time() - t0:.0f} s; archive holds {n_dst} repo files; "
          f"verification failures: {bad}", flush=True)
    sys.exit(1 if bad or n_src != n_dst else 0)


if __name__ == "__main__":
    main()
