"""Download and combine official Mouse MSigDB collections for GSEA."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


COLLECTION_TEMPLATES = (
    "mh.all.v{version}.symbols.gmt",
    "m5.go.bp.v{version}.symbols.gmt",
    "m2.cp.wikipathways.v{version}.symbols.gmt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--version", default="2025.1.Mm")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"https://data.broadinstitute.org/gsea-msigdb/msigdb/release/{args.version}"
    files: list[Path] = []
    manifest: list[dict] = []

    for template in COLLECTION_TEMPLATES:
        name = template.format(version=args.version)
        destination = args.output_dir / name
        url = f"{base_url}/{name}"
        if args.force or not destination.exists():
            print(f"Downloading {url}", flush=True)
            urllib.request.urlretrieve(url, destination)
        line_count = sum(1 for _ in destination.open(encoding="utf-8"))
        if destination.stat().st_size == 0 or line_count == 0:
            raise RuntimeError(f"Downloaded GMT was empty: {destination}")
        files.append(destination)
        manifest.append({
            "collection_file": name,
            "url": url,
            "bytes": destination.stat().st_size,
            "pathway_n": line_count,
            "sha256": sha256(destination),
        })

    combined = args.output_dir / (
        f"mouse_go_bp_hallmark_wikipathways_v{args.version}.symbols.gmt"
    )
    with combined.open("wb") as output:
        for path in files:
            content = path.read_bytes()
            output.write(content)
            if content and not content.endswith(b"\n"):
                output.write(b"\n")
    combined_lines = sum(1 for _ in combined.open(encoding="utf-8"))
    metadata = {
        "msigdb_release": args.version,
        "species": "Mus musculus",
        "gene_identifier": "mouse gene symbol",
        "combined_gmt": combined.name,
        "combined_pathway_n": combined_lines,
        "combined_sha256": sha256(combined),
        "collections": manifest,
    }
    (args.output_dir / "mouse_msigdb_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
