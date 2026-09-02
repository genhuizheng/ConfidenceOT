"""Download and combine official Human MSigDB collections for cancer GSEA."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


COLLECTION_TEMPLATES = (
    "h.all.v{version}.symbols.gmt",
    "c5.go.bp.v{version}.symbols.gmt",
    "c2.cp.reactome.v{version}.symbols.gmt",
    "c2.cp.wikipathways.v{version}.symbols.gmt",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path, retries: int = 3) -> str:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    errors: list[str] = []
    request = urllib.request.Request(
        url, headers={"User-Agent": "ConfidenceOT-HumanMSigDB/1.0"}
    )
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
            return "python-urllib"
        except (OSError, urllib.error.URLError) as error:
            errors.append(f"urllib attempt {attempt}: {error}")
            temporary.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(2**attempt)

    for program, arguments, engine in (
        ("curl", ["--http1.1", "--fail", "--location", "--silent", "--show-error",
                  "--retry", "10", "--retry-all-errors", "--connect-timeout", "30",
                  "--output", str(temporary), url], "curl-http1.1"),
        ("wget", ["--https-only", "--tries=10", "--timeout=30",
                  "--output-document", str(temporary), url], "wget"),
    ):
        executable = shutil.which(program)
        if executable is None:
            continue
        try:
            subprocess.run([executable, *arguments], check=True)
            temporary.replace(destination)
            return engine
        except (OSError, subprocess.CalledProcessError) as error:
            errors.append(f"{program}: {error}")
            temporary.unlink(missing_ok=True)
    raise RuntimeError("All secure download methods failed:\n" + "\n".join(errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--version", default="2025.1.Hs")
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
        engine = "existing-file"
        if args.force or not destination.exists():
            print(f"Downloading {url}", flush=True)
            engine = download(url, destination)
        pathway_n = sum(1 for line in destination.open(encoding="utf-8") if line.strip())
        if not destination.stat().st_size or not pathway_n:
            raise RuntimeError(f"Empty GMT: {destination}")
        files.append(destination)
        manifest.append({"collection_file": name, "url": url, "bytes": destination.stat().st_size,
                         "pathway_n": pathway_n, "sha256": sha256(destination),
                         "download_engine": engine})

    combined = args.output_dir / f"human_hallmark_go_bp_reactome_wikipathways_v{args.version}.symbols.gmt"
    with combined.open("wb") as output:
        for path in files:
            content = path.read_bytes()
            output.write(content)
            if content and not content.endswith(b"\n"):
                output.write(b"\n")
    metadata = {
        "msigdb_release": args.version, "species": "Homo sapiens",
        "gene_identifier": "human gene symbol", "combined_gmt": combined.name,
        "combined_pathway_n": sum(1 for line in combined.open(encoding="utf-8") if line.strip()),
        "combined_sha256": sha256(combined), "collections": manifest,
    }
    (args.output_dir / "human_msigdb_manifest.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
