#!/bin/bash
# Download the lineage visualization package from TACC and check it against its manifest. Run locally
# (Git Bash) from the repository root after submit_export.sh has finished on TACC.
#
#   bash lineage_validation/local_lineage_visualization/download_from_tacc.sh            # core only
#   bash lineage_validation/local_lineage_visualization/download_from_tacc.sh --couplings
#
# Writes lineage_validation/local_lineage_visualization/data/local_lineage_visualization_export/.
# scp asks for the TACC password and MFA code itself; nothing is stored here.
set -eo pipefail

remote=${TACC_REMOTE:-ghzheng@vista.tacc.utexas.edu}
root=/scratch/10119/ghzheng/OT_project/lineage_validation
dest=lineage_validation/local_lineage_visualization/data
mkdir -p "$dest"
files=(local_lineage_visualization_core.tar.gz)
[[ "${1:-}" == --couplings ]] && files+=(local_lineage_visualization_couplings.tar)
for f in "${files[@]}"; do
    [[ -e "$dest/$f" ]] && { echo "already here: $dest/$f (delete it to download again)"; continue; }
    scp "$remote:$root/$f" "$dest/"
done
for f in "${files[@]}"; do
    tar -xf "$dest/$f" -C "$dest"
done
python=${PYTHON:-/d/miniconda/envs/sc_sim/python.exe}
"$python" - "$dest/local_lineage_visualization_export" <<'EOF'
import hashlib, sys
from pathlib import Path
import pandas as pd
root = Path(sys.argv[1])
manifest = pd.read_csv(root / "MANIFEST.tsv", sep="\t")
bad, absent = [], []
for path, md5 in zip(manifest["path"], manifest["md5"]):
    file = root / path
    if not file.exists():
        absent.append(path)
        continue
    if hashlib.md5(file.read_bytes()).hexdigest() != md5:
        bad.append(path)
couplings = sum(p.startswith("couplings/") for p in absent)
print(f"{len(manifest)} files in the manifest; {len(manifest) - len(absent)} present, {len(bad)} md5 mismatches, "
      f"{len(absent)} absent ({couplings} of them couplings, expected without --couplings)")
for path in bad[:20]:
    print("MD5 MISMATCH", path)
for path in [p for p in absent if not p.startswith("couplings/")][:20]:
    print("ABSENT", path)
sys.exit(1 if bad or any(not p.startswith("couplings/") for p in absent) else 0)
EOF
