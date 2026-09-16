#!/bin/bash
#SBATCH -J cot_prep_pca
#SBATCH -o /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/prep_pca_%j.out
#SBATCH -e /scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results/logs/prep_pca_%j.err
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Splits the GSE271675 prostate object into per-specimen H5ADs and writes a
# pair manifest. Batch rather than interactive because reading 646 million
# nonzeros needs more memory than a login node grants -- an earlier attempt
# there failed allocating 89 MiB.
#
# No `set -u`. It fires on things that are fine here, such as an empty array
# expansion, while missing the failure that actually cost this project two job
# numbers: a path variable that was defined but empty, which `-u` permits. The
# explicit non-empty absolute-path check below is the protection that matters.
set -eo pipefail

source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_ENV:-/scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot}"

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
result=${CANCER_COT_ROOT:-/scratch/10119/ghzheng/primary_metastatic_cancer/confidenceot_results}
source_h5ad=${PROSTATE_H5AD:-/scratch/10119/ghzheng/AllPatients_Malignant_harmony.h5ad}
output=${PROSTATE_PREPARED_ROOT:-$result/prepared_GSE271675_20260916}

# Two submissions were lost earlier to a path variable arriving empty, so the
# job refuses to start rather than creating directories at the filesystem root.
for name in repo result source_h5ad output; do
  value=${!name}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$source_h5ad" ]]; then
  echo "Source H5AD does not exist: $source_h5ad" >&2
  exit 2
fi

cd "$repo"
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$output" "$result/logs"

echo "===== PREPARE START $(date --iso-8601=seconds) ====="
python cancer_metastasis/34_prepare_prostate_pairs.py \
  "$source_h5ad" "$output" --dataset-id "${PROSTATE_DATASET_ID:-GSE271675}"
echo "===== PREPARE COMPLETE $(date --iso-8601=seconds) ====="

# The per-cell depth distribution decides the subsampling target, and multiome
# nuclei do not match the whole-cell 3' depths the ovarian target of 3,119 came
# from. It is reported here and chosen in a separate step rather than guessed.
python - "$output/pair_manifest_eligible.csv" <<'PY'
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

manifest = pd.read_csv(sys.argv[1])
depths = []
for row in manifest.to_dict("records"):
    for side in ("source", "target"):
        for path in json.loads(str(row[f"{side}_h5ads_json"])):
            data = ad.read_h5ad(path)
            depths.append(
                np.asarray(sparse.csr_matrix(data.X).sum(axis=1), dtype=np.int64).ravel()
            )
            del data
pooled = np.concatenate(depths)
quantiles = {f"q{int(q * 100):02d}": int(np.quantile(pooled, q))
             for q in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90)}
print("pooled per-cell depth over every prepared specimen:")
print(json.dumps({"cells": int(pooled.size), **quantiles}, indent=2), flush=True)
PY
