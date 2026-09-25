#!/bin/bash
# A drop-in Rscript that runs inside the benchmark's container.
#
#   CONFIDENCEOT_RSCRIPT=benchmark/tacc/rscript.sh
#
# Everything about apptainer lives here, so the generation job calls one
# command and does not care which of the three R routes is in use.
#
# The one subtlety is the library path. The account has Splatter compiled
# against the conda R 4.5.3 on aarch64, and R_LIBS_USER points at it whenever
# the conda route is configured. Apptainer passes the host environment into
# the container, so without the override below the image's R 4.4.1 would find
# those packages first and load objects built against a different R against a
# different libc. That fails late and blames Splatter.

set -euo pipefail

sif=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/confidenceot-splatter.sif}
if [[ ! -f "$sif" ]]; then
    echo "no image at $sif" >&2
    echo "  build it: sbatch benchmark/tacc/build_container.slurm" >&2
    exit 4
fi

module load tacc-apptainer > /dev/null 2>&1 || true

# The image's own site library wins over anything the host set.
export APPTAINERENV_R_LIBS_USER=/usr/local/lib/R/site-library
export APPTAINERENV_R_LIBS_SITE=/usr/local/lib/R/site-library

binds=(-B /scratch)
[[ -n "${WORK:-}" && -d "${WORK}" ]] && binds+=(-B "$WORK")
[[ -n "${HOME:-}" && -d "${HOME}" ]] && binds+=(-B "$HOME")

exec apptainer exec "${binds[@]}" "$sif" Rscript "$@"
