#!/bin/bash
#SBATCH -J r_seurat
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/depth_screen/logs/r_seurat_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/depth_screen/logs/r_seurat_%j.err
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Check for Seurat in the R container and install it if missing, in one
# submission.
#
# Apptainer is refused on the TACC login nodes, so even asking "is Seurat
# there?" costs a job. This does the asking and the installing together rather
# than spending two trips through the queue to find out.
#
# The container is a rocker r-ver image whose /usr/local/lib/R is read only, so
# packages go to a host directory that is bound in and named by R_LIBS_USER.
# That directory is what CONFIDENCEOT_R_LIBS then points the screen at.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits.
#
# Usage:
#   sbatch scripts/tacc/prepare_r_seurat.slurm
set -eo pipefail

container=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
libs=${CONFIDENCEOT_R_LIBS:-/scratch/10119/ghzheng/r_libs}
threads=${SLURM_CPUS_PER_TASK:-8}

for pair in "container:$container" "libs:$libs"; do
  name=${pair%%:*}; value=${pair#*:}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$container" ]]; then
  echo "no container at $container" >&2
  exit 2
fi

module load tacc-apptainer
mkdir -p "$libs" "$(dirname "${SLURM_JOB_ID:+x}")" 2>/dev/null || true
mkdir -p "$libs"

run() {
  apptainer exec -B /scratch --env R_LIBS_USER="$libs" "$container" Rscript "$@"
}

echo "== what the container has, before anything is installed"
run --vanilla -e '
cat(R.version.string, "\n")
cat("libPaths:\n"); print(.libPaths())
for (p in c("Matrix", "Seurat", "sctransform"))
  cat(sprintf("  %-12s %s\n", p,
      tryCatch(as.character(packageVersion(p)),
               error = function(e) "MISSING")))
'

echo
echo "== installing whatever is missing into $libs"
# Seurat pulls a long dependency chain; Ncpus matters here. Matrix and
# sctransform come along with Seurat but are named so a partial state repairs.
run --vanilla -e '
libs <- Sys.getenv("R_LIBS_USER")
.libPaths(c(libs, .libPaths()))
wanted <- c("Matrix", "Seurat", "sctransform")
missing <- wanted[!vapply(wanted, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing) == 0) {
  cat("nothing to install\n")
} else {
  cat("installing:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, lib = libs,
                   repos = "https://cloud.r-project.org",
                   Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8")))
}
'

echo
echo "== verifying, the way the bridge will use it"
run --vanilla -e '
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({library(Matrix); library(Seurat)})
cat("Seurat", as.character(packageVersion("Seurat")), "loads\n")
# The exact call the bridge makes, on a tiny matrix, so a version mismatch in
# GetAssayData surfaces here rather than inside the screen.
set.seed(1)
counts <- matrix(rpois(2000, 3), nrow = 100)
rownames(counts) <- paste0("g", 1:100); colnames(counts) <- paste0("c", 1:20)
object <- SCTransform(CreateSeuratObject(counts = as(counts, "CsparseMatrix")),
                      vst.flavor = "v2", verbose = FALSE,
                      return.only.var.genes = FALSE,
                      variable.features.n = 100)
residuals <- tryCatch(
  GetAssayData(object, assay = "SCT", layer = "scale.data"),
  error = function(e) GetAssayData(object, assay = "SCT", slot = "scale.data"))
cat("SCTransform returned", nrow(residuals), "genes x", ncol(residuals),
    "cells\n")
if (nrow(residuals) == 0) stop("empty scale.data")
cat("READY\n")
'

echo
echo "=================================================="
echo "Seurat is usable. Submit the container configurations with:"
echo
echo "  export CONFIDENCEOT_R_LIBS=$libs"
echo "  sbatch --array=9-10 scripts/tacc/screen_depth_configurations.slurm"
echo
echo "(the screen job loads tacc-apptainer and builds the invocation itself)"
