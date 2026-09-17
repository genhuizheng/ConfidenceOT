#!/bin/bash
#SBATCH -J r_sctransform
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/depth_screen/logs/r_sct_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/depth_screen/logs/r_sct_%j.err
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=END,FAIL

# Install and verify sctransform in the R container, in one submission.
#
# Replaces the Seurat attempt, which failed for a reason worth recording: the
# rocker r-ver image has compilers but not the development headers, so 19 of
# Seurat's dependencies would not build -- httpuv on zlib.h, openssl and curl
# on their own headers, and pkg-config missing entirely. Every one of those is
# from Seurat's interactive and plotting chain, none of which this comparison
# touches. The image is read only, so they cannot be added.
#
# SCTransform's algorithm is not in Seurat. It is in the standalone
# `sctransform` package -- Seurat::SCTransform is a wrapper around
# sctransform::vst -- whose dependencies are Matrix, Rcpp, RcppArmadillo,
# matrixStats, dplyr and ggplot2, none of which need those headers. So the
# external comparison is against the package that implements the method, which
# is the more defensible reference anyway.
#
# Apptainer is refused on the TACC login nodes, so asking and installing are
# done together rather than spending two trips through the queue.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits.
#
# Usage:
#   sbatch scripts/tacc/prepare_r_sctransform.slurm
set -eo pipefail

container=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
libs=${CONFIDENCEOT_R_LIBS:-/scratch/10119/ghzheng/r_libs}

for pair in "container:$container" "libs:$libs"; do
  name=${pair%%:*}
  value=${pair#*:}
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
mkdir -p "$libs"
work=$(mktemp -d "${SCRATCH:-/tmp}/r_sct_XXXXXX")
trap 'rm -rf "$work"' EXIT

run_r() {
  apptainer exec -B /scratch --env R_LIBS_USER="$libs" "$container" \
    Rscript --vanilla "$1"
}

# The R is written to files rather than passed with -e. Quoting R through bash
# is how the earlier version acquired a literal tab and three broken string
# literals.
cat > "$work/inspect.R" <<'INSPECT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
cat(R.version.string, "\n")
cat("libPaths:\n")
print(.libPaths())
for (package in c("Matrix", "sctransform", "Rcpp", "RcppArmadillo")) {
  version <- tryCatch(as.character(packageVersion(package)),
                      error = function(e) "MISSING")
  cat(sprintf("  %-14s %s\n", package, version))
}
INSPECT

cat > "$work/install.R" <<'INSTALL'
libs <- Sys.getenv("R_LIBS_USER")
.libPaths(c(libs, .libPaths()))
wanted <- c("Matrix", "sctransform")
have <- vapply(wanted, requireNamespace, logical(1), quietly = TRUE)
if (all(have)) {
  cat("nothing to install\n")
} else {
  missing <- wanted[!have]
  cat("installing:", paste(missing, collapse = ", "), "\n")
  install.packages(missing, lib = libs,
                   repos = "https://cloud.r-project.org",
                   Ncpus = as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "8")))
}
INSTALL

# The exact call the bridge makes, on a tiny matrix, so an API change or a
# half-built dependency surfaces here in a short job rather than inside a
# screen task an hour in.
cat > "$work/verify.R" <<'VERIFY'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({
  library(Matrix)
  library(sctransform)
})
cat("sctransform", as.character(packageVersion("sctransform")), "loads\n")
set.seed(1)
counts <- matrix(rpois(4000, 3), nrow = 200)
rownames(counts) <- paste0("g", seq_len(200))
colnames(counts) <- paste0("c", seq_len(20))
fit <- vst(as(counts, "CsparseMatrix"), vst.flavor = "v2",
           residual_type = "pearson", min_cells = 1,
           return_cell_attr = FALSE, verbosity = 0)
if (is.null(fit$y) || nrow(fit$y) == 0 || ncol(fit$y) == 0) {
  stop("vst returned no residuals")
}
cat("vst returned", nrow(fit$y), "genes x", ncol(fit$y), "cells\n")
cat("READY\n")
VERIFY

echo "== what the container has now"
run_r "$work/inspect.R"

echo
echo "== installing whatever is missing into $libs"
run_r "$work/install.R"

echo
echo "== verifying the way the bridge will use it"
run_r "$work/verify.R"

echo
echo "=================================================="
echo "sctransform is usable. Submit the container configurations with:"
echo
echo "  export CONFIDENCEOT_R_LIBS=$libs"
echo "  sbatch --array=9-10 scripts/tacc/screen_depth_configurations.slurm"
echo
echo "(the screen job loads tacc-apptainer and builds the invocation itself)"
