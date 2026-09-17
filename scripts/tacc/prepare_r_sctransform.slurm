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

# Install sctransform into whichever R the environment helper finds, and verify
# it the way the bridge will use it, in one submission.
#
# It does not name an R itself. r_environment.sh searches, and on this account
# it lands on one of the conda R environments under $SCRATCH/conda_envs rather
# than on the container -- which is the better outcome: the container has
# compilers but not the development headers and is read only, which is why the
# Seurat attempt lost 19 packages to zlib.h, libcurl and a missing pkg-config.
#
# SCTransform's algorithm is not in Seurat anyway. It is in the standalone
# `sctransform` package, which Seurat::SCTransform wraps, and whose dependency
# chain needs none of those headers. Comparing against the package that
# implements the method is also the more defensible reference.
#
# Installs go to R_LIBS_USER, a directory of ours, so a shared conda
# environment that other work depends on is never written into.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits.
#
# Usage:
#   sbatch scripts/tacc/prepare_r_sctransform.slurm
#
# To pin the R rather than let the helper search:
#   sbatch --export=ALL,CONFIDENCEOT_RSCRIPT=/path/to/Rscript \
#       scripts/tacc/prepare_r_sctransform.slurm
set -eo pipefail

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_DEPTH_SCREEN_ROOT:-/scratch/10119/ghzheng/OT_project/depth_screen}

for pair in "repo:$repo" "root:$root"; do
  name=${pair%%:*}
  value=${pair#*:}
  if [[ -z "$value" || "$value" != /* ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -f "$repo/scripts/tacc/r_environment.sh" ]]; then
  echo "no r_environment.sh under $repo; did the pull land?" >&2
  exit 2
fi

mkdir -p "$root/logs"

echo "== finding R"
# shellcheck source=/dev/null
source "$repo/scripts/tacc/r_environment.sh"
if [[ "${CONFIDENCEOT_R_STATUS:-}" == "no-r" ]]; then
  echo "no R to install into" >&2
  exit 2
fi
if [[ "${CONFIDENCEOT_R_STATUS:-}" == "ready" ]]; then
  echo
  echo "sctransform already loads; nothing to install."
  echo "submit the R configurations with:"
  echo "  export CONFIDENCEOT_RSCRIPT='$CONFIDENCEOT_RSCRIPT'"
  echo "  sbatch --array=9-10 scripts/tacc/screen_depth_configurations.slurm"
  exit 0
fi
if [[ -z "${R_LIBS_USER:-}" ]]; then
  echo "R_LIBS_USER is unset, so there is nowhere of ours to install into" >&2
  exit 2
fi
mkdir -p "$R_LIBS_USER"

work=$(mktemp -d "${SCRATCH:-/tmp}/r_sct_XXXXXX")
trap 'rm -rf "$work"' EXIT

run_r() {
  $CONFIDENCEOT_RSCRIPT --vanilla "$1"
}

# The R goes into files rather than through `-e`. Quoting R through bash is how
# the earlier version of this job acquired a literal tab where a \t escape
# belonged, and three string literals broken across newlines.
cat > "$work/inspect.R" <<'INSPECT'
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
cat(R.version.string, "\n")
cat("platform:", R.version$platform, "\n")
cat("libPaths:\n")
print(.libPaths())
for (package in c("Matrix", "sctransform", "Rcpp", "RcppArmadillo",
                  "matrixStats", "Seurat")) {
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
# half-built dependency surfaces here in a short job rather than an hour into a
# screen task.
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

echo
echo "== what this R has now"
run_r "$work/inspect.R"

echo
echo "== installing whatever is missing into $R_LIBS_USER"
run_r "$work/install.R"

echo
echo "== verifying the way the bridge will use it"
run_r "$work/verify.R"

echo
echo "=================================================="
echo "sctransform is usable. Submit the R configurations with:"
echo
echo "  export CONFIDENCEOT_RSCRIPT='$CONFIDENCEOT_RSCRIPT'"
echo "  export R_LIBS_USER='$R_LIBS_USER'"
echo "  sbatch --array=9-10 scripts/tacc/screen_depth_configurations.slurm"
