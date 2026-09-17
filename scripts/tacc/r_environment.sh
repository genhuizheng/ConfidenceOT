#!/bin/bash
# Find R and its package library, and export what the SCTransform bridge needs.
#
# Source this; do not execute it:
#   source scripts/tacc/r_environment.sh
#
# Sets CONFIDENCEOT_RSCRIPT and R_LIBS_USER, and leaves CONFIDENCEOT_R_STATUS
# as "ready", "no-packages" or "no-r" so a caller can branch without parsing
# messages.
#
# Preference order, cheapest and most reliable first:
#
#   1. CONFIDENCEOT_RSCRIPT, if the caller already set it.
#   2. The conda R environment at $WORK/envs/r-omics. This is the one to want:
#      a real R binary that runs anywhere, including the login nodes, with its
#      packages already installed under $WORK/Rlibs.
#   3. Rscript on PATH.
#   4. The rocker container, which needs the tacc-apptainer module and is
#      refused on the login nodes. Kept last because installing into it fails:
#      the image has compilers but not the development headers, so anything
#      needing zlib.h, libcurl or libssl will not build, and the image is read
#      only.
#
# The library path is computed the way the lab's own rstudio.slurm computes it,
# platform and R major.minor, so it points at the packages that are already
# there rather than at a fresh empty directory.

if [[ -n "${CONFIDENCEOT_RSCRIPT:-}" ]]; then
  _cot_r_source="caller-supplied"
elif [[ -x "${WORK:-/nonexistent}/envs/r-omics/bin/Rscript" ]]; then
  export CONFIDENCEOT_RSCRIPT="$WORK/envs/r-omics/bin/Rscript"
  _cot_r_source="conda r-omics"
elif command -v Rscript > /dev/null 2>&1; then
  export CONFIDENCEOT_RSCRIPT="Rscript"
  _cot_r_source="PATH"
else
  _cot_r_sif=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
  if [[ -f "$_cot_r_sif" ]]; then
    module load tacc-apptainer 2> /dev/null || true
    export CONFIDENCEOT_RSCRIPT="apptainer exec -B /scratch -B ${WORK:-/scratch} $_cot_r_sif Rscript"
    _cot_r_source="container $_cot_r_sif"
  else
    _cot_r_source="none"
  fi
fi

if [[ "$_cot_r_source" == "none" ]]; then
  export CONFIDENCEOT_R_STATUS="no-r"
  echo "   no R found: set CONFIDENCEOT_RSCRIPT, or check \$WORK/envs/r-omics" >&2
  return 0 2> /dev/null || exit 0
fi

echo "   R via $_cot_r_source: $CONFIDENCEOT_RSCRIPT"

# Only compute the library path when the caller has not fixed it, and only when
# $WORK exists -- the layout is the lab's, not something to invent elsewhere.
if [[ -z "${R_LIBS_USER:-}" && -n "${WORK:-}" && -d "$WORK/Rlibs" ]]; then
  _cot_r_platform=$($CONFIDENCEOT_RSCRIPT --vanilla -e 'cat(R.version$platform)' 2> /dev/null)
  _cot_r_version=$($CONFIDENCEOT_RSCRIPT --vanilla -e 'cat(paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = "."))' 2> /dev/null)
  if [[ -n "$_cot_r_platform" && -n "$_cot_r_version" ]]; then
    _cot_candidate="$WORK/Rlibs/${_cot_r_platform}-library/${_cot_r_version}"
    if [[ -d "$_cot_candidate" ]]; then
      export R_LIBS_USER="$_cot_candidate"
      echo "   R_LIBS_USER $R_LIBS_USER"
    else
      echo "   note: $_cot_candidate does not exist; leaving R_LIBS_USER unset"
    fi
  fi
elif [[ -n "${R_LIBS_USER:-}" ]]; then
  echo "   R_LIBS_USER $R_LIBS_USER (from the caller)"
fi

# Does the bridge's own dependency actually load? Only sctransform matters;
# Seurat::SCTransform is a wrapper around sctransform::vst and pulls in a
# dependency chain this comparison never touches.
if $CONFIDENCEOT_RSCRIPT --vanilla -e 'suppressMessages({library(Matrix); library(sctransform)}); cat("   ok   sctransform", as.character(packageVersion("sctransform")), "\n")' 2> /dev/null; then
  export CONFIDENCEOT_R_STATUS="ready"
else
  export CONFIDENCEOT_R_STATUS="no-packages"
  echo "   R runs but sctransform will not load. Install it with:"
  echo "     $CONFIDENCEOT_RSCRIPT --vanilla -e 'install.packages(\"sctransform\", repos=\"https://cloud.r-project.org\")'"
fi
