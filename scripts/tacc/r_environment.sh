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
# It searches rather than assumes. The previous version hardcoded
# $WORK/envs/r-omics/bin/Rscript from the lab's RStudio notes; that path does
# not exist on this account, and a wrong guess reads as "no R" when there may
# be an R two directories away. Search order:
#
#   1. CONFIDENCEOT_RSCRIPT, if the caller already set it. Always wins.
#   2. Known conda and env layouts under $WORK and $SCRATCH.
#   3. A bounded find under $WORK, then $SCRATCH, for any */bin/Rscript.
#   4. Rscript on PATH, including after trying the Rstats module.
#   5. The rocker container, last: it needs the tacc-apptainer module, is
#      refused on the login nodes, and cannot be installed into, because the
#      image has compilers but not the development headers and is read only.
#
# Every place it looked is printed when nothing is found, so the next step is
# to set CONFIDENCEOT_RSCRIPT by hand rather than to guess again.
#
# The library path is computed the way the lab's rstudio.slurm computes it,
# platform and R major.minor, so it points at packages that are already there
# rather than at a fresh empty directory.

_cot_r_source=""
_cot_r_looked=()

_cot_try_rscript() {
  # $1 = candidate path or command, $2 = how to describe it
  local candidate=$1 description=$2
  [[ -z "$candidate" ]] && return 1
  _cot_r_looked+=("$candidate")
  if [[ -x "$candidate" ]]; then
    export CONFIDENCEOT_RSCRIPT="$candidate"
    _cot_r_source=$description
    return 0
  fi
  return 1
}

if [[ -n "${CONFIDENCEOT_RSCRIPT:-}" ]]; then
  _cot_r_source="caller-supplied"
else
  for _cot_candidate in \
      "${WORK:-/nonexistent}/envs/r-omics/bin/Rscript" \
      "${WORK:-/nonexistent}/apps/miniforge3/envs/r-omics/bin/Rscript" \
      "${WORK:-/nonexistent}/miniforge3/envs/r-omics/bin/Rscript" \
      "${WORK:-/nonexistent}/apps/miniconda3/envs/r-omics/bin/Rscript" \
      "${SCRATCH:-/nonexistent}/conda_envs/r-omics/bin/Rscript" \
      "${SCRATCH:-/nonexistent}/envs/r-omics/bin/Rscript"; do
    _cot_try_rscript "$_cot_candidate" "conda r-omics" && break
  done
fi

# Nothing at a known layout, so look for one. Bounded depth and -type f keep
# this to seconds even on a Lustre filesystem; the first hit wins.
if [[ -z "$_cot_r_source" ]]; then
  for _cot_root in "${WORK:-}" "${SCRATCH:-}"; do
    [[ -z "$_cot_root" || ! -d "$_cot_root" ]] && continue
    _cot_r_looked+=("find $_cot_root -maxdepth 5 -name Rscript")
    _cot_found=$(find "$_cot_root" -maxdepth 5 -type f -name Rscript \
                   -perm -u+x 2> /dev/null | head -n 1)
    if [[ -n "$_cot_found" ]]; then
      export CONFIDENCEOT_RSCRIPT="$_cot_found"
      _cot_r_source="found under $_cot_root"
      break
    fi
  done
fi

# The module system may carry one. Loading it changes PATH for the rest of the
# shell, which is what a caller of this file wants anyway.
if [[ -z "$_cot_r_source" ]]; then
  _cot_r_looked+=("module load Rstats")
  module load Rstats > /dev/null 2>&1 || true
  if command -v Rscript > /dev/null 2>&1; then
    export CONFIDENCEOT_RSCRIPT="$(command -v Rscript)"
    _cot_r_source="PATH"
  fi
fi

if [[ -z "$_cot_r_source" ]]; then
  _cot_r_sif=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
  _cot_r_looked+=("$_cot_r_sif")
  if [[ -f "$_cot_r_sif" ]]; then
    module load tacc-apptainer 2> /dev/null || true
    export CONFIDENCEOT_RSCRIPT="apptainer exec -B /scratch -B ${WORK:-/scratch} $_cot_r_sif Rscript"
    _cot_r_source="container $_cot_r_sif"
  fi
fi

if [[ -z "$_cot_r_source" ]]; then
  export CONFIDENCEOT_R_STATUS="no-r"
  echo "   no R found. Looked at:" >&2
  for _cot_candidate in "${_cot_r_looked[@]}"; do
    echo "     $_cot_candidate" >&2
  done
  echo "   set CONFIDENCEOT_RSCRIPT to an Rscript and source this again" >&2
  return 0 2> /dev/null || exit 0
fi

echo "   R via $_cot_r_source: $CONFIDENCEOT_RSCRIPT"

# Only compute the library path when the caller has not fixed it, and only
# when the directory the lab's layout would put it in actually exists.
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

# Does the bridge's own dependency load? Only sctransform matters:
# Seurat::SCTransform is a wrapper around sctransform::vst and drags in a
# dependency chain this comparison never touches.
if $CONFIDENCEOT_RSCRIPT --vanilla -e 'suppressMessages({library(Matrix); library(sctransform)}); cat("   ok   sctransform", as.character(packageVersion("sctransform")), "\n")' 2> /dev/null; then
  export CONFIDENCEOT_R_STATUS="ready"
else
  export CONFIDENCEOT_R_STATUS="no-packages"
  echo "   R runs but sctransform will not load. Install it with:"
  echo "     $CONFIDENCEOT_RSCRIPT --vanilla -e 'install.packages(\"sctransform\", repos=\"https://cloud.r-project.org\")'"
fi
