#!/bin/bash
# Find an R that can run the SCTransform bridge, and export what it needs.
#
# Source this; do not execute it:
#   source scripts/tacc/r_environment.sh
#
# Sets CONFIDENCEOT_RSCRIPT and R_LIBS_USER, and leaves CONFIDENCEOT_R_STATUS
# as "ready", "no-packages" or "no-r" so a caller can branch without parsing
# messages.
#
# It chooses by capability, not by name. The earlier version hardcoded
# $WORK/envs/r-omics/bin/Rscript from the lab's RStudio notes; that path does
# not exist on this account, and the two R environments that do exist
# ($SCRATCH/conda_envs/geomx_r_env and .../infercnv_r) were installed for
# other projects, so which of them carries sctransform is not something to
# guess at. So: collect every plausible R, then prefer the first one that can
# actually load sctransform. Only if none can does it fall back to the first R
# that runs at all, compute a library directory of ours, and report
# "no-packages" so the install job can fill it.
#
# Candidates, in order:
#   1. CONFIDENCEOT_RSCRIPT, if the caller set it. Always wins outright.
#   2. Known conda and env layouts under $WORK and $SCRATCH.
#   3. A depth-limited find for any */bin/Rscript under $SCRATCH, then $WORK.
#   4. Rscript on PATH, including after trying the Rstats module.
#   5. The rocker container, last: it needs the tacc-apptainer module, is
#      refused on the login nodes, and cannot be installed into, because the
#      image has compilers but not the development headers and is read only.
#      That is what took 19 packages down with the Seurat attempt.
#
# Every place it looked is printed when nothing works, because the useful next
# step is then to set CONFIDENCEOT_RSCRIPT by hand, not to guess again.

_cot_r_candidates=()
_cot_r_describe=()

_cot_r_add() {
  # $1 = command, $2 = description. Absolute paths must be executable; a bare
  # command or a container invocation is taken on trust and probed below.
  local command=$1 description=$2
  [[ -z "$command" ]] && return 0
  if [[ "$command" == /* && "$command" != *" "* && ! -x "$command" ]]; then
    _cot_r_candidates+=("$command")
    _cot_r_describe+=("$description (absent)")
    return 0
  fi
  _cot_r_candidates+=("$command")
  _cot_r_describe+=("$description")
}

_cot_r_runs() {
  # Can this command run R at all?
  $1 --vanilla -e 'cat(R.version$platform)' > /dev/null 2>&1
}

_cot_r_has_sctransform() {
  $1 --vanilla -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); suppressMessages({library(Matrix); library(sctransform)})' > /dev/null 2>&1
}

if [[ -n "${CONFIDENCEOT_RSCRIPT:-}" ]]; then
  _cot_r_add "$CONFIDENCEOT_RSCRIPT" "caller-supplied"
else
  for _cot_path in \
      "${WORK:-/nonexistent}/envs/r-omics/bin/Rscript" \
      "${WORK:-/nonexistent}/apps/miniforge3/envs/r-omics/bin/Rscript" \
      "${WORK:-/nonexistent}/miniforge3/envs/r-omics/bin/Rscript" \
      "${SCRATCH:-/nonexistent}/conda_envs/r-omics/bin/Rscript" \
      "${SCRATCH:-/nonexistent}/conda_envs/geomx_r_env/bin/Rscript" \
      "${SCRATCH:-/nonexistent}/conda_envs/infercnv_r/bin/Rscript"; do
    _cot_r_add "$_cot_path" "known layout"
  done

  # Anything else on the filesystem. Bounded depth and -type f keep this to
  # seconds even on Lustre; duplicates of the paths above are harmless.
  for _cot_root in "${SCRATCH:-}" "${WORK:-}"; do
    [[ -z "$_cot_root" || ! -d "$_cot_root" ]] && continue
    while IFS= read -r _cot_found; do
      [[ -n "$_cot_found" ]] && _cot_r_add "$_cot_found" "found under $_cot_root"
    done < <(find "$_cot_root" -maxdepth 5 -type f -name Rscript -perm -u+x \
               2> /dev/null | head -n 8)
  done

  # The module system may carry one; loading it changes PATH for the rest of
  # the shell, which is what a caller of this file wants anyway.
  module load Rstats > /dev/null 2>&1 || true
  if command -v Rscript > /dev/null 2>&1; then
    _cot_r_add "$(command -v Rscript)" "PATH"
  fi

  _cot_r_sif=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/r-ver-4.4.sif}
  if [[ -f "$_cot_r_sif" ]]; then
    module load tacc-apptainer > /dev/null 2>&1 || true
    _cot_r_add "apptainer exec -B /scratch -B ${WORK:-/scratch} $_cot_r_sif Rscript" \
               "container"
  fi
fi

# First pass: an R that already has sctransform needs nothing else.
_cot_r_choice=""
_cot_r_source=""
_cot_r_first_runnable=""
_cot_r_first_runnable_source=""
for _cot_index in "${!_cot_r_candidates[@]}"; do
  _cot_command=${_cot_r_candidates[$_cot_index]}
  _cot_description=${_cot_r_describe[$_cot_index]}
  [[ "$_cot_description" == *"(absent)" ]] && continue
  _cot_r_runs "$_cot_command" || continue
  if [[ -z "$_cot_r_first_runnable" ]]; then
    _cot_r_first_runnable=$_cot_command
    _cot_r_first_runnable_source=$_cot_description
  fi
  if _cot_r_has_sctransform "$_cot_command"; then
    _cot_r_choice=$_cot_command
    _cot_r_source=$_cot_description
    break
  fi
done

if [[ -z "$_cot_r_choice" && -n "$_cot_r_first_runnable" ]]; then
  _cot_r_choice=$_cot_r_first_runnable
  _cot_r_source=$_cot_r_first_runnable_source
fi

if [[ -z "$_cot_r_choice" ]]; then
  export CONFIDENCEOT_R_STATUS="no-r"
  echo "   no working R found. Looked at:" >&2
  for _cot_index in "${!_cot_r_candidates[@]}"; do
    echo "     ${_cot_r_candidates[$_cot_index]}  [${_cot_r_describe[$_cot_index]}]" >&2
  done
  echo "   set CONFIDENCEOT_RSCRIPT to an Rscript and source this again" >&2
  return 0 2> /dev/null || exit 0
fi

export CONFIDENCEOT_RSCRIPT="$_cot_r_choice"
echo "   R via $_cot_r_source: $CONFIDENCEOT_RSCRIPT"

# The library directory. R is version and platform specific about compiled
# packages, so it carries both, exactly as the lab's rstudio.slurm writes it.
# Order: the caller's, then the lab's layout if it is there, then one of ours
# under $SCRATCH -- which is what happens here, since $WORK/Rlibs does not
# exist. Ours, rather than the conda environment's own, so installing for this
# comparison cannot disturb the project that environment was built for.
if [[ -n "${R_LIBS_USER:-}" ]]; then
  echo "   R_LIBS_USER $R_LIBS_USER (from the caller)"
else
  _cot_r_platform=$($CONFIDENCEOT_RSCRIPT --vanilla -e 'cat(R.version$platform)' 2> /dev/null)
  _cot_r_version=$($CONFIDENCEOT_RSCRIPT --vanilla -e 'cat(paste(R.version$major, strsplit(R.version$minor, ".", fixed = TRUE)[[1]][1], sep = "."))' 2> /dev/null)
  if [[ -z "$_cot_r_platform" || -z "$_cot_r_version" ]]; then
    echo "   note: could not ask R for its platform; leaving R_LIBS_USER unset"
  else
    _cot_r_leaf="${_cot_r_platform}-library/${_cot_r_version}"
    if [[ -n "${WORK:-}" && -d "$WORK/Rlibs/$_cot_r_leaf" ]]; then
      export R_LIBS_USER="$WORK/Rlibs/$_cot_r_leaf"
    else
      export R_LIBS_USER="${CONFIDENCEOT_R_LIBS:-${SCRATCH:-/tmp}/OT_project/r_libs}/$_cot_r_leaf"
    fi
    echo "   R_LIBS_USER $R_LIBS_USER"
  fi
fi

# Re-check with R_LIBS_USER in place: the first pass ran before it was known,
# so a package installed there by an earlier run would have been missed.
if _cot_r_has_sctransform "$CONFIDENCEOT_RSCRIPT"; then
  export CONFIDENCEOT_R_STATUS="ready"
  $CONFIDENCEOT_RSCRIPT --vanilla -e '.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths())); cat("   ok   sctransform", as.character(packageVersion("sctransform")), "\n")' 2> /dev/null
else
  export CONFIDENCEOT_R_STATUS="no-packages"
  echo "   this R runs but sctransform will not load. Install and verify it:"
  echo "     sbatch scripts/tacc/prepare_r_sctransform.slurm"
  echo "   (it installs into R_LIBS_USER, leaving the shared environment"
  echo "    untouched, then runs the bridge's own vst call)"
fi
