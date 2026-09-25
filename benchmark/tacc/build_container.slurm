#!/bin/bash
#SBATCH -J cot_bench_sif
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/sif_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/benchmark_technical/logs/sif_%j.err
#
# Build the Splatter image from benchmark/tacc/splatter.def.
#
# On a compute node, because apptainer is refused on the login nodes here.
# With --fakeroot, because the %post section installs Debian packages and
# there is no root on a shared cluster; if fakeroot is not configured for this
# account the build fails with a clear message and the fallback is a remote
# build, which is printed rather than attempted silently.
#
# The build is the verification. splatter.def ends by running the same
# splatSimulate call the benchmark makes and checking the object that comes
# back, so an image that finishes building is an image that can generate.

# No set -u. TACC's module files and conda's shell functions read
# variables they do not define -- module load tacc-apptainer sources a
# completion script that reads BASH_COMPLETION_DEBUG -- and under set -u
# that kills the job in seconds with nothing in the log. Every expansion
# below carries its own default.
set -eo pipefail
source /home1/10119/ghzheng/.bashrc

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
root=${CONFIDENCEOT_BENCH_ROOT:-/scratch/10119/ghzheng/OT_project/benchmark_technical}
sif=${CONFIDENCEOT_R_SIF:-/scratch/10119/ghzheng/containers/confidenceot-splatter.sif}
definition=$repo/benchmark/tacc/splatter.def
force=${CONFIDENCEOT_BENCH_FORCE:-0}

mkdir -p "$root/logs" "$(dirname "$sif")"
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-/scratch/10119/ghzheng/apptainer_cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$SCRATCH/apptainer_tmp}
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

if [[ "$force" != "1" && -f "$sif" ]]; then
    echo "$sif already exists; set CONFIDENCEOT_BENCH_FORCE=1 to rebuild"
    exit 0
fi

module load tacc-apptainer
echo "architecture: $(uname -m)"
echo "building $sif from $definition"

if ! apptainer build --fakeroot "$sif" "$definition"; then
    echo >&2
    echo "the build failed. Read the message above before choosing:" >&2
    echo "  setgroups / Method http has died  -> apt dropping privileges "\
         "inside the namespace; %post already disables its sandbox, so this "\
         "would mean the fix did not apply" >&2
    echo "  fakeroot not configured           -> use one of:" >&2
    echo "  apptainer build --remote $sif $definition   # needs a sylabs token" >&2
    echo "  or keep using the conda library, which already has splatter:" >&2
    echo "  CONFIDENCEOT_R_LIBRARY=$repo/.r-library" >&2
    exit 5
fi

echo
echo "built $sif"

# The check, against the finished image. Loading the packages is weaker than
# it looks -- Bioconductor packages built from source on aarch64 can install
# cleanly and misbehave at run time -- so this runs the call the benchmark
# makes, with the same arguments, and looks at the object that comes back.
#
# Outside the build on purpose: a failure here leaves the image on disk, so
# fixing the check costs a minute rather than another twenty.
if ! apptainer exec -B /scratch "$sif" R -q --no-save <<'VERIFY'
suppressPackageStartupMessages({
    library(splatter)
    library(SingleCellExperiment)
    library(SummarizedExperiment)
    library(Matrix)
    library(jsonlite)
})
params <- setParams(newSplatParams(),
                    batchCells     = c(50, 50),
                    batch.facLoc   = 0,
                    batch.facScale = 0,
                    group.prob     = rep(1 / 5, 5),
                    dropout.type   = "none",
                    nGenes         = 200,
                    seed           = 1)
sim <- splatSimulate(params, method = "groups", verbose = FALSE)
matrix <- counts(sim)
stopifnot(
    identical(dim(matrix), c(200L, 100L)),
    sum(matrix) > 0,
    length(unique(colData(sim)$Group)) == 5,
    length(unique(colData(sim)$Batch)) == 2,
    all(is.finite(rowData(sim)$GeneMean))
)
writeMM(as(matrix, "CsparseMatrix"), tempfile())
cat("splatter", as.character(packageVersion("splatter")),
    "simulated", sum(matrix), "counts on", R.version$platform, "\n")
VERIFY
then
    echo >&2
    echo "the image built but does not generate. It is still at $sif, so "\
         "re-run this job to check it again after a fix; nothing has to be "\
         "rebuilt." >&2
    exit 6
fi
