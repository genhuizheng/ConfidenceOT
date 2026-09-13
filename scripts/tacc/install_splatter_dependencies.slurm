#!/bin/bash
#SBATCH -J cot_splat_deps
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -A MCB26031
#SBATCH -o /scratch/10119/ghzheng/OT_project/splatter_install_%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/splatter_install_%j.err

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate "${CONFIDENCEOT_R_ENV:-/scratch/10119/ghzheng/conda_envs/infercnv_r}"
set -u

repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
library=${CONFIDENCEOT_R_LIBRARY:-$repo/.r-library}
mkdir -p "$library"
export R_LIBS_USER="$library"

Rscript - <<'RS'
options(repos = c(CRAN = "https://cloud.r-project.org"))
library <- Sys.getenv("R_LIBS_USER")
if (!requireNamespace("BiocManager", quietly = TRUE)) {
    install.packages("BiocManager", lib = library)
}
BiocManager::install(
    c("splatter", "SingleCellExperiment", "SummarizedExperiment"),
    lib = library,
    ask = FALSE,
    update = FALSE
)
stopifnot(
    requireNamespace("splatter", quietly = TRUE),
    requireNamespace("SingleCellExperiment", quietly = TRUE),
    requireNamespace("SummarizedExperiment", quietly = TRUE),
    requireNamespace("Matrix", quietly = TRUE)
)
cat("Splatter dependencies available in", library, "\n")
RS
