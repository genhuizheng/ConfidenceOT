#!/bin/bash
#SBATCH -p gg
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 144
#SBATCH -t 06:00:00
#SBATCH -A MCB26031
#SBATCH -J cnvtools_env
# The R environment of the four CNV tools (CNV_TOOL_APIS.md), built on a compute node (aarch64) under /work, which the
# scratch purge does not touch (the old /scratch infercnv_r environment lost libblas). Pinned sources:
#   inferCNV   bioconda bioconductor-infercnv (with jags / r-rjags: the package imports rjags even without the HMM)
#   CopyKAT    navinlabcode/copykat 01924c0
#   SCEVAN     AntonioDeFalco/SCEVAN 5a49b88, after yaGST (miccec/yaGST 56227df); the repository ships x86-64 .o files
#              in src/, removed before building
#   CONICSmat  diazlab/CONICS 053cf26, subfolder CONICSmat, plus zoo (used, not declared); the region files of the
#              repository root are copied to $ENV/share/conics_regions
# Writes $ENV/BUILD_REPORT.txt (versions, an import check per package). Submitted by hand:
#   mkdir -p /work/10119/ghzheng/vista/envs/logs
#   sbatch -o /work/10119/ghzheng/vista/envs/logs/cnvtools_env_%j.out lineage_validation/rna_cnv_clade_20261123/cnv_tools/build_r_env.slurm
# No set -e / -u / pipefail (TACC rules); every step reports its own status.

ENV=${CNVTOOLS_ENV:-/work/10119/ghzheng/vista/envs/cnvtools_r}
CONDA=${CONDA_EXE_PATH:-/work/10119/ghzheng/vista/miniconda3/bin/conda}
SRC=$ENV.src
REPORT=$ENV/BUILD_REPORT.txt
echo "build $(date -Iseconds) on $(hostname) $(uname -m)"
if [[ ! -x "$ENV/bin/R" ]]; then
    "$CONDA" create -y -p "$ENV" -c conda-forge -c bioconda --override-channels \
        "r-base=4.3" bioconductor-infercnv jags r-rjags r-remotes r-biocmanager r-jsonlite r-zoo r-data.table r-matrix \
        r-mixtools r-pheatmap r-beanplot r-rtsne r-mcmcpack r-gplots r-parallelly git compilers make \
        || { echo "conda create failed"; exit 1; }
fi
export PATH="$ENV/bin:$PATH"
mkdir -p "$SRC"
echo "R $(R --version | head -1)" > "$REPORT"

fetch() {  # repo commit dir
    local repo=$1 commit=$2 dir=$SRC/$3
    [[ -d "$dir/.git" ]] || git clone -q "https://github.com/$repo.git" "$dir" || { echo "$repo: clone failed" >> "$REPORT"; return 1; }
    git -C "$dir" checkout -q "$commit" || { echo "$repo: checkout $commit failed" >> "$REPORT"; return 1; }
    echo "$repo $(git -C "$dir" rev-parse --short HEAD)" >> "$REPORT"
}
install_dir() {  # dir: dependencies from CRAN / Bioconductor, then the package itself
    Rscript -e "options(repos = BiocManager::repositories()); remotes::install_deps('$1', dependencies = c('Depends','Imports','LinkingTo'), upgrade = 'never')" \
        >> "$SRC/install.log" 2>&1
    R CMD INSTALL "$1" >> "$SRC/install.log" 2>&1 || echo "$1: R CMD INSTALL failed (see $SRC/install.log)" >> "$REPORT"
}

fetch navinlabcode/copykat 01924c0 copykat && install_dir "$SRC/copykat"
fetch miccec/yaGST 56227df yaGST && install_dir "$SRC/yaGST"
if fetch AntonioDeFalco/SCEVAN 5a49b88 SCEVAN; then
    rm -f "$SRC"/SCEVAN/src/*.o "$SRC"/SCEVAN/src/*.so
    install_dir "$SRC/SCEVAN"
fi
if fetch diazlab/CONICS 053cf26 CONICS; then
    install_dir "$SRC/CONICS/CONICSmat"
    mkdir -p "$ENV/share/conics_regions"
    cp "$SRC"/CONICS/chromosome_*_positions_*.txt "$ENV/share/conics_regions/" 2>/dev/null
    ls "$ENV/share/conics_regions" >> "$REPORT"
fi

for p in infercnv rjags copykat yaGST SCEVAN CONICSmat zoo Matrix jsonlite; do
    Rscript -e "suppressPackageStartupMessages(ok <- requireNamespace('$p', quietly = TRUE)); cat('$p', ok, if (ok) as.character(packageVersion('$p')) else '', '\n')" >> "$REPORT" 2>&1
done
cat "$REPORT"
echo "done $(date -Iseconds)"
