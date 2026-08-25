#!/bin/bash
#SBATCH -J cot_bin_gsea
#SBATCH -o /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_gsea_scanpy/logs/%j.out
#SBATCH -e /scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_gsea_scanpy/logs/%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH -t 02:00:00
#SBATCH -A MCB26031
#SBATCH --mail-user=genhuizheng@utexas.edu
#SBATCH --mail-type=FAIL,END

set -eo pipefail
source /home1/10119/ghzheng/.bashrc
conda activate /scratch/10119/ghzheng/conda_envs/worldmodel_withconfidenceot

repo=/scratch/10119/ghzheng/OT_project/code/ConfidenceOT
deg_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg_scanpy
gmt=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/gene_sets/msigdb_2025.1_Mm/mouse_go_bp_hallmark_wikipathways_v2025.1.Mm.symbols.gmt
gsea_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_gsea_scanpy
figure_root=/scratch/10119/ghzheng/OT_project/mouse_embryo_results/spatial_bin_deg_gsea_figures

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd "$repo"
mkdir -p "$gsea_root/logs" "$figure_root"
test -s "$gmt" || { echo "Missing GMT: $gmt" >&2; exit 1; }

python mouse_embryo/10_run_spatial_bin_gseapy.py \
  "$deg_root" \
  "$gmt" \
  "$gsea_root" \
  --permutations 2000 \
  --threads 16 \
  --minimum-size 10 \
  --maximum-size 500

python mouse_embryo/11_visualize_spatial_bin_programs.py \
  "$deg_root" \
  "$figure_root" \
  --gsea-root "$gsea_root"

echo "COMPLETE Mouse MSigDB GSEA and figures"
