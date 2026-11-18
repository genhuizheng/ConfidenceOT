#!/bin/bash
# OT depth benchmark on TACC (ot_depth_benchmark_20261117/PLAN.md). The user submits; nothing runs by itself.
#
#   bash ot_depth_benchmark_20261117/tacc/submit_otdb.sh [--dry-run] [--after ID] [--hours H] STAGE
#
# STAGE
#   vcc_prepare     VCC 2025 training h5ad -> CSR arrays, cell table, UNITS.tsv, AUDIT.json (read AUDIT.json before fit)
#   vcc_fit         every VCC unit (150 target genes x control references A/B + control vs control), C1 / C2 x 6 solvers
#   norman_prepare  Norman 2019 (scPerturb v1.4) the same way (the local run is the test of this stage)
#   norman_fit
#   splatter_s1     Splatter N = 5,000 paired root, replicates 1-3, every level x case, no couplings saved
#   splatter_s2     the S2 depth-identity conflict on the same root (L0_matched all_shared)
#   learned_vcc     CINEMA-OT / CellOT (run_learned.py) on VCC, in the otbench env (see setup_learned_env.sh)
#   learned_norman
#
# Downloads are not done by a job (compute nodes may have no outbound network); run them on the login node first:
#   vcc     curl (see the README section "Data"), 15,482,497,461 bytes, GCS generation 1765904883947296
#   norman  curl from Zenodo record 13350497, md5 c870e6967d91c017d9da827bab183cd6
# Output root: $OTDB_ROOT (default /scratch/10119/ghzheng/OT_project/ot_depth_benchmark_20261117), one folder per dataset.
# No set -e / -u / pipefail (TACC rules).

dry_run=0; after=""; hours=""; stage=""
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --after) after=$2; shift 2 ;;
        --hours) hours=$2; shift 2 ;;
        *) stage=$1; shift ;;
    esac
done
root=${OTDB_ROOT:-/scratch/10119/ghzheng/OT_project/ot_depth_benchmark_20261117}
data=${OTDB_DATA:-/scratch/10119/ghzheng/OT_project/data}
splatter_root=${OTDB_SPLATTER_ROOT:-/scratch/10119/ghzheng/OT_project/splatter_n5000_default_paired_20261103}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root/logs"
case "$stage" in
    vcc_prepare)    st=perturb;  h=${hours:-2};  args="--dataset vcc --h5ad $data/vcc2025/adata_Training.h5ad --out $root/vcc --stage prepare" ;;
    vcc_fit)        st=perturb;  h=${hours:-12}; args="--dataset vcc --out $root/vcc --stage fit --jobs ${OTDB_JOBS:-48}" ;;
    norman_prepare) st=perturb;  h=${hours:-1};  args="--dataset norman --h5ad $data/norman2019/NormanWeissman2019_filtered.h5ad --out $root/norman --stage prepare" ;;
    norman_fit)     st=perturb;  h=${hours:-8};  args="--dataset norman --out $root/norman --stage fit --jobs ${OTDB_JOBS:-48}" ;;
    splatter_s1)    st=splatter; h=${hours:-24}; args="--stage s1 --data-root $splatter_root --conditions-subdir N5000 --replicates 1 2 3 --out $root/splatter_n5000 --no-save-couplings --jobs ${OTDB_JOBS:-24}" ;;
    splatter_s2)    st=splatter; h=${hours:-24}; args="--stage s2 --data-root $splatter_root --conditions-subdir N5000 --replicates 1 2 3 --out $root/splatter_n5000 --no-save-couplings --jobs ${OTDB_JOBS:-24}" ;;
    learned_vcc)    st=learned;  h=${hours:-24}; args="--perturb-root $root/vcc --jobs ${OTDB_JOBS:-24}" ;;
    learned_norman) st=learned;  h=${hours:-24}; args="--perturb-root $root/norman --jobs ${OTDB_JOBS:-24}" ;;
    *) echo "unknown stage '$stage'"; exit 1 ;;
esac
dep=""; [[ -n "$after" ]] && dep="--dependency=afterok:$after"
cmd=(sbatch --parsable -J "otdb_$stage" -t "$(printf '%02d:00:00' "$h")" -o "$root/logs/${stage}_%j.out" $dep
     --export=ALL,OTDB_STAGE=$st,OTDB_ARGS="$args" "$here/otdb.slurm")
if (( dry_run )); then echo "DRY RUN ${cmd[*]}"; exit 0; fi
id=$("${cmd[@]}" | tail -n 1); echo "submitted $stage: $id"
