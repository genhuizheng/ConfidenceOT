#!/bin/bash
# Four more CNV tools on the 15 lineage CNV units (11 KP-Tracer mice, 4 Quinn experiments), for the RNA + CNV arms of
# submit_rna_cnv.sh --tool (user 2026-11-23). The user submits; nothing runs by itself.
#
#   bash lineage_validation/rna_cnv_clade_20261123/cnv_tools/submit_cnv_tools.sh [--dry-run] env|export|status
#   bash lineage_validation/rna_cnv_clade_20261123/cnv_tools/submit_cnv_tools.sh [--dry-run] run TOOL [TOOL ...]
#
# env     build_r_env.slurm: the R environment under /work (6 h, one gg node)
# export  01_export_r_inputs.py on a gg node (the venv): the infercnvpy units -> $base/cnv_inputs/<dataset>/<patient>/
# run     one gg job per tool: 02_run_cnv_tool.R on every unit, $par units at once with $threads threads each ->
#         $base/cnv_exports/<tool>/<dataset>/<patient>/; a unit with SUCCESS is skipped, so a resubmission continues
# status  per tool: units with SUCCESS / FAILED and the coverage of each (observation cells exported of all)
# TOOL    infercnv | copykat | scevan | conicsmat
# CNVTOOLS_ONLY="3466 30k" runs only those units (a pilot); the full run later skips them (SUCCESS)
# No set -e / -u / pipefail (TACC rules).

dry_run=0; stage=""; tools=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        env|export|run|status) stage=$1; shift ;;
        infercnv|copykat|scevan|conicsmat) tools+=("$1"); shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$stage" ]] || { echo "usage: bash $0 [--dry-run] env|export|status | run TOOL..." >&2; exit 2; }
repo=${CONFIDENCEOT_REPO:-/scratch/10119/ghzheng/OT_project/code/ConfidenceOT}
here=$repo/lineage_validation/rna_cnv_clade_20261123/cnv_tools
base=${RNACNV_BASE:-/scratch/10119/ghzheng/OT_project/v3_grid_rnacnv_20261123}
venv=${V3_VENV:-/scratch/10119/ghzheng/OT_project/envs/otdb_otbench}
renv=${CNVTOOLS_ENV:-/work/10119/ghzheng/vista/envs/cnvtools_r}
kp_units=/scratch/10119/ghzheng/OT_project/kp_tracer_confidenceot/cnv_units/infercnvpy/units.tsv
qn_units=/scratch/10119/ghzheng/OT_project/quinn2021_confidenceot/cnv_units/infercnvpy/units.tsv
par=${CNVTOOLS_PAR:-4}; threads=${CNVTOOLS_THREADS:-32}; hours=${CNVTOOLS_HOURS:-48}
only=${CNVTOOLS_ONLY:-}   # optional pilot: space-separated patients (e.g. "3466 30k"); empty = every unit
logs=$base/logs/cnv_tools

sub() { if (( dry_run )); then echo "DRY RUN  sbatch $*" | cut -c1-600 >&2; echo DRY; else sbatch --parsable "$@"; fi; }
(( dry_run )) || mkdir -p "$logs" /work/10119/ghzheng/vista/envs/logs || exit 1

case $stage in
env)
    sub -o /work/10119/ghzheng/vista/envs/logs/cnvtools_env_%j.out "$here/build_r_env.slurm"
    ;;
export)
    sub -p gg -N 1 -n 1 -c 16 -t 03:00:00 -A MCB26031 -J cnv_export -o "$logs/export_%j.out" \
        --wrap "module load gcc python3/3.11.8 2> /dev/null || module load python3; export PATH=$venv/bin:\$PATH OMP_NUM_THREADS=4; cd $repo && python $here/01_export_r_inputs.py --units $kp_units $qn_units --out $base/cnv_inputs"
    ;;
run)
    (( ${#tools[@]} )) || { echo "run needs at least one TOOL" >&2; exit 2; }
    [[ -x "$renv/bin/Rscript" ]] || (( dry_run )) || { echo "missing $renv/bin/Rscript (env first)" >&2; exit 1; }
    queued=$(squeue -u "$USER" -h -r 2> /dev/null | wc -l)
    (( dry_run || queued + ${#tools[@]} <= 40 )) || { echo "$queued queued; ${#tools[@]} more would pass the cap of 40" >&2; exit 3; }
    for t in "${tools[@]}"; do
        # one line per unit: dataset patient species; Quinn LM0 has no OT unit
        loop="export PATH=$renv/bin:\$PATH OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; cat $kp_units $qn_units | awk -F'\t' -v only=\"$only\" '\$2 != \"LM0\" && (only == \"\" || index(\" \" only \" \", \" \" \$2 \" \")) {print \$1, \$2, (\$1 == \"KPTracer\" ? \"mouse\" : \"human\")}' | xargs -P $par -L 1 bash -c 'mkdir -p $base/cnv_exports/$t/\$0 && Rscript $here/02_run_cnv_tool.R $t $base/cnv_inputs/\$0/\$1 $base/cnv_exports/$t/\$0/\$1 \$2 $threads > $logs/${t}_\$0_\$1.log 2>&1 && echo DONE \$0 \$1 || echo FAILED \$0 \$1'"
        id=$(sub -p gg -N 1 -n 1 -c 144 -t "$hours:00:00" -A MCB26031 -J "cnv_$t" -o "$logs/run_${t}_%j.out" --wrap "$loop") || exit 1
        echo "$t job $id"
    done
    ;;
status)
    for t in infercnv copykat scevan conicsmat; do
        d=$base/cnv_exports/$t
        [[ -d "$d" ]] || { echo "$t: not run"; continue; }
        echo "$t: SUCCESS $(ls $d/*/*/SUCCESS 2> /dev/null | wc -l), FAILED $(ls $d/*/*/FAILED 2> /dev/null | wc -l)"
        for c in $d/*/*/coverage.json; do [[ -f "$c" ]] && echo "  $(basename $(dirname $(dirname $c)))/$(basename $(dirname $c)): $(tr -d '\n ' < $c)"; done
        for f in $d/*/*/FAILED; do [[ -f "$f" ]] && echo "  FAILED $(basename $(dirname $f)): $(head -c 300 $f)"; done
    done
    ;;
esac
