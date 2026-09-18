#!/bin/bash
# Pack the depth screen's summaries into one tarball, ready to scp.
#
# The figure has to be drawn wherever the Splatter benchmark report is, and
# that report is 47 MB and lives on the workstation, so the screen travels to
# the benchmark rather than the other way round. Its summaries are about 130 KB
# for all eleven configurations.
#
# One tarball rather than a glob of files: make_journal_figure.py finds
# configurations by directory, as <label>/depth_null_arm_summary.csv, and
# `scp */depth_null_*` would flatten that into a single directory where every
# configuration's file overwrites the last.
#
# Only the three small per-configuration files go in. Everything else the
# screen wrote -- logs, intermediate matrices -- stays on scratch.
#
# No `set -u`: the explicit checks below catch the defined-but-empty variable
# that -u permits.
#
# Usage:
#   bash scripts/tacc/export_depth_screen.sh
#   bash scripts/tacc/export_depth_screen.sh SCREEN_ROOT OUTPUT_TARBALL
set -eo pipefail

root=${1:-${CONFIDENCEOT_DEPTH_SCREEN_ROOT:-/scratch/10119/ghzheng/OT_project/depth_screen}}
tarball=${2:-${SCRATCH:-/tmp}/depth_screen_summaries.tgz}

for pair in "root:$root" "tarball:$tarball"; do
  name=${pair%%:*}
  value=${pair#*:}
  # A drive letter counts as absolute too, the same way
  # screen_depth_configurations.sh allows it, so this runs on a workstation
  # copy of the screen as well as on the cluster.
  if [[ -z "$value" || ( "$value" != /* && "$value" != [A-Za-z]:* ) ]]; then
    echo "$name must be a non-empty absolute path, found: '$value'" >&2
    exit 2
  fi
done
if [[ ! -d "$root" ]]; then
  echo "no screen root at $root" >&2
  exit 2
fi

members=()
skipped=()
for directory in "$root"/*/; do
  label=$(basename "$directory")
  if [[ ! -f "$directory/depth_null_arm_summary.csv" ]]; then
    skipped+=("$label")
    continue
  fi
  for name in depth_null_arm_summary.csv depth_null_replicates.csv \
              depth_null_report.json; do
    [[ -f "$directory/$name" ]] && members+=("$label/$name")
  done
  echo "   $label"
done

if (( ${#members[@]} == 0 )); then
  echo "nothing to export: no configuration under $root has a summary yet" >&2
  exit 2
fi

tar czf "$tarball" -C "$root" "${members[@]}"

echo
echo "wrote $tarball  ($(du -h "$tarball" | cut -f1))"
if (( ${#skipped[@]} )); then
  echo "no summary, so not in it (${#skipped[@]}): ${skipped[*]:0:12}"
fi
echo
echo "=================================================="
echo "from the workstation, not from here:"
echo
echo "  scp ${USER:-ghzheng}@vista.tacc.utexas.edu:$tarball ."
echo "  mkdir -p depth_screen_from_tacc"
echo "  tar xzf $(basename "$tarball") -C depth_screen_from_tacc"
echo
echo "then draw the figure against that directory:"
echo
echo "  python scripts/make_journal_figure.py \\"
echo "      --benchmark-root benchmark_results/splatter_n1000_5000_10000/report \\"
echo "      --screen-root depth_screen_from_tacc \\"
echo "      --out benchmark_results/journal_figure"
