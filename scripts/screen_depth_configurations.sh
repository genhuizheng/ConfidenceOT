#!/bin/bash
# Screen every candidate depth treatment against the same simulated ground
# truth, so "which one works" is a table rather than an argument.
#
# The configuration that matters most is `rank_ds`: ranking *plus* read
# equalisation. The simulation had no equalisation step at all, so its
# gene-ranking arm measured ranking alone -- which is why that arm still
# carried a 0.35 depth effect at high depth spread while the production cancer
# pipeline reaches 0.49 on real prostate data. The two were never the same
# thing.
#
# `cosine` isolates the PI's suggestion: L2-normalising each cell discards
# magnitude and nothing else, so it should help where depth acts as a scale and
# not where it acts through dropout or composition.
#
# No `set -u`: the explicit path check below catches the defined-but-empty
# variable that -u permits.
#
# Usage:
#   bash scripts/screen_depth_configurations.sh OUTPUT_ROOT [extra args...]
set -eo pipefail

root=${1:?"usage: screen_depth_configurations.sh OUTPUT_ROOT [extra args]"}
shift || true

if [[ "$root" != /* && "$root" != [A-Za-z]:* ]]; then
  echo "OUTPUT_ROOT must be an absolute path, found: '$root'" >&2
  exit 2
fi

repo=${CONFIDENCEOT_REPO:-$(cd "$(dirname "$0")/.." && pwd)}
export PYTHONPATH="$repo/src:$repo/cancer_metastasis:$repo${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python=${PYTHON:-python}
script="$repo/scripts/validate_depth_null_specificity.py"

# label | flags
configurations=(
  "logcpm|--representation log_cpm"
  "logcpm_ds|--representation log_cpm --equalise-depth"
  "logcpm_cos|--representation log_cpm --cost cosine"
  "rank256|--representation rank_value --rank-top-n 256"
  "rank256_ds|--representation rank_value --rank-top-n 256 --equalise-depth"
  "rank256_ds_cos|--representation rank_value --rank-top-n 256 --equalise-depth --cost cosine"
  "pearson_ds|--representation pearson_residuals --equalise-depth"
  "pearson_ds_cos|--representation pearson_residuals --equalise-depth --cost cosine"
)

mkdir -p "$root"
for entry in "${configurations[@]}"; do
  label=${entry%%|*}
  flags=${entry#*|}
  out="$root/$label"
  if [[ -f "$out/depth_null_arm_summary.csv" ]]; then
    echo "== $label already done, skipping"
    continue
  fi
  echo "=================================================="
  echo "== $label   $flags"
  # shellcheck disable=SC2086
  "$python" "$script" "$out" $flags "$@" > "$root/$label.log" 2>&1 \
    || { echo "FAILED $label, see $root/$label.log"; continue; }
  echo "   done -> $out/depth_null_arm_summary.csv"
done

echo
echo "== collecting"
"$python" - "$root" <<'PY'
import glob, os, sys
import pandas as pd
root = sys.argv[1]
frames = []
for path in sorted(glob.glob(os.path.join(root, "*", "depth_null_arm_summary.csv"))):
    frame = pd.read_csv(path)
    frame.insert(0, "configuration", os.path.basename(os.path.dirname(path)))
    frames.append(frame)
if not frames:
    print("no summaries written"); raise SystemExit(1)
all_arms = pd.concat(frames, ignore_index=True)
all_arms.to_csv(os.path.join(root, "screen_all_arms.csv"), index=False)

# The headline: how much depth effect is left, per configuration and arm.
key = "auc_total_counts"
if key in all_arms:
    all_arms["depth_effect"] = (all_arms[key] - 0.5).abs()
    wide = all_arms.pivot_table(index="configuration", columns="arm",
                                values="depth_effect", aggfunc="median")
    order = [c for c in ("homogeneous_depth_cv0", "homogeneous_depth_cv_low",
                         "homogeneous_depth_cv_mid", "homogeneous_depth_cv_high",
                         "perturbed_depth_cv0") if c in wide.columns]
    wide = wide[order]
    pd.set_option("display.width", 200)
    print("\nsize of the depth effect (|AUC - 0.5|, lower is better):")
    print(wide.round(3).to_string())
    wide.to_csv(os.path.join(root, "screen_depth_effect.csv"))
    if "perturbed_depth_cv0" in all_arms.arm.unique():
        power = all_arms[all_arms.arm.eq("perturbed_depth_cv0")]
        cols = [c for c in ("perturbed_f1", "perturbed_recall",
                            "perturbed_precision") if c in power.columns]
        if cols:
            print("\npositive control, does it still detect the known 20%:")
            print(power.groupby("configuration")[cols].median().round(3).to_string())
PY
