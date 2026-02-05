#!/usr/bin/env bash
set -euo pipefail

# GPU training/eval using a FILTERED speed column from CSV (dq_filt_rad_s) as qd ground-truth.
# This script:
#   1) Creates a copy of the CSV with an extra filtered speed column (does not modify the original)
#   2) Prepares dataset using qd_col=dq_filt_rad_s (no additional filtering)
#   3) Trains with open-loop sine metric selection
#   4) Evaluates + saves plots
#
# Usage:
#   conda activate nerd_py310
#   bash scripts/run_real_csv_closed_loop_gpu_qdfilt.sh
#
# Optional env overrides:
#   SRC_CSV=real_data/xxx.csv
#   CUTOFF_HZ=20
#   METHOD=zero_phase_one_pole

cfg="configs/real_csv_closed_loop_gpu.json"
ts="$(date +%Y%m%d_%H%M%S)"

SRC_CSV="${SRC_CSV:-real_data/coverage_capture_20260204_103129.csv}"
METHOD="${METHOD:-zero_phase_one_pole}"
CUTOFF_HZ="${CUTOFF_HZ:-20}"

export PYTHONPATH="."

filt_csv="results/$(basename "${SRC_CSV%.csv}")_with_qd_filt.csv"

if [[ ! -f "${filt_csv}" ]]; then
  python3 scripts/add_filtered_speed_column.py \
    --config "${cfg}" \
    --csv "${SRC_CSV}" \
    --method "${METHOD}" \
    --cutoff_hz "${CUTOFF_HZ}"
else
  echo "use existing: ${filt_csv}"
fi

ds="results/real_csv_closed_loop_dataset_qdfilt_${ts}.npz"
st="results/real_csv_closed_loop_stats_qdfilt_${ts}.npz"
w="results/real_csv_closed_loop_model_gpu_qdfilt_${ts}.pt"
md="results/summary_closed_loop_csv_gpu_qdfilt_${ts}.md"
png="results/plot_reversals_sine_qdfilt_${ts}.png"
cyc="results/plot_full_cycle_sine_qdfilt_${ts}.png"

python3 scripts/prepare_closed_loop_csv.py \
  --config "${cfg}" \
  --csv "${filt_csv}" \
  --qd_col dq_filt_rad_s \
  --qd_filter_hz 0 \
  --qd_use_raw \
  --out "${ds}" \
  --stats "${st}"

python3 scripts/train_closed_loop_csv.py \
  --config "${cfg}" \
  --dataset "${ds}" \
  --out "${w}" \
  --csv "${filt_csv}" \
  --qd_col dq_filt_rad_s

python3 scripts/eval_closed_loop_csv.py \
  --config "${cfg}" \
  --csv "${filt_csv}" \
  --qd_col dq_filt_rad_s \
  --qd_filter_hz 0 --qd_use_raw \
  --dataset "${ds}" \
  --weights "${w}" \
  --device cuda \
  --stages sine,pos_sweep,vel_step \
  --horizon_steps 300 \
  --baseline \
  --out_md "${md}" \
  --plot_reversals \
  --plot_horizon_steps 4000 \
  --reversal_window_s 0.6 \
  --out_png "${png}" \
  --plot_full_cycle \
  --out_full_png "${cyc}"

echo "done: ${ds}"
echo "done: ${w}"
echo "done: ${md}"
echo "done: ${png}"
echo "done: ${cyc}"

