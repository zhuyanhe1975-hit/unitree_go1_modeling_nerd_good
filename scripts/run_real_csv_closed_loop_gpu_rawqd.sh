#!/usr/bin/env bash
set -euo pipefail

# GPU training/eval using RAW speed from CSV (dq_rad_s) as qd ground-truth.
#
# Usage:
#   conda activate nerd_py310
#   bash scripts/run_real_csv_closed_loop_gpu_rawqd.sh

cfg="configs/real_csv_closed_loop_gpu_rawqd.json"
ts="$(date +%Y%m%d_%H%M%S)"

export PYTHONPATH="."

python3 scripts/prepare_closed_loop_csv.py \
  --config "${cfg}" \
  --out "results/real_csv_closed_loop_dataset_rawqd_${ts}.npz" \
  --stats "results/real_csv_closed_loop_stats_rawqd_${ts}.npz" \
  --qd_col dq_rad_s \
  --qd_filter_hz 0 \
  --qd_use_raw

python3 scripts/train_closed_loop_csv.py \
  --config "${cfg}" \
  --dataset "results/real_csv_closed_loop_dataset_rawqd_${ts}.npz" \
  --out "results/real_csv_closed_loop_model_gpu_rawqd_${ts}.pt" \
  --csv "real_data/coverage_capture_20260204_103129.csv" \
  --qd_col dq_rad_s

python3 scripts/eval_closed_loop_csv.py \
  --config "${cfg}" \
  --csv "real_data/coverage_capture_20260204_103129.csv" \
  --qd_col dq_rad_s \
  --qd_filter_hz 0 --qd_use_raw \
  --dataset "results/real_csv_closed_loop_dataset_rawqd_${ts}.npz" \
  --weights "results/real_csv_closed_loop_model_gpu_rawqd_${ts}.pt" \
  --device cuda \
  --stages sine,pos_sweep,vel_step \
  --horizon_steps 300 \
  --baseline \
  --out_md "results/summary_closed_loop_csv_gpu_rawqd_${ts}.md" \
  --plot_reversals \
  --plot_horizon_steps 4000 \
  --reversal_window_s 0.6 \
  --out_png "results/plot_reversals_sine_rawqd_${ts}.png" \
  --plot_full_cycle \
  --out_full_png "results/plot_full_cycle_sine_rawqd_${ts}.png"

echo "done: results/*_${ts}.*"

