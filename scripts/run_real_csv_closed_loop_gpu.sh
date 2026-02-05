#!/usr/bin/env bash
set -euo pipefail

# Run inside an activated conda env, e.g.:
#   conda activate nerd_py310
#
# This script assumes:
#   - tau_ff = 0 (no commanded feedforward torque during capture)
#   - configs/real_csv_closed_loop_gpu.json uses sim.device=cuda for training

cfg="configs/real_csv_closed_loop_gpu.json"
ts="$(date +%Y%m%d_%H%M%S)"

export PYTHONPATH="."

python3 scripts/prepare_closed_loop_csv.py \
  --config "${cfg}" \
  --out "results/real_csv_closed_loop_dataset_${ts}.npz" \
  --stats "results/real_csv_closed_loop_stats_${ts}.npz"

python3 scripts/train_closed_loop_csv.py \
  --config "${cfg}" \
  --dataset "results/real_csv_closed_loop_dataset_${ts}.npz" \
  --out "results/real_csv_closed_loop_model_gpu_${ts}.pt"

python3 scripts/eval_closed_loop_csv.py \
  --config "${cfg}" \
  --dataset "results/real_csv_closed_loop_dataset_${ts}.npz" \
  --weights "results/real_csv_closed_loop_model_gpu_${ts}.pt" \
  --device cuda \
  --stages sine,pos_sweep,vel_step \
  --horizon_steps 300 \
  --baseline \
  --out_md "results/summary_closed_loop_csv_gpu_${ts}.md" \
  --plot_reversals \
  --plot_horizon_steps 4000 \
  --reversal_window_s 0.6 \
  --out_png "results/plot_reversals_sine_gpu_${ts}.png" \
  --plot_full_cycle \
  --out_full_png "results/plot_full_cycle_sine_gpu_${ts}.png"

echo "done: results/*_${ts}.*"

