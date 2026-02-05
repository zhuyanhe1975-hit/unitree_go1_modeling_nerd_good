# Closed-loop digital twin eval summary

- csv: `results/coverage_capture_20260204_103129_with_qd_filt.csv`
- dataset: `results/real_csv_closed_loop_dataset.npz`
- weights: `results/real_csv_closed_loop_model_gpu.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`
- qd_col: `dq_filt_rad_s` (raw plot uses `dq_rad_s` if present)
- qd_filter: `zero_phase_one_pole` @ `0.0` Hz, use_filtered=`False`
- qd_filter_delta_rmse(raw->filt): `0.277711` rad/s

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.00840644 | 0.0230517 | 0.153712 | 0.352382 |
| pos_sweep | 300 | 0.00670265 | 0.0193656 | 0.126362 | 0.430791 |
| vel_step | 300 | 0.00725534 | 0.0253631 | 0.303314 | 1.60454 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.69239 | 1.37205 | 0.872434 | 1.43571 |
| pos_sweep | 300 | 0.0729265 | 0.101166 | 0.257875 | 0.746252 |
| vel_step | 300 | 0.745709 | 2.13058 | 2.36037 | 4.92485 |
