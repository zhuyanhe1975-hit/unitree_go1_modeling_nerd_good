# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `results/real_csv_closed_loop_dataset_rawqd_20260205_094831.npz`
- weights: `results/real_csv_closed_loop_model_gpu_rawqd_20260205_094831.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`
- qd_col: `dq_rad_s` (raw plot uses `dq_rad_s` if present)
- qd_filter: `zero_phase_one_pole` @ `0.0` Hz, use_filtered=`False`
- qd_filter_delta_rmse(raw->filt): `0` rad/s

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.00995438 | 0.0211509 | 0.336256 | 0.929608 |
| pos_sweep | 300 | 0.00683041 | 0.0173323 | 0.272666 | 0.782646 |
| vel_step | 300 | 0.00870694 | 0.0251252 | 0.501115 | 1.39099 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
