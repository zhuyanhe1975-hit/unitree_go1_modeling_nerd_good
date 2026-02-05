# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `results/real_csv_closed_loop_dataset_20260204_221555.npz`
- weights: `results/real_csv_closed_loop_model_gpu_20260204_221555.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.0163142 | 0.0321604 | 0.408866 | 1.61586 |
| pos_sweep | 300 | 0.0121726 | 0.0244511 | 0.328566 | 1.13722 |
| vel_step | 300 | 0.0184984 | 0.0585689 | 0.527247 | 1.74539 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
