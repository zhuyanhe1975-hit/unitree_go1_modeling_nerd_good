# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `results/real_csv_closed_loop_dataset_20260204_220642.npz`
- weights: `results/real_csv_closed_loop_model_gpu_20260204_220642.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.00788733 | 0.0201546 | 0.342664 | 1.01068 |
| pos_sweep | 300 | 0.0216842 | 0.038451 | 0.273334 | 0.818385 |
| vel_step | 300 | 0.00895902 | 0.022559 | 0.548618 | 2.3982 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
