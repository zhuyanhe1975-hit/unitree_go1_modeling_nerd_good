# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `results/real_csv_closed_loop_dataset_20260204_222221.npz`
- weights: `results/real_csv_closed_loop_model_gpu_20260204_222221.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.0085944 | 0.0231843 | 0.348294 | 1.03555 |
| pos_sweep | 300 | 0.0055186 | 0.0139455 | 0.29453 | 1.02072 |
| vel_step | 300 | 0.0066798 | 0.0283421 | 0.566811 | 2.66946 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
