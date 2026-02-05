# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `results/real_csv_closed_loop_dataset.npz`
- weights: `results/real_csv_closed_loop_model_smoke.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cpu`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.0346293 | 0.0692041 | 0.698512 | 1.69482 |
| pos_sweep | 300 | 0.0311979 | 0.0645783 | 0.681943 | 1.7788 |
| vel_step | 300 | 0.9441 | 2.68339 | 9.29227 | 28.9745 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
