# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `/tmp/unitree_go1_modeling_nerd_good_runs/real_csv_closed_loop_dataset.npz`
- weights: `/tmp/unitree_go1_modeling_nerd_good_runs/real_csv_closed_loop_model_smoke.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cuda`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.0689586 | 0.148342 | 0.720431 | 1.69146 |
| pos_sweep | 300 | 0.0582572 | 0.131694 | 0.551615 | 1.45973 |
| vel_step | 300 | 0.985876 | 2.76534 | 2.70656 | 9.43199 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
