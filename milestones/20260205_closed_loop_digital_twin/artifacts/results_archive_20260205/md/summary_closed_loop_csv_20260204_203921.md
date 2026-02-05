# Closed-loop digital twin eval summary

- csv: `real_data/coverage_capture_20260204_103129.csv`
- dataset: `/tmp/unitree_go1_modeling_nerd_good_runs/real_csv_closed_loop_dataset.npz`
- weights: `/tmp/unitree_go1_modeling_nerd_good_runs/real_csv_closed_loop_model_smoke.pt`
- stages: `sine, pos_sweep, vel_step`
- horizon_steps: `300`
- device: `cpu`

### Model (open-loop)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.0648171 | 0.129038 | 0.627322 | 1.66096 |
| pos_sweep | 300 | 0.0269583 | 0.0662364 | 0.416258 | 1.04612 |
| vel_step | 300 | 1.35925 | 3.72447 | 4.52355 | 14.0567 |

### Baseline (open-loop, constant qd)

| stage | horizon | q_rmse | q_maxabs | qd_rmse | qd_maxabs |
| --- | --- | --- | --- | --- | --- |
| sine | 300 | 0.844748 | 1.63843 | 1.07724 | 2.18439 |
| pos_sweep | 300 | 0.10984 | 0.148609 | 0.330418 | 1.03084 |
| vel_step | 300 | 0.839679 | 2.33328 | 2.49085 | 6.2832 |
