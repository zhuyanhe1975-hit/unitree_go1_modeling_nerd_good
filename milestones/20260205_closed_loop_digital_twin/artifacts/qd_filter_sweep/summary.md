# qd filter cutoff sweep

- cfg: `configs/real_csv_closed_loop_sweep_gpu.json`
- device: `cuda`
- horizon_steps: `300`

- filter: `zero_phase_one_pole`
- score: `qd_rmse + 0.25 * qd_filter_delta_rmse` (smaller is better; penalizes over-smoothing)

| cutoff_hz | q_rmse (sine) | qd_rmse (sine) | qd_filter_delta_rmse | score | reversal plot | full-cycle plot |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.00852063 | 0.178549 | 0.277711 | 0.247977 | `cutoff_20Hz/plot_reversals_sine.png` | `cutoff_20Hz/plot_full_cycle_sine.png` |
| 30 | 0.00884696 | 0.204595 | 0.235547 | 0.263482 | `cutoff_30Hz/plot_reversals_sine.png` | `cutoff_30Hz/plot_full_cycle_sine.png` |
| 10 | 0.0434065 | 0.176488 | 0.353054 | 0.264752 | `cutoff_10Hz/plot_reversals_sine.png` | `cutoff_10Hz/plot_full_cycle_sine.png` |
| 40 | 0.00685226 | 0.215897 | 0.206922 | 0.267628 | `cutoff_40Hz/plot_reversals_sine.png` | `cutoff_40Hz/plot_full_cycle_sine.png` |
| 60 | 0.00711303 | 0.239058 | 0.168834 | 0.281266 | `cutoff_60Hz/plot_reversals_sine.png` | `cutoff_60Hz/plot_full_cycle_sine.png` |
| 15 | 0.00685982 | 0.240937 | 0.30854 | 0.318072 | `cutoff_15Hz/plot_reversals_sine.png` | `cutoff_15Hz/plot_full_cycle_sine.png` |

**Recommendation:** cutoff_hz=`20` (min score).
- reversal: `cutoff_20Hz/plot_reversals_sine.png`
- full-cycle: `cutoff_20Hz/plot_full_cycle_sine.png`
