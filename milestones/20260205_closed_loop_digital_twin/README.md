# Milestone (2026-02-05): Closed-loop command-conditioned digital twin (Unitree GO-M8010-6, 1-DoF)

本里程碑整理了“单关节数字孪生体（闭环指令驱动、开环状态滚动预测）”的关键实现与结果，用于后续控制算法设计与强化学习。

## 目标与约束

- 目标：学习/辨识一个可用于 **长期 open-loop rollout** 的关节动力学预测器（数字孪生体），支持低速换向等场景。
- 关键约束：部署时的观测只能来自 **指令** 与 **孪生体内部状态**，不能直接使用真机实时 `q/qd` 观测。
- Unitree 底层关节力矩合成（孪生体内部计算的“等效施加力矩”）：

`tau_cmd_hat = kp*(q_ref - q_hat) + kd*(qd_ref - qd_hat) + tau_ff`

本阶段采集时未发送 `tau_ff`，因此在建模中使用 `tau_ff = 0`。

## 数据

见 `data/`：
- `coverage_capture_20260204_103129.csv`：原始实机采集（包含 `q_rad`, `dq_rad_s`, `tau_Nm`, `q_ref_rad`, `dq_ref_rad_s`）。
- `coverage_capture_20260204_103129_with_qd_filt.csv`：在原始 CSV 基础上新增列 `dq_filt_rad_s`（零相位滤波速度），**不覆盖原始列**。

## 速度滤波与选择

为避免网络追噪声，我们将速度监督目标改为滤波速度列 `dq_filt_rad_s`，并用 sweep 选取截止频率：
- sweep 汇总：`artifacts/qd_filter_sweep/summary.md`
- 推荐（折中“换向细节”与“不过度平滑”）：`zero_phase_one_pole @ 20 Hz`

## 模型输入特征（增强版 / full）

为更好学习摩擦/死区/滞回，使用 `feature_set=full`：

`[sin(q_hat), cos(q_hat), qd_hat, e_q, e_qd, kp, kd, tau_cmd_hat, dt]`

其中 `e_q=q_ref-q_hat`，`e_qd=qd_ref-qd_hat`，`tau_cmd_hat=kp*e_q+kd*e_qd+tau_ff`。

## 训练策略：用 open-loop 指标选 best（只看 sine）

训练过程中不仅监控一步 `val_mse`，还定期做 `sine` 段的 open-loop rollout（`horizon_steps=300`）并以 `open_loop_qd_rmse` 作为 **选 best** 指标，避免出现“一步更准但开环更差”的情况。

## 关键结果（已归档）

### 最优（滤波速度监督 + full 特征）

`artifacts/qdfilt_full/`：
- `summary_closed_loop_csv_gpu_qdfilt_20260205_104512.md`
- `plot_reversals_sine_qdfilt_20260205_104512.png`
- `plot_full_cycle_sine_qdfilt_20260205_104512.png`
- `real_csv_closed_loop_model_gpu_qdfilt_20260205_104512.pt`
- `real_csv_closed_loop_dataset_qdfilt_20260205_104512.npz`

亮点：`vel_step` 的 `qd_rmse` 相比早期版本显著下降（见 summary）。

### 对照（raw qd 监督，不滤波）

`artifacts/rawqd_full/`：用于说明不滤波速度会明显伤害 `qd` 的可预测性（尤其开环）。

### `tau_Nm` 与 PD 合成力矩一致性检验（诊断）

`artifacts/tau_pd_vs_tauNm/`：
- `compare_tau_pd_rawqd.png`
- `compare_tau_pd_qdfilt.png`
- `compare_tau_pd_kp1_kd001.png`

说明：`tau_Nm` 是电流反馈按比例换算得到的力矩估计（非外置传感器）；本对比用于判断 `tau_Nm` 与 `kp*(q_ref-q)+kd*(qd_ref-qd)` 的一致性水平。

## 复现实验（参考命令）

- 滤波版一键（生成/使用 `dq_filt_rad_s` 作为监督列）：`scripts/run_real_csv_closed_loop_gpu_qdfilt.sh`
- rawqd 一键：`scripts/run_real_csv_closed_loop_gpu_rawqd.sh`
- sweep 截止频率：`scripts/sweep_qd_filter_cutoff.sh`
- 生成 `dq_filt_rad_s` 列：`scripts/add_filtered_speed_column.py`

