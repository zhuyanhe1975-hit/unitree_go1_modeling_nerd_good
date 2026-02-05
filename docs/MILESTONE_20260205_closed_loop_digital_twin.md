# Milestone (2026-02-05): Closed-loop command-conditioned digital twin (GO-M8010-6, 1-DoF)

本里程碑聚焦 **单关节数字孪生体**：训练时用真机日志做监督，但推理/评估时只能用 **指令 + 孪生体内部状态** 做 open-loop rollout，从而可用于后续控制与 RL。

> 归档的可复现实验数据、模型、图片见：`milestones/20260205_closed_loop_digital_twin/`

## 约束与建模假设

- 观测约束：部署时不能读取真机 `q/qd` 作为观测；可用量仅包括指令（`q_ref/qd_ref/kp/kd`）与孪生体内部状态（`q_hat/qd_hat`）。
- 力矩合成：Unitree 关节的底层驱动力矩生成按：

`tau_cmd_hat = kp*(q_ref - q_hat) + kd*(qd_ref - qd_hat) + tau_ff`

本阶段采集数据未发送 `tau_ff`，因此使用 `tau_ff=0`。

## 数据与预处理

### 原始 CSV

原始采集：`real_data/coverage_capture_20260204_103129.csv`

### 速度滤波（零相位）

目标：避免模型学习 `qd` 的高频噪声（否则一步拟合变好但 open-loop 漂移更严重）。

- 使用 **零相位一阶低通**（forward-backward one-pole，filtfilt-style）生成 `dq_filt_rad_s` 列。
- 截止频率 sweep（平衡“换向细节”与“不过度平滑”）推荐 `20 Hz`。

归档：
- sweep 汇总与最佳示例图：`milestones/20260205_closed_loop_digital_twin/artifacts/qd_filter_sweep/`
- 带滤波列的 CSV：`milestones/20260205_closed_loop_digital_twin/data/coverage_capture_20260204_103129_with_qd_filt.csv`

## 特征增强（full feature_set）

为更好学习摩擦/死区/滞回，将输入特征从 minimal 扩展到 full：

`[sin(q_hat), cos(q_hat), qd_hat, e_q, e_qd, kp, kd, tau_cmd_hat, dt]`

其中：
- `e_q=q_ref-q_hat`
- `e_qd=qd_ref-qd_hat`
- `tau_cmd_hat=kp*e_q+kd*e_qd+tau_ff`

## 训练策略：用 open-loop 指标选最优 checkpoint（只看 sine）

观察到：长时间训练会让一步 teacher-forcing 继续变好，但 open-loop 反而变差（exposure bias + 过拟合）。

因此训练阶段定期评估 `sine` 段 open-loop rollout（例如 300 steps），并以 `open_loop_qd_rmse` 作为选 best 指标（同时保留 `val_mse` 作为诊断）。

## 结果摘要（最优 / 对照）

请以归档 summary 为准：

- 最优（qdfilt + full features）：`milestones/20260205_closed_loop_digital_twin/artifacts/qdfilt_full/summary_closed_loop_csv_gpu_qdfilt_20260205_104512.md`
- 对照（rawqd + full features）：`milestones/20260205_closed_loop_digital_twin/artifacts/rawqd_full/summary_closed_loop_csv_gpu_rawqd_20260205_094831.md`

同时提供关键可视化：
- 低速换向窗口：`.../plot_reversals_*.png`
- 完整正弦周期：`.../plot_full_cycle_*.png`

## `tau_Nm` 与 PD 合成力矩一致性诊断

`tau_Nm` 为电流反馈按比例换算得到的力矩估计（非外置传感器）。我们对比了：

`tau_pd = kp*(q_ref - q) + kd*(qd_ref - qd) + tau_ff`（此处 `tau_ff=0`）

用于判断日志 `tau_Nm` 与 PD 合成力矩的相符程度与可能的滞后/偏置。

归档图：`milestones/20260205_closed_loop_digital_twin/artifacts/tau_pd_vs_tauNm/`

## 复现入口（一键脚本）

- 滤波版（自动生成/使用 `dq_filt_rad_s`）：`scripts/run_real_csv_closed_loop_gpu_qdfilt.sh`
- rawqd 版：`scripts/run_real_csv_closed_loop_gpu_rawqd.sh`

