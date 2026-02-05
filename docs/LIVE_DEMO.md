# Live Demo: 真机 + 闭环指令驱动数字孪生（同步滚动）

目标：让真机关节按给定指令运行，同时数字孪生体**只用指令 + 自身状态**做 open-loop rollout，最终对比长期预测误差。

数字孪生体每步使用 Unitree 力矩合成公式（在孪生体内部计算等效施加力矩）：

`tau_cmd_hat = kp*(q_ref - q_hat) + kd*(qd_ref - qd_hat) + tau_ff`

并用已训练的 closed-loop 模型输出 `delta_q, delta_qd` 进行状态更新。

## 前置

- 确保已编译/可导入 `unitree_actuator_sdk`：
  - 见 `README.md` 中 “Unitree SDK Python 模块” 部分
- 确保已训练并得到：
  - 权重 `paths.real_csv_model`（例如 `results/real_csv_closed_loop_model_gpu_qdfilt_*.pt`）
  - 统计量 `paths.real_csv_stats`（例如 `results/real_csv_closed_loop_stats_qdfilt_*.npz`）

注意：`weights` 与 `stats` 必须来自**同一次**数据准备/训练（特征维度必须一致）。如果出现维度不匹配报错，请在命令行显式传入配套的 `--weights` 与 `--stats`。

## 运行（示例）

在仓库根目录：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/demo_live_closed_loop_digital_twin.py \
  --hw_config config.json \
  --model_config configs/real_csv_closed_loop_gpu.json \
  --device cuda \
  --duration_s 20 --rate_hz 200 \
  --q_center 1.0 --amp 0.2 --freq_hz 0.1 \
  --plot
```

也可以直接用“自动加载最佳模型”的 demo 配置（推荐）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/demo_live_closed_loop_digital_twin.py \
  --config configs/live_demo_best_qdfilt_full.json
```

为避免启动瞬间的指令位置跳跃，demo 默认会在前 `ramp_s` 秒内平滑地从当前 `q0` 过渡到 `q_center + amp*cos(wt)`。

输出（默认在 `results/`，不会进 git）：
- `results/live_twin_<tag>_<timestamp>.csv`
- `results/live_twin_<tag>_<timestamp>.png`（如果加 `--plot` 且装了 matplotlib）

## 安全

- `Ctrl-C` 会停止运行并发送一次“kp=kd=tau_ff=0”的停机命令。
- 首次运行建议先用 `--dry_run`（不发送指令，仅读取并跑孪生体）确认环境与串口权限：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/demo_live_closed_loop_digital_twin.py --dry_run --plot
```
