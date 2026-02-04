# nerd_1dof_goM8010_6

单关节真实动力学建模（Unitree GO-M8010-6），目标是学习/辨识一个可用于预测的关节动力学模型（不包含补偿器相关内容）。

## 参考实现：Neural Robot Dynamics (NeRD)

本项目的单关节 Warp 环境 + “ground-truth vs neural” 的切换方式，参考了：
`/home/yhzhu/AI/neural-robot-dynamics`（NVLabs NeRD, CoRL 2025）的整体架构思想。

为了避免把整套 NeRD 代码搬进来，这里提供了一个轻量兼容层：
- `nerd_compat/joint_1dof_neural_env.py`：`Joint1DofNeuralEnvironment`（`ground-truth`=Warp Featherstone；`neural`=本仓库训练的 state-delta Transformer）。
- `scripts/rollout_compare_neural.py`：对同一条力矩序列，比较 ground-truth vs neural 的长时滚动误差（类似 NeRD 的 passive motion eval）。

## 目录结构（精简版）

- `config.json`：全局唯一参数入口（仿真/训练/实机采集/路径）。
- `scripts/`：所有 CLI 脚本（仿真采集、real 采集、训练/finetune、评估、对比等）。
- `pipeline/`：数据准备、模型定义、训练调度等可复用模块。
- `assets/`：单关节 MJCF（包含摩擦/回差等结构）。
- `custom_envs/joint_1dof_env.py`：Warp 单关节环境（用于 sim rollout）。

原工程的完整功能可参考：`/home/yhzhu/myWorks/nerd_1dof`。

## 快速开始

### Python 环境（conda: nerd_py310）

你指定的环境是 `nerd_py310`，建议所有命令都用：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 <script>.py ...
```

### 完整链条（推荐）

1) generation（仿真生成原始 log）：
- `python3 scripts/generate.py`

2) prepare（构建监督数据集 + 统计量）：
- `python3 scripts/prepare.py --mode sim`

3) train（训练仿真动力学模型）：
- `python3 scripts/train.py`

4) finetune（采集真实数据 + 构建数据集 + 微调）：
- 采集：`python3 scripts/collect_real_data.py`（输出到 `paths.real_log`）
- prepare：`python3 scripts/prepare.py --mode real`（使用 `paths.stats_npz` 做归一化）
- finetune：`python3 scripts/finetune.py`

5) eval（评估/画图）：
- `python3 scripts/eval.py`

### 快速自检（CPU smoke）

如果你只是想验证“仿真生成→prepare→train→neural rollout”整条链路能跑通，可用：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/generate.py --config configs/smoke_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/prepare.py --mode sim --config configs/smoke_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/train.py --mode sim --config configs/smoke_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/rollout_compare_neural.py --config configs/smoke_cpu.json --device cpu
```

### （可选）NeRD 风格：ground-truth vs neural rollout 对比

在你完成 `generate -> prepare -> train` 得到 `runs/sim_dataset.npz` 和 `runs/sim_model.pt` 后：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/rollout_compare_neural.py \
  --device cuda --num_envs 1 --steps 2000 --profile chirp
```

产物：
- `runs/rollout_compare_neural.npz`
- `runs/rollout_compare_neural.png`（若装了 matplotlib）

## Milestone: Torque-Delta Feedforward Compensation (2026-02-04)

在位置闭环模式下（给定 `q_ref/dq_ref/kp/kd`），我们验证了一个可重复的阶段性成果：
使用 **torque-delta 一步预测**构造前馈力矩，可显著降低正弦轨迹跟踪误差，尤其在低速换向/低频场景更明显。

- 说明文档：`docs/MILESTONE_20260204_torque_delta_ff.md`
- 关键 demo：`scripts/demo_ff_sine.py`（baseline vs feedforward 对比）
- torque-delta 管线：`inverse_torque/`（prepare/train/eval）

快速跑硬件 demo（示例）：

```bash
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both \
  --ff_type torque_delta \
  --amp 0.2 --freq 0.1 --duration 20 \
  --kp 1 --kd 0.01 \
  --tau_ff_limit 0.15 --tau_ff_scale 1 \
  --ff_update_div 1
```

## 实机采集前：Unitree SDK Python 模块

如果你在 `python3 scripts/collect_real_data.py` 遇到：
`ModuleNotFoundError: No module named 'unitree_actuator_sdk'`
通常是因为 SDK 自带的 `.so` 与当前 Python 版本不匹配（例如 SDK 里是 `cpython-38`，你在用 `py310`）。

用当前 Python 重新编译一次：
- `python3 scripts/build_unitree_sdk.py --sdk_src "/home/yhzhu/Industrial Robot/unitree_actuator_sdk"`

编译输出会在 `runs/unitree_sdk_build/lib/`，把它写到 `config.json`：
- `real.unitree_sdk_lib = "runs/unitree_sdk_build/lib"`

也可以直接设置环境变量（更通用）：
- `UNITREE_ACTUATOR_SDK_LIB=/path/to/unitree_actuator_sdk/lib`

## 实机采集注意事项（关键）

- 采集 chirp 力矩时，请确保 `config.json` 里 `real.kp=0` 且 `real.kd=0`，否则反馈的 `tau_out` 会混入内部位置/速度环的力矩分量，导致 `tau_cmd` 与 `tau_out` 形状差异很大。
- `config.json` 的 `data.real.action_key` 决定 real 数据集训练/评估用哪一个力矩作为输入：
  - `"tau_cmd"`：下发的力矩指令
  - `"tau_out"`：SDK 反馈的力矩（kp/kd=0 时更接近“真实作用力矩”）
- 如果后续确认 SDK 的 `data.tau` 是电机侧力矩，可把 `data.real.tau_out_scale_to_out=true`，在进入模型前按 `(gear_ratio * efficiency)` 缩放到输出端等效力矩。

## 闭环指令驱动的数字孪生（不使用真机状态作为观测）

你的约束是：数字孪生体部署时的观测只能来自**指令**（以及孪生体自身状态），不能直接读取真机的 `q/qd`。
因此我们提供了一个“command-conditioned”的管线：用真机日志做监督训练，但推理/评估用 open-loop rollout（只用初始状态 + 指令序列 + 孪生体自身滚动状态）。

并使用 Unitree 底层力矩合成公式在孪生体内部构造“等效施加力矩”：

`tau_cmd_hat = kp*(q_ref - q_hat) + kd*(qd_ref - qd_hat) + tau_ff`

训练数据特征（每步，最小版）：

`[sin(q_hat), cos(q_hat), qd_hat, tau_cmd_hat, dt]`

示例（CPU，输出写到 `/tmp/`）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/prepare_closed_loop_csv.py --config configs/real_csv_closed_loop_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/train_closed_loop_csv.py --config configs/real_csv_closed_loop_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/eval_closed_loop_csv.py --config configs/real_csv_closed_loop_cpu.json --device cpu --stage sine --horizon_steps 300
```

如果你想先快速验证能跑通，可用更小的 smoke 配置（训练更快）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/train_closed_loop_csv.py --config configs/real_csv_closed_loop_smoke_cpu.json
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/eval_closed_loop_csv.py --config configs/real_csv_closed_loop_smoke_cpu.json --device cpu --stage sine --horizon_steps 300
```

批量评估多个 stage 并输出 markdown 汇总（保存到 `/tmp/.../summary_closed_loop_csv_*.md`）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/eval_closed_loop_csv.py \
  --config configs/real_csv_closed_loop_smoke_cpu.json \
  --device cpu \
  --stages sine,pos_sweep,vel_step \
  --horizon_steps 300 \
  --baseline
```

画“低速换向”开环预测对比图（只用指令 + 孪生体自身状态滚动；换向点由 `qd_ref` 过零检测）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/eval_closed_loop_csv.py \
  --config configs/real_csv_closed_loop_smoke_cpu.json \
  --device cpu \
  --stage sine \
  --horizon_steps 2000 \
  --plot_reversals \
  --plot_horizon_steps 4000 \
  --reversal_window_s 0.6 \
  --reversal_max_events 4 \
  --reversal_speed_th 0.5
```

画“完整正弦周期”的开环预测结果图（从 `qd_ref` 过零点自动截取一个完整周期）：

```bash
conda run -n nerd_py310 PYTHONPATH=. python3 scripts/eval_closed_loop_csv.py \
  --config configs/real_csv_closed_loop_smoke_cpu.json \
  --device cpu \
  --stage sine \
  --horizon_steps 2000 \
  --plot_full_cycle \
  --plot_horizon_steps 4000
```
