from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = REPO_ROOT / "config.json"


def _run(cmd: list[str]) -> None:
    print(f"\n[cmd] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    try:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[error] command failed with code {e.returncode}")


def _input_cfg() -> Path:
    cfg = input(f"Config path (default: {DEFAULT_CFG}): ").strip()
    return Path(cfg) if cfg else DEFAULT_CFG


def do_train_residual() -> None:
    cfg = _input_cfg()
    mode = input("Residual mode (sim/real/real_scratch) [real]: ").strip() or "real"
    py = sys.executable
    _run([py, "scripts/train_residual.py", "--mode", mode, "--config", str(cfg)])


def do_collect_real() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/collect_real_data.py", "--config", str(cfg)])


def do_sim_data() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/generate.py", "--config", str(cfg)])
    _run([py, "scripts/prepare_torque.py", "--mode", "sim", "--config", str(cfg)])


def do_real_data() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/collect_real_data.py", "--config", str(cfg)])
    _run([py, "scripts/prepare_torque.py", "--mode", "real", "--config", str(cfg)])


def do_train_base() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/train_torque.py", "--mode", "sim", "--config", str(cfg)])


def do_finetune() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/prepare_torque.py", "--mode", "real", "--config", str(cfg)])
    _run([py, "scripts/train_torque.py", "--mode", "finetune", "--config", str(cfg)])


def do_train_real_scratch() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/prepare_torque.py", "--mode", "real", "--config", str(cfg)])
    _run([py, "scripts/train_torque.py", "--mode", "real_scratch", "--config", str(cfg)])


def do_train_residual() -> None:
    cfg = _input_cfg()
    mode = input("Residual mode (sim/real/real_scratch) [real]: ").strip() or "real"
    py = sys.executable
    _run([py, "scripts/train_residual.py", "--mode", mode, "--config", str(cfg)])


def do_eval_with_residual() -> None:
    cfg = _input_cfg()
    target = input("Eval target (sim/real/scratch/residual/all) [real]: ").strip() or "real"
    res = input("Residual model path (optional, auto-detect if empty): ").strip()
    py = sys.executable
    cmd = [py, "scripts/eval_torque.py", "--mode", target, "--config", str(cfg)]
    if res:
        cmd += ["--residual_model", res]
    _run(cmd)


def main() -> None:
    menu = {
        "1": ("仿真数据集生成（采集+处理）", do_sim_data),
        "2": ("真机数据集生成（采集+处理）", do_real_data),
        "3": ("nerd base: 仿真预训练力矩模型", do_train_base),
        "4": ("nerd finetune: 仿真预训基础上用真机微调", do_finetune),
        "5": ("nerd from scratch: 真机数据直接训练", do_train_real_scratch),
        "6": ("phys+nerd residual: 残差模型训练", do_train_residual),
        "7": ("评估（sim/real/scratch/residual）", do_eval_with_residual),
        "q": ("退出", None),
    }
    while True:
        print("\n=== Compensation Guide ===")
        for k, (label, _) in menu.items():
            print(f"{k}) {label}")
        choice = input("选择操作: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice not in menu:
            print("无效选项，请重选。")
            continue
        _, fn = menu[choice]
        if fn:
            fn()


if __name__ == "__main__":
    main()
