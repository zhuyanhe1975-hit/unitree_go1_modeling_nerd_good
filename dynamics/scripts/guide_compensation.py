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


def do_prepare_torque() -> None:
    cfg = _input_cfg()
    raw = input("Raw log path (default: paths.real_log): ").strip()
    out = input("Output torque dataset (default: runs/torque_dataset.npz): ").strip()
    stats = input("Output stats path (default: runs/torque_stats.npz): ").strip()
    py = sys.executable
    cmd = [py, "scripts/prepare_torque.py", "--config", str(cfg)]
    if raw:
        cmd += ["--raw", raw]
    if out:
        cmd += ["--out", out]
    if stats:
        cmd += ["--stats", stats]
    _run(cmd)


def do_train_torque() -> None:
    cfg = _input_cfg()
    dataset = input("Torque dataset path (default: runs/torque_dataset.npz): ").strip()
    out = input("Output torque model (default: runs/torque_model.pt): ").strip()
    py = sys.executable
    cmd = [py, "scripts/train_torque.py", "--config", str(cfg)]
    if dataset:
        cmd += ["--dataset", dataset]
    if out:
        cmd += ["--out", out]
    _run(cmd)


def do_eval_with_residual() -> None:
    cfg = _input_cfg()
    target = input("Eval target (real/real_scratch/sim/all) [real]: ").strip() or "real"
    res_model = input("Residual model path (optional, default auto-detect): ").strip()
    py = sys.executable
    cmd = [py, "scripts/eval.py", "--model", target, "--config", str(cfg)]
    if res_model:
        cmd += ["--residual_model", res_model]
    _run(cmd)


def main() -> None:
    menu = {
        "1": ("采集补偿数据（实机）", do_collect_real),
        "2": ("准备力矩补偿数据集", do_prepare_torque),
        "3": ("训练力矩补偿模型", do_train_torque),
        "4": ("残差模型训练", do_train_residual),
        "5": ("评估（含残差）", do_eval_with_residual),
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
