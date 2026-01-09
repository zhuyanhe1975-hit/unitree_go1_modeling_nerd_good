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


def do_sim_and_pretrain() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/generate.py", "--config", str(cfg)])
    _run([py, "scripts/prepare.py", "--mode", "sim", "--config", str(cfg)])
    _run([py, "scripts/train.py", "--mode", "sim", "--config", str(cfg)])


def do_collect_real() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/collect_real_data.py", "--config", str(cfg)])


def do_finetune() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/prepare.py", "--mode", "real", "--config", str(cfg)])
    _run([py, "scripts/finetune.py", "--config", str(cfg)])


def do_eval() -> None:
    cfg = _input_cfg()
    model = input("Eval target (sim/real/real_scratch/all) [all]: ").strip() or "all"
    py = sys.executable
    _run([py, "scripts/eval.py", "--model", model, "--config", str(cfg)])


def do_train_real_scratch() -> None:
    cfg = _input_cfg()
    py = sys.executable
    _run([py, "scripts/prepare.py", "--mode", "real", "--config", str(cfg)])
    _run([py, "scripts/train.py", "--mode", "real", "--config", str(cfg)])


def main() -> None:
    menu = {
        "1": ("仿真数据生成 + 预训练", do_sim_and_pretrain),
        "2": ("真机数据采集", do_collect_real),
        "3": ("真机数据 finetune", do_finetune),
        "4": ("真机数据 scratch 训练", do_train_real_scratch),
        "5": ("评估", do_eval),
        "q": ("退出", None),
    }

    while True:
        print("\n=== NERD 1-DOF Guide ===")
        for k, (label, _) in menu.items():
            print(f"{k}) {label}")
        choice = input("选择操作: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            break
        if choice not in menu:
            print("无效选项，请重选。")
            continue
        _, fn = menu[choice]
        if fn is not None:
            fn()


if __name__ == "__main__":
    main()
