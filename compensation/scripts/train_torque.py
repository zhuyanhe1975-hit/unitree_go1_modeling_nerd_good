from __future__ import annotations

import argparse
import os
import subprocess
import sys

from pipeline.comp_torque import train_torque_model
from pipeline.config import load_cfg
from project_config import ensure_dir, get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "finetune", "real_scratch"], default="sim")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--base", default=None, help="base model for finetune")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if args.mode == "sim":
        dataset = args.dataset or str(get(cfg, "paths.sim_dataset"))
        out = args.out or str(get(cfg, "paths.sim_model"))
        base = args.base
    elif args.mode == "finetune":
        dataset = args.dataset or str(get(cfg, "paths.real_dataset"))
        base = args.base or str(get(cfg, "paths.sim_model"))
        out = args.out or str(get(cfg, "paths.real_model"))
        if not os.path.exists(base):
            print(f"[info] base sim model {base} not found, auto-running sim pretrain")
            # run sim pipeline: generate -> prepare -> train sim
            subprocess.run(
                [sys.executable, "scripts/generate.py", "--config", str(args.config or os.path.join(root, "config.json"))],
                cwd=root,
                check=True,
                env={**os.environ, "PYTHONPATH": root},
            )
            subprocess.run(
                [sys.executable, "scripts/prepare_torque.py", "--mode", "sim", "--config", str(args.config or os.path.join(root, "config.json"))],
                cwd=root,
                check=True,
                env={**os.environ, "PYTHONPATH": root},
            )
            subprocess.run(
                [sys.executable, "scripts/train_torque.py", "--mode", "sim", "--config", str(args.config or os.path.join(root, "config.json"))],
                cwd=root,
                check=True,
                env={**os.environ, "PYTHONPATH": root},
            )
    else:  # real_scratch
        dataset = args.dataset or str(get(cfg, "paths.real_dataset"))
        base = args.base
        out = args.out or str(get(cfg, "paths.real_model_scratch"))

    ensure_dir(os.path.dirname(out) or ".")
    train_torque_model(cfg, dataset_npz=dataset, out_weights=out, base_weights=base)
    print(f"saved torque model: {out}")


if __name__ == "__main__":
    main()
