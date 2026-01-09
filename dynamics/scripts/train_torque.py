from __future__ import annotations

import argparse
import os

from pipeline.comp_torque import train_torque_model
from pipeline.config import load_cfg
from project_config import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="runs/torque_dataset.npz")
    ap.add_argument("--out", default="runs/torque_model.pt")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(os.path.dirname(args.out) or ".")
    train_torque_model(cfg, dataset_npz=args.dataset, out_weights=args.out)
    print(f"saved torque model: {args.out}")


if __name__ == "__main__":
    main()
