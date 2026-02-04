from __future__ import annotations

import argparse
import os

from pipeline.config import load_cfg
from project_config import ensure_dir

from inverse_torque.friction_model import train_friction_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="runs/friction_dataset.npz")
    ap.add_argument("--out", default="runs/friction_model.pt")
    ap.add_argument("--split", choices=["train", "all"], default="train")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(os.path.dirname(args.out) or ".")
    train_friction_model(cfg, dataset_npz=args.dataset, out_weights=args.out, split="train" if args.split == "train" else "all")
    print(f"saved friction model: {args.out}")


if __name__ == "__main__":
    main()

