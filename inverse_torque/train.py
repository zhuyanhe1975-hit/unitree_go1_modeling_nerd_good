from __future__ import annotations

import argparse
import os

from inverse_torque.torque_delta import train_torque_delta_model
from pipeline.config import load_cfg
from project_config import ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="runs/torque_delta_dataset.npz")
    ap.add_argument("--out", default="runs/torque_delta_model.pt")
    ap.add_argument("--split", choices=["train", "all"], default="train")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(os.path.dirname(args.out) or ".")
    # By default, train on the time-series train split (see dataset's train_idx).
    if args.split == "train":
        import numpy as np

        ds = dict(np.load(args.dataset, allow_pickle=True))
        train_idx = ds.get("train_idx", None)
        if train_idx is None:
            train_torque_delta_model(cfg, dataset_npz=args.dataset, out_weights=args.out)
        else:
            x = ds["x"].astype(np.float32)[train_idx]
            y = ds["y"].astype(np.float32)[train_idx]
            # Write a small temporary dataset for reusing the existing trainer.
            tmp = os.path.join(os.path.dirname(args.dataset) or ".", "_torque_delta_train_tmp.npz")
            np.savez(tmp, x=x, y=y)
            train_torque_delta_model(cfg, dataset_npz=tmp, out_weights=args.out)
            try:
                os.remove(tmp)
            except Exception:
                pass
    else:
        train_torque_delta_model(cfg, dataset_npz=args.dataset, out_weights=args.out)
    print(f"saved torque-delta model: {args.out}")


if __name__ == "__main__":
    main()
