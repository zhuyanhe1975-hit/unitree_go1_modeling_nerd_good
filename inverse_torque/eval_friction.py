from __future__ import annotations

import argparse
import os

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir


def _load_model(weights_path: str, device: str, cfg: dict) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", cfg["model"]["embed_dim"])),
        num_layers=int(ckpt.get("num_layers", cfg["model"]["num_layers"])),
        num_heads=int(ckpt.get("num_heads", cfg["model"]["num_heads"])),
        history_len=int(ckpt.get("history_len", cfg["model"]["history_len"])),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def _mse(model: CausalTransformer, x: np.ndarray, y: np.ndarray, device: str) -> float:
    xb = torch.from_numpy(x).float().to(device)
    yb = torch.from_numpy(y).float().to(device)
    pred = model(xb)
    return float(torch.mean((pred - yb) ** 2).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="runs/friction_dataset.npz")
    ap.add_argument("--model", default="runs/friction_model.pt")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _resolve_device(cfg)

    ds = dict(np.load(args.dataset, allow_pickle=True))
    x_all = ds["x"].astype(np.float32)
    y_all = ds["y"].astype(np.float32)
    y_mean = ds["y_mean"].astype(np.float32)
    y_std = ds["y_std"].astype(np.float32)

    if args.split == "train" and "train_idx" in ds:
        idx = ds["train_idx"].astype(np.int64)
    elif args.split == "val" and "val_idx" in ds:
        idx = ds["val_idx"].astype(np.int64)
    else:
        idx = None

    x = x_all[idx] if idx is not None else x_all
    y = y_all[idx] if idx is not None else y_all

    model = _load_model(args.model, device=device, cfg=cfg)
    mse_n = _mse(model, x, y, device=device)
    print(f"friction_model mse_norm={mse_n:.6f} (split={args.split})")

    if plt is None or not args.plot:
        return

    n = min(2000, x.shape[0])
    with torch.no_grad():
        pred_n = model(torch.from_numpy(x[:n]).float().to(device)).cpu().numpy()
    y_pred = pred_n * y_std + y_mean
    y_gt = y[:n] * y_std + y_mean

    plt.figure(figsize=(12, 5))
    plt.plot(y_gt.reshape(-1), label="tau_fric (gt)", alpha=0.7)
    plt.plot(y_pred.reshape(-1), label="tau_fric (pred)", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(os.path.dirname(args.dataset) or ".", f"eval_friction_{args.split}.png")
    ensure_dir(os.path.dirname(out) or ".")
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

