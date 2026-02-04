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


def _load_model(weights_path: str, device: str, fallback_cfg: dict) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", None) or fallback_cfg["model"]["embed_dim"]),
        num_layers=int(ckpt.get("num_layers", None) or fallback_cfg["model"]["num_layers"]),
        num_heads=int(ckpt.get("num_heads", None) or fallback_cfg["model"]["num_heads"]),
        history_len=int(ckpt.get("history_len", None) or fallback_cfg["model"]["history_len"]),
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
    ap.add_argument("--dataset", default="runs/torque_delta_dataset.npz")
    ap.add_argument("--model", default="runs/torque_delta_model.pt")
    ap.add_argument("--plot", action="store_true", help="save eval_torque_delta.png")
    ap.add_argument("--split", choices=["val", "train", "all"], default="val")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _resolve_device(cfg)

    ds = dict(np.load(args.dataset, allow_pickle=True))
    x_all = ds["x"].astype(np.float32)
    y_all = ds["y"].astype(np.float32)
    x_mean = ds["x_mean"].astype(np.float32)
    x_std = ds["x_std"].astype(np.float32)
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

    model = _load_model(args.model, device=device, fallback_cfg=cfg)
    mse_norm = _mse(model, x, y, device=device)
    print(f"torque_delta_model mse_norm={mse_norm:.6f}")

    if plt is None or not args.plot:
        return

    n_plot = min(3000, x.shape[0])
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x[:n_plot]).float().to(device)).cpu().numpy()
    delta_pred = pred_norm * y_std + y_mean  # [N,1]
    delta_true = y[:n_plot] * y_std + y_mean

    # Recover tau_out[k-1] from the (de-normalized) tau_out channel in x.
    # tau_out is always the last feature in our dataset (see prepare_torque_delta_dataset),
    # and our windows end at (k-1), so last element equals tau_out[k-1].
    tau_prev = (x[:n_plot, -1, -1] * x_std[-1] + x_mean[-1]).reshape(-1)  # tau_out[k-1]
    tau_gt = tau_prev + delta_true.reshape(-1)  # tau_out[k]
    tau_pred = tau_prev + delta_pred.reshape(-1)  # tau_out[k] (pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(tau_gt, label="tau_out[k] (gt)", alpha=0.7)
    plt.plot(tau_pred, label="tau_out[k] (pred from delta)", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot((tau_pred - tau_gt), label="error (Nm)", color="r", alpha=0.8)
    plt.axhline(0.0, color="k", lw=0.8)
    plt.grid(True, alpha=0.3)
    plt.legend()

    suffix = "" if args.split == "val" else f"_{args.split}"
    out = os.path.join(os.path.dirname(args.dataset) or ".", f"eval_torque_delta{suffix}.png")
    ensure_dir(os.path.dirname(out) or ".")
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
