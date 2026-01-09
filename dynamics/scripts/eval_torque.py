from __future__ import annotations

import argparse
import os

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from pipeline.comp_torque import TorqueDataset
from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir


def _load_model(weights_path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=ckpt.get("embed_dim", None) or 64,
        num_layers=ckpt.get("num_layers", None) or 2,
        num_heads=ckpt.get("num_heads", None) or 4,
        history_len=ckpt.get("history_len", None) or 10,
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
    ap.add_argument("--dataset", default="runs/torque_dataset.npz")
    ap.add_argument("--model", default="runs/torque_model.pt")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _resolve_device(cfg)

    ds = dict(np.load(args.dataset, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)
    x_mean = ds["x_mean"].astype(np.float32)
    x_std = ds["x_std"].astype(np.float32)
    y_mean = ds["y_mean"].astype(np.float32)
    y_std = ds["y_std"].astype(np.float32)

    model = _load_model(args.model, device=device)
    mse_norm = _mse(model, x, y, device=device)
    print(f"torque_model mse_norm={mse_norm:.6f}")

    if plt is None:
        return

    n_plot = min(2000, x.shape[0])
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x[:n_plot]).float().to(device)).cpu().numpy()
    delta_tau_pred = pred_norm * y_std + y_mean
    delta_tau_true = y[:n_plot] * y_std + y_mean

    tau_cmd = (x[:n_plot, -1, -1] * x_std[-1] + x_mean[-1]).reshape(-1)
    tau_out_pred = tau_cmd + delta_tau_pred.reshape(-1)
    tau_out_true = tau_cmd + delta_tau_true.reshape(-1)

    plt.figure(figsize=(12, 5))
    plt.plot(tau_out_true, label="tau_out (gt)", alpha=0.7)
    plt.plot(tau_out_pred, label="tau_out (pred)", alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    out = os.path.join(os.path.dirname(args.dataset) or ".", "eval_torque.png")
    ensure_dir(os.path.dirname(out) or ".")
    plt.tight_layout()
    plt.savefig(out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
