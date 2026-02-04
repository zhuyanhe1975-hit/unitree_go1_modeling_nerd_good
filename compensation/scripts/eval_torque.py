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
from project_config import ensure_dir, get


def _load_model(weights_path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", 64)),
        num_layers=int(ckpt.get("num_layers", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
        history_len=int(ckpt.get("history_len", 10)),
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
    ap.add_argument("--mode", choices=["sim", "real", "real_scratch", "residual", "all"], default="real")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--residual_model", default=None, help="optional residual model path")
    ap.add_argument("--mode_combo", choices=["base", "base+residual"], default="base+residual")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = _resolve_device(cfg)

    def _dt_for_mode(mode: str) -> float:
        return float(get(cfg, "sim.frame_dt")) if mode == "sim" else float(get(cfg, "real.dt"))

    def eval_one(mode: str, dataset: str, model_path: str, residual_path: str | None) -> None:
        ds = dict(np.load(dataset, allow_pickle=True))
        x = ds["x"].astype(np.float32)
        y = ds["y"].astype(np.float32)
        y_mean = ds["y_mean"].astype(np.float32)
        y_std = ds["y_std"].astype(np.float32)
        dt = _dt_for_mode(mode)
        hist = int(get(cfg, "model.history_len"))

        model = _load_model(model_path, device=device)
        model_res = (
            _load_model(residual_path, device=device)
            if residual_path and args.mode_combo == "base+residual"
            else None
        )
        mse_norm = _mse(model, x, y, device=device)
        print(f"[{mode}] base torque_model mse_norm={mse_norm:.6f}")
        if model_res is not None:
            mse_res = _mse(model_res, x, y, device=device)
            print(f"[{mode}] residual_model mse_norm={mse_res:.6f}")

        if plt is None:
            return

        n_plot = min(2000, x.shape[0])
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(x[:n_plot]).float().to(device))
            if model_res is not None:
                pred_norm = pred_norm + model_res(torch.from_numpy(x[:n_plot]).float().to(device))
            pred_norm = pred_norm.cpu().numpy()
        tau_pred = pred_norm * y_std + y_mean
        tau_true = y[:n_plot] * y_std + y_mean
        t_full = (np.arange(n_plot) + hist) * dt

        plt.figure(figsize=(12, 7))
        plt.subplot(2, 1, 1)
        plt.plot(t_full, tau_true.reshape(-1), label="tau (gt)", alpha=0.7)
        plt.plot(
            t_full,
            tau_pred.reshape(-1),
            label="tau (pred)" + (" + residual" if model_res is not None else ""),
            alpha=0.7,
        )
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        err_full = (tau_pred - tau_true).reshape(-1)
        plt.plot(t_full, err_full, label="tau error (pred - gt)", color="r", alpha=0.8)
        plt.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.legend()
        name_map = {
            "sim": "eval_torque_sim.png",
            "real": "eval_torque_real.png",
            "real_scratch": "eval_torque_scratch.png",
            "residual": "eval_torque_residual.png",
        }
        out_name = name_map.get(mode, f"eval_torque_{mode}.png")
        out = os.path.join(os.path.dirname(dataset) or ".", out_name)
        ensure_dir(os.path.dirname(out) or ".")
        plt.tight_layout()
        plt.savefig(out)
        print(f"saved: {out}")

    # Resolve single or all
    modes = [args.mode] if args.mode != "all" else ["sim", "real", "real_scratch", "residual"]

    for m in modes:
        if m == "sim":
            dataset = args.dataset or str(get(cfg, "paths.sim_dataset"))
            model_path = args.model or str(get(cfg, "paths.sim_model"))
            residual_path = None
        elif m == "real_scratch":
            dataset = args.dataset or str(get(cfg, "paths.real_dataset"))
            model_path = args.model or str(get(cfg, "paths.real_model_scratch"))
            residual_path = None
        else:
            dataset = args.dataset or str(get(cfg, "paths.real_dataset"))
            model_path = args.model or str(get(cfg, "paths.real_model"))
            residual_path = args.residual_model
            if residual_path is None and "." in model_path:
                stem, ext = model_path.rsplit(".", 1)
                cand = f"{stem}_residual.{ext}"
                if os.path.exists(cand):
                    residual_path = cand
            if m == "residual":
                args.mode_combo = "base+residual"
        eval_one(m, dataset, model_path, residual_path)

if __name__ == "__main__":
    main()
