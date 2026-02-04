from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

# ensure local imports work when run from anywhere
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from project_config import ensure_dir, get


def _load_model(path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(path, map_location=torch.device(device))
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


def build_windows(ds: dict, hist: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(ds["q_out"], dtype=np.float32).reshape(-1)
    qd = np.asarray(ds["qd_out"], dtype=np.float32).reshape(-1)
    tau_out = np.asarray(ds.get("tau_out", []), dtype=np.float32).reshape(-1)
    tau_cmd = np.asarray(ds.get("tau_cmd", []), dtype=np.float32).reshape(-1)
    tau_raw = np.asarray(ds.get("tau_out_raw", []), dtype=np.float32).reshape(-1)
    if tau_out.size == q.shape[0]:
        tau = tau_out
    elif tau_cmd.size == q.shape[0]:
        tau = tau_cmd
    elif tau_raw.size == q.shape[0]:
        tau = tau_raw
    else:
        raise KeyError("log missing torque feedback (tau_out/tau_cmd/tau_out_raw)")
    temp = np.asarray(ds.get("temp", np.zeros_like(q)), dtype=np.float32).reshape(-1)
    qdd = np.zeros_like(qd, dtype=np.float32)
    if len(qd) > 1:
        qdd[1:] = (qd[1:] - qd[:-1]) / float(ds.get("dt", 0.01))
    cmd_qd = np.asarray(ds.get("cmd_qd", np.zeros_like(q)), dtype=np.float32).reshape(-1)
    cmd_q = np.asarray(ds.get("cmd_q", np.zeros_like(q)), dtype=np.float32).reshape(-1)

    cols = [q[:, None], qd[:, None], qdd[:, None], temp[:, None], cmd_qd[:, None], cmd_q[:, None]]
    feat_full = np.concatenate(cols, axis=-1)  # [T, D]

    xs = []
    ys = []
    for k in range(hist - 1, len(q)):
        xs.append(feat_full[k - (hist - 1) : k + 1])
        ys.append(tau[k])
    return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.float32).reshape(-1, 1), tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--log", default=None, help="raw log npz (default: paths.real_log)")
    ap.add_argument("--model", default=None, help="torque model (default: paths.real_model)")
    ap.add_argument("--tail_s", type=float, default=10.0, help="tail window length in seconds")
    ap.add_argument("--out", default=None, help="output plot path")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    log_path = args.log or str(get(cfg, "paths.real_log"))
    model_path = args.model or str(get(cfg, "paths.real_model"))
    ds = dict(np.load(log_path, allow_pickle=True))

    t = np.asarray(ds.get("t", None), dtype=np.float32)
    if t.size > 0:
        tmax = float(t.max())
        mask = t >= (tmax - float(args.tail_s))
        for k, v in list(ds.items()):
            if isinstance(v, np.ndarray) and v.shape and v.shape[0] == t.shape[0]:
                ds[k] = v[mask]

    hist = int(get(cfg, "model.history_len"))
    x, y, tau_full = build_windows(ds, hist=hist)

    stats = dict(np.load(str(get(cfg, "paths.real_dataset")), allow_pickle=True))
    x_mean = stats["x_mean"][: x.shape[-1]]
    x_std = stats["x_std"][: x.shape[-1]]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    x_n = (x - x_mean) / x_std

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_model(model_path, device=device)
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_n).float().to(device)).cpu().numpy()
    tau_pred = pred_norm * y_std + y_mean  # model outputs are normalized, convert back
    tau_true = y  # ground truth already in physical units

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(t[-len(tau_true) :], tau_true.reshape(-1), label="tau_out (gt)", alpha=0.7)
        plt.plot(t[-len(tau_pred) :], tau_pred.reshape(-1), label="tau_pred", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(t[-len(tau_true) :], (tau_pred - tau_true).reshape(-1), label="tau error", color="r", alpha=0.8)
        plt.axhline(0.0, color="k", linestyle="--", alpha=0.5)
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_plot = args.out or os.path.join(get(cfg, "paths.runs_dir"), "check_tail_torque.png")
        ensure_dir(os.path.dirname(out_plot) or ".")
        plt.tight_layout()
        plt.savefig(out_plot)
        print(f"saved plot: {out_plot}")
    except Exception:
        pass

    print(f"tau_pred range: [{tau_pred.min():.4f}, {tau_pred.max():.4f}]")
    print(f"tau_true range: [{tau_true.min():.4f}, {tau_true.max():.4f}]")


if __name__ == "__main__":
    main()
