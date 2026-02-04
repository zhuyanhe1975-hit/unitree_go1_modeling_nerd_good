from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict

import numpy as np
import torch

from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from project_config import get


def _load_model(cfg: Dict[str, Any], weights_path: str, device: str) -> tuple[CausalTransformer, dict]:
    """
    Loads a checkpoint produced by pipeline/train.py.

    Note: older checkpoints may only store {model,input_dim,output_dim}. In that case we
    use config.json model hyperparams to rebuild the architecture.
    """
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", get(cfg, "model.embed_dim"))),
        num_layers=int(ckpt.get("num_layers", get(cfg, "model.num_layers"))),
        num_heads=int(ckpt.get("num_heads", get(cfg, "model.num_heads"))),
        history_len=int(ckpt.get("history_len", get(cfg, "model.history_len"))),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def _open_loop_segment(
    model: CausalTransformer,
    stats: dict,
    *,
    t: np.ndarray,
    stage: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    tau_ff: np.ndarray,
    q_ref: np.ndarray,
    qd_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    stage_name: str,
    H: int,
    horizon_steps: int,
    device: str,
) -> dict:
    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    gap_thr = max(0.05, 10.0 * dt_med) if dt_med > 0 else 0.05
    cuts = [0]
    for i in range(1, len(t)):
        if stage[i] != stage[i - 1] or (t[i] - t[i - 1]) > gap_thr:
            cuts.append(i)
    cuts.append(len(t))

    pick = None
    for i in range(len(cuts) - 1):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if str(stage[a]) != str(stage_name):
            continue
        if (b - a) >= (H + int(horizon_steps) + 2):
            pick = (a, b)
            break
    if pick is None:
        raise ValueError(f"no segment found for stage='{stage_name}' with horizon={horizon_steps} and H={H}")
    a, _b = pick

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    q_hist = q[a : a + H].astype(np.float64).copy()
    qd_hist = qd[a : a + H].astype(np.float64).copy()
    q_pred = float(q_hist[-1])
    qd_pred = float(qd_hist[-1])

    e_q = np.zeros((horizon_steps,), dtype=np.float64)
    e_qd = np.zeros((horizon_steps,), dtype=np.float64)
    x_buf = np.zeros((1, H, int(x_mean.shape[0])), dtype=np.float32)

    for i in range(horizon_steps):
        k = a + (H - 1) + i
        sl = slice(k - (H - 1), k + 1)

        qref_h = q_ref[sl].astype(np.float64)
        qdref_h = qd_ref[sl].astype(np.float64)
        kp_h = kp[sl].astype(np.float64)
        kd_h = kd[sl].astype(np.float64)
        tau_ff_h = tau_ff[sl].astype(np.float64)
        tt = t[sl].astype(np.float64)

        dt_h = np.zeros((H,), dtype=np.float64)
        if len(tt) >= 2:
            dt_h[0] = float(np.median(np.diff(tt)))
            dt_h[1:] = np.diff(tt)

        tau_cmd_hat = kp_h * (qref_h - q_hist) + kd_h * (qdref_h - qd_hist) + tau_ff_h
        feat = np.stack([np.sin(q_hist), np.cos(q_hist), qd_hist, tau_cmd_hat, dt_h], axis=-1).astype(np.float32)
        x_buf[0] = (feat - x_mean[None, None, :]) / x_std[None, None, :]

        pred_n = model(torch.from_numpy(x_buf).to(device)).detach().cpu().numpy().reshape(-1).astype(np.float32)
        delta = pred_n * y_std + y_mean

        q_pred = q_pred + float(delta[0])
        qd_pred = qd_pred + float(delta[1])

        q_hist[:-1] = q_hist[1:]
        qd_hist[:-1] = qd_hist[1:]
        q_hist[-1] = q_pred
        qd_hist[-1] = qd_pred

        e_q[i] = q_pred - float(q[k + 1])
        e_qd[i] = qd_pred - float(qd[k + 1])

    return {
        "stage": str(stage_name),
        "horizon_steps": int(horizon_steps),
        "q_rmse": float(np.sqrt(np.mean(e_q**2))),
        "q_maxabs": float(np.max(np.abs(e_q))),
        "qd_rmse": float(np.sqrt(np.mean(e_qd**2))),
        "qd_maxabs": float(np.max(np.abs(e_qd))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="CSV path (default: paths.real_csv)")
    ap.add_argument("--dataset", default=None, help="prepared dataset npz (default: paths.real_csv_dataset)")
    ap.add_argument("--weights", default=None, help="model weights (default: paths.real_csv_model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stage", default="sine")
    ap.add_argument("--horizon_steps", type=int, default=300)
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_cfg(args.config)

    csv_path = args.csv or str(get(cfg, "paths.real_csv"))
    dataset_npz = args.dataset or str(get(cfg, "paths.real_csv_dataset"))
    weights_path = args.weights or str(get(cfg, "paths.real_csv_model"))
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing csv: {csv_path}")
    if not os.path.exists(dataset_npz):
        raise SystemExit(f"missing dataset: {dataset_npz} (run scripts/prepare_closed_loop_csv.py first)")
    if not os.path.exists(weights_path):
        raise SystemExit(f"missing weights: {weights_path} (run scripts/train_closed_loop_csv.py first)")

    ds = dict(np.load(dataset_npz, allow_pickle=True))
    stats = {k: ds[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    H = int(ds["x"].shape[1])

    model, _ = _load_model(cfg, weights_path, device=str(args.device))

    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)
    with torch.no_grad():
        pred = model(torch.from_numpy(x).to(args.device)).detach().cpu().numpy().astype(np.float32)
    mse_norm = float(np.mean((pred - y) ** 2))
    pred_phys = pred * stats["y_std"][None, :] + stats["y_mean"][None, :]
    y_phys = y * stats["y_std"][None, :] + stats["y_mean"][None, :]
    err = pred_phys - y_phys
    print(
        f"[teacher-forcing] one-step mse_norm={mse_norm:.6g} "
        f"rmse_dq={float(np.sqrt(np.mean(err[:,0]**2))):.6g} "
        f"rmse_dqd={float(np.sqrt(np.mean(err[:,1]**2))):.6g}"
    )

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if len(rows) < 50:
        raise SystemExit("csv too short")

    t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    stage = np.array([r["stage"] for r in rows], dtype=object)
    q = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
    qd = np.array([float(r["dq_rad_s"]) for r in rows], dtype=np.float64)
    tau_ff = np.array([float(r["tau_Nm"]) for r in rows], dtype=np.float64)
    q_ref = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
    qd_ref = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)

    if "kp" in rows[0]:
        kp = np.array([float(r["kp"]) for r in rows], dtype=np.float64)
    else:
        kp = np.full((len(rows),), float(get(cfg, "real.kp", required=False, default=0.0)), dtype=np.float64)
    if "kd" in rows[0]:
        kd = np.array([float(r["kd"]) for r in rows], dtype=np.float64)
    else:
        kd = np.full((len(rows),), float(get(cfg, "real.kd", required=False, default=0.0)), dtype=np.float64)

    res = _open_loop_segment(
        model,
        stats,
        t=t,
        stage=stage,
        q=q,
        qd=qd,
        tau_ff=tau_ff,
        q_ref=q_ref,
        qd_ref=qd_ref,
        kp=kp,
        kd=kd,
        stage_name=str(args.stage),
        H=H,
        horizon_steps=int(args.horizon_steps),
        device=str(args.device),
    )
    print(
        f"[open-loop stage={res['stage']}] horizon={res['horizon_steps']} "
        f"q_rmse={res['q_rmse']:.6g} q_maxabs={res['q_maxabs']:.6g} "
        f"qd_rmse={res['qd_rmse']:.6g} qd_maxabs={res['qd_maxabs']:.6g}"
    )


if __name__ == "__main__":
    main()
