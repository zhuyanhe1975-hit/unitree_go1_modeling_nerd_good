from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pipeline.features import state_to_features
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from project_config import ensure_dir, get


def _one_pole_lpf(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if cutoff_hz <= 0 or len(x) == 0:
        return x
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    alpha = float(dt) / (rc + float(dt))
    y = np.empty_like(x)
    y[0] = x[0]
    for k in range(1, len(x)):
        y[k] = y[k - 1] + alpha * (x[k] - y[k - 1])
    return y


class _Dataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _compute_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = x.mean(axis=(0, 1)).astype(np.float32)
    x_std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = (y.std(axis=0) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def prepare_friction_dataset(
    cfg: Dict[str, Any],
    raw_npz: str,
    out_npz: str,
    stats_npz: str | None = None,
    qd_source: str = "from_log",
    qd_lpf_hz: float = 30.0,
) -> None:
    """
    Prepare a dataset to predict *friction-like* torque from state only.

    Target construction (motor/out side consistent with your log conventions):
      tau_fric[k] = tau_out[k] - J * qdd[k]

    Inputs:
      x[k] = history of [sin(q), cos(q), qd, (temp?)] up to time k
    Target:
      y[k] = tau_fric[k]

    Note:
    - This is a practical \"effective friction\" target. In closed-loop real data it can include
      unmodeled effects beyond pure Coulomb/viscous friction.
    """
    ds = dict(np.load(raw_npz, allow_pickle=True))
    t = np.asarray(ds.get("t", []), dtype=np.float64).reshape(-1)
    q = np.asarray(ds.get("q_out", []), dtype=np.float64).reshape(-1)
    qd = np.asarray(ds.get("qd_out", []), dtype=np.float64).reshape(-1)
    tau_out = np.asarray(ds.get("tau_out", []), dtype=np.float64).reshape(-1)
    temp = np.asarray(ds.get("temp", []), dtype=np.float64).reshape(-1) if "temp" in ds else None

    if len(t) < 3 or not (len(t) == len(q) == len(qd) == len(tau_out)):
        raise ValueError("raw log must contain t, q_out, qd_out, tau_out with equal lengths and len>=3")

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("invalid dt computed from t")

    if qd_source == "from_q":
        qd2 = np.zeros_like(q)
        qd2[1:] = (q[1:] - q[:-1]) / dt
        qd = qd2
    elif qd_source != "from_log":
        raise ValueError("qd_source must be 'from_log' or 'from_q'")

    if qd_lpf_hz > 0:
        qd = _one_pole_lpf(qd, dt=dt, cutoff_hz=float(qd_lpf_hz))

    qdd = np.zeros_like(qd)
    qdd[1:] = (qd[1:] - qd[:-1]) / dt

    J = float(get(cfg, "motor.rotor_inertia_kg_m2", required=False, default=0.0))
    tau_fric = (tau_out - J * qdd).astype(np.float32).reshape(-1, 1)

    feat_state = state_to_features(q, qd).astype(np.float32)  # [T,3]
    if temp is not None and len(temp) == len(t):
        feat_full = np.concatenate([feat_state, temp.astype(np.float32).reshape(-1, 1)], axis=-1)  # [T,4]
    else:
        feat_full = feat_state  # [T,3]

    H = int(get(cfg, "model.history_len"))
    T = int(len(t))
    if T < H:
        raise ValueError("trajectory too short for history_len")

    xs = []
    ys = []
    for k in range(H - 1, T):
        xs.append(feat_full[k - (H - 1) : k + 1])  # [H,D]
        ys.append(tau_fric[k])  # [1]

    x = np.stack(xs, axis=0).astype(np.float32)  # [N,H,D]
    y = np.stack(ys, axis=0).astype(np.float32).reshape(-1, 1)  # [N,1]

    val_ratio = float(get(cfg, "data.prepare.val_ratio", required=False, default=0.1))
    val_ratio = float(np.clip(val_ratio, 0.0, 0.5))
    n = int(x.shape[0])
    n_train = int(round(n * (1.0 - val_ratio)))
    n_train = max(1, min(n - 1, n_train)) if n >= 2 else n
    train_idx = np.arange(0, n_train, dtype=np.int64)
    val_idx = np.arange(n_train, n, dtype=np.int64)

    stats = _compute_stats(x, y)
    if stats_npz is not None:
        ensure_dir(os.path.dirname(stats_npz) or ".")
        np.savez(stats_npz, **stats)

    x_n = (x - stats["x_mean"]) / stats["x_std"]
    y_n = (y - stats["y_mean"]) / stats["y_std"]

    ensure_dir(os.path.dirname(out_npz) or ".")
    np.savez(
        out_npz,
        x=x_n,
        y=y_n,
        x_mean=stats["x_mean"],
        x_std=stats["x_std"],
        y_mean=stats["y_mean"],
        y_std=stats["y_std"],
        train_idx=train_idx,
        val_idx=val_idx,
        dt=np.array([dt], dtype=np.float64),
        qd_source=np.array([qd_source], dtype=object),
        qd_lpf_hz=np.array([qd_lpf_hz], dtype=np.float64),
        stats_path=np.array([stats_npz or ""], dtype=object),
        parent_raw=np.array([raw_npz], dtype=object),
    )


def train_friction_model(
    cfg: Dict[str, Any],
    dataset_npz: str,
    out_weights: str,
    split: str = "train",
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x_all = ds["x"].astype(np.float32)
    y_all = ds["y"].astype(np.float32)

    if split == "train" and "train_idx" in ds:
        idx = ds["train_idx"].astype(np.int64)
    elif split == "val" and "val_idx" in ds:
        idx = ds["val_idx"].astype(np.int64)
    else:
        idx = None

    x = x_all[idx] if idx is not None else x_all
    y = y_all[idx] if idx is not None else y_all

    device = _resolve_device(cfg)
    model = CausalTransformer(
        input_dim=int(x.shape[-1]),
        output_dim=int(y.shape[-1]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)

    bs = int(batch_size or get(cfg, "train.batch_size"))
    ep = int(epochs or get(cfg, "train.epochs"))
    learning_rate = float(lr or get(cfg, "train.lr"))

    loader = DataLoader(_Dataset(x, y), batch_size=bs, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for e in range(ep):
        model.train()
        total = 0.0
        n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * int(xb.shape[0])
            n += int(xb.shape[0])
        if (e + 1) % max(1, ep // 5) == 0 or e == 0:
            print(f"[friction] epoch {e+1}/{ep} mse={total/max(1,n):.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": x.shape[-1],
            "output_dim": y.shape[-1],
            "embed_dim": int(get(cfg, "model.embed_dim")),
            "num_layers": int(get(cfg, "model.num_layers")),
            "num_heads": int(get(cfg, "model.num_heads")),
            "history_len": int(get(cfg, "model.history_len")),
        },
        out_weights,
    )

