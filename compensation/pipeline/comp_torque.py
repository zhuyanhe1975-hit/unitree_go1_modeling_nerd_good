from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from project_config import ensure_dir, get
from .features import state_to_features
from .model import CausalTransformer
from .train import _resolve_device


class TorqueDataset(Dataset):
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
    y_mean = y.mean(axis=(0, 1)).astype(np.float32)
    y_std = (y.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def prepare_torque_dataset(
    cfg: Dict[str, Any],
    raw_npz: str,
    out_npz: str,
    stats_npz: str | None = None,
) -> None:
    """
    Build inverse-dynamics dataset:
      input: history of state features (sin q, cos q, qd, optional qdd/temp)
      target: torque (tau_out if available else tau_cmd) in motor-side units.
    """
    ds = dict(np.load(raw_npz, allow_pickle=True))
    q = np.asarray(ds["q_out"], dtype=np.float64).reshape(-1)
    qd = np.asarray(ds["qd_out"], dtype=np.float64).reshape(-1)
    tau_cmd = np.asarray(ds.get("tau_cmd", []), dtype=np.float64).reshape(-1)
    tau_out = np.asarray(ds.get("tau_out", []), dtype=np.float64).reshape(-1)
    temp = np.asarray(ds.get("temp", []), dtype=np.float64).reshape(-1)
    qdd = np.asarray(ds.get("qdd_out", []), dtype=np.float64).reshape(-1) if "qdd_out" in ds else None

    tau_target = tau_out if tau_out.size == q.size else tau_cmd
    if tau_target.size != q.size:
        raise KeyError("raw log missing usable torque target (tau_out or tau_cmd)")
    if q.shape != qd.shape:
        raise ValueError("q/qd shape mismatch")

    H = int(get(cfg, "model.history_len"))
    T = q.shape[0]
    if T < H:
        raise ValueError("trajectory too short for history_len")

    feat_state = state_to_features(q, qd).astype(np.float32)  # [T,3]
    cols = [feat_state]
    if qdd is not None and qdd.size == T:
        cols.append(qdd.astype(np.float32).reshape(-1, 1))
    if temp.size == T:
        cols.append(temp.astype(np.float32).reshape(-1, 1))
    feat_full = np.concatenate(cols, axis=-1)  # [T, D]

    # Delta torque target: tau[k+1] - tau[k]
    delta_tau = (tau_target[1:] - tau_target[:-1]).astype(np.float32)  # [T-1]

    xs = []
    ys = []
    # window ends at index k (>= H-1) and predicts delta_tau at k (which corresponds to tau[k+1]-tau[k])
    for k in range(H - 1, T - 1):
        x_win = feat_full[k - (H - 1) : k + 1]  # [H, Din]
        y_t = delta_tau[k]  # scalar
        xs.append(x_win)
        ys.append(y_t)

    x = np.stack(xs, axis=0).astype(np.float32)  # [N,H,D]
    y = np.array(ys, dtype=np.float32).reshape(-1, 1)  # [N,1]

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
        stats_path=np.array([stats_npz or ""], dtype=object),
        parent_raw=np.array([raw_npz], dtype=object),
    )


def train_torque_model(
    cfg: Dict[str, Any],
    dataset_npz: str,
    out_weights: str,
    base_weights: str | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)

    ckpt = None
    if base_weights and os.path.exists(base_weights):
        maybe_ckpt = torch.load(base_weights, map_location=torch.device(device))
        in_match = int(maybe_ckpt["input_dim"]) == int(x.shape[-1])
        out_match = int(maybe_ckpt["output_dim"]) == int(y.shape[-1])
        if in_match and out_match:
            ckpt = maybe_ckpt
        else:
            print(f"[warn] base_weights dim mismatch (ckpt {maybe_ckpt['input_dim']}->{maybe_ckpt['output_dim']}, data {x.shape[-1]}->{y.shape[-1]}), training from scratch")

    if ckpt is not None:
        model = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(get(cfg, "model.embed_dim")),
            num_layers=int(get(cfg, "model.num_layers")),
            num_heads=int(get(cfg, "model.num_heads")),
            history_len=int(get(cfg, "model.history_len")),
        ).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        model = CausalTransformer(
            input_dim=int(x.shape[-1]),
            output_dim=int(y.shape[-1]),
            embed_dim=int(get(cfg, "model.embed_dim")),
            num_layers=int(get(cfg, "model.num_layers")),
            num_heads=int(get(cfg, "model.num_heads")),
            history_len=int(get(cfg, "model.history_len")),
        ).to(device)

    bs = int(batch_size or get(cfg, "train.batch_size"))
    ep = int(epochs or (get(cfg, "train.finetune_epochs") if base_weights else get(cfg, "train.epochs")))
    learning_rate = float(lr or (get(cfg, "train.finetune_lr") if base_weights else get(cfg, "train.lr")))

    loader = DataLoader(TorqueDataset(x, y), batch_size=bs, shuffle=True)
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
            phase = "finetune" if base_weights else "train"
            print(f"[torque-{phase}] epoch {e+1}/{ep} mse={total/max(1,n):.6f}")

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
