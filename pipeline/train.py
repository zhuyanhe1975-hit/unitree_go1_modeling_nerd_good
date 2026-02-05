from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from project_config import get
from .model import CausalTransformer
from .prepare_closed_loop import _build_features, _feature_names_for_set


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def _resolve_device(cfg: Dict[str, Any]) -> str:
    dev = str(get(cfg, "sim.device"))
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] sim.device is cuda but CUDA not available; falling back to cpu")
        return "cpu"
    return dev


def _open_loop_sine_qd_rmse(
    cfg: Dict[str, Any],
    model: CausalTransformer,
    stats: dict,
    *,
    H: int,
    device: str,
) -> float:
    """
    Lightweight open-loop validation for closed-loop (command-conditioned) real-CSV modeling.

    This measures what we actually care about: long-horizon rollout stability under command-only
    observations. We intentionally implement a minimal 'sine' segment evaluator here so the
    trainer can select the checkpoint that best rollouts, not just best one-step teacher forcing.
    """
    import csv

    csv_path = str(get(cfg, "train.open_loop_val.csv", required=False, default="")) or str(get(cfg, "paths.real_csv"))
    stage_name = str(get(cfg, "train.open_loop_val.stage", required=False, default="sine"))
    horizon_steps = int(get(cfg, "train.open_loop_val.horizon_steps", required=False, default=300))
    qd_col = str(get(cfg, "data.real.qd_col", required=False, default="dq_rad_s"))
    feature_set = str(get(cfg, "data.real.feature_set", required=False, default="minimal"))
    feature_names = _feature_names_for_set(feature_set)

    # tau_ff policy
    use_tau_ff_from_csv = bool(get(cfg, "data.real.use_tau_ff_from_csv", required=False, default=False))

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
        cols = list(r.fieldnames or [])

    required = ["t_s", "stage", "q_rad", qd_col, "tau_Nm", "q_ref_rad", "dq_ref_rad_s"]
    missing = [k for k in required if k not in cols]
    if missing:
        raise RuntimeError(f"open-loop val missing columns: {missing} in {csv_path}")

    t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    stage = np.array([r["stage"] for r in rows], dtype=object)
    q = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
    qd = np.array([float(r[qd_col]) for r in rows], dtype=np.float64)
    tau_ff = (
        np.array([float(r["tau_Nm"]) for r in rows], dtype=np.float64)
        if use_tau_ff_from_csv
        else np.zeros((len(rows),), dtype=np.float64)
    )
    q_ref = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
    qd_ref = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)

    if "kp" in cols:
        kp = np.array([float(r["kp"]) for r in rows], dtype=np.float64)
    else:
        kp = np.full((len(rows),), float(get(cfg, "real.kp", required=False, default=0.0)), dtype=np.float64)
    if "kd" in cols:
        kd = np.array([float(r["kd"]) for r in rows], dtype=np.float64)
    else:
        kd = np.full((len(rows),), float(get(cfg, "real.kd", required=False, default=0.0)), dtype=np.float64)

    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    gap_thr = max(0.05, 10.0 * dt_med) if dt_med > 0 else 0.05

    # Find first usable segment for the stage.
    cuts = [0]
    for i in range(1, len(t)):
        if stage[i] != stage[i - 1] or (t[i] - t[i - 1]) > gap_thr:
            cuts.append(i)
    cuts.append(len(t))
    a = None
    for i in range(len(cuts) - 1):
        s0, s1 = int(cuts[i]), int(cuts[i + 1])
        if str(stage[s0]) != stage_name:
            continue
        if (s1 - s0) >= (H + horizon_steps + 2):
            a = s0
            break
    if a is None:
        raise RuntimeError(f"open-loop val: no segment for stage={stage_name!r} (need >= H+horizon+2)")

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]

    q_hist = q[a : a + H].astype(np.float64).copy()
    qd_hist = qd[a : a + H].astype(np.float64).copy()
    q_pred = float(q_hist[-1])
    qd_pred = float(qd_hist[-1])

    x_buf = np.zeros((1, H, int(x_mean.shape[0])), dtype=np.float32)
    e_qd = np.zeros((horizon_steps,), dtype=np.float64)

    model.eval()
    with torch.no_grad():
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

            feat = _build_features(
                q=q_hist,
                qd=qd_hist,
                q_ref=qref_h,
                qd_ref=qdref_h,
                kp=kp_h,
                kd=kd_h,
                tau_ff=tau_ff_h,
                dt=dt_h,
                feature_names=feature_names,
            )
            x_buf[0] = (feat - x_mean[None, None, :]) / x_std[None, None, :]

            pred_n = model(torch.from_numpy(x_buf).to(device)).detach().cpu().numpy().reshape(-1).astype(np.float32)
            delta = pred_n * y_std + y_mean

            q_pred = q_pred + float(delta[0])
            qd_pred = qd_pred + float(delta[1])

            q_hist[:-1] = q_hist[1:]
            qd_hist[:-1] = qd_hist[1:]
            q_hist[-1] = q_pred
            qd_hist[-1] = qd_pred

            e_qd[i] = qd_pred - float(qd[k + 1])

    return float(np.sqrt(np.mean(e_qd**2)))


def train_model(cfg: Dict[str, Any], dataset_npz: str, out_weights: str) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    # Split
    n = x.shape[0]
    val_ratio = float(get(cfg, "data.prepare.val_ratio"))
    n_val = max(1, int(n * val_ratio))
    idx = np.arange(n)
    np.random.default_rng(int(get(cfg, "project.seed"))).shuffle(idx)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    device = _resolve_device(cfg)
    model = CausalTransformer(
        input_dim=int(x.shape[-1]),
        output_dim=int(y.shape[-1]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)

    opt = optim.Adam(model.parameters(), lr=float(get(cfg, "train.lr")))
    loss_fn = nn.MSELoss()

    bs = int(get(cfg, "train.batch_size"))
    tr_loader = DataLoader(WindowDataset(x_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader = DataLoader(WindowDataset(x_val, y_val), batch_size=bs, shuffle=False)

    epochs = int(get(cfg, "train.epochs"))
    save_best = bool(get(cfg, "train.save_best", required=False, default=True))
    save_last = bool(get(cfg, "train.save_last", required=False, default=False))
    best_metric = float("inf")
    best_epoch = -1
    best_state: dict | None = None

    patience = int(get(cfg, "train.early_stop.patience", required=False, default=0))
    min_delta = float(get(cfg, "train.early_stop.min_delta", required=False, default=0.0))
    no_improve = 0

    open_loop_enabled = bool(get(cfg, "train.open_loop_val.enabled", required=False, default=False))
    open_loop_every = int(get(cfg, "train.open_loop_val.every", required=False, default=20))
    stats = {k: ds[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    last_open_loop = float("nan")
    metric_name = "val_mse"

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += float(loss.item()) * int(xb.shape[0])
                va_n += int(xb.shape[0])

        tr_mse = tr_loss / max(1, tr_n)
        va_mse = va_loss / max(1, va_n)

        # Model selection metric:
        # - default: one-step val_mse
        # - if open-loop validation enabled: only update best/early-stop on open-loop eval steps,
        #   using open-loop qd_rmse as the selection metric.
        metric = float(va_mse)
        metric_name = "val_mse"
        eval_step = True
        improved = False

        should_run_open_loop = open_loop_enabled and (ep % max(1, open_loop_every) == 0 or ep == 0)
        if open_loop_enabled and should_run_open_loop:
            try:
                last_open_loop = _open_loop_sine_qd_rmse(cfg, model, stats, H=int(x.shape[1]), device=device)
                metric = float(last_open_loop)
                metric_name = "open_loop_qd_rmse"
            except Exception as e:
                print(f"[warn] open-loop val failed: {e}")
                # Fall back to one-step val_mse for this eval step only.
                metric = float(va_mse)
                metric_name = "val_mse"
        elif open_loop_enabled:
            # Between open-loop eval steps, do not update best/early-stop based on val_mse,
            # otherwise we would incorrectly label many epochs as "[best]" while the
            # open-loop metric was not evaluated.
            eval_step = False
            no_improve = no_improve  # explicit: keep counter unchanged

        if eval_step:
            improved = (best_metric - metric) > float(min_delta)
            if improved:
                best_metric = float(metric)
                best_epoch = int(ep)
                no_improve = 0
                if save_best:
                    best_state = {
                        "model": model.state_dict(),
                        "input_dim": x.shape[-1],
                        "output_dim": y.shape[-1],
                        "embed_dim": int(get(cfg, "model.embed_dim")),
                        "num_layers": int(get(cfg, "model.num_layers")),
                        "num_heads": int(get(cfg, "model.num_heads")),
                        "history_len": int(get(cfg, "model.history_len")),
                        "best_epoch": int(ep + 1),
                        "best_metric_name": str(metric_name),
                        "best_metric_value": float(best_metric),
                    }
                    torch.save(best_state, out_weights)
            else:
                no_improve += 1

        if (ep + 1) % max(1, epochs // 10) == 0 or ep == 0 or should_run_open_loop or improved:
            msg = f"epoch {ep+1}/{epochs} train_mse={tr_mse:.6f} val_mse={va_mse:.6f}"
            if should_run_open_loop and math.isfinite(last_open_loop):
                msg += f" open_loop_qd_rmse={last_open_loop:.6f}"
            if improved:
                msg += " [best]"
            print(msg)

        if patience > 0 and eval_step and no_improve >= patience:
            print(
                f"[early-stop] no val improvement >{min_delta} for {patience} evals; "
                f"best_epoch={best_epoch+1} best_{metric_name}={best_metric:.6f}"
            )
            break

    if save_last:
        out_last = out_weights.replace(".pt", "_last.pt") if out_weights.endswith(".pt") else (out_weights + "_last.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "input_dim": x.shape[-1],
                "output_dim": y.shape[-1],
                "embed_dim": int(get(cfg, "model.embed_dim")),
                "num_layers": int(get(cfg, "model.num_layers")),
                "num_heads": int(get(cfg, "model.num_heads")),
                "history_len": int(get(cfg, "model.history_len")),
                "last_epoch": int(ep + 1),
                "last_val_mse": float(va_mse),
            },
            out_last,
        )

    if not save_best:
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
    else:
        # Ensure we saved at least once (in case epochs==0 or val never improved).
        if best_state is None:
            best_state = {
                "model": model.state_dict(),
                "input_dim": x.shape[-1],
                "output_dim": y.shape[-1],
                "embed_dim": int(get(cfg, "model.embed_dim")),
                "num_layers": int(get(cfg, "model.num_layers")),
                "num_heads": int(get(cfg, "model.num_heads")),
                "history_len": int(get(cfg, "model.history_len")),
                "best_epoch": int(ep + 1),
                "best_metric_name": "val_mse",
                "best_metric_value": float(va_mse),
            }
            torch.save(best_state, out_weights)


def finetune_model(cfg: Dict[str, Any], dataset_npz: str, base_weights: str, out_weights: str) -> None:
    ds = dict(np.load(dataset_npz, allow_pickle=True))
    x = ds["x"].astype(np.float32)
    y = ds["y"].astype(np.float32)

    device = _resolve_device(cfg)
    ckpt = torch.load(base_weights, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(get(cfg, "model.embed_dim")),
        num_layers=int(get(cfg, "model.num_layers")),
        num_heads=int(get(cfg, "model.num_heads")),
        history_len=int(get(cfg, "model.history_len")),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    opt = optim.Adam(model.parameters(), lr=float(get(cfg, "train.finetune_lr")))
    loss_fn = nn.MSELoss()
    bs = int(get(cfg, "train.batch_size"))
    loader = DataLoader(WindowDataset(x, y), batch_size=bs, shuffle=True)

    epochs = int(get(cfg, "train.finetune_epochs"))
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            print(f"finetune epoch {ep+1}/{epochs} mse={tr_loss/max(1,tr_n):.6f}")

    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": ckpt["input_dim"],
            "output_dim": ckpt["output_dim"],
            "embed_dim": int(get(cfg, "model.embed_dim")),
            "num_layers": int(get(cfg, "model.num_layers")),
            "num_heads": int(get(cfg, "model.num_heads")),
            "history_len": int(get(cfg, "model.history_len")),
        },
        out_weights,
    )
