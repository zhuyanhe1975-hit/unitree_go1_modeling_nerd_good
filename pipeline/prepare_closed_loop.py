from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import numpy as np

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


def _compute_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = x.mean(axis=(0, 1)).astype(np.float32)
    x_std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = (y.std(axis=0) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _segment_indices(t: np.ndarray, stage: np.ndarray, gap_thr_s: float, min_len: int) -> list[tuple[int, int]]:
    cuts = [0]
    for i in range(1, len(t)):
        if stage[i] != stage[i - 1] or (float(t[i]) - float(t[i - 1])) > float(gap_thr_s):
            cuts.append(i)
    cuts.append(len(t))
    segs = []
    for i in range(len(cuts) - 1):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if b - a >= int(min_len):
            segs.append((a, b))
    return segs


def prepare_closed_loop_csv_dataset(
    cfg: Dict[str, Any],
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
    out_npz: str,
    stats_npz: str | None = None,
) -> None:
    """
    Prepare a *command-conditioned* dataset for single-joint digital-twin prediction.

    Deployment constraint:
      Digital twin observations must come from commands (q_ref, qd_ref, kp, kd, tau_ff, dt)
      and the digital twin internal state (q_hat, qd_hat). No direct access to measured
      real motor state at inference time.

    We use the Unitree joint low-level torque synthesis model:
      tau_cmd_hat = kp*(q_ref - q_hat) + kd*(qd_ref - qd_hat) + tau_ff

    For training windows we teacher-force q_hat=q, qd_hat=qd from the log, but at runtime
    tau_cmd_hat is computed from the twin's predicted state.

    Feature per step (D=5, minimal):
      [sin(q_hat), cos(q_hat), qd_hat, tau_cmd_hat, dt]
    Target:
      y = [delta_q, delta_qd] to next step
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    stage = np.asarray(stage, dtype=object).reshape(-1)
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    tau_ff = np.asarray(tau_ff, dtype=np.float64).reshape(-1)
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    qd_ref = np.asarray(qd_ref, dtype=np.float64).reshape(-1)
    kp = np.asarray(kp, dtype=np.float64).reshape(-1)
    kd = np.asarray(kd, dtype=np.float64).reshape(-1)

    n = int(len(t))
    if not (len(stage) == len(q) == len(qd) == len(tau_ff) == len(q_ref) == len(qd_ref) == len(kp) == len(kd) == n):
        raise ValueError("all input arrays must have the same length")
    if n < 50:
        raise ValueError("log too short")

    # dt series
    if n >= 2:
        dt_med = float(np.median(np.diff(t)))
    else:
        dt_med = 0.0

    gap_thr = max(0.05, 10.0 * dt_med) if dt_med > 0 else 0.05
    dt = np.zeros((n,), dtype=np.float64)
    dt[0] = dt_med
    dt[1:] = np.clip(np.diff(t), 0.0, gap_thr)

    # Optional LPF for qd
    qd_lpf_hz = float(get(cfg, "data.real.qd_lpf_hz", required=False, default=0.0))
    if qd_lpf_hz > 0 and dt_med > 0:
        qd = _one_pole_lpf(qd, dt=dt_med, cutoff_hz=qd_lpf_hz)

    H = int(get(cfg, "model.history_len"))
    segs = _segment_indices(t, stage, gap_thr_s=gap_thr, min_len=H + 2)
    if len(segs) == 0:
        raise ValueError("no valid continuous segments found after segmentation")

    xs = []
    ys = []
    for a, b in segs:
        for k in range(a + (H - 1), b - 1):
            sl = slice(k - (H - 1), k + 1)
            qh = q[sl]
            qdh = qd[sl]
            qrh = q_ref[sl]
            qdrh = qd_ref[sl]
            kph = kp[sl]
            kdh = kd[sl]
            tau_ff_h = tau_ff[sl]
            dth = dt[sl]

            # teacher forcing uses log state as twin state here
            tau_cmd_hat = (kph * (qrh - qh) + kdh * (qdrh - qdh) + tau_ff_h).astype(np.float64)

            feat = np.stack([np.sin(qh), np.cos(qh), qdh, tau_cmd_hat, dth], axis=-1).astype(np.float32)  # [H,5]
            y = np.array([q[k + 1] - q[k], qd[k + 1] - qd[k]], dtype=np.float32)  # [2]

            xs.append(feat)
            ys.append(y)

    x = np.stack(xs, axis=0).astype(np.float32)  # [N,H,5]
    y = np.stack(ys, axis=0).astype(np.float32)  # [N,2]

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
        feature_names=np.array(["sin_q", "cos_q", "qd", "tau_cmd_hat", "dt"], dtype=object),
        dt_med=np.array([dt_med], dtype=np.float64),
        gap_thr=np.array([gap_thr], dtype=np.float64),
        num_segments=np.array([len(segs)], dtype=np.int64),
        stats_path=np.array([stats_npz or ""], dtype=object),
    )


def load_prepared_closed_loop(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = dict(np.load(npz_path, allow_pickle=True))
    return ds["x"].astype(np.float32), ds["y"].astype(np.float32)

