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


def _zero_phase_one_pole_lpf(x: np.ndarray, dt: float, cutoff_hz: float) -> np.ndarray:
    """
    Zero-phase variant via forward-backward filtering (filtfilt-style) using the one-pole LPF.

    This removes phase lag at the cost of stronger smoothing (effectively higher order).
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if cutoff_hz <= 0 or len(x) == 0:
        return x
    y_f = _one_pole_lpf(x, dt=dt, cutoff_hz=cutoff_hz)
    y_b = _one_pole_lpf(y_f[::-1], dt=dt, cutoff_hz=cutoff_hz)[::-1]
    return y_b


def _get_qd_filter_cfg(cfg: Dict[str, Any]) -> tuple[str, float, bool]:
    """
    Returns (method, cutoff_hz, use_filtered).

    Backward compatible:
      - If data.real.qd_filter.* missing, falls back to data.real.qd_lpf_hz.
      - If cutoff_hz<=0, use_filtered defaults to False.
    """
    cutoff = float(get(cfg, "data.real.qd_filter.cutoff_hz", required=False, default=np.nan))
    if not np.isfinite(cutoff):
        cutoff = float(get(cfg, "data.real.qd_lpf_hz", required=False, default=0.0))
    method = str(get(cfg, "data.real.qd_filter.method", required=False, default="one_pole"))

    default_use = bool(cutoff > 0)
    use_filtered = bool(get(cfg, "data.real.qd_use_filtered", required=False, default=default_use))
    if cutoff <= 0:
        use_filtered = False
    return method, cutoff, use_filtered


def _compute_stats(x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = x.mean(axis=(0, 1)).astype(np.float32)
    x_std = (x.std(axis=(0, 1)) + 1e-6).astype(np.float32)
    y_mean = y.mean(axis=0).astype(np.float32)
    y_std = (y.std(axis=0) + 1e-6).astype(np.float32)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _feature_names_for_set(feature_set: str) -> list[str]:
    fs = str(feature_set).strip().lower()
    if fs in ("minimal", "min"):
        return ["sin_q", "cos_q", "qd", "tau_cmd_hat", "dt"]
    if fs in ("full", "rich"):
        # Keep tau_cmd_hat even though it's redundant with (kp,kd,e_q,e_qd,tau_ff),
        # since it is a meaningful "simulated applied torque" signal.
        return ["sin_q", "cos_q", "qd", "e_q", "e_qd", "kp", "kd", "tau_cmd_hat", "dt"]
    raise ValueError(f"unknown feature_set: {feature_set!r} (expected 'minimal' or 'full')")


def _build_features(
    *,
    q: np.ndarray,
    qd: np.ndarray,
    q_ref: np.ndarray,
    qd_ref: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    tau_ff: np.ndarray,
    dt: np.ndarray,
    feature_names: list[str],
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    qd = np.asarray(qd, dtype=np.float64).reshape(-1)
    q_ref = np.asarray(q_ref, dtype=np.float64).reshape(-1)
    qd_ref = np.asarray(qd_ref, dtype=np.float64).reshape(-1)
    kp = np.asarray(kp, dtype=np.float64).reshape(-1)
    kd = np.asarray(kd, dtype=np.float64).reshape(-1)
    tau_ff = np.asarray(tau_ff, dtype=np.float64).reshape(-1)
    dt = np.asarray(dt, dtype=np.float64).reshape(-1)

    e_q = q_ref - q
    e_qd = qd_ref - qd
    tau_cmd_hat = kp * e_q + kd * e_qd + tau_ff

    cols: list[np.ndarray] = []
    for name in feature_names:
        if name == "sin_q":
            cols.append(np.sin(q))
        elif name == "cos_q":
            cols.append(np.cos(q))
        elif name == "qd":
            cols.append(qd)
        elif name == "e_q":
            cols.append(e_q)
        elif name == "e_qd":
            cols.append(e_qd)
        elif name == "kp":
            cols.append(kp)
        elif name == "kd":
            cols.append(kd)
        elif name == "tau_cmd_hat":
            cols.append(tau_cmd_hat)
        elif name == "dt":
            cols.append(dt)
        else:
            raise ValueError(f"unknown feature name: {name!r}")

    feat = np.stack(cols, axis=-1).astype(np.float32)
    return feat


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

    # Optional filtering for qd (keep raw qd intact)
    qd_raw = qd.astype(np.float64).copy()
    method, cutoff_hz, use_filtered = _get_qd_filter_cfg(cfg)
    qd_filt = qd_raw
    if cutoff_hz > 0 and dt_med > 0:
        if method == "zero_phase_one_pole":
            qd_filt = _zero_phase_one_pole_lpf(qd_raw, dt=dt_med, cutoff_hz=cutoff_hz)
        elif method == "one_pole":
            qd_filt = _one_pole_lpf(qd_raw, dt=dt_med, cutoff_hz=cutoff_hz)
        else:
            raise ValueError(f"unknown qd filter method: {method!r} (expected 'one_pole' or 'zero_phase_one_pole')")

    qd_used = qd_filt if use_filtered else qd_raw

    feature_set = str(get(cfg, "data.real.feature_set", required=False, default="minimal"))
    feature_names = _feature_names_for_set(feature_set)

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
            qdh = qd_used[sl]
            qrh = q_ref[sl]
            qdrh = qd_ref[sl]
            kph = kp[sl]
            kdh = kd[sl]
            tau_ff_h = tau_ff[sl]
            dth = dt[sl]

            feat = _build_features(
                q=qh,
                qd=qdh,
                q_ref=qrh,
                qd_ref=qdrh,
                kp=kph,
                kd=kdh,
                tau_ff=tau_ff_h,
                dt=dth,
                feature_names=feature_names,
            )  # [H,D]
            y = np.array([q[k + 1] - q[k], qd_used[k + 1] - qd_used[k]], dtype=np.float32)  # [2]

            xs.append(feat)
            ys.append(y)

    x = np.stack(xs, axis=0).astype(np.float32)  # [N,H,D]
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
        feature_set=np.array([feature_set], dtype=object),
        feature_names=np.array(feature_names, dtype=object),
        dt_med=np.array([dt_med], dtype=np.float64),
        gap_thr=np.array([gap_thr], dtype=np.float64),
        num_segments=np.array([len(segs)], dtype=np.int64),
        qd_filter_method=np.array([method], dtype=object),
        qd_filter_cutoff_hz=np.array([float(cutoff_hz)], dtype=np.float64),
        qd_use_filtered=np.array([bool(use_filtered)], dtype=np.bool_),
        stats_path=np.array([stats_npz or ""], dtype=object),
    )


def load_prepared_closed_loop(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    ds = dict(np.load(npz_path, allow_pickle=True))
    return ds["x"].astype(np.float32), ds["y"].astype(np.float32)
