from __future__ import annotations

import argparse
import csv
import os
import time
from typing import Any, Dict
from pathlib import Path

import numpy as np
import torch

from pipeline.config import load_cfg
from pipeline.model import CausalTransformer
from pipeline.prepare_closed_loop import (
    _build_features,
    _feature_names_for_set,
    _get_qd_filter_cfg,
    _one_pole_lpf,
    _zero_phase_one_pole_lpf,
)
from project_config import ensure_dir, get


def _load_model(cfg: Dict[str, Any], weights_path: str, device: str) -> tuple[CausalTransformer, dict]:
    """
    Loads a checkpoint produced by pipeline/train.py.

    Note: older checkpoints may only store {model,input_dim,output_dim}. In that case we
    use config.json model hyperparams to rebuild the architecture.
    """
    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] --device is cuda but torch.cuda.is_available() is False; falling back to cpu")
        dev = "cpu"

    ckpt = torch.load(weights_path, map_location=torch.device(dev))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", get(cfg, "model.embed_dim"))),
        num_layers=int(ckpt.get("num_layers", get(cfg, "model.num_layers"))),
        num_heads=int(ckpt.get("num_heads", get(cfg, "model.num_heads"))),
        history_len=int(ckpt.get("history_len", get(cfg, "model.history_len"))),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def _find_stage_segment(t: np.ndarray, stage: np.ndarray, stage_name: str, *, H: int, horizon_steps: int) -> tuple[int, int]:
    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    gap_thr = max(0.05, 10.0 * dt_med) if dt_med > 0 else 0.05
    cuts = [0]
    for i in range(1, len(t)):
        if stage[i] != stage[i - 1] or (t[i] - t[i - 1]) > gap_thr:
            cuts.append(i)
    cuts.append(len(t))

    for i in range(len(cuts) - 1):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if str(stage[a]) != str(stage_name):
            continue
        if (b - a) >= (H + int(horizon_steps) + 2):
            return a, b
    raise ValueError(f"no segment found for stage='{stage_name}' with horizon={horizon_steps} and H={H}")


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
    a, _b = _find_stage_segment(t, stage, stage_name, H=H, horizon_steps=horizon_steps)

    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    feature_names = list(stats.get("feature_names", []))
    if not feature_names:
        feature_names = ["sin_q", "cos_q", "qd", "tau_cmd_hat", "dt"]

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


@torch.no_grad()
def _open_loop_rollout_trace(
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
    """
    Open-loop rollout trace for visualization:
      - seed (q_hat, qd_hat) with real log for the first H steps
      - roll forward using commands + the twin's internal state only
    """
    a, b = _find_stage_segment(t, stage, stage_name, H=H, horizon_steps=horizon_steps)
    x_mean = stats["x_mean"]
    x_std = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std = stats["y_std"]
    feature_names = list(stats.get("feature_names", []))
    if not feature_names:
        feature_names = ["sin_q", "cos_q", "qd", "tau_cmd_hat", "dt"]

    k0 = a + (H - 1)
    k1 = k0 + int(horizon_steps)
    if k1 + 1 >= b:
        raise ValueError("segment too short for requested horizon")

    q_hist = q[k0 - (H - 1) : k0 + 1].astype(np.float64).copy()
    qd_hist = qd[k0 - (H - 1) : k0 + 1].astype(np.float64).copy()
    q_pred = float(q_hist[-1])
    qd_pred = float(qd_hist[-1])

    x_buf = np.zeros((1, H, int(x_mean.shape[0])), dtype=np.float32)
    q_hat = np.zeros((horizon_steps + 1,), dtype=np.float64)
    qd_hat = np.zeros((horizon_steps + 1,), dtype=np.float64)
    tau_cmd_hat = np.zeros((horizon_steps,), dtype=np.float64)

    q_hat[0] = q_pred
    qd_hat[0] = qd_pred

    t_seg = t[k0 : k0 + horizon_steps + 1].astype(np.float64).copy()
    q_gt = q[k0 : k0 + horizon_steps + 1].astype(np.float64).copy()
    qd_gt = qd[k0 : k0 + horizon_steps + 1].astype(np.float64).copy()
    qref_seg = q_ref[k0 : k0 + horizon_steps + 1].astype(np.float64).copy()
    qdref_seg = qd_ref[k0 : k0 + horizon_steps + 1].astype(np.float64).copy()

    for i in range(horizon_steps):
        k = k0 + i
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

        tau_cmd_h = kp_h * (qref_h - q_hist) + kd_h * (qdref_h - qd_hist) + tau_ff_h
        tau_cmd_hat[i] = float(tau_cmd_h[-1])

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

        q_hat[i + 1] = q_pred
        qd_hat[i + 1] = qd_pred

    return {
        "stage": str(stage_name),
        "t": t_seg,
        "q_gt": q_gt,
        "qd_gt": qd_gt,
        "q_hat": q_hat,
        "qd_hat": qd_hat,
        "q_ref": qref_seg,
        "qd_ref": qdref_seg,
        "tau_cmd_hat": tau_cmd_hat,
        "H": int(H),
        "k0": int(k0),
    }


def _reversal_centers_from_qd_ref(qd_ref: np.ndarray, *, speed_th: float) -> np.ndarray:
    qd_ref = np.asarray(qd_ref, dtype=np.float64).reshape(-1)
    s = np.sign(qd_ref)
    s[s == 0.0] = 1.0
    ch = np.where((s[1:] * s[:-1]) < 0.0)[0] + 1
    if ch.size == 0:
        return ch
    if speed_th > 0:
        ch = ch[np.abs(qd_ref[ch]) <= float(speed_th)]
    return ch


def _plot_reversal_windows(
    trace: dict,
    *,
    out_png: str,
    window_s: float,
    max_events: int,
    speed_th: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = trace["t"]
    q_gt = trace["q_gt"]
    qd_gt = trace["qd_gt"]
    qd_gt_raw = trace.get("qd_gt_raw", None)
    q_hat = trace["q_hat"]
    qd_hat = trace["qd_hat"]
    q_ref = trace["q_ref"]
    qd_ref = trace["qd_ref"]

    dt_med = float(np.median(np.diff(t)))
    half = max(5, int(round(float(window_s) / max(1e-9, dt_med))))

    centers = _reversal_centers_from_qd_ref(qd_ref, speed_th=float(speed_th))
    if centers.size == 0:
        raise ValueError("no reversal found in qd_ref for the selected segment")
    centers = centers[: int(max_events)]

    n = int(len(centers))
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(12, max(3.0, 2.4 * n)), sharex=False)
    if n == 1:
        axes = np.array([axes])

    for i, c in enumerate(centers):
        a = max(0, int(c) - half)
        b = min(len(t) - 1, int(c) + half)
        tt = t[a : b + 1] - t[int(c)]

        axq = axes[i, 0]
        axd = axes[i, 1]

        axq.plot(tt, q_ref[a : b + 1], label="q_ref", color="tab:gray", lw=1.0, alpha=0.8)
        axq.plot(tt, q_gt[a : b + 1], label="q_gt", color="k", lw=1.2, alpha=0.8)
        axq.plot(tt, q_hat[a : b + 1], label="q_hat (open-loop)", color="tab:red", lw=1.2)
        axq.axvline(0.0, color="tab:blue", lw=0.8, alpha=0.6)
        axq.grid(True, alpha=0.25)
        axq.set_ylabel("q (rad)")
        axq.set_title(f"reversal #{i+1} (t=0 at qd_ref sign-change)")

        axd.plot(tt, qd_ref[a : b + 1], label="qd_ref", color="tab:gray", lw=1.0, alpha=0.8)
        if qd_gt_raw is not None:
            axd.plot(tt, qd_gt_raw[a : b + 1], label="qd_gt_raw", color="0.7", lw=0.8, alpha=0.7)
        axd.plot(tt, qd_gt[a : b + 1], label="qd_gt", color="k", lw=1.2, alpha=0.85)
        axd.plot(tt, qd_hat[a : b + 1], label="qd_hat (open-loop)", color="tab:red", lw=1.2)
        axd.axvline(0.0, color="tab:blue", lw=0.8, alpha=0.6)
        axd.grid(True, alpha=0.25)
        axd.set_ylabel("qd (rad/s)")

        if i == 0:
            axq.legend(loc="best", fontsize=9)
            axd.legend(loc="best", fontsize=9)

    for ax in axes[-1]:
        ax.set_xlabel("time around reversal (s)")

    fig.suptitle(f"Open-loop digital twin around low-speed reversals (stage={trace['stage']})", y=0.995)
    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    fig.savefig(out_png, dpi=160)


def _pick_full_cycle_window_from_qd_ref(qd_ref: np.ndarray) -> tuple[int, int]:
    """
    Picks a window spanning one full sine-like cycle using qd_ref sign changes.

    qd_ref crosses 0 at turning points, so one full cycle spans two reversals:
      reversal 0 -> reversal 2
    """
    ch = _reversal_centers_from_qd_ref(qd_ref, speed_th=0.0)
    if ch.size < 3:
        raise ValueError("not enough reversals in qd_ref to define a full cycle (need >=3)")
    i0 = int(ch[0])
    i2 = int(ch[2])
    if i2 <= i0:
        raise ValueError("invalid reversal ordering")
    return i0, i2


def _plot_full_cycle(trace: dict, *, out_png: str) -> None:
    """
    Plot one full cycle of the sine segment (open-loop prediction vs ground-truth).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = trace["t"]
    q_gt = trace["q_gt"]
    qd_gt = trace["qd_gt"]
    qd_gt_raw = trace.get("qd_gt_raw", None)
    q_hat = trace["q_hat"]
    qd_hat = trace["qd_hat"]
    q_ref = trace["q_ref"]
    qd_ref = trace["qd_ref"]

    i0, i2 = _pick_full_cycle_window_from_qd_ref(qd_ref)
    sl = slice(i0, i2 + 1)
    tt = t[sl] - t[i0]

    e_q = q_hat[sl] - q_gt[sl]
    e_qd = qd_hat[sl] - qd_gt[sl]

    fig = plt.figure(figsize=(12, 9))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(tt, q_ref[sl], label="q_ref", color="tab:gray", lw=1.0, alpha=0.8)
    ax1.plot(tt, q_gt[sl], label="q_gt", color="k", lw=1.2, alpha=0.8)
    ax1.plot(tt, q_hat[sl], label="q_hat (open-loop)", color="tab:red", lw=1.2)
    ax1.grid(True, alpha=0.25)
    ax1.set_ylabel("q (rad)")
    ax1.set_title(f"Open-loop prediction on one full sine cycle (stage={trace['stage']})")
    ax1.legend(loc="best", fontsize=9)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(tt, qd_ref[sl], label="qd_ref", color="tab:gray", lw=1.0, alpha=0.8)
    if qd_gt_raw is not None:
        ax2.plot(tt, qd_gt_raw[sl], label="qd_gt_raw", color="0.7", lw=0.8, alpha=0.7)
    ax2.plot(tt, qd_gt[sl], label="qd_gt", color="k", lw=1.2, alpha=0.85)
    ax2.plot(tt, qd_hat[sl], label="qd_hat (open-loop)", color="tab:red", lw=1.2)
    ax2.grid(True, alpha=0.25)
    ax2.set_ylabel("qd (rad/s)")
    ax2.legend(loc="best", fontsize=9)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(tt, e_q, label="q_hat - q_gt", color="tab:blue", lw=1.2)
    ax3.plot(tt, e_qd, label="qd_hat - qd_gt", color="tab:orange", lw=1.2, alpha=0.9)
    ax3.axhline(0.0, color="k", lw=0.8, alpha=0.4)
    ax3.grid(True, alpha=0.25)
    ax3.set_xlabel("time (s)")
    ax3.set_ylabel("error")
    ax3.legend(loc="best", fontsize=9)

    fig.tight_layout()
    ensure_dir(os.path.dirname(out_png) or ".")
    fig.savefig(out_png, dpi=160)


def _open_loop_baseline(
    *,
    t: np.ndarray,
    stage: np.ndarray,
    q: np.ndarray,
    qd: np.ndarray,
    stage_name: str,
    H: int,
    horizon_steps: int,
) -> dict:
    """
    A minimal baseline that respects the same deployment constraint (no real q/qd after init):
      - use last qd from init window and integrate q with dt
      - keep qd constant
    """
    a, _b = _find_stage_segment(t, stage, stage_name, H=H, horizon_steps=horizon_steps)
    k0 = a + (H - 1)
    q_pred = float(q[k0])
    qd_pred = float(qd[k0])
    e_q = np.zeros((horizon_steps,), dtype=np.float64)
    e_qd = np.zeros((horizon_steps,), dtype=np.float64)
    for i in range(horizon_steps):
        k = k0 + i
        dt_k = float(t[k + 1] - t[k])
        q_pred = q_pred + qd_pred * dt_k
        # qd_pred constant
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


def _fmt(x: float) -> str:
    if not np.isfinite(x):
        return "nan"
    ax = abs(float(x))
    if ax >= 1e3 or (ax > 0 and ax < 1e-3):
        return f"{x:.3e}"
    return f"{x:.6g}"


def _to_md_table(rows: list[dict[str, Any]], *, title: str) -> str:
    headers = ["stage", "horizon", "q_rmse", "q_maxabs", "qd_rmse", "qd_maxabs"]
    lines = [f"### {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        line = "| " + " | ".join(
            [
                str(r["stage"]),
                str(r["horizon_steps"]),
                _fmt(float(r["q_rmse"])),
                _fmt(float(r["q_maxabs"])),
                _fmt(float(r["qd_rmse"])),
                _fmt(float(r["qd_maxabs"])),
            ]
        ) + " |"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="CSV path (default: paths.real_csv)")
    ap.add_argument("--dataset", default=None, help="prepared dataset npz (default: paths.real_csv_dataset)")
    ap.add_argument("--weights", default=None, help="model weights (default: paths.real_csv_model)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stage", default="sine", help="single stage name (used if --stages not provided)")
    ap.add_argument(
        "--stages",
        default="",
        help="comma-separated stage list to evaluate, e.g. 'sine,pos_sweep,vel_step' (overrides --stage)",
    )
    ap.add_argument("--horizon_steps", type=int, default=300)
    ap.add_argument("--baseline", action="store_true", help="also compute a simple open-loop baseline")
    ap.add_argument("--out_md", default="", help="write a markdown summary to this path (default: <runs_dir>/summary_*.md)")
    ap.add_argument(
        "--use_tau_ff_from_csv",
        action="store_true",
        help="use feedforward torque from CSV (otherwise tau_ff=0). Use only if tau_ff was actually commanded.",
    )
    ap.add_argument("--qd_col", default="", help="qd column name in CSV (default: data.real.qd_col or dq_rad_s)")
    ap.add_argument("--qd_filter_method", default="", help="one_pole or zero_phase_one_pole (override config)")
    ap.add_argument("--qd_filter_hz", type=float, default=float("nan"), help="qd filter cutoff Hz (override config)")
    qd_src = ap.add_mutually_exclusive_group()
    qd_src.add_argument("--qd_use_filtered", action="store_true", help="use filtered qd as ground-truth (override config)")
    qd_src.add_argument("--qd_use_raw", action="store_true", help="use raw qd as ground-truth (override config)")
    ap.add_argument("--plot_reversals", action="store_true", help="save a plot focused on low-speed reversal windows")
    ap.add_argument("--plot_horizon_steps", type=int, default=0, help="override horizon for plotting trace (default: --horizon_steps)")
    ap.add_argument("--reversal_window_s", type=float, default=0.5, help="half-window size around reversal (seconds)")
    ap.add_argument("--reversal_max_events", type=int, default=4, help="max number of reversals to plot")
    ap.add_argument(
        "--reversal_speed_th",
        type=float,
        default=0.5,
        help="filter reversals where |qd_ref| at crossing exceeds this threshold (rad/s). 0 disables.",
    )
    ap.add_argument("--out_png", default="", help="output png path (default: <runs_dir>/plot_reversals_*.png)")
    ap.add_argument("--plot_full_cycle", action="store_true", help="plot one full sine cycle open-loop trace")
    ap.add_argument("--out_full_png", default="", help="output png path for full-cycle plot (default: results/plot_full_cycle_*.png)")
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_cfg(args.config)

    # Optional overrides (mutate cfg in-memory)
    if str(args.qd_filter_method).strip() or np.isfinite(float(args.qd_filter_hz)):
        cfg.setdefault("data", {}).setdefault("real", {}).setdefault("qd_filter", {})
        if str(args.qd_filter_method).strip():
            cfg["data"]["real"]["qd_filter"]["method"] = str(args.qd_filter_method).strip()
        if np.isfinite(float(args.qd_filter_hz)):
            cfg["data"]["real"]["qd_filter"]["cutoff_hz"] = float(args.qd_filter_hz)
    if bool(args.qd_use_filtered) or bool(args.qd_use_raw):
        cfg.setdefault("data", {}).setdefault("real", {})
        cfg["data"]["real"]["qd_use_filtered"] = bool(args.qd_use_filtered)

    runs_dir = str(get(cfg, "paths.runs_dir"))
    ensure_dir(runs_dir)

    repo_root = Path(__file__).resolve().parents[1]
    results_dir = repo_root / "results"

    csv_path = args.csv or str(get(cfg, "paths.real_csv"))
    dataset_npz = args.dataset or str(get(cfg, "paths.real_csv_dataset"))
    weights_path = args.weights or str(get(cfg, "paths.real_csv_model"))
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing csv: {csv_path}")
    if not os.path.exists(dataset_npz):
        cand = results_dir / Path(dataset_npz).name
        if cand.exists():
            dataset_npz = str(cand)
            print(f"[warn] dataset not found at config path; using: {dataset_npz}")
        else:
            raise SystemExit(f"missing dataset: {dataset_npz} (run scripts/prepare_closed_loop_csv.py first)")
    if not os.path.exists(weights_path):
        cand = results_dir / Path(weights_path).name
        if cand.exists():
            weights_path = str(cand)
            print(f"[warn] weights not found at config path; using: {weights_path}")
        else:
            raise SystemExit(f"missing weights: {weights_path} (run scripts/train_closed_loop_csv.py first)")

    ds = dict(np.load(dataset_npz, allow_pickle=True))
    stats = {k: ds[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]}
    if "feature_names" in ds:
        stats["feature_names"] = list(ds["feature_names"].tolist())
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
    qd_col = str(args.qd_col).strip() or str(get(cfg, "data.real.qd_col", required=False, default="dq_rad_s"))
    if qd_col not in rows[0]:
        raise SystemExit(f"missing qd_col: {qd_col} (available: {list(rows[0].keys())})")
    qd_col_data = np.array([float(r[qd_col]) for r in rows], dtype=np.float64)
    qd_raw = np.array([float(r["dq_rad_s"]) for r in rows], dtype=np.float64) if "dq_rad_s" in rows[0] else qd_col_data
    use_tau_ff_from_csv = bool(
        args.use_tau_ff_from_csv or get(cfg, "data.real.use_tau_ff_from_csv", required=False, default=False)
    )
    if use_tau_ff_from_csv:
        tau_key = None
        for k in ["tau_ff_Nm", "tau_ff", "tau_Nm"]:
            if k in rows[0]:
                tau_key = k
                break
        if tau_key is None:
            raise SystemExit("no tau_ff column found (expected one of: tau_ff_Nm, tau_ff, tau_Nm)")
        tau_ff = np.array([float(r[tau_key]) for r in rows], dtype=np.float64)
    else:
        tau_ff = np.zeros((len(rows),), dtype=np.float64)
    q_ref = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
    qd_ref = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)

    # Apply the same qd filtering policy used for dataset prep (but keep CSV dq_rad_s for plotting if present).
    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    qd_method, qd_cutoff_hz, qd_use_filtered = _get_qd_filter_cfg(cfg)
    qd_base = qd_col_data
    qd_filt = qd_base
    if float(qd_cutoff_hz) > 0 and dt_med > 0:
        if qd_method == "zero_phase_one_pole":
            qd_filt = _zero_phase_one_pole_lpf(qd_base, dt=dt_med, cutoff_hz=float(qd_cutoff_hz))
        elif qd_method == "one_pole":
            qd_filt = _one_pole_lpf(qd_base, dt=dt_med, cutoff_hz=float(qd_cutoff_hz))
        else:
            raise SystemExit(f"unknown qd filter method: {qd_method!r}")
    qd = qd_filt if bool(qd_use_filtered) else qd_base
    qd_filter_delta_rmse = float(np.sqrt(np.mean((qd - qd_raw) ** 2))) if len(qd_raw) else 0.0

    if "kp" in rows[0]:
        kp = np.array([float(r["kp"]) for r in rows], dtype=np.float64)
    else:
        kp = np.full((len(rows),), float(get(cfg, "real.kp", required=False, default=0.0)), dtype=np.float64)
    if "kd" in rows[0]:
        kd = np.array([float(r["kd"]) for r in rows], dtype=np.float64)
    else:
        kd = np.full((len(rows),), float(get(cfg, "real.kd", required=False, default=0.0)), dtype=np.float64)

    if str(args.stages).strip():
        stages = [s.strip() for s in str(args.stages).split(",") if s.strip()]
    else:
        stages = [str(args.stage)]

    rows_model = []
    rows_base = []
    for stg in stages:
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
            stage_name=str(stg),
            H=H,
            horizon_steps=int(args.horizon_steps),
            device=str(args.device),
        )
        rows_model.append(res)
        print(
            f"[open-loop stage={res['stage']}] horizon={res['horizon_steps']} "
            f"q_rmse={res['q_rmse']:.6g} q_maxabs={res['q_maxabs']:.6g} "
            f"qd_rmse={res['qd_rmse']:.6g} qd_maxabs={res['qd_maxabs']:.6g}"
        )
        if args.baseline:
            b = _open_loop_baseline(
                t=t, stage=stage, q=q, qd=qd, stage_name=str(stg), H=H, horizon_steps=int(args.horizon_steps)
            )
            rows_base.append(b)

    # Save a small markdown summary for batch runs.
    if len(stages) > 1 or args.out_md:
        if args.out_md:
            out_md = str(args.out_md)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_md = os.path.join(runs_dir, f"summary_closed_loop_csv_{ts}.md")
        ensure_dir(os.path.dirname(out_md) or ".")
        md = []
        md.append("# Closed-loop digital twin eval summary")
        md.append("")
        md.append(f"- csv: `{csv_path}`")
        md.append(f"- dataset: `{dataset_npz}`")
        md.append(f"- weights: `{weights_path}`")
        md.append(f"- stages: `{', '.join(stages)}`")
        md.append(f"- horizon_steps: `{int(args.horizon_steps)}`")
        md.append(f"- device: `{args.device}`")
        md.append(f"- qd_col: `{qd_col}` (raw plot uses `dq_rad_s` if present)")
        md.append(f"- qd_filter: `{qd_method}` @ `{float(qd_cutoff_hz)}` Hz, use_filtered=`{bool(qd_use_filtered)}`")
        md.append(f"- qd_filter_delta_rmse(raw->filt): `{qd_filter_delta_rmse:.6g}` rad/s")
        md.append("")
        md.append(_to_md_table(rows_model, title="Model (open-loop)"))
        if rows_base:
            md.append(_to_md_table(rows_base, title="Baseline (open-loop, constant qd)"))
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md).rstrip() + "\n")
        print(f"saved: {out_md}")

    if args.plot_reversals:
        plot_h = int(args.plot_horizon_steps) if int(args.plot_horizon_steps) > 0 else int(args.horizon_steps)
        plot_stage = stages[0]
        trace = _open_loop_rollout_trace(
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
            stage_name=str(plot_stage),
            H=H,
            horizon_steps=int(plot_h),
            device=str(args.device),
        )
        trace["qd_gt_raw"] = qd_raw[trace["k0"] : trace["k0"] + int(plot_h) + 1].astype(np.float64).copy()
        if args.out_png:
            out_png = str(args.out_png)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_png = os.path.join(runs_dir, f"plot_reversals_{trace['stage']}_{ts}.png")
        _plot_reversal_windows(
            trace,
            out_png=out_png,
            window_s=float(args.reversal_window_s),
            max_events=int(args.reversal_max_events),
            speed_th=float(args.reversal_speed_th),
        )
        print(f"saved: {out_png}")

    if args.plot_full_cycle:
        plot_h = int(args.plot_horizon_steps) if int(args.plot_horizon_steps) > 0 else max(int(args.horizon_steps), 4000)
        plot_stage = stages[0]
        trace = _open_loop_rollout_trace(
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
            stage_name=str(plot_stage),
            H=H,
            horizon_steps=int(plot_h),
            device=str(args.device),
        )
        trace["qd_gt_raw"] = qd_raw[trace["k0"] : trace["k0"] + int(plot_h) + 1].astype(np.float64).copy()
        if args.out_full_png:
            out_full = str(args.out_full_png)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_full = str(results_dir / f"plot_full_cycle_{trace['stage']}_{ts}.png")
        _plot_full_cycle(trace, out_png=out_full)
        print(f"saved: {out_full}")


if __name__ == "__main__":
    main()
