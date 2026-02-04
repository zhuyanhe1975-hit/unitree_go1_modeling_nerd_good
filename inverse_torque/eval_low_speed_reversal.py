from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

from pipeline.features import state_to_features
from pipeline.model import CausalTransformer
from pipeline.train import _resolve_device
from pipeline.config import load_cfg
from project_config import ensure_dir, get


@dataclass
class Segment:
    center: int
    start: int
    end: int
    score: float


def _load_model(weights_path: str, device: str, cfg: dict) -> CausalTransformer:
    ckpt = torch.load(weights_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", None) or get(cfg, "model.embed_dim")),
        num_layers=int(ckpt.get("num_layers", None) or get(cfg, "model.num_layers")),
        num_heads=int(ckpt.get("num_heads", None) or get(cfg, "model.num_heads")),
        history_len=int(ckpt.get("history_len", None) or get(cfg, "model.history_len")),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _select_low_speed_reversal_segments(
    t: np.ndarray,
    qd: np.ndarray,
    tau: np.ndarray,
    history_len: int,
    v_th: float,
    window_s: float,
    hold_s: float,
    num: int,
    min_gap_s: float,
    max_gap_s: float,
) -> list[Segment]:
    if len(t) < 3:
        return []
    dt = float(np.median(np.diff(t)))
    half = int(max(1, round(window_s / dt)))
    hold = int(max(1, round(hold_s / dt)))
    gap = int(max(1, round(min_gap_s / dt)))

    # qd sign-change indices (between i and i+1)
    zc = np.where(qd[:-1] * qd[1:] <= 0)[0]
    cands: list[Segment] = []
    for i in zc:
        if i < 1:
            continue
        if not (abs(qd[i]) <= v_th and abs(qd[i + 1]) <= v_th):
            continue
        center = i + 1
        start = max(history_len, center - half)
        end = min(len(t) - 1, center + half)
        if start <= 1 or end - start < 10:
            continue
        # Ensure the segment is time-contiguous (avoid hidden large jumps in t).
        local_dt = np.diff(t[start : end + 1])
        if local_dt.size > 0 and float(np.max(local_dt)) > max_gap_s:
            continue
        # Require that around the reversal, velocity stays low for a short duration (robust to single-sample noise).
        hs = max(0, center - hold)
        he = min(len(qd) - 1, center + hold)
        if hs < 0 or he <= hs:
            continue
        qd_abs = np.abs(qd[hs : he + 1])
        if float(np.quantile(qd_abs, 0.9)) > v_th:
            continue
        # Score: torque activity near reversal (captures stick-slip / friction bumps).
        local = tau[start : end + 1]
        score = float(np.mean(np.abs(np.diff(local)))) + 0.1 * float(np.mean(np.abs(local)))
        cands.append(Segment(center=center, start=start, end=end, score=score))

    # Sort by score desc, then pick with non-overlap-ish gap.
    cands.sort(key=lambda s: s.score, reverse=True)
    picked: list[Segment] = []
    for s in cands:
        if len(picked) >= num:
            break
        if all(abs(s.center - p.center) >= gap for p in picked):
            picked.append(s)
    return picked


def _build_features(q: np.ndarray, qd: np.ndarray, tau_out: np.ndarray, temp: np.ndarray | None, din: int) -> np.ndarray:
    feat_state = state_to_features(q, qd).astype(np.float32)  # [T,3]
    tau_col = tau_out.astype(np.float32).reshape(-1, 1)
    if din == 5:
        if temp is None:
            # If the model expects temp but raw log lacks it, use zeros.
            temp_col = np.zeros_like(tau_col, dtype=np.float32)
        else:
            temp_col = temp.astype(np.float32).reshape(-1, 1)
        return np.concatenate([feat_state, temp_col, tau_col], axis=-1)
    return np.concatenate([feat_state, tau_col], axis=-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="raw log npz (default: paths.real_log)")
    ap.add_argument("--dataset", default="runs/torque_delta_dataset.npz", help="for normalization stats")
    ap.add_argument("--model", default="runs/torque_delta_model.pt")
    ap.add_argument(
        "--qd_select",
        choices=["from_log", "from_q"],
        default="from_q",
        help="which velocity to use for low-speed reversal selection (does NOT change model inputs)",
    )
    ap.add_argument("--v_th", type=float, default=0.15, help="low-speed threshold for reversal detection (rad/s)")
    ap.add_argument("--window_s", type=float, default=0.8, help="half window size around reversal (seconds)")
    ap.add_argument("--hold_s", type=float, default=0.15, help="require low speed within +/-hold_s around reversal")
    ap.add_argument("--num", type=int, default=5, help="number of segments to evaluate")
    ap.add_argument("--min_gap_s", type=float, default=1.0, help="min time gap between selected segments (seconds)")
    ap.add_argument("--max_gap_s", type=float, default=0.02, help="reject segments with dt gaps larger than this (seconds)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    raw = args.raw or str(get(cfg, "paths.real_log"))
    lg = dict(np.load(raw, allow_pickle=True))

    t = np.asarray(lg.get("t", []), dtype=np.float64).reshape(-1)
    q = np.asarray(lg.get("q_out", []), dtype=np.float64).reshape(-1)
    qd = np.asarray(lg.get("qd_out", []), dtype=np.float64).reshape(-1)
    tau = np.asarray(lg.get("tau_out", []), dtype=np.float64).reshape(-1)
    temp = np.asarray(lg.get("temp", []), dtype=np.float64).reshape(-1) if "temp" in lg else None

    if not (len(t) == len(q) == len(qd) == len(tau)) or len(t) == 0:
        raise ValueError("raw log must contain t, q_out, qd_out, tau_out with equal lengths")

    dt = float(np.median(np.diff(t))) if len(t) >= 2 else float("nan")
    H = int(get(cfg, "model.history_len"))

    # Use a more reliable velocity for selecting low-speed reversals if requested.
    if args.qd_select == "from_q" and len(t) >= 2:
        qd_sel = np.zeros_like(q)
        qd_sel[1:] = (q[1:] - q[:-1]) / dt
    else:
        qd_sel = qd

    ds = dict(np.load(args.dataset, allow_pickle=True))
    x_mean = ds["x_mean"].astype(np.float32)
    x_std = ds["x_std"].astype(np.float32)
    y_mean = ds["y_mean"].astype(np.float32)
    y_std = ds["y_std"].astype(np.float32)
    din = int(x_mean.shape[0])

    segments = _select_low_speed_reversal_segments(
        t=t,
        qd=qd_sel,
        tau=tau,
        history_len=H,
        v_th=args.v_th,
        window_s=args.window_s,
        hold_s=args.hold_s,
        num=args.num,
        min_gap_s=args.min_gap_s,
        max_gap_s=args.max_gap_s,
    )
    if len(segments) == 0:
        raise RuntimeError("no low-speed reversal segment found; try increasing --v_th or --window_s")

    device = _resolve_device(cfg)
    model = _load_model(args.model, device=device, cfg=cfg)

    feat_full = _build_features(q=q, qd=qd, tau_out=tau, temp=temp if temp is not None and len(temp) == len(t) else None, din=din)

    out_dir = str(get(cfg, "paths.runs_dir"))
    ensure_dir(out_dir)
    out_md = os.path.join(out_dir, "eval_torque_delta_low_speed.md")

    lines: list[str] = []
    lines.append("# Torque-Delta Evaluation: Low-Speed Reversal Segments")
    lines.append("")
    lines.append(f"raw: {raw}")
    lines.append(f"dt(median): {dt:.6f}s")
    lines.append(f"model: {args.model}")
    lines.append(f"dataset(stats): {args.dataset}")
    lines.append(f"reversal selection: v_th={args.v_th:g} rad/s, window_s={args.window_s:g}, num={len(segments)}")
    lines.append(f"qd_select: {args.qd_select}, hold_s={args.hold_s:g}, max_gap_s={args.max_gap_s:g}")
    lines.append("")

    def _metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
        err = pred - gt
        return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

    # Evaluate each segment with teacher forcing (one-step): use GT history tau_out in x.
    for si, seg in enumerate(segments, start=1):
        k0 = seg.start
        k1 = seg.end
        ks = np.arange(k0, k1 + 1, dtype=np.int64)

        # Build per-step windows (ending at k-1) and predict delta_tau_out[k]
        pred_delta = np.zeros((len(ks),), dtype=np.float64)
        for j, k in enumerate(ks):
            if k < H:
                pred_delta[j] = 0.0
                continue
            x_win = feat_full[k - H : k].astype(np.float32)  # [H,D], last row is (k-1)
            x_n = (x_win - x_mean) / x_std
            with torch.no_grad():
                y_n = model(torch.from_numpy(x_n[None, ...]).float().to(device)).detach().cpu().numpy().reshape(-1)
            pred_delta[j] = float(y_n[0] * y_std[0] + y_mean[0])

        # Convert delta predictions to tau_out[k] predictions using tau_out[k-1] (GT).
        tau_gt = tau[ks]
        tau_prev = tau[ks - 1]
        tau_pred = tau_prev + pred_delta

        # Baselines
        tau_persist = tau_prev  # delta=0
        tau_prev2 = tau[ks - 2].copy()
        tau_prev2[ks < 2] = tau_prev[ks < 2]  # guard
        tau_prev_delta = tau_prev + (tau_prev - tau_prev2)  # delta[k] ≈ delta[k-1]

        mae, rmse = _metrics(tau_pred, tau_gt)
        mae0, rmse0 = _metrics(tau_persist, tau_gt)
        mae1, rmse1 = _metrics(tau_prev_delta, tau_gt)

        lines.append(f"## Segment {si}")
        lines.append(f"- center_idx: {seg.center}, t≈{t[seg.center]:.3f}s, score={seg.score:.6g}")
        lines.append(f"- range: idx {k0}..{k1} (t≈{t[k0]:.3f}s..{t[k1]:.3f}s), n={len(ks)}")
        lines.append("- metrics (predict tau_out[k]):")
        lines.append(f"  - model: MAE={mae:.6f} Nm, RMSE={rmse:.6f} Nm")
        lines.append(f"  - baseline(persist): MAE={mae0:.6f} Nm, RMSE={rmse0:.6f} Nm")
        lines.append(f"  - baseline(prev-delta): MAE={mae1:.6f} Nm, RMSE={rmse1:.6f} Nm")

        if plt is not None:
            out_png = os.path.join(out_dir, f"eval_torque_delta_low_speed_seg{si}.png")
            tt = t[ks]
            plt.figure(figsize=(12, 7))
            plt.subplot(3, 1, 1)
            plt.plot(tt, qd[ks], label="qd_out (from log)", color="tab:gray", alpha=0.55)
            plt.plot(tt, qd_sel[ks], label="qd_sel (for selection)", color="tab:gray", alpha=0.9)
            plt.axhline(0.0, color="k", lw=0.8)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(3, 1, 2)
            plt.plot(tt, tau_gt, label="tau_out[k] (gt)", alpha=0.75)
            plt.plot(tt, tau_pred, label="tau_out[k] (pred)", alpha=0.75)
            plt.plot(tt, tau_persist, label="persist baseline", alpha=0.6)
            plt.grid(True, alpha=0.3)
            plt.legend()

            plt.subplot(3, 1, 3)
            plt.plot(tt, (tau_pred - tau_gt), label="error (pred-gt)", color="r", alpha=0.85)
            plt.axhline(0.0, color="k", lw=0.8)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_png)

            lines.append(f"- plot: runs/{os.path.basename(out_png)}")
        lines.append("")

    ensure_dir(os.path.dirname(out_md) or ".")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("saved:", out_md)


if __name__ == "__main__":
    main()
