from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from project_config import ensure_dir


@dataclass
class FitResult:
    kp: float
    kd: float
    bias: float
    rmse: float
    corr: float
    lag_steps: int
    dt_med: float


def _read_csv(path: str, qd_col: str) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
        cols = list(r.fieldnames or [])
    if len(rows) < 3:
        raise SystemExit("csv too short")

    required = ["t_s", "stage", "q_rad", "tau_Nm", "q_ref_rad", "dq_ref_rad_s", qd_col]
    missing = [k for k in required if k not in cols]
    if missing:
        raise SystemExit(f"missing columns: {missing} (available: {cols})")

    out: dict[str, np.ndarray] = {}
    out["t"] = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    out["stage"] = np.array([r["stage"] for r in rows], dtype=object)
    out["q"] = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
    out["qd"] = np.array([float(r[qd_col]) for r in rows], dtype=np.float64)
    out["tau"] = np.array([float(r["tau_Nm"]) for r in rows], dtype=np.float64)
    out["q_ref"] = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
    out["qd_ref"] = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)
    return out


def _fit_kp_kd_bias(
    e_q: np.ndarray,
    e_qd: np.ndarray,
    tau: np.ndarray,
    *,
    lag_steps: int,
    use_bias: bool,
) -> tuple[float, float, float]:
    # Align tau to error features using lag (positive lag means tau is delayed)
    if lag_steps > 0:
        eq = e_q[:-lag_steps]
        ed = e_qd[:-lag_steps]
        tt = tau[lag_steps:]
    elif lag_steps < 0:
        ls = -lag_steps
        eq = e_q[ls:]
        ed = e_qd[ls:]
        tt = tau[:-ls]
    else:
        eq, ed, tt = e_q, e_qd, tau

    X = np.stack([eq, ed], axis=1).astype(np.float64)
    if use_bias:
        X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)

    # Solve least squares
    w, *_ = np.linalg.lstsq(X, tt.astype(np.float64), rcond=None)
    kp = float(w[0])
    kd = float(w[1])
    bias = float(w[2]) if use_bias else 0.0
    return kp, kd, bias


def _metrics(
    e_q: np.ndarray,
    e_qd: np.ndarray,
    tau: np.ndarray,
    *,
    kp: float,
    kd: float,
    bias: float,
    lag_steps: int,
) -> tuple[float, float]:
    if lag_steps > 0:
        eq = e_q[:-lag_steps]
        ed = e_qd[:-lag_steps]
        tt = tau[lag_steps:]
    elif lag_steps < 0:
        ls = -lag_steps
        eq = e_q[ls:]
        ed = e_qd[ls:]
        tt = tau[:-ls]
    else:
        eq, ed, tt = e_q, e_qd, tau

    pred = kp * eq + kd * ed + bias
    err = pred - tt
    rmse = float(np.sqrt(np.mean(err**2)))
    # correlation (guard constant vectors)
    if float(np.std(pred)) < 1e-9 or float(np.std(tt)) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred, tt)[0, 1])
    return rmse, corr


def _best_lag(
    e_q: np.ndarray,
    e_qd: np.ndarray,
    tau: np.ndarray,
    *,
    kp: float,
    kd: float,
    bias: float,
    max_abs_lag: int,
) -> int:
    # brute force lag search by minimizing rmse
    best = (float("inf"), 0)
    for lag in range(-int(max_abs_lag), int(max_abs_lag) + 1):
        rmse, _ = _metrics(e_q, e_qd, tau, kp=kp, kd=kd, bias=bias, lag_steps=lag)
        if rmse < best[0]:
            best = (rmse, lag)
    return int(best[1])


def _segment_indices(t: np.ndarray, stage: np.ndarray, gap_thr_s: float, min_len: int) -> list[tuple[int, int]]:
    cuts = [0]
    for i in range(1, len(t)):
        if stage[i] != stage[i - 1] or (float(t[i]) - float(t[i - 1])) > float(gap_thr_s):
            cuts.append(i)
    cuts.append(len(t))
    segs: list[tuple[int, int]] = []
    for i in range(len(cuts) - 1):
        a, b = int(cuts[i]), int(cuts[i + 1])
        if b - a >= int(min_len):
            segs.append((a, b))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--qd_col", default="dq_rad_s", help="qd column to use as vel_now (e.g. dq_filt_rad_s)")
    ap.add_argument("--kp", type=float, default=float("nan"), help="fixed kp (optional)")
    ap.add_argument("--kd", type=float, default=float("nan"), help="fixed kd (optional)")
    ap.add_argument("--tau_ff", type=float, default=0.0, help="assumed commanded tau_ff (default: 0)")
    ap.add_argument("--fit", action="store_true", help="fit kp,kd (and bias) by least squares")
    ap.add_argument("--no_bias", action="store_true", help="when --fit, do not fit a constant bias term")
    ap.add_argument("--max_abs_lag", type=int, default=10, help="search best lag in [-N,+N] steps")
    ap.add_argument("--stage", default="", help="limit to this stage name (optional)")
    ap.add_argument("--out_png", default="", help="output plot png (optional)")
    args = ap.parse_args()

    d = _read_csv(str(args.csv), qd_col=str(args.qd_col))
    t = d["t"]
    stage = d["stage"]
    q = d["q"]
    qd = d["qd"]
    tau = d["tau"]
    q_ref = d["q_ref"]
    qd_ref = d["qd_ref"]

    if str(args.stage).strip():
        m = np.array([str(s) == str(args.stage).strip() for s in stage], dtype=bool)
        if int(m.sum()) < 5:
            raise SystemExit(f"stage filter too small: {args.stage!r}")
        t, stage, q, qd, tau, q_ref, qd_ref = (arr[m] for arr in (t, stage, q, qd, tau, q_ref, qd_ref))

    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    gap_thr = max(0.05, 10.0 * dt_med) if dt_med > 0 else 0.05

    e_q = (q_ref - q).astype(np.float64)
    e_qd = (qd_ref - qd).astype(np.float64)
    tau0 = tau.astype(np.float64) - float(args.tau_ff)

    # Use only segments where q_ref/qd_ref change (exclude long static holds if present)
    segs = _segment_indices(t, stage, gap_thr_s=gap_thr, min_len=50)
    if len(segs) == 0:
        segs = [(0, len(t))]

    # Concatenate segments for fitting/metrics
    idx = np.zeros((len(t),), dtype=bool)
    for a, b in segs:
        idx[a:b] = True
    e_q = e_q[idx]
    e_qd = e_qd[idx]
    tau0 = tau0[idx]
    t2 = t[idx]

    use_bias = not bool(args.no_bias)

    kp = float(args.kp)
    kd = float(args.kd)
    bias = 0.0

    if args.fit or (not np.isfinite(kp) and not np.isfinite(kd)):
        # initial lag=0 fit, then refine lag and refit
        kp, kd, bias = _fit_kp_kd_bias(e_q, e_qd, tau0, lag_steps=0, use_bias=use_bias)
        lag = _best_lag(e_q, e_qd, tau0, kp=kp, kd=kd, bias=bias, max_abs_lag=int(args.max_abs_lag))
        kp, kd, bias = _fit_kp_kd_bias(e_q, e_qd, tau0, lag_steps=lag, use_bias=use_bias)
    else:
        lag = _best_lag(e_q, e_qd, tau0, kp=kp, kd=kd, bias=bias, max_abs_lag=int(args.max_abs_lag))

    rmse, corr = _metrics(e_q, e_qd, tau0, kp=kp, kd=kd, bias=bias, lag_steps=lag)
    print("PD torque fit vs tau_Nm")
    print(f"- csv: {args.csv}")
    print(f"- qd_col: {args.qd_col}")
    print(f"- dt_med: {dt_med:.6g} s")
    print(f"- best_lag_steps: {lag}  (~{lag*dt_med:.6g} s)")
    print(f"- kp: {kp:.6g}")
    print(f"- kd: {kd:.6g}")
    print(f"- bias: {bias:.6g} Nm")
    print(f"- rmse: {rmse:.6g} Nm")
    print(f"- corr: {corr:.6g}")

    out_png = str(args.out_png).strip()
    if out_png:
        if lag > 0:
            eq = e_q[:-lag]
            ed = e_qd[:-lag]
            tt = tau0[lag:]
            tt_t = t2[:-lag]
        elif lag < 0:
            ls = -lag
            eq = e_q[ls:]
            ed = e_qd[ls:]
            tt = tau0[:-ls]
            tt_t = t2[ls:]
        else:
            eq, ed, tt, tt_t = e_q, e_qd, tau0, t2

        pred = kp * eq + kd * ed + bias
        err = pred - tt

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(12, 9))
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(tt_t - tt_t[0], tt, label="tau_Nm (log)", color="k", lw=1.0, alpha=0.8)
        ax1.plot(tt_t - tt_t[0], pred, label="tau_pd (kp*e_q+kd*e_qd+bias)", color="tab:blue", lw=1.0)
        ax1.grid(True, alpha=0.25)
        ax1.set_ylabel("tau (Nm)")
        ax1.legend(loc="best", fontsize=9)

        ax2 = plt.subplot(3, 1, 2, sharex=ax1)
        ax2.plot(tt_t - tt_t[0], err, color="tab:orange", lw=1.0)
        ax2.axhline(0.0, color="k", lw=0.8, alpha=0.4)
        ax2.grid(True, alpha=0.25)
        ax2.set_ylabel("tau_pd - tau_Nm (Nm)")

        ax3 = plt.subplot(3, 1, 3)
        ax3.scatter(tt, pred, s=4, alpha=0.35)
        lo = float(min(tt.min(), pred.min()))
        hi = float(max(tt.max(), pred.max()))
        ax3.plot([lo, hi], [lo, hi], color="k", lw=0.8, alpha=0.5)
        ax3.grid(True, alpha=0.25)
        ax3.set_xlabel("tau_Nm (log)")
        ax3.set_ylabel("tau_pd (pred)")
        ax3.set_title(f"rmse={rmse:.3g}Nm corr={corr:.3g} lag={lag} steps")

        fig.tight_layout()
        ensure_dir(os.path.dirname(out_png) or ".")
        fig.savefig(out_png, dpi=160)
        print(f"saved: {out_png}")


if __name__ == "__main__":
    main()

