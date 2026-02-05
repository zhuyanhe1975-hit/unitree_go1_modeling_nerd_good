from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pipeline.config import load_cfg
from pipeline.prepare_closed_loop import _one_pole_lpf, _zero_phase_one_pole_lpf
from project_config import ensure_dir, get


def _filter_qd(qd: np.ndarray, dt: float, method: str, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0 or dt <= 0 or len(qd) == 0:
        return qd
    if method == "zero_phase_one_pole":
        return _zero_phase_one_pole_lpf(qd, dt=dt, cutoff_hz=cutoff_hz)
    if method == "one_pole":
        return _one_pole_lpf(qd, dt=dt, cutoff_hz=cutoff_hz)
    raise ValueError(f"unknown method: {method!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="used for default csv path and default filter params")
    ap.add_argument("--csv", default=None, help="input CSV path (default: paths.real_csv)")
    ap.add_argument("--out", default="", help="output CSV path (default: results/<name>_with_qd_filt.csv)")
    ap.add_argument("--qd_col", default="dq_rad_s", help="raw qd column name (default: dq_rad_s)")
    ap.add_argument("--out_col", default="dq_filt_rad_s", help="new filtered qd column name (default: dq_filt_rad_s)")
    ap.add_argument("--method", default="", help="one_pole or zero_phase_one_pole (default from config)")
    ap.add_argument("--cutoff_hz", type=float, default=float("nan"), help="cutoff frequency in Hz (default from config)")
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_cfg(args.config)
    csv_path = args.csv or str(get(cfg, "paths.real_csv"))
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing csv: {csv_path}")

    method = str(args.method).strip() or str(get(cfg, "data.real.qd_filter.method", required=False, default="one_pole"))
    cutoff_hz = float(args.cutoff_hz)
    if not np.isfinite(cutoff_hz):
        cutoff_hz = float(get(cfg, "data.real.qd_filter.cutoff_hz", required=False, default=np.nan))
    if not np.isfinite(cutoff_hz):
        cutoff_hz = float(get(cfg, "data.real.qd_lpf_hz", required=False, default=0.0))

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
        fieldnames = list(r.fieldnames or [])

    if len(rows) < 2:
        raise SystemExit("csv too short")
    if "t_s" not in rows[0]:
        raise SystemExit("missing required column: t_s")
    if args.qd_col not in rows[0]:
        raise SystemExit(f"missing required column: {args.qd_col} (available: {list(rows[0].keys())})")

    t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    qd = np.array([float(r[args.qd_col]) for r in rows], dtype=np.float64)
    dt_med = float(np.median(np.diff(t))) if len(t) >= 2 else 0.0
    qd_f = _filter_qd(qd, dt=dt_med, method=method, cutoff_hz=cutoff_hz)

    if args.out_col in fieldnames:
        raise SystemExit(f"output column already exists: {args.out_col}")
    out_fieldnames = fieldnames + [args.out_col]

    if args.out:
        out_path = Path(args.out)
    else:
        repo_root = Path(__file__).resolve().parents[1]
        results_dir = repo_root / "results"
        ensure_dir(str(results_dir))
        stem = Path(csv_path).stem
        out_path = results_dir / f"{stem}_with_qd_filt.csv"

    ensure_dir(str(out_path.parent))
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fieldnames)
        w.writeheader()
        for i, row in enumerate(rows):
            row2 = dict(row)
            row2[args.out_col] = f"{float(qd_f[i]):.10g}"
            w.writerow(row2)

    print(f"saved: {out_path}")
    print(f"- method: {method}")
    print(f"- cutoff_hz: {cutoff_hz}")
    print(f"- dt_med: {dt_med}")


if __name__ == "__main__":
    main()

