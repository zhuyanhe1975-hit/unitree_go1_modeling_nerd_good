from __future__ import annotations

import argparse
import csv
import os
from typing import Any, Dict

import numpy as np

from pipeline.config import load_cfg
from pipeline.prepare_closed_loop import prepare_closed_loop_csv_dataset
from project_config import ensure_dir, get


def _col_or_default(rows: list[dict[str, str]], key: str, default: float) -> np.ndarray:
    if len(rows) == 0:
        return np.array([], dtype=np.float64)
    if key in rows[0]:
        return np.array([float(r[key]) for r in rows], dtype=np.float64)
    return np.full((len(rows),), float(default), dtype=np.float64)


def _infer_tau_ff_from_rows(rows: list[dict[str, str]]) -> np.ndarray:
    """
    Best-effort feedforward torque extraction from a CSV row dict.

    Accepted column names (first found wins):
      - tau_ff_Nm
      - tau_ff
      - tau_Nm  (legacy; often a measured torque, so use with care)
    """
    if len(rows) == 0:
        return np.array([], dtype=np.float64)
    keys = ["tau_ff_Nm", "tau_ff", "tau_Nm"]
    for k in keys:
        if k in rows[0]:
            return np.array([float(r[k]) for r in rows], dtype=np.float64)
    raise KeyError(f"no tau_ff column found (tried: {keys})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="real CSV path (default: paths.real_csv)")
    ap.add_argument("--out", default=None, help="output dataset npz (default: paths.real_csv_dataset)")
    ap.add_argument("--stats", default=None, help="output stats npz (default: paths.real_csv_stats)")
    ap.add_argument("--kp", type=float, default=None, help="kp if CSV does not contain a kp column")
    ap.add_argument("--kd", type=float, default=None, help="kd if CSV does not contain a kd column")
    ap.add_argument("--qd_col", default="", help="qd column name in CSV (default: data.real.qd_col or dq_rad_s)")
    ap.add_argument(
        "--use_tau_ff_from_csv",
        action="store_true",
        help="use feedforward torque from CSV (otherwise tau_ff=0). Use only if tau_ff was actually commanded.",
    )
    ap.add_argument("--qd_filter_method", default="", help="one_pole or zero_phase_one_pole (override config)")
    ap.add_argument("--qd_filter_hz", type=float, default=float("nan"), help="qd filter cutoff Hz (override config)")
    qd_src = ap.add_mutually_exclusive_group()
    qd_src.add_argument("--qd_use_filtered", action="store_true", help="use filtered qd for training (override config)")
    qd_src.add_argument("--qd_use_raw", action="store_true", help="use raw qd for training (override config)")
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

    csv_path = args.csv or str(get(cfg, "paths.real_csv"))
    if not os.path.exists(csv_path):
        raise SystemExit(f"missing csv: {csv_path}")

    out_npz = args.out or str(get(cfg, "paths.real_csv_dataset"))
    stats_npz = args.stats or str(get(cfg, "paths.real_csv_stats", required=False, default=""))
    if not stats_npz:
        stats_npz = None

    kp_default = float(args.kp) if args.kp is not None else float(get(cfg, "real.kp", required=False, default=0.0))
    kd_default = float(args.kd) if args.kd is not None else float(get(cfg, "real.kd", required=False, default=0.0))

    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if len(rows) < 50:
        raise SystemExit("csv too short")

    qd_col = str(args.qd_col).strip() or str(get(cfg, "data.real.qd_col", required=False, default="dq_rad_s"))
    required = ["t_s", "stage", "q_rad", qd_col, "q_ref_rad", "dq_ref_rad_s"]
    missing = [k for k in required if k not in rows[0]]
    if missing:
        raise SystemExit(f"missing required columns: {missing} (available: {list(rows[0].keys())})")

    use_tau_ff_from_csv = bool(
        args.use_tau_ff_from_csv or get(cfg, "data.real.use_tau_ff_from_csv", required=False, default=False)
    )

    t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    stage = np.array([r["stage"] for r in rows], dtype=object)
    q = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
    qd = np.array([float(r[qd_col]) for r in rows], dtype=np.float64)
    if use_tau_ff_from_csv:
        try:
            tau_ff = _infer_tau_ff_from_rows(rows)
        except KeyError as e:
            raise SystemExit(str(e)) from e
    else:
        tau_ff = np.zeros((len(rows),), dtype=np.float64)
    q_ref = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
    qd_ref = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)

    kp = _col_or_default(rows, "kp", kp_default)
    kd = _col_or_default(rows, "kd", kd_default)

    prepare_closed_loop_csv_dataset(
        cfg,
        t=t,
        stage=stage,
        q=q,
        qd=qd,
        tau_ff=tau_ff,
        q_ref=q_ref,
        qd_ref=qd_ref,
        kp=kp,
        kd=kd,
        out_npz=out_npz,
        stats_npz=stats_npz,
    )
    print(f"saved: {out_npz}")
    print(f"- qd_col: {qd_col}")
    if stats_npz:
        print(f"saved: {stats_npz}")


if __name__ == "__main__":
    main()
