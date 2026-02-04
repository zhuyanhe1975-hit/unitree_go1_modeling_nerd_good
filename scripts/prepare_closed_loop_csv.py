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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="real CSV path (default: paths.real_csv)")
    ap.add_argument("--out", default=None, help="output dataset npz (default: paths.real_csv_dataset)")
    ap.add_argument("--stats", default=None, help="output stats npz (default: paths.real_csv_stats)")
    ap.add_argument("--kp", type=float, default=None, help="kp if CSV does not contain a kp column")
    ap.add_argument("--kd", type=float, default=None, help="kd if CSV does not contain a kd column")
    args = ap.parse_args()

    cfg: Dict[str, Any] = load_cfg(args.config)
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

    required = ["t_s", "stage", "q_rad", "dq_rad_s", "tau_Nm", "q_ref_rad", "dq_ref_rad_s"]
    missing = [k for k in required if k not in rows[0]]
    if missing:
        raise SystemExit(f"missing required columns: {missing} (available: {list(rows[0].keys())})")

    t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
    stage = np.array([r["stage"] for r in rows], dtype=object)
    q = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
    qd = np.array([float(r["dq_rad_s"]) for r in rows], dtype=np.float64)
    tau_ff = np.array([float(r["tau_Nm"]) for r in rows], dtype=np.float64)
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
    if stats_npz:
        print(f"saved: {stats_npz}")


if __name__ == "__main__":
    main()

