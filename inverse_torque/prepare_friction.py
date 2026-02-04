from __future__ import annotations

import argparse

from pipeline.config import load_cfg
from project_config import get

from inverse_torque.friction_model import prepare_friction_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--raw", default=None, help="raw log npz (default: paths.real_log)")
    ap.add_argument("--out", default="runs/friction_dataset.npz")
    ap.add_argument("--stats", default="runs/friction_stats.npz")
    ap.add_argument("--qd_source", choices=["from_log", "from_q"], default="from_q")
    ap.add_argument("--qd_lpf_hz", type=float, default=30.0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    raw = args.raw or str(get(cfg, "paths.real_log"))

    prepare_friction_dataset(
        cfg,
        raw_npz=raw,
        out_npz=args.out,
        stats_npz=args.stats,
        qd_source=args.qd_source,
        qd_lpf_hz=args.qd_lpf_hz,
    )
    print(f"saved friction dataset: {args.out}")
    print(f"saved friction stats: {args.stats}")


if __name__ == "__main__":
    main()

