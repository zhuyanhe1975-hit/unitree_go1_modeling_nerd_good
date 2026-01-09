from __future__ import annotations

import argparse

from pipeline.comp_torque import prepare_torque_dataset
from pipeline.config import load_cfg
from project_config import get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["sim", "real"], default="real")
    ap.add_argument("--raw", default=None, help="override raw log npz")
    ap.add_argument("--out", default=None, help="override output torque dataset npz")
    ap.add_argument("--stats", default=None, help="override stats npz (optional)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.mode == "sim":
        raw = args.raw or str(get(cfg, "paths.sim_raw_log"))
        out = args.out or str(get(cfg, "paths.sim_dataset"))
    else:
        raw = args.raw or str(get(cfg, "paths.real_log"))
        out = args.out or str(get(cfg, "paths.real_dataset"))
    stats = args.stats or str(get(cfg, "paths.stats_npz"))

    prepare_torque_dataset(cfg, raw_npz=raw, out_npz=out, stats_npz=stats)
    print(f"saved torque dataset: {out}")
    if stats:
        print(f"saved torque stats: {stats}")


if __name__ == "__main__":
    main()
