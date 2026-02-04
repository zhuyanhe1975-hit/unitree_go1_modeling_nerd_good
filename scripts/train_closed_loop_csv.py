from __future__ import annotations

import argparse
import os

from pipeline.config import load_cfg
from pipeline.train import train_model
from project_config import ensure_dir, get


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default=None, help="override dataset npz (default: paths.real_csv_dataset)")
    ap.add_argument("--out", default=None, help="override output weights (default: paths.real_csv_model)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))

    ds = args.dataset or str(get(cfg, "paths.real_csv_dataset"))
    out = args.out or str(get(cfg, "paths.real_csv_model"))
    if not os.path.exists(ds):
        raise SystemExit(f"missing dataset: {ds} (run scripts/prepare_closed_loop_csv.py first)")
    ensure_dir(os.path.dirname(out) or ".")

    train_model(cfg, dataset_npz=ds, out_weights=out)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

