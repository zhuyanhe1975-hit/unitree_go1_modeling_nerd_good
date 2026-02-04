from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np

from nerd_compat.joint_1dof_neural_env import Joint1DofNeuralEnvironment, PredictorConfig
from pipeline.config import load_cfg
from project_config import ensure_dir, get


def _make_env_cfg(cfg: Dict[str, Any], num_envs: int, device: str) -> Dict[str, Any]:
    return {
        "num_envs": int(num_envs),
        "frame_dt": float(get(cfg, "sim.frame_dt")),
        "sim_substeps": int(get(cfg, "sim.sim_substeps")),
        "asset_mjcf": str(get(cfg, "paths.asset_mjcf")),
        "mjcf_override": dict(get(cfg, "sim.mjcf_override")),
        "runs_dir": str(get(cfg, "paths.runs_dir")),
        "device": str(device),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", default=None, help="state-delta model weights (default: paths.sim_model)")
    ap.add_argument("--dataset", default=None, help="prepared dataset npz with stats (default: paths.sim_dataset)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num_envs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--profile", choices=["chirp", "sine", "random"], default="chirp")
    ap.add_argument("--amp", type=float, default=0.4)
    ap.add_argument("--f0_hz", type=float, default=0.1)
    ap.add_argument("--f1_hz", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    ensure_dir(get(cfg, "paths.runs_dir"))
    dt = float(get(cfg, "sim.frame_dt"))

    weights = args.weights or str(get(cfg, "paths.sim_model"))
    dataset = args.dataset or str(get(cfg, "paths.sim_dataset"))
    if not os.path.exists(weights):
        raise SystemExit(f"missing weights: {weights} (run training first, e.g. `python3 scripts/train.py --mode sim`)")
    if not os.path.exists(dataset):
        raise SystemExit(f"missing dataset: {dataset} (run prepare first, e.g. `python3 scripts/prepare.py --mode sim`)")

    env_cfg = _make_env_cfg(cfg, num_envs=args.num_envs, device=args.device)
    pred_cfg = PredictorConfig(weights_path=weights, dataset_npz=dataset, device="cpu" if args.device == "cpu" else args.device)

    env = Joint1DofNeuralEnvironment(env_cfg, predictor_cfg=pred_cfg, default_env_mode="ground-truth")

    rng = np.random.default_rng(args.seed)
    t = np.arange(args.steps, dtype=np.float64) * dt
    if args.profile == "chirp":
        k = (args.f1_hz - args.f0_hz) / max(1e-6, float(args.steps) * dt)
        phase = 2.0 * np.pi * (args.f0_hz * t + 0.5 * k * t * t)
        tau = args.amp * np.sin(phase)
    elif args.profile == "sine":
        tau = args.amp * np.sin(2.0 * np.pi * args.f0_hz * t)
    else:
        tau = rng.uniform(low=-args.amp, high=args.amp, size=(args.steps,))

    tau = tau.astype(np.float64)

    # rollout ground-truth
    env.set_env_mode("ground-truth")
    env.reset(q0=0.0, qd0=0.0, tau0=float(tau[0]))
    gt = []
    for k in range(args.steps):
        st = env.step(np.full((args.num_envs, 1), tau[k], dtype=np.float64))
        gt.append(st.copy())
    gt = np.stack(gt, axis=0)  # [T,K,2]

    # rollout neural
    env.set_env_mode("neural")
    env.reset(q0=0.0, qd0=0.0, tau0=float(tau[0]))
    nn = []
    for k in range(args.steps):
        st = env.step(np.full((args.num_envs, 1), tau[k], dtype=np.float64))
        nn.append(st.copy())
    nn = np.stack(nn, axis=0)  # [T,K,2]

    # metrics (K=1 by default)
    e = nn[..., 0] - gt[..., 0]
    rmse = float(np.sqrt(np.mean(e**2)))
    maxabs = float(np.max(np.abs(e)))
    print(f"q RMSE={rmse:.6g}, maxabs={maxabs:.6g}")

    out_dir = str(get(cfg, "paths.runs_dir"))
    out_npz = os.path.join(out_dir, "rollout_compare_neural.npz")
    np.savez(out_npz, t=t, tau=tau, gt=gt, nn=nn, rmse=rmse, maxabs=maxabs)
    print(f"saved: {out_npz}")

    # optional plot
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 7))
        plt.subplot(3, 1, 1)
        plt.plot(t, tau, label="tau")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t, gt[:, 0, 0], label="q (gt)", color="k", alpha=0.6)
        plt.plot(t, nn[:, 0, 0], label="q (neural)", color="r", lw=1.0)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(t, (nn[:, 0, 0] - gt[:, 0, 0]), label="q error (neural-gt)", color="b")
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_png = os.path.join(out_dir, "rollout_compare_neural.png")
        plt.tight_layout()
        plt.savefig(out_png)
        print(f"saved: {out_png}")
    except Exception:
        pass


if __name__ == "__main__":
    main()

