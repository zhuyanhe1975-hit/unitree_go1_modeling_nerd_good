from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict

import numpy as np
import torch

from pipeline.features import state_to_features
from pipeline.model import CausalTransformer
from project_config import ensure_dir, get, load_config
from collect_real_data import RealMotorInterface


def _load_model(model_path: str, device: str) -> CausalTransformer:
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model = CausalTransformer(
        input_dim=int(ckpt["input_dim"]),
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", 64)),
        num_layers=int(ckpt.get("num_layers", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
        history_len=int(ckpt.get("history_len", 10)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def run_velocity_test(cfg: Dict[str, Any], out_log: str, model_path: str, apply_ff: bool = True) -> None:
    # load stats for normalization
    ds = dict(np.load(str(get(cfg, "paths.real_dataset")), allow_pickle=True))
    stats = {
        "x_mean": ds["x_mean"].astype(np.float32),
        "x_std": ds["x_std"].astype(np.float32),
        "y_mean": ds["y_mean"].astype(np.float32),
        "y_std": ds["y_std"].astype(np.float32),
    }
    hist = int(get(cfg, "model.history_len"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(model_path, map_location=torch.device(device))
    input_dim = int(ckpt["input_dim"])
    model = CausalTransformer(
        input_dim=input_dim,
        output_dim=int(ckpt["output_dim"]),
        embed_dim=int(ckpt.get("embed_dim", 64)),
        num_layers=int(ckpt.get("num_layers", 2)),
        num_heads=int(ckpt.get("num_heads", 4)),
        history_len=int(ckpt.get("history_len", 10)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Init hardware via shared interface
    max_torque = float(get(cfg, "real.max_torque"))
    motor = RealMotorInterface(cfg, max_torque_nm=max_torque)

    dt = float(get(cfg, "real.dt"))
    ps = get(cfg, "real.pos_sine", required=False, default={})
    ps_test = get(cfg, "real.pos_sine_test", required=False, default={})
    duration = float(ps_test.get("duration", ps.get("duration", get(cfg, "real.duration"))))
    max_speed = float(get(cfg, "real.max_speed_rad_s", required=False, default=6.28))
    freq = float(ps_test.get("freq_hz", ps.get("freq_hz", 0.2)))
    amp = float(ps_test.get("amplitude_rad", ps.get("amplitude_rad", 0.1)))

    log = {"t": [], "q_out": [], "qd_out": [], "tau_out": [], "cmd_q": [], "cmd_qd": [], "tau_pred": []}

    q_hist = []
    qd_hist = []
    q_offset = None

    # Helper to send position/velocity/torque command through RealMotorInterface internals
    def _send_cmd(q: float, dq: float, kp: float, kd: float, tau: float) -> tuple[float, float, float]:
        motor.cmd.q = float(q)
        motor.cmd.dq = float(dq)
        motor.cmd.kp = float(kp)
        motor.cmd.kd = float(kd)
        motor.cmd.tau = float(tau)
        motor.serial.sendRecv(motor.cmd, motor.data)
        return float(motor.data.q), float(motor.data.dq), float(motor.data.tau)

    # Warmup: read current state for 1s to latch initial position, avoid jumps
    warmup_end = time.perf_counter() + 1.0
    while time.perf_counter() < warmup_end:
        q_m, qd_m, _ = _send_cmd(
            q=0.0,
            dq=0.0,
            kp=float(get(cfg, "real.pos_sine.kp", 0.2)),
            kd=float(get(cfg, "real.pos_sine.kd", 0.01)),
            tau=0.0,
        )
        q_offset = float(q_m)
        time.sleep(dt)

    start = time.perf_counter()
    next_time = start
    try:
        while True:
            now = time.perf_counter()
            t = now - start
            if t > duration:
                break

            omega = 2 * np.pi * freq
            if q_offset is None:
                # latch initial position to avoid jump
                q_m, _, _ = motor.get_state()
                q_offset = float(q_m)
            q_cmd = q_offset + float(amp * np.sin(omega * t))
            qd_cmd = float(amp * omega * np.cos(omega * t))

            # Build feedforward torque when history is ready (match training features: [q, qd, qdd?, temp?, cmd_qd?, cmd_q?])
            if len(q_hist) >= hist:
                q_arr = np.array(q_hist[-hist:], dtype=np.float32).reshape(-1, 1)
                qd_arr = np.array(qd_hist[-hist:], dtype=np.float32).reshape(-1, 1)
                qdd_arr = np.zeros_like(qd_arr)
                if len(qd_hist) >= hist + 1:
                    qd_hist_arr = np.array(qd_hist[-(hist + 1) :], dtype=np.float32)
                    qdd_arr = ((qd_hist_arr[1:] - qd_hist_arr[:-1]) / dt).reshape(-1, 1)
                temp_arr = np.zeros_like(q_arr)
                cmd_qd_arr = np.full_like(q_arr, qd_cmd, dtype=np.float32)
                cmd_q_arr = np.array(log["cmd_q"][-hist:], dtype=np.float32).reshape(-1, 1) if len(log["cmd_q"]) >= hist else np.full_like(q_arr, q_cmd, dtype=np.float32)

                feats = [q_arr, qd_arr]
                if stats["x_mean"].shape[0] >= 3:
                    feats.append(qdd_arr)
                if sum(f.shape[-1] for f in feats) < input_dim:
                    feats.append(temp_arr)
                if sum(f.shape[-1] for f in feats) < input_dim:
                    feats.append(cmd_qd_arr)
                if sum(f.shape[-1] for f in feats) < input_dim:
                    feats.append(cmd_q_arr)
                while sum(f.shape[-1] for f in feats) < input_dim:
                    feats.append(np.zeros_like(q_arr))

                feat_full = np.concatenate(feats, axis=-1)[:, :input_dim]

                x_mean = stats["x_mean"][:input_dim]
                x_std = stats["x_std"][:input_dim]
                x_n = (feat_full - x_mean) / x_std
                with torch.no_grad():
                    pred_norm = model(torch.from_numpy(x_n[None, ...]).float().to(device)).cpu().numpy()
                tau_pred = float(pred_norm.reshape(-1) * stats["y_std"] + stats["y_mean"])
            else:
                tau_pred = 0.0

            tau_pred = float(np.clip(tau_pred, -max_torque, max_torque))

            # Safety: run pure position loop; optionally apply feedforward torque if requested.
            q_m, qd_m, tau_m = _send_cmd(
                q=q_cmd,
                dq=qd_cmd,
                kp=float(get(cfg, "real.pos_sine.kp", required=False, default=0.2)),
                kd=float(get(cfg, "real.pos_sine.kd", required=False, default=0.01)),
                tau=tau_pred if apply_ff else 0.0,
            )

            q_hist.append(float(q_m))
            qd_hist.append(float(qd_m))

            log["t"].append(t)
            log["q_out"].append(float(q_m))
            log["qd_out"].append(float(qd_m))
            log["tau_out"].append(float(tau_m))
            log["cmd_q"].append(q_cmd)
            log["cmd_qd"].append(qd_cmd)
            log["tau_pred"].append(tau_pred)

            next_time += dt
            sleep = next_time - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        try:
            _send_cmd(
                q=0.0,
                dq=0.0,
                kp=float(get(cfg, "real.pos_sine.kp", 0.2)),
                kd=float(get(cfg, "real.pos_sine.kd", 0.01)),
                tau=0.0,
            )
        except Exception:
            pass

    for k in log:
        log[k] = np.asarray(log[k], dtype=np.float32)
    ensure_dir(os.path.dirname(out_log) or ".")
    np.savez(out_log, **log)
    print(f"saved test log: {out_log}")
    print(f"tau_pred (not applied) range: [{log['tau_pred'].min():.4f}, {log['tau_pred'].max():.4f}]")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 9))
        plt.subplot(3, 1, 1)
        plt.plot(log["t"], log["cmd_q"], label="q_cmd", alpha=0.7)
        plt.plot(log["t"], log["q_out"], label="q_out", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(log["t"], log["cmd_qd"], label="qd_cmd", alpha=0.7)
        plt.plot(log["t"], log["qd_out"], label="qd_out", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(log["t"], log["tau_out"], label="tau_out", alpha=0.7)
        plt.plot(log["t"], log["tau_pred"], label="tau_pred_ff", alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()

        out_plot = os.path.join(os.path.dirname(out_log) or ".", "eval_torque_real_test.png")
        plt.tight_layout()
        plt.savefig(out_plot)
        print(f"saved: {out_plot}")
    except Exception:
        print("[warn] matplotlib not available; skipping plot")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, help="torque model to evaluate (default real_model)")
    ap.add_argument("--out", default=None, help="test log output npz")
    ap.add_argument("--apply_ff", action="store_true", help="actually apply feedforward torque (default: off)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_log = args.out or str(get(cfg, "paths.real_log_test"))
    model_path = args.model or str(get(cfg, "paths.real_model"))
    run_velocity_test(cfg, out_log=out_log, model_path=model_path, apply_ff=bool(args.apply_ff))


if __name__ == "__main__":
    main()
