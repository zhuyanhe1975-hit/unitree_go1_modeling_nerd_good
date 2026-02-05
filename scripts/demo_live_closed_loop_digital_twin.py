from __future__ import annotations

# Allow running this script via `python scripts/...py` without requiring `PYTHONPATH=.`.
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import csv
import signal
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from project_config import ensure_dir, get, load_config
from pipeline.model import CausalTransformer
from pipeline.prepare_closed_loop import _build_features, _feature_names_for_set


def _try_add_unitree_sdk_path(cfg: Dict[str, Any]) -> None:
    def _try_add(p: str) -> bool:
        p = os.path.abspath(p)
        if not os.path.isdir(p):
            return False
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        if p not in ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = (p + (":" + ld if ld else ""))
        sys.path.insert(0, p)
        return True

    env_sdk = os.environ.get("UNITREE_ACTUATOR_SDK_LIB", "").strip()
    sdk_lib = str(get(cfg, "real.unitree_sdk_lib", required=False, default="") or "").strip()

    ok = False
    if env_sdk:
        ok = _try_add(env_sdk)
    if (not ok) and sdk_lib:
        ok = _try_add(sdk_lib)
    if not ok:
        # Best-effort fallback: known local SDK path.
        _try_add("/home/yhzhu/Industrial Robot/unitree_actuator_sdk/lib")


class UnitreeGoM8010_6:
    """
    Minimal Unitree GO-M8010-6 interface for position/velocity PD + feedforward torque.

    We always run in FOC mode and send:
      q_des, dq_des, kp, kd, tau_ff

    Read back:
      q_now, dq_now, tau_Nm (SDK's motor-side estimate from current feedback)
    """

    def __init__(self, hw_cfg: Dict[str, Any]):
        _try_add_unitree_sdk_path(hw_cfg)
        try:
            import unitree_actuator_sdk as u  # type: ignore
        except Exception as e:
            raise ImportError(
                "Cannot import unitree_actuator_sdk. Set env UNITREE_ACTUATOR_SDK_LIB to the folder containing "
                "`unitree_actuator_sdk*.so`, or set `real.unitree_sdk_lib` in hw config. "
                f"Original error: {e}"
            ) from e

        self.u = u
        port = str(get(hw_cfg, "real.serial_port"))
        self.serial = u.SerialPort(port)

        self.cmd = u.MotorCmd()
        self.data = u.MotorData()
        self.data.motorType = u.MotorType.GO_M8010_6
        self.cmd.motorType = u.MotorType.GO_M8010_6
        self.cmd.mode = u.queryMotorMode(u.MotorType.GO_M8010_6, u.MotorMode.FOC)
        self.cmd.id = int(get(hw_cfg, "real.motor_id"))

        # Prime.
        self.cmd.q = 0.0
        self.cmd.dq = 0.0
        self.cmd.kp = float(get(hw_cfg, "real.kp", required=False, default=0.0))
        self.cmd.kd = float(get(hw_cfg, "real.kd", required=False, default=0.0))
        self.cmd.tau = 0.0
        self.serial.sendRecv(self.cmd, self.data)

    def send_pd(self, *, q_des: float, dq_des: float, kp: float, kd: float, tau_ff: float) -> None:
        self.cmd.q = float(q_des)
        self.cmd.dq = float(dq_des)
        self.cmd.kp = float(kp)
        self.cmd.kd = float(kd)
        self.cmd.tau = float(tau_ff)
        self.serial.sendRecv(self.cmd, self.data)

    def read(self) -> Tuple[float, float, float]:
        return float(self.data.q), float(self.data.dq), float(self.data.tau)

    def stop(self) -> None:
        try:
            self.send_pd(q_des=float(self.data.q), dq_des=0.0, kp=0.0, kd=0.0, tau_ff=0.0)
        except Exception:
            pass


@dataclass
class ClosedLoopTwin:
    model: Any
    stats: Dict[str, np.ndarray]
    feature_names: List[str]
    H: int
    device: str

    # history buffers (length H)
    q_hist: np.ndarray
    qd_hist: np.ndarray
    qref_hist: np.ndarray
    qdref_hist: np.ndarray
    kp_hist: np.ndarray
    kd_hist: np.ndarray
    tau_ff_hist: np.ndarray
    dt_hist: np.ndarray

    # current predicted state (for this tick)
    q_hat: float
    qd_hat: float

    @staticmethod
    def load_from_files(
        *,
        model_cfg: Dict[str, Any],
        weights_path: str,
        stats_path: str,
        device: str,
        feature_set: str,
    ) -> "ClosedLoopTwin":
        import torch

        stats_npz = dict(np.load(stats_path, allow_pickle=True))
        stats = {k: stats_npz[k].astype(np.float32) for k in ["x_mean", "x_std", "y_mean", "y_std"]}

        ckpt = torch.load(weights_path, map_location=torch.device(device))
        model = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(get(model_cfg, "model.embed_dim")),
            num_layers=int(get(model_cfg, "model.num_layers")),
            num_heads=int(get(model_cfg, "model.num_heads")),
            history_len=int(get(model_cfg, "model.history_len")),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        H = int(get(model_cfg, "model.history_len"))
        # Prefer the feature schema saved during dataset preparation to avoid mismatches
        # between the current config and the stats/model that are being loaded.
        has_feature_names_in_npz = "feature_names" in stats_npz
        if "feature_names" in stats_npz:
            raw = stats_npz["feature_names"]
            try:
                feature_names = [str(x) for x in list(raw.tolist())]
            except Exception:
                feature_names = [str(x) for x in list(raw)]
        else:
            feature_names = _feature_names_for_set(feature_set)

        d_stats = int(stats["x_mean"].shape[0])
        d_feat = int(len(feature_names))
        d_ckpt = int(ckpt["input_dim"])

        # If stats/ckpt agree but the requested feature_set does not, infer the correct schema.
        # This makes the demo robust when the user points at an older minimal (D=5) model while
        # the current config requests full (D=9), or vice versa.
        if (not has_feature_names_in_npz) and (d_stats == d_ckpt) and (d_feat != d_ckpt):
            inferred_set: Optional[str] = None
            if d_ckpt == 5:
                inferred_set = "minimal"
            elif d_ckpt == 9:
                inferred_set = "full"
            if inferred_set is not None:
                feature_names = _feature_names_for_set(inferred_set)
                d_feat = int(len(feature_names))
                print(
                    f"[demo][warn] feature_set mismatch: requested={feature_set!r} but (stats,weights) imply "
                    f"{inferred_set!r} (D={d_ckpt}). Using inferred feature_names for the demo."
                )

        if not (d_stats == d_feat == d_ckpt):
            raise SystemExit(
                "Feature dimension mismatch for live demo:\n"
                f"- stats x_mean dim: {d_stats}\n"
                f"- checkpoint input_dim: {d_ckpt}\n"
                f"- feature_names len: {d_feat}\n\n"
                "This usually means you're mixing a (stats, weights) pair prepared with a different "
                "`data.real.feature_set` (minimal vs full).\n"
                "Fix: pass a matching pair via `--weights ... --stats ...`, or set `--feature_set minimal/full`, "
                "or re-run "
                "`scripts/prepare_closed_loop_csv.py` and `scripts/train_closed_loop_csv.py` with a consistent config."
            )

        def _zeros() -> np.ndarray:
            return np.zeros((H,), dtype=np.float64)

        return ClosedLoopTwin(
            model=model,
            stats=stats,
            feature_names=feature_names,
            H=H,
            device=str(device),
            q_hist=_zeros(),
            qd_hist=_zeros(),
            qref_hist=_zeros(),
            qdref_hist=_zeros(),
            kp_hist=_zeros(),
            kd_hist=_zeros(),
            tau_ff_hist=_zeros(),
            dt_hist=_zeros(),
            q_hat=0.0,
            qd_hat=0.0,
        )

    def reset(self, *, q0: float, qd0: float, dt0: float, q_ref0: float, qd_ref0: float, kp0: float, kd0: float, tau_ff0: float) -> None:
        self.q_hat = float(q0)
        self.qd_hat = float(qd0)

        # Fill history with the initial state/command.
        self.q_hist[:] = float(q0)
        self.qd_hist[:] = float(qd0)
        self.qref_hist[:] = float(q_ref0)
        self.qdref_hist[:] = float(qd_ref0)
        self.kp_hist[:] = float(kp0)
        self.kd_hist[:] = float(kd0)
        self.tau_ff_hist[:] = float(tau_ff0)
        self.dt_hist[:] = float(dt0)

    def step(self, *, q_ref: float, qd_ref: float, kp: float, kd: float, tau_ff: float, dt: float) -> Tuple[float, float, float]:
        """
        Advance the twin by one step using ONLY:
          command (q_ref, qd_ref, kp, kd, tau_ff, dt)
          and twin internal state (q_hat, qd_hat)

        Returns:
          (q_hat_next, qd_hat_next, tau_cmd_hat_used)
        """
        import torch

        # Roll history
        self.q_hist[:-1] = self.q_hist[1:]
        self.qd_hist[:-1] = self.qd_hist[1:]
        self.qref_hist[:-1] = self.qref_hist[1:]
        self.qdref_hist[:-1] = self.qdref_hist[1:]
        self.kp_hist[:-1] = self.kp_hist[1:]
        self.kd_hist[:-1] = self.kd_hist[1:]
        self.tau_ff_hist[:-1] = self.tau_ff_hist[1:]
        self.dt_hist[:-1] = self.dt_hist[1:]

        self.q_hist[-1] = float(self.q_hat)
        self.qd_hist[-1] = float(self.qd_hat)
        self.qref_hist[-1] = float(q_ref)
        self.qdref_hist[-1] = float(qd_ref)
        self.kp_hist[-1] = float(kp)
        self.kd_hist[-1] = float(kd)
        self.tau_ff_hist[-1] = float(tau_ff)
        self.dt_hist[-1] = float(dt)

        feat = _build_features(
            q=self.q_hist,
            qd=self.qd_hist,
            q_ref=self.qref_hist,
            qd_ref=self.qdref_hist,
            kp=self.kp_hist,
            kd=self.kd_hist,
            tau_ff=self.tau_ff_hist,
            dt=self.dt_hist,
            feature_names=self.feature_names,
        ).astype(np.float32)  # [H,D]

        x_mean = self.stats["x_mean"]
        x_std = self.stats["x_std"]
        y_mean = self.stats["y_mean"]
        y_std = self.stats["y_std"]

        x_n = (feat - x_mean[None, :]) / x_std[None, :]
        x_buf = torch.from_numpy(x_n[None, :, :]).float().to(self.device)  # [1,H,D]
        with torch.no_grad():
            pred_n = self.model(x_buf).detach().cpu().numpy().reshape(-1).astype(np.float64)  # [2]
        delta = pred_n * y_std.astype(np.float64) + y_mean.astype(np.float64)
        dq, dqd = float(delta[0]), float(delta[1])

        e_q = float(q_ref) - float(self.q_hat)
        e_qd = float(qd_ref) - float(self.qd_hat)
        tau_cmd_hat = float(kp) * e_q + float(kd) * e_qd + float(tau_ff)

        self.q_hat = float(self.q_hat + dq)
        self.qd_hat = float(self.qd_hat + dqd)
        return self.q_hat, self.qd_hat, tau_cmd_hat


def _profile_sine(t: float, *, q0: float, q_center: float, amp: float, freq_hz: float, ramp_s: float) -> Tuple[float, float]:
    """
    Sine reference with a smooth start to avoid an initial position jump.

    During the first `ramp_s` seconds we smoothly blend from a hold at q0 to a sine around q_center:
      q_ref(t) = q_center + a(t)*cos(w t) + b(t)
      a(t) = amp * s(t)
      b(t) = (q0 - q_center) * (1 - s(t))
      s(t) = 0.5*(1 - cos(pi*u)), u=clamp(t/ramp_s,0,1)

    This ensures:
      - at t=0: q_ref=q0 and qd_ref=0
      - after ramp: q_ref=q_center + amp*cos(wt), qd_ref=-amp*w*sin(wt)
    """
    w = 2.0 * np.pi * float(freq_hz)
    if ramp_s <= 1e-9:
        s = 1.0
        ds = 0.0
    else:
        u = float(np.clip(t / float(ramp_s), 0.0, 1.0))
        s = 0.5 * (1.0 - float(np.cos(np.pi * u)))
        ds = (0.5 * np.pi / float(ramp_s) * float(np.sin(np.pi * u))) if u < 1.0 else 0.0

    a = float(amp) * s
    da = float(amp) * ds
    b = (float(q0) - float(q_center)) * (1.0 - s)
    db = (float(q0) - float(q_center)) * (-ds)

    c = float(np.cos(w * t))
    sn = float(np.sin(w * t))
    q_ref = float(q_center + a * c + b)
    qd_ref = float(da * c + a * (-w * sn) + db)
    return q_ref, qd_ref


def _default_out_paths(out_dir: str, tag: str) -> Tuple[str, str]:
    ensure_dir(out_dir)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"live_twin_{tag}_{ts}"
    return os.path.join(out_dir, stem + ".csv"), os.path.join(out_dir, stem + ".png")


def main() -> None:
    ap = argparse.ArgumentParser(description="Live closed-loop (command-conditioned) digital twin demo (Unitree GO-M8010-6)")
    ap.add_argument(
        "--config",
        default="",
        help="optional demo config json (fields: hw_config, model_config, weights, stats, device, feature_set, "
        "duration_s, rate_hz, kp, kd, tau_ff, q_center, amp, freq_hz, plot, dry_run, out_dir, tag)",
    )
    ap.add_argument("--hw_config", default="config.json", help="hardware config containing real.serial_port/motor_id/unitree_sdk_lib")
    ap.add_argument("--model_config", default="configs/real_csv_closed_loop_gpu.json", help="model config for closed-loop twin")
    ap.add_argument("--weights", default="", help="override weights path (default: paths.real_csv_model from model_config)")
    ap.add_argument("--stats", default="", help="override stats npz path (default: paths.real_csv_stats from model_config)")
    ap.add_argument("--device", default="cuda", help="cpu or cuda (model inference device)")
    ap.add_argument("--out_dir", default="results", help="output directory (default: results/)")
    ap.add_argument("--tag", default="sine", help="tag for output filenames")
    ap.add_argument("--dry_run", action="store_true", help="do not send commands to motor (still reads and runs twin)")
    ap.add_argument("--duration_s", type=float, default=20.0, help="demo duration seconds")
    ap.add_argument("--rate_hz", type=float, default=200.0, help="control loop rate (Hz)")
    ap.add_argument("--kp", type=float, default=float("nan"), help="override kp (default: model_config real.kp)")
    ap.add_argument("--kd", type=float, default=float("nan"), help="override kd (default: model_config real.kd)")
    ap.add_argument("--tau_ff", type=float, default=0.0, help="feedforward torque command (default 0)")
    ap.add_argument("--feature_set", default="", help="override feature_set (default: model_config data.real.feature_set)")

    # sine profile parameters
    ap.add_argument("--q_center", type=float, default=1.0, help="sine center position (rad)")
    ap.add_argument("--amp", type=float, default=0.2, help="sine amplitude (rad)")
    ap.add_argument("--freq_hz", type=float, default=0.1, help="sine frequency (Hz)")
    ap.add_argument("--ramp_s", type=float, default=1.0, help="smoothly blend from q0 to sine over this duration (s)")
    ap.add_argument("--plot", action="store_true", help="save a png plot (requires matplotlib)")

    args = ap.parse_args()

    demo_cfg: Dict[str, Any] = {}
    if str(args.config).strip():
        demo_cfg = load_config(str(args.config))

    def _pick(key: str, default: Any) -> Any:
        cli_v = getattr(args, key)
        # For argparse "store_true", default is False and we want config to be able to set True.
        if isinstance(cli_v, bool):
            return cli_v if cli_v else bool(demo_cfg.get(key, default))
        # For strings: prefer CLI if non-empty.
        if isinstance(cli_v, str):
            return cli_v if cli_v.strip() else demo_cfg.get(key, default)
        # For floats/ints: if CLI equals argparse default, allow config override.
        return demo_cfg.get(key, cli_v)

    hw_config_path = str(_pick("hw_config", "config.json"))
    model_config_path = str(_pick("model_config", "configs/real_csv_closed_loop_gpu.json"))

    hw_cfg = load_config(hw_config_path)
    model_cfg = load_config(model_config_path)

    weights = str(_pick("weights", "")).strip() or str(get(model_cfg, "paths.real_csv_model"))
    stats = str(_pick("stats", "")).strip() or str(get(model_cfg, "paths.real_csv_stats"))
    device = str(_pick("device", "cuda"))
    feature_set = str(_pick("feature_set", "")).strip() or str(
        get(model_cfg, "data.real.feature_set", required=False, default="minimal")
    )

    kp_cli = float(getattr(args, "kp"))
    kd_cli = float(getattr(args, "kd"))
    kp = float(_pick("kp", kp_cli)) if np.isfinite(kp_cli) else float(demo_cfg.get("kp", get(model_cfg, "real.kp", required=False, default=0.0)))
    kd = float(_pick("kd", kd_cli)) if np.isfinite(kd_cli) else float(demo_cfg.get("kd", get(model_cfg, "real.kd", required=False, default=0.0)))
    tau_ff = float(_pick("tau_ff", float(getattr(args, "tau_ff"))))

    out_dir = str(_pick("out_dir", "results"))
    tag = str(_pick("tag", "sine"))
    out_csv, out_png = _default_out_paths(out_dir, tag)

    if str(args.config).strip():
        print("[demo] demo_config:", str(args.config))
    print("[demo] hw_config:", hw_config_path)
    print("[demo] model_config:", model_config_path)
    print("[demo] weights:", weights)
    print("[demo] stats:", stats)
    print(f"[demo] device={device} feature_set(requested)={feature_set} kp={kp:g} kd={kd:g} tau_ff={tau_ff:g}")
    duration_s = float(_pick("duration_s", float(getattr(args, "duration_s"))))
    rate_hz = float(_pick("rate_hz", float(getattr(args, "rate_hz"))))
    dry_run = bool(_pick("dry_run", bool(getattr(args, "dry_run"))))
    plot = bool(_pick("plot", bool(getattr(args, "plot"))))
    print(f"[demo] duration_s={duration_s:g} rate_hz={rate_hz:g} dry_run={dry_run}")
    print(f"[demo] out_csv: {out_csv}")
    if plot:
        print(f"[demo] out_png: {out_png}")

    motor = UnitreeGoM8010_6(hw_cfg)
    twin = ClosedLoopTwin.load_from_files(
        model_cfg=model_cfg,
        weights_path=weights,
        stats_path=stats,
        device=device,
        feature_set=feature_set,
    )
    print(f"[demo] feature_names(loaded)={twin.feature_names} (D={len(twin.feature_names)})")

    stop_flag = {"stop": False}

    def _on_sigint(_sig, _frm):
        stop_flag["stop"] = True

    signal.signal(signal.SIGINT, _on_sigint)

    dt_nom = 1.0 / max(1e-6, float(rate_hz))
    t0 = time.time()
    last = t0

    # Initial read and twin reset.
    q0, qd0, tau0 = motor.read()
    q_center = float(_pick("q_center", float(getattr(args, "q_center"))))
    amp = float(_pick("amp", float(getattr(args, "amp"))))
    freq_hz = float(_pick("freq_hz", float(getattr(args, "freq_hz"))))
    ramp_s = float(_pick("ramp_s", float(getattr(args, "ramp_s"))))

    qref0, qdref0 = _profile_sine(0.0, q0=float(q0), q_center=q_center, amp=amp, freq_hz=freq_hz, ramp_s=ramp_s)
    twin.reset(q0=q0, qd0=qd0, dt0=dt_nom, q_ref0=qref0, qd_ref0=qdref0, kp0=kp, kd0=kd, tau_ff0=tau_ff)

    fields = [
        "t_s",
        "stage",
        "q_ref_rad",
        "dq_ref_rad_s",
        "kp",
        "kd",
        "tau_ff",
        "q_rad",
        "dq_rad_s",
        "tau_Nm",
        "q_hat_rad",
        "dq_hat_rad_s",
        "tau_cmd_hat_Nm",
        "tau_pd_meas_Nm",
        "err_q_rad",
        "err_dq_rad_s",
        "dt_s",
    ]

    rows: List[Dict[str, float | str]] = []
    print("[demo] running... (Ctrl-C to stop)")

    # Optional: hold the motor at start for a short settle time (avoid jumping to q_center).
    settle_s = 0.3
    while time.time() - t0 < settle_s and not stop_flag["stop"]:
        qref, qdref = float(q0), 0.0
        if not dry_run:
            motor.send_pd(q_des=qref, dq_des=qdref, kp=kp, kd=kd, tau_ff=tau_ff)
        time.sleep(dt_nom)

    while not stop_flag["stop"]:
        now = time.time()
        t_s = now - t0
        if t_s >= float(duration_s):
            break

        dt = float(np.clip(now - last, 0.0, 0.05))
        last = now

        # commands for this tick
        qref, qdref = _profile_sine(t_s, q0=float(q0), q_center=q_center, amp=amp, freq_hz=freq_hz, ramp_s=ramp_s)

        if not dry_run:
            motor.send_pd(q_des=qref, dq_des=qdref, kp=kp, kd=kd, tau_ff=tau_ff)
        q, qd, tau = motor.read()

        # error is computed against current twin prediction (which does not observe q/qd)
        err_q = float(twin.q_hat - q)
        err_qd = float(twin.qd_hat - qd)

        # Diagnostic: PD torque computed from measured state (not available to twin at deployment)
        tau_pd_meas = float(kp) * float(qref - q) + float(kd) * float(qdref - qd) + float(tau_ff)

        # Advance twin one step (predict next tick)
        q_hat_next, qd_hat_next, tau_cmd_hat = twin.step(q_ref=qref, qd_ref=qdref, kp=kp, kd=kd, tau_ff=tau_ff, dt=dt)
        _ = (q_hat_next, qd_hat_next)  # keep for clarity

        rows.append(
            {
                "t_s": float(t_s),
                "stage": "sine",
                "q_ref_rad": float(qref),
                "dq_ref_rad_s": float(qdref),
                "kp": float(kp),
                "kd": float(kd),
                "tau_ff": float(tau_ff),
                "q_rad": float(q),
                "dq_rad_s": float(qd),
                "tau_Nm": float(tau),
                "q_hat_rad": float(twin.q_hat),
                "dq_hat_rad_s": float(twin.qd_hat),
                "tau_cmd_hat_Nm": float(tau_cmd_hat),
                "tau_pd_meas_Nm": float(tau_pd_meas),
                "err_q_rad": float(err_q),
                "err_dq_rad_s": float(err_qd),
                "dt_s": float(dt),
            }
        )

        # crude rate control
        sleep_s = dt_nom - (time.time() - now)
        if sleep_s > 0:
            time.sleep(float(sleep_s))

    print("[demo] stopping motor...")
    motor.stop()

    ensure_dir(os.path.dirname(out_csv) or ".")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"saved: {out_csv} ({len(rows)} rows)")

    if plot:
        try:
            import matplotlib.pyplot as plt

            t = np.array([float(r["t_s"]) for r in rows], dtype=np.float64)
            q_ref = np.array([float(r["q_ref_rad"]) for r in rows], dtype=np.float64)
            qd_ref = np.array([float(r["dq_ref_rad_s"]) for r in rows], dtype=np.float64)
            q = np.array([float(r["q_rad"]) for r in rows], dtype=np.float64)
            qd = np.array([float(r["dq_rad_s"]) for r in rows], dtype=np.float64)
            q_hat = np.array([float(r["q_hat_rad"]) for r in rows], dtype=np.float64)
            qd_hat = np.array([float(r["dq_hat_rad_s"]) for r in rows], dtype=np.float64)
            tau = np.array([float(r["tau_Nm"]) for r in rows], dtype=np.float64)
            tau_cmd_hat = np.array([float(r["tau_cmd_hat_Nm"]) for r in rows], dtype=np.float64)
            tau_pd_meas = np.array([float(r["tau_pd_meas_Nm"]) for r in rows], dtype=np.float64)

            plt.figure(figsize=(12, 9))
            ax1 = plt.subplot(3, 1, 1)
            ax1.set_title("Live closed-loop digital twin (open-loop rollout)")
            ax1.plot(t, q_ref, label="q_ref", color="tab:gray", lw=1.0, alpha=0.8)
            ax1.plot(t, q, label="q_gt", color="k", lw=1.0, alpha=0.8)
            ax1.plot(t, q_hat, label="q_hat", color="r", lw=1.3, alpha=0.9)
            ax1.set_ylabel("q (rad)")
            ax1.grid(True)
            ax1.legend()

            ax2 = plt.subplot(3, 1, 2, sharex=ax1)
            ax2.plot(t, qd_ref, label="qd_ref", color="tab:gray", lw=1.0, alpha=0.8)
            ax2.plot(t, qd, label="qd_gt", color="k", lw=1.0, alpha=0.6)
            ax2.plot(t, qd_hat, label="qd_hat", color="r", lw=1.3, alpha=0.9)
            ax2.set_ylabel("qd (rad/s)")
            ax2.grid(True)
            ax2.legend()

            ax3 = plt.subplot(3, 1, 3, sharex=ax1)
            ax3.plot(t, tau, label="tau_Nm (feedback est.)", color="k", lw=1.0, alpha=0.6)
            ax3.plot(t, tau_cmd_hat, label="tau_cmd_hat (twin)", color="tab:blue", lw=1.2, alpha=0.9)
            ax3.plot(t, tau_pd_meas, label="tau_pd (meas state, diag)", color="tab:orange", lw=1.2, alpha=0.8)
            ax3.set_ylabel("tau (Nm)")
            ax3.set_xlabel("time (s)")
            ax3.grid(True)
            ax3.legend()

            plt.tight_layout()
            plt.savefig(out_png, dpi=140)
            print(f"saved: {out_png}")
        except ImportError:
            print("[warn] matplotlib not available; skip plot")


if __name__ == "__main__":
    main()
