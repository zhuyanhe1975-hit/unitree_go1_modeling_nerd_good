from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np

from custom_envs.joint_1dof_env import Joint1DofEnv
from pipeline.features import state_to_features


@dataclass(frozen=True)
class PredictorConfig:
    weights_path: str
    dataset_npz: str
    device: str = "cpu"


class StateDeltaPredictor:
    """
    One-step state-delta predictor compatible with this repo's training pipeline:

      input x: history of [sin(q), cos(q), qd, tau] (normalized)
      output y: [delta_q, delta_qd] (normalized)
    """

    def __init__(self, cfg: PredictorConfig):
        import torch

        from pipeline.model import CausalTransformer

        ds = dict(np.load(cfg.dataset_npz, allow_pickle=True))
        self.s_mean = ds["s_mean"].astype(np.float32)
        self.s_std = ds["s_std"].astype(np.float32)
        self.a_mean = ds["a_mean"].astype(np.float32)
        self.a_std = ds["a_std"].astype(np.float32)
        self.d_mean = ds["d_mean"].astype(np.float32)
        self.d_std = ds["d_std"].astype(np.float32)

        ckpt = torch.load(cfg.weights_path, map_location=torch.device(cfg.device))
        self.model = CausalTransformer(
            input_dim=int(ckpt["input_dim"]),
            output_dim=int(ckpt["output_dim"]),
            embed_dim=int(ckpt.get("embed_dim", 64)),
            num_layers=int(ckpt.get("num_layers", 2)),
            num_heads=int(ckpt.get("num_heads", 4)),
            history_len=int(ckpt.get("history_len", ds["x"].shape[1])),
        ).to(cfg.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

        self.device = str(cfg.device)
        self.H = int(ckpt.get("history_len", ds["x"].shape[1]))

        self._torch = torch
        self._q_hist = None
        self._qd_hist = None
        self._tau_hist = None
        self._x_buf = None

    def reset(self, q0: np.ndarray, qd0: np.ndarray, tau0: Optional[np.ndarray] = None) -> None:
        q0 = np.asarray(q0, dtype=np.float64).reshape(-1)
        qd0 = np.asarray(qd0, dtype=np.float64).reshape(-1)
        if q0.shape != qd0.shape:
            raise ValueError(f"q0/qd0 shape mismatch: {q0.shape} vs {qd0.shape}")
        if tau0 is None:
            tau0 = np.zeros_like(q0)
        tau0 = np.asarray(tau0, dtype=np.float64).reshape(-1)
        if tau0.shape != q0.shape:
            raise ValueError(f"tau0 shape mismatch: {tau0.shape} vs {q0.shape}")

        B = int(q0.shape[0])
        self._q_hist = np.repeat(q0[:, None], self.H, axis=1).astype(np.float64)  # [B,H]
        self._qd_hist = np.repeat(qd0[:, None], self.H, axis=1).astype(np.float64)  # [B,H]
        self._tau_hist = np.repeat(tau0[:, None], self.H, axis=1).astype(np.float64)  # [B,H]
        self._x_buf = np.zeros((B, self.H, 4), dtype=np.float32)

    def step(self, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            tau: [B] or [B,1] torque command at current step
        Returns:
            (q_next[B], qd_next[B])
        """
        if self._q_hist is None or self._qd_hist is None or self._tau_hist is None or self._x_buf is None:
            raise RuntimeError("predictor not reset; call reset(q0, qd0) first")

        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if tau.shape[0] != self._q_hist.shape[0]:
            raise ValueError(f"tau batch mismatch: got {tau.shape[0]}, expected {self._q_hist.shape[0]}")

        # shift and append action
        self._tau_hist[:, :-1] = self._tau_hist[:, 1:]
        self._tau_hist[:, -1] = tau

        q = self._q_hist
        qd = self._qd_hist
        feat = state_to_features(q, qd).astype(np.float32)  # [B,H,3]
        act = self._tau_hist.astype(np.float32)[:, :, None]  # [B,H,1]

        # normalize and pack
        self._x_buf[..., :3] = (feat - self.s_mean[None, None, :]) / self.s_std[None, None, :]
        self._x_buf[..., 3:] = (act - self.a_mean[None, None, :]) / self.a_std[None, None, :]

        with self._torch.no_grad():
            xb = self._torch.from_numpy(self._x_buf).to(self.device)
            pred_n = self.model(xb).detach().cpu().numpy().astype(np.float32)  # [B,2]

        delta = pred_n * self.d_std[None, :] + self.d_mean[None, :]  # [B,2]
        q_next = q[:, -1] + delta[:, 0].astype(np.float64)
        qd_next = qd[:, -1] + delta[:, 1].astype(np.float64)

        # shift and append state
        self._q_hist[:, :-1] = self._q_hist[:, 1:]
        self._qd_hist[:, :-1] = self._qd_hist[:, 1:]
        self._q_hist[:, -1] = q_next
        self._qd_hist[:, -1] = qd_next

        return q_next, qd_next


EnvMode = Literal["ground-truth", "neural"]


class Joint1DofNeuralEnvironment:
    """
    A minimal NeRD-like wrapper for this repo's 1-DOF joint environment.

    - ground-truth: Warp (Featherstone) simulation via Joint1DofEnv
    - neural: roll forward using a learned state-delta model (CausalTransformer)
    """

    def __init__(
        self,
        env_cfg: Dict[str, Any],
        *,
        predictor_cfg: Optional[PredictorConfig] = None,
        default_env_mode: EnvMode = "ground-truth",
    ):
        self.env = Joint1DofEnv(env_cfg, device=str(env_cfg.get("device", "cuda")))
        self.predictor = StateDeltaPredictor(predictor_cfg) if predictor_cfg is not None else None
        self.env_mode: EnvMode = "ground-truth"
        self._q_neural = None
        self._qd_neural = None
        self.set_env_mode(default_env_mode)

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    def set_env_mode(self, mode: EnvMode) -> None:
        if mode not in ("ground-truth", "neural"):
            raise ValueError("mode must be 'ground-truth' or 'neural'")
        if mode == "neural" and self.predictor is None:
            raise RuntimeError("neural mode requires predictor_cfg (weights_path + dataset_npz)")
        self.env_mode = mode

    def reset(self, q0: float = 0.0, qd0: float = 0.0, tau0: float = 0.0) -> np.ndarray:
        # Reset the Warp env and then overwrite initial state for determinism
        # (Joint1DofEnv.reset randomizes q by default).
        _ = self.env.reset()

        # We only support 1 DoF per env; state layout is [q, qd].
        q = np.full((self.num_envs,), float(q0), dtype=np.float64)
        qd = np.full((self.num_envs,), float(qd0), dtype=np.float64)

        self._set_warp_state(q=q, qd=qd)

        self._q_neural = q.copy()
        self._qd_neural = qd.copy()
        if self.predictor is not None:
            self.predictor.reset(q0=q, qd0=qd, tau0=np.full_like(q, float(tau0)))

        # Return a consistent [K,2] state
        return np.stack([q, qd], axis=-1).astype(np.float32)

    def step(self, tau: np.ndarray) -> np.ndarray:
        """
        Args:
            tau: [K,1] torque action in Nm (same as Joint1DofEnv.step)
        Returns:
            next_state: [K,2] of (q, qd)
        """
        tau = np.asarray(tau, dtype=np.float64)
        if tau.ndim == 1:
            tau = tau[:, None]
        if tau.shape[0] != self.num_envs or tau.shape[1] != 1:
            raise ValueError(f"expected tau shape [K,1] with K={self.num_envs}, got {tau.shape}")

        if self.env_mode == "ground-truth":
            st = self.env.step(actions=_to_torch_like(tau, device=str(self.env.device)))
            return st.detach().cpu().numpy().astype(np.float32)

        if self._q_neural is None or self._qd_neural is None:
            raise RuntimeError("env not reset; call reset() first")

        q_next, qd_next = self.predictor.step(tau[:, 0])
        self._q_neural = q_next
        self._qd_neural = qd_next
        return np.stack([q_next, qd_next], axis=-1).astype(np.float32)

    def _set_warp_state(self, *, q: np.ndarray, qd: np.ndarray) -> None:
        import warp as wp

        q = np.asarray(q, dtype=np.float32).reshape(-1)
        qd = np.asarray(qd, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.num_envs or qd.shape[0] != self.num_envs:
            raise ValueError("q/qd must be [num_envs] for the 1-DOF env")

        with wp.ScopedDevice(str(self.env.device)):
            total_dof = int(self.env.state.joint_q.shape[0])
            if self.env.dof_per_env <= 0:
                raise RuntimeError("invalid dof_per_env")
            if total_dof != self.num_envs * self.env.dof_per_env:
                raise RuntimeError("unexpected warp state layout")

            q_full = np.zeros((total_dof,), dtype=np.float32)
            qd_full = np.zeros((total_dof,), dtype=np.float32)
            for env_i in range(self.num_envs):
                idx = env_i * self.env.dof_per_env
                q_full[idx] = float(q[env_i])
                qd_full[idx] = float(qd[env_i])

            self.env.state.joint_q.assign(wp.from_numpy(q_full, device=str(self.env.device), dtype=wp.float32))
            self.env.state.joint_qd.assign(wp.from_numpy(qd_full, device=str(self.env.device), dtype=wp.float32))
            self.env.control.joint_act.zero_()


def _to_torch_like(x: np.ndarray, device: str):
    import torch

    return torch.from_numpy(np.asarray(x, dtype=np.float32)).to(device)
