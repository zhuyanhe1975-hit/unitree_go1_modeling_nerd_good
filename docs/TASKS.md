# Tasks: Torque-Delta Feedforward Improvement Roadmap

Scope: keep the current successful approach (**train from real `runs/real_log.npz`**) and iteratively improve torque-delta feedforward tracking performance and robustness.

Key entrypoints:
- Offline: `inverse_torque/prepare.py`, `inverse_torque/train.py`, `inverse_torque/eval_low_speed_reversal.py`
- Online demo: `scripts/demo_ff_sine.py` (`--ff_type torque_delta`)

---

## Phase 0 — Baseline Freeze (Now)
- [ ] Record the current “known-good” default demo setting (amp/freq/kp/kd/limit/scale/update_div) in a short table.
- [ ] Define the success metrics we will always report:
- [ ] `rmse_e_q`, `maxabs_e_q`, `meanabs_tau_out`, `meanabs_tau_ff`, `loop_dt_median`, `loop_dt_p90`
- [ ] Define one standard track (low-speed) and one stress track (higher speed) for regression testing.

Deliverables:
- A consistent “golden run” command line saved in this doc.
- A regression expectation: “new model must not degrade baseline by >X%”.

---

## Phase 1 — Evaluation Harness (Make Progress Measurable)
- [ ] Add a small offline evaluator that replays `ff_demo_*.npz` and prints a single summary table for multiple runs.
- [ ] Add a “segment metrics” report focusing on low-speed reversals:
- [ ] Detect reversal regions by `qd` and compute error statistics only inside those regions.
- [ ] Produce a single `runs/summary_*.md` per experiment batch (no screenshots required).

Acceptance:
- One command produces a markdown summary comparing multiple experiments.

---

## Phase 2 — Better Real Logs (Add the Missing “Controller Context”)
Goal: make the model see what the position loop is asking for.

- [ ] Extend `scripts/demo_ff_sine.py` logs to include:
- [ ] `kp`, `kd` (per step)
- [ ] `tau_cmd_ff` (the actual feedforward we send)
- [ ] `qd_from_q` (computed from q and dt)
- [ ] Optionally: `qdd_from_q` (lightly filtered)
- [ ] Extend the log to include `q_ref`, `qd_ref` (already present), and compute/store:
- [ ] `e_q = q_ref - q`
- [ ] `e_qd = qd_ref - qd`
- [ ] Make sure `runs/real_log.npz` schema is documented in one place.

Acceptance:
- A new `real_log` contains enough information to train either:
- [ ] A pure dynamics-ish model (state driven), or
- [ ] A controller-aware model (error/ref driven).

---

## Phase 3 — Torque-Delta Model v2 (Feature Upgrade)
Goal: improve predictability in the exact deployed regime (position closed-loop).

- [ ] Create a new dataset version (do not break the current one):
- [ ] Output: `runs/torque_delta_dataset_v2.npz`
- [ ] Input features per step (suggested):
- [ ] `sin(q)`, `cos(q)`, `qd`
- [ ] `q_ref`, `qd_ref` (or `e_q`, `e_qd`)
- [ ] `tau_out` (k-1 history only)
- [ ] Optional: `temp`
- [ ] Keep the same target: `delta_tau_out[k]`
- [ ] Update `scripts/demo_ff_sine.py` to allow selecting which model/dataset to use (v1 vs v2).

Acceptance:
- v2 beats v1 on the standard low-speed track without causing timing issues.

---

## Phase 4 — Training Improvements (Make Delta Learning Less Noisy)
- [ ] Add optional smoothing before differencing:
- [ ] Low-pass filter `tau_out` then compute `delta_tau_out`
- [ ] Add robust loss option:
- [ ] Huber loss (or clipped MSE) to reduce sensitivity to torque quantization spikes
- [ ] Add sample weighting around “friction-critical regions”:
- [ ] Higher weight when `|qd| < v_th`
- [ ] Higher weight near sign changes of `qd`
- [ ] Add early-stopping using the time-series validation split (avoid overfit).

Acceptance:
- Validation metrics improve and become more stable across runs.

---

## Phase 5 — Online Integration (Make It Safer + More Effective)
- [ ] Add a `torque_delta_mode` option in the demo:
- [ ] `tau_hat`: `tau_ff = tau_out_prev + delta_pred` (current)
- [ ] `delta_only`: `tau_ff = delta_pred` (more “compensator-like”)
- [ ] Add “gate near reversal” option:
- [ ] Apply feedforward only when `|qd| < v_gate` or within a short window around reversals
- [ ] Add a simple “confidence limiter”:
- [ ] If model output jumps too fast, clamp delta more aggressively than `tau_ff_slew`

Acceptance:
- The `delta_only` and `gated` modes do not destabilize tracking, and can still improve reversal error.

---

## Phase 6 — Generalization Matrix (Prove It’s Not a One-Off)
- [ ] Run a grid on real hardware (or as many as safe):
- [ ] `amp ∈ {0.1, 0.2, 0.3}`
- [ ] `freq ∈ {0.1, 0.2, 0.5}`
- [ ] `kp/kd` two settings: “soft” and “stiff”
- [ ] Summarize improvements as a table: baseline vs ff across the grid.

Acceptance:
- The model improves most cells in the low-speed/low-frequency regime and does not catastrophically degrade elsewhere.

---

## Phase 7 — Packaging / Release (Optional)
- [ ] Add a “quickstart” section dedicated to torque-delta feedforward (with minimal steps).
- [ ] Add versioned configs for experiments (checked in):
- [ ] `configs/ff_demo_soft.json`
- [ ] `configs/ff_demo_stiff.json`
- [ ] Add a script to collect + train + run demo in one guided flow.

Acceptance:
- Someone else can reproduce the milestone behavior from scratch by following 5–8 commands.

