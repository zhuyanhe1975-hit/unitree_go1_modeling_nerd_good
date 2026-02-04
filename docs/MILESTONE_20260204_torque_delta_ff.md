# Milestone (2026-02-04): Torque-Delta Feedforward Compensation

This repo reached a stage where **torque-delta feedforward** provides a clear and repeatable tracking improvement
in real hardware (Unitree GO-M8010-6) under position closed-loop.

## What We Built

1) **Torque-delta model** (one-step):

- Predict: `delta_tau_out[k] = tau_out[k] - tau_out[k-1]`
- Online construct (used as feedforward):
  - `tau_hat[k] = tau_out[k-1] + delta_tau_pred[k]`
  - `tau_ff[k] = tau_hat[k]` (optionally scaled/limited)

2) **Real-time demo** on hardware for apples-to-apples comparisons:

- Same sine trajectory, same `kp/kd`, same `dt`
- Compare:
  - baseline: position loop only (`tau_ff=0`)
  - ff: position loop + feedforward (`tau_ff != 0`)

3) **Robust execution**:

- Loop timing instrumentation (`loop_dt_median`, `loop_dt_p90`) to ensure comparisons are fair.
- Controls for safety and stability: `tau_ff_limit`, `tau_ff_slew`, `ff_update_div`.

## Key Code

- Torque-delta pipeline:
  - `inverse_torque/prepare.py`
  - `inverse_torque/train.py`
  - `inverse_torque/eval.py`
  - `inverse_torque/eval_low_speed_reversal.py`
- Hardware demo:
  - `scripts/demo_ff_sine.py`

## Repro Steps (Local)

1) Train torque-delta model (offline):

```bash
PYTHONPATH=. python3 inverse_torque/prepare.py
PYTHONPATH=. python3 inverse_torque/train.py
```

2) Run hardware demo:

```bash
PYTHONPATH=. python3 scripts/demo_ff_sine.py \
  --mode both \
  --ff_type torque_delta \
  --amp 0.2 --freq 0.1 --duration 20 \
  --kp 1 --kd 0.01 \
  --tau_ff_limit 0.15 --tau_ff_scale 1 \
  --ff_update_div 1
```

Artifacts saved under `runs/`:
- `ff_demo_report_*.md`
- `ff_demo_compare_*.png`
- `ff_demo_baseline_*.npz`, `ff_demo_ff_*.npz`

## Observed Results (Representative)

On low-frequency/low-speed sine tracking (position closed-loop), torque-delta feedforward showed:

- Large reduction in `e_q` RMS and peak errors (often 50–70% RMS reduction) when baseline gains are modest.
- Improvements remain visible with higher `kp/kd`, but the marginal gain reduces as the feedback loop dominates.

These results are captured in the `runs/ff_demo_report_*.md` and `runs/ff_demo_compare_*.png` artifacts generated on 2026-02-04.

## Notes / Caveats

- This feedforward form (`tau_ff = tau_hat`) can become "forceful" because it is on the same order as measured `tau_out`.
  Use `tau_ff_scale` and `tau_ff_limit` conservatively.
- For reproducible Unitree SDK imports, prefer setting `UNITREE_ACTUATOR_SDK_LIB` to the folder containing
  `unitree_actuator_sdk*.so`, and/or set `real.unitree_sdk_lib` in `config.json`.

