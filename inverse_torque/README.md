# Inverse Torque Pipeline (State -> Torque Delta)

This subfolder provides a torque-delta prediction pipeline (inverse dynamics flavored),
separate from the original state-delta prediction flow.

## Goal
Given a history of states and the last measured torque, predict the **increment**
to the current measured torque:

- **input**: history of `[sin(q), cos(q), qd, (temp?), tau_out]`
- **target**: `delta_tau_out[k] = tau_out[k] - tau_out[k-1]`

## Files
- `prepare.py`: build `runs/torque_delta_dataset.npz`
- `train.py`: train `runs/torque_delta_model.pt`
- `eval.py`: evaluate & (optional) plot `runs/eval_torque_delta.png`

## Usage
Run from project root with the same environment:

```bash
PYTHONPATH=. python3 inverse_torque/prepare.py
PYTHONPATH=. python3 inverse_torque/train.py
PYTHONPATH=. python3 inverse_torque/eval.py --plot
```

## Notes
- Uses `paths.real_log` as the raw log by default.
- This pipeline assumes your raw log contains `tau_out` (measured/estimated actual torque).
- This pipeline is compatible with the current `config.json`.
