#!/usr/bin/env bash
set -euo pipefail

# Sweep qd filter cutoff and generate:
#   - per-cutoff open-loop metrics
#   - reversal plots (low-speed direction changes)
#
# Usage:
#   conda activate nerd_py310
#   bash scripts/sweep_qd_filter_cutoff.sh
#
# Optional env overrides:
#   CFG=...                 (default: configs/real_csv_closed_loop_sweep_gpu.json)
#   CUTOFFS="10 15 20 30"   (default: "10 15 20 30 40 60")
#   DEVICE=cuda             (default: cuda)
#   HORIZON=300             (default: 300)
#   PLOT_HORIZON=4000       (default: 4000)
#   WINDOW_S=0.6            (default: 0.6)
#   OUTDIR=results/...      (default: results/qd_filter_sweep_<timestamp>)

CFG="${CFG:-configs/real_csv_closed_loop_sweep_gpu.json}"
CUTOFFS="${CUTOFFS:-10 15 20 30 40 60}"
DEVICE="${DEVICE:-cuda}"
HORIZON="${HORIZON:-300}"
PLOT_HORIZON="${PLOT_HORIZON:-4000}"
WINDOW_S="${WINDOW_S:-0.6}"

ts="$(date +%Y%m%d_%H%M%S)"
outdir="${OUTDIR:-results/qd_filter_sweep_${ts}}"
mkdir -p "${outdir}"

export PYTHONPATH="."

for hz in ${CUTOFFS}; do
  sub="${outdir}/cutoff_${hz}Hz"
  mkdir -p "${sub}"

  ds="${sub}/dataset.npz"
  st="${sub}/stats.npz"
  w="${sub}/model.pt"
  md="${sub}/summary.md"
  png="${sub}/plot_reversals_sine.png"
  cyc="${sub}/plot_full_cycle_sine.png"

  if [[ -f "${md}" && -f "${png}" && -f "${cyc}" ]]; then
    echo "skip cutoff=${hz}Hz (already have ${md})"
    continue
  fi

  python3 scripts/prepare_closed_loop_csv.py \
    --config "${CFG}" \
    --qd_filter_method zero_phase_one_pole \
    --qd_filter_hz "${hz}" \
    --qd_use_filtered \
    --out "${ds}" \
    --stats "${st}"

  python3 scripts/train_closed_loop_csv.py \
    --config "${CFG}" \
    --dataset "${ds}" \
    --out "${w}"

  python3 scripts/eval_closed_loop_csv.py \
    --config "${CFG}" \
    --dataset "${ds}" \
    --weights "${w}" \
    --qd_filter_method zero_phase_one_pole \
    --qd_filter_hz "${hz}" \
    --qd_use_filtered \
    --device "${DEVICE}" \
    --stages sine \
    --horizon_steps "${HORIZON}" \
    --out_md "${md}" \
    --plot_reversals \
    --plot_horizon_steps "${PLOT_HORIZON}" \
    --reversal_window_s "${WINDOW_S}" \
    --out_png "${png}" \
    --plot_full_cycle \
    --out_full_png "${cyc}" \
    >/dev/null

  echo "done cutoff=${hz}Hz"
done

python3 - "${outdir}" "${CFG}" "${DEVICE}" "${HORIZON}" "${CUTOFFS}" <<'PY'
import sys
from pathlib import Path
import math

outdir = Path(sys.argv[1])
cfg = sys.argv[2]
device = sys.argv[3]
horizon = sys.argv[4]
cutoffs = [c for c in sys.argv[5].split() if c.strip()]

def parse_one(md_path: Path) -> tuple[float, float, float]:
    s = md_path.read_text(encoding="utf-8")
    lines = s.splitlines()
    row = None
    for l in lines:
        if l.startswith("| sine |"):
            row = l
            break
    if row is None:
        raise RuntimeError(f"missing sine row in {md_path}")
    cols = [c.strip() for c in row.strip("|").split("|")]
    q_rmse = float(cols[2])
    qd_rmse = float(cols[4])
    delta = float("nan")
    for l in lines:
        if l.strip().startswith("- qd_filter_delta_rmse"):
            delta = float(l.split("`")[1])
            break
    return q_rmse, qd_rmse, delta

rows = []
for hz in cutoffs:
    sub = outdir / f"cutoff_{hz}Hz"
    md = sub / "summary.md"
    png = sub / "plot_reversals_sine.png"
    cyc = sub / "plot_full_cycle_sine.png"
    if not md.exists():
        continue
    qrmse, qdrmse, dlt = parse_one(md)
    score = qdrmse + (0.25 * dlt if math.isfinite(dlt) else 0.0)
    rows.append((score, float(hz), qrmse, qdrmse, dlt, png, cyc))

rows.sort()
out = []
out.append("# qd filter cutoff sweep")
out.append("")
out.append(f"- cfg: `{cfg}`")
out.append(f"- device: `{device}`")
out.append(f"- horizon_steps: `{horizon}`")
out.append("")
out.append("- filter: `zero_phase_one_pole`")
out.append("- score: `qd_rmse + 0.25 * qd_filter_delta_rmse` (smaller is better; penalizes over-smoothing)")
out.append("")
out.append("| cutoff_hz | q_rmse (sine) | qd_rmse (sine) | qd_filter_delta_rmse | score | reversal plot | full-cycle plot |")
out.append("| --- | --- | --- | --- | --- | --- | --- |")
for score, hz, qrmse, qdrmse, dlt, png, cyc in rows:
    rel_png = png.relative_to(outdir)
    rel_cyc = cyc.relative_to(outdir)
    out.append(
        f"| {int(hz)} | {qrmse:.6g} | {qdrmse:.6g} | {dlt:.6g} | {score:.6g} | `{rel_png.as_posix()}` | `{rel_cyc.as_posix()}` |"
    )

if rows:
    best = rows[0]
    out.append("")
    out.append(f"**Recommendation:** cutoff_hz=`{int(best[1])}` (min score).")
    out.append(f"- reversal: `{best[5].relative_to(outdir).as_posix()}`")
    out.append(f"- full-cycle: `{best[6].relative_to(outdir).as_posix()}`")

(outdir / "summary.md").write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
PY

echo "saved: ${outdir}/summary.md"
