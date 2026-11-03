#!/bin/bash
# The readout's one constant (referees 2026-11-03): row D trained with BHR at four other phi (slopes about 544, 272,
# 68, 34 against 136 at phi 0.0556), and with a softmax of the group sums at BHR's slope 136 fixed (fixtemp_group),
# three seeds each. Writes results/pilot_caltech10fct_e50_{bhr,fixtemp_group}_amp_pt<value>_..._s<seed>.json.
# Run:  bash readout_sweep.sh   (in the release: bash src/readout_sweep.sh; about 15 runs of row D on a GPU; a done run is skipped)
cd "$(dirname "$0")/.." && mkdir -p logs
PY=${PY:-python3}
ROW_D="--rot rzryrx --order morton --n_out 7 --sim fast --ent brickwall --dataset caltech10fct --n_qubits 12
  --n_train 2644 --balanced --aug_crop 0.8 --epochs 50 --device cuda --optical 4 --center_image --mean_plane 0.5
  --fourier_mask --fourier_linear --fourier_phase"
run() {   # mode, pt_ratio, seed
  f=results/pilot_caltech10fct_e50_$1_amp_pt$2_q12_d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s$3.json
  [ -f "$f" ] && { echo "done: $f"; return; }
  $PY src/pilot_fmnist.py --mode "$1" --pt_ratio "$2" --seed "$3" $ROW_D > "logs/readout_sweep_$1_pt$2_s$3.log" 2>&1
  tail -1 "logs/readout_sweep_$1_pt$2_s$3.log"
}
for s in 0 1 2; do
  run fixtemp_group 136 $s
  for phi in 0.0139 0.0278 0.111 0.222; do run bhr $phi $s; done
done
