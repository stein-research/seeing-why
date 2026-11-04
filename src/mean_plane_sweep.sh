#!/bin/bash
# The means plane's share of the energy, f (referee B, second round, Q9): f = 0.5 was set before any run; row D trained
# again with f = 0.25 and 0.75, three seeds each, beside the paper's f = 0.5. Writes
# results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d0rzmo_cimp<f>_..._s<seed>.json and prints the table
# (results/mean_plane_sweep.json).
# Run:  bash mean_plane_sweep.sh   (in the release: bash src/mean_plane_sweep.sh; 6 runs of row D, about a minute each on a GPU)
cd "$(dirname "$0")/.." && mkdir -p logs
PY=${PY:-python3}
ROW_D="--mode bhr --pt_ratio 0.0556 --rot rzryrx --order morton --n_out 7 --sim fast --ent brickwall --dataset caltech10fct
  --n_qubits 12 --n_train 2644 --balanced --aug_crop 0.8 --epochs 50 --device cuda --optical 4 --center_image
  --fourier_mask --fourier_linear --fourier_phase"
f_run() { echo "results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d0rzmo_cimp$1_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s$2.json"; }
for s in 0 1 2; do
  for f in 0.25 0.75; do
    [ -f "$(f_run $f $s)" ] && { echo "done: $(f_run $f $s)"; continue; }
    $PY src/pilot_fmnist.py --mean_plane "$f" --seed "$s" $ROW_D > "logs/mean_plane_sweep_f${f}_s${s}.log" 2>&1
    tail -1 "logs/mean_plane_sweep_f${f}_s${s}.log"
  done
done
$PY - <<'EOF'
import json, numpy as np
out = {}
for f in ("0.25", "0.5", "0.75"):
    r = [json.load(open(f"results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d0rzmo_cimp{f}_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s{s}.json"))
         for s in range(3)]
    acc = np.array([x["test_exact"] for x in r]) * 100
    auc = np.array([x["roc_auc_macro"] for x in r])
    out[f] = dict(acc=acc.mean(), acc_sd=acc.std(ddof=1), auc=auc.mean(), auc_sd=auc.std(ddof=1), seeds=acc.tolist())
    print(f"f = {f:4s}  accuracy {acc.mean():.1f} +- {acc.std(ddof=1):.1f} %  ROC-AUC {auc.mean():.3f} +- {auc.std(ddof=1):.3f}"
          f"  seeds {', '.join(f'{a:.1f}' for a in acc)}")
json.dump(out, open("results/mean_plane_sweep.json", "w"), indent=1)
EOF
