#!/usr/bin/env python3
"""The readout's one constant (readout_sweep.sh): row D's test accuracy and ROC-AUC (mean and spread over three seeds)
for BHR at each phi, with its slope per unit of probability kappa / (1 - b_g) (kappa = ln[(1 - eps)(G - 1)/eps]
/ phi, b_g = 13/128 for the paper's group sizes), beside a softmax of the group sums at BHR's slope 136 fixed, one with
a learned temperature and the raw sums.
Run:  python src/readout_sweep_report.py  -> table, results/readout_sweep.json
"""
import json
import math
import pathlib

import numpy as np

from bssg import BornHeadroomReadout

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
TAIL = "_q12_d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s{}.json"


def row(mode, pt):
    runs = [json.loads((HERE / "results" / f"pilot_caltech10fct_e50_{mode}_amp_pt{pt}{TAIL.format(s)}").read_text())
            for s in range(3)]
    acc = np.array([r["test_exact"] for r in runs]) * 100
    auc = np.array([r["roc_auc_macro"] for r in runs])
    return dict(acc=float(acc.mean()), acc_sd=float(acc.std(ddof=1)), auc=float(auc.mean()), auc_sd=float(auc.std(ddof=1)),
                seeds=[float(a) for a in acc])


def main():
    out = {}
    b = 13 / 128
    for phi in ("0.0139", "0.0278", "0.0556", "0.111", "0.222"):
        kappa = math.log((1 - BornHeadroomReadout.EPS) * 9 / BornHeadroomReadout.EPS) / float(phi)
        out[f"BHR phi {phi}"] = dict(slope=kappa / (1 - b), **row("bhr", phi))
    out["softmax, slope 136 fixed"] = dict(slope=136.0, **row("fixtemp_group", "136"))
    out["softmax, learned temperature"] = dict(slope=None, **row("temp_group", "0.0556"))
    out["raw sums"] = dict(slope=1.0, **row("raw_group", "0.0556"))
    for k, r in out.items():
        s = f"{r['slope']:5.0f}" if r["slope"] else "    -"
        print(f"{k:30s} slope {s}  accuracy {r['acc']:.1f} +- {r['acc_sd']:.1f} %  ROC-AUC {r['auc']:.3f} +- {r['auc_sd']:.3f}"
              f"  seeds {', '.join(f'{a:.1f}' for a in r['seeds'])}")
    (HERE / "results" / "readout_sweep.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
