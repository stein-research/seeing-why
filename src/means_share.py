#!/usr/bin/env python3
"""The means plane's share of the true-class evidence (referees A item 3 and B item 3, second round). The pixel maps
add the evidence of the three colour planes at each pixel and leave the means plane (the fourth plane of the mean-plane
encoding) out; its evidence is E_means = the sum over the means plane's amplitudes, and the planes add up to
P_g - b_g. Rows D (plates) and D' (qubits), seeds 0-2, the 300 Caltech-10 test pictures: the median of
E_means / (P_g - b_g), its quartiles, and the share of pictures where |E_means| > |P_g - b_g|.
Run:  python src/means_share.py  -> numbers, results/means_share.json
"""
import json
import pathlib

import numpy as np

from attention_in_box import maps

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
C = "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_"
RUNS = {"plates (row D)": C + "d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s{}.json",
        "qubits (row D')": C + "d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s{}.json"}


def main():
    out = {}
    for name, f in RUNS.items():
        share, over, rows = [], [], []
        for s in range(3):
            run = json.loads((HERE / "results" / f.format(s)).read_text())
            d = np.load(HERE / "data" / f"{run['dataset']}.npz")
            E = maps(run, d["x_val"], d["y_val"], planes=True)            # (300, 4): R, G, B, means
            tot = E.sum(1)
            share.append(E[:, 3] / tot)
            over.append(np.abs(E[:, 3]) > np.abs(tot))
            rows.append(dict(seed=s, median=float(np.median(E[:, 3] / tot)),
                             q=np.percentile(E[:, 3] / tot, [25, 75]).tolist(),
                             over=float(over[-1].mean()), median_abs_means=float(np.median(np.abs(E[:, 3]))),
                             median_abs_total=float(np.median(np.abs(tot)))))
        sh, ov = np.concatenate(share), np.concatenate(over)
        out[name] = dict(median=float(np.median(sh)), q=np.percentile(sh, [25, 75]).tolist(), over=float(ov.mean()),
                         seeds=rows)
        print(f"{name}: means plane / (P - b) median {100 * np.median(sh):+.1f} % (quartiles "
              f"{100 * np.percentile(sh, 25):+.1f}, {100 * np.percentile(sh, 75):+.1f}); |E_means| > |P - b| in "
              f"{100 * ov.mean():.1f} % of pictures; per seed medians "
              + ", ".join(f"{100 * r['median']:+.1f}" for r in rows))
    (HERE / "results" / "means_share.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
