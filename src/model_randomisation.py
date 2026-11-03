#!/usr/bin/env python3
"""Model-randomisation check of the evidence maps (referee B, second round; Adebayo et al. 2018): do the maps change
when the trained weights are destroyed? For row D (phase plates) and D' (qubits), seed 0, the 300 Caltech-10 test
pictures, the true class's evidence of the trained model against models whose trained values are permuted at random
(each a fixed draw, numpy default_rng(0)), cascading from the top:
  - the plates (row D) or the brickwall circuit (D') permuted, the Fourier-plane mask kept;
  - both permuted (mask too).
Reported, as medians over pictures: the Spearman correlation of each randomised model's signed and |e| maps with the
trained model's, and of each model's |e| map with the encoded light of the pixel (sum over colour planes of its
amplitude squared): what any map of this encoding shares, trained or not.
Run:  python src/model_randomisation.py  -> numbers, results/model_randomisation.json
"""
import copy
import json
import pathlib

import numpy as np
from scipy.stats import spearmanr

import fastsim
from attention_in_box import maps

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
C = "results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_"
RUNS = {"plates (row D)": C + "d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s0.json",
        "qubits (row D')": C + "d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json"}


def parts(run):
    """(rows of the trained top part: plates or circuit, rows of the Fourier-plane mask), as slices of the weights."""
    w = np.asarray(run["weights"])
    n, nr = w.shape[1], w.shape[2]
    if run.get("optical"):
        fr = fastsim.fourier_rows(n, nr, 32, 2 ** n // 1024, run.get("fourier_phase", False))
        return slice(1 + fr, None), slice(1, 1 + fr)
    return slice(0, run["depth"] + 1), slice(run["depth"] + 1, None)


def permuted(run, rows, rng):
    r = copy.deepcopy(run)
    w = np.asarray(r["weights"], dtype=float)
    for sl in rows:
        block = w[sl]
        w[sl] = rng.permutation(block.reshape(-1)).reshape(block.shape)
    r["weights"] = w.tolist()
    return r


def med_spearman(A, B):
    return float(np.median([spearmanr(a.ravel(), b.ravel())[0] for a, b in zip(A, B)]))


def main():
    out = {}
    for name, f in RUNS.items():
        run = json.loads((HERE / f).read_text())
        d = np.load(HERE / "data" / f"{run['dataset']}.npz")
        X, Y = d["x_val"], d["y_val"]
        light = maps(run, X, Y, signed=True)[0]                  # the encoded light per pixel
        top, mask = parts(run)
        rng = np.random.default_rng(0)
        models = {"trained": run, "top permuted": permuted(run, [top], rng), "all permuted": permuted(run, [top, mask], rng)}
        ev = {k: maps(m, X, Y, signed=True)[2] for k, m in models.items()}
        r = {}
        for k, e in ev.items():
            r[k] = dict(signed_vs_trained=med_spearman(e, ev["trained"]),
                        abs_vs_trained=med_spearman(np.abs(e), np.abs(ev["trained"])),
                        abs_vs_light=med_spearman(np.abs(e), light))
            print(f"{name:16s} {k:13s} signed vs trained {r[k]['signed_vs_trained']:+.2f}, |e| vs trained "
                  f"{r[k]['abs_vs_trained']:+.2f}, |e| vs light {r[k]['abs_vs_light']:+.2f}", flush=True)
        out[name] = r
    (HERE / "results" / "model_randomisation.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
