#!/usr/bin/env python3
"""Paired tests for the accuracy steps of the test set on hardware (referee A, second round): the same 300 pictures
classified exactly, through the fitted loading without noise, and on ibm_kingston; exact McNemar tests (two-sided
binomial on the discordant pairs) for exact -> loading, loading -> chip and exact -> chip, for D' (with the lens
front end) and A' (plain amplitudes, no mask), and D' against A' at each step (the same pictures).
Run:  python src/hw_mcnemar.py  -> numbers, results/hw_mcnemar.json
"""
import json
import pathlib

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
MODELS = {"D'": "ibm_testset_pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5__L8",
          "A'": "ibm_testset_pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_crop0.8__L8"}


def mcnemar(a, b):
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return n01, n10, float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0


def correct(folder):
    f = HERE / "results" / folder
    pz = np.load(f / "preps.npz")
    chip = json.loads((f / "chip_ibm_kingston_r0.json").read_text())
    y = pz["y"]
    assert all(chip[str(i)]["label"] == y[i] for i in range(len(y)))
    return y, {"exact": pz["q_exact"].argmax(1) == y, "loading": pz["q_prep"].argmax(1) == y,
               "chip": np.array([chip[str(i)]["pred"] == chip[str(i)]["label"] for i in range(len(y))])}


def main():
    ok, ys, out = {}, {}, {}
    for m, folder in MODELS.items():
        ys[m], ok[m] = correct(folder)
        out[m] = {k: float(v.mean()) for k, v in ok[m].items()}
        for a, b in (("exact", "loading"), ("loading", "chip"), ("exact", "chip")):
            n01, n10, p = mcnemar(ok[m][a], ok[m][b])
            out[m][f"{a} -> {b}"] = dict(lost=n01, gained=n10, p=p)
            print(f"{m} {a:7s} -> {b:7s}: {100 * ok[m][a].mean():.1f} -> {100 * ok[m][b].mean():.1f} %, right only "
                  f"before {n01}, only after {n10}, exact McNemar p = {p:.3g}")
    assert (ys["D'"] == ys["A'"]).all()                         # the same pictures in the same order
    for s in ("exact", "loading", "chip"):
        n01, n10, p = mcnemar(ok["D'"][s], ok["A'"][s])
        out[f"D' vs A', {s}"] = dict(only_D=n01, only_A=n10, p=p)
        d, a = ok["D'"][s], ok["A'"][s]
        print(f"D' vs A' {s:7s}: {100 * d.mean():.1f} vs {100 * a.mean():.1f} %, right only "
              f"with D' {n01}, only with A' {n10}, exact McNemar p = {p:.3g}")
    (HERE / "results" / "hw_mcnemar.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
