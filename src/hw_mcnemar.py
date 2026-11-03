#!/usr/bin/env python3
"""Paired tests for the accuracy steps of the test set on hardware (referee A, second round): the same 300 pictures
classified exactly, through the fitted loading without noise, and on ibm_kingston; exact McNemar tests (two-sided
binomial on the discordant pairs) for exact -> loading, loading -> chip and exact -> chip.
Run:  python src/hw_mcnemar.py  -> numbers, results/hw_mcnemar.json
"""
import json
import pathlib

import numpy as np
from scipy.stats import binomtest

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root


def main():
    f = next((HERE / "results").glob("ibm_testset_*L8"))
    pz = np.load(f / "preps.npz")
    chip = json.loads((f / "chip_ibm_kingston_r0.json").read_text())
    y = pz["y"]
    ok = {"exact": pz["q_exact"].argmax(1) == y, "loading": pz["q_prep"].argmax(1) == y,
          "chip": np.array([chip[str(i)]["pred"] == chip[str(i)]["label"] for i in range(len(y))])}
    assert all(chip[str(i)]["label"] == y[i] for i in range(len(y)))
    out = {k: float(v.mean()) for k, v in ok.items()}
    for a, b in (("exact", "loading"), ("loading", "chip"), ("exact", "chip")):
        n01, n10 = int((ok[a] & ~ok[b]).sum()), int((~ok[a] & ok[b]).sum())
        p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
        out[f"{a} -> {b}"] = dict(lost=n01, gained=n10, p=float(p))
        print(f"{a:7s} -> {b:7s}: {100 * ok[a].mean():.1f} -> {100 * ok[b].mean():.1f} %, right only before {n01}, "
              f"only after {n10}, exact McNemar p = {p:.3g}")
    (HERE / "results" / "hw_mcnemar.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
