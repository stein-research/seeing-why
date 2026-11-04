#!/usr/bin/env python3
"""Shot noise of a measured evidence map (referee B, second round, Q6): the variance of one block's evidence for
the two estimators of Section 2.6, against a Hadamard-test estimate of the same quantity, from the exact class
probabilities of the 30 random pictures' circuits (results/<test-set run>/random/states.npz, the circuits sent).

On the chip each circuit prepares a normalised state v/|v| and N shots estimate its class probability P(v) with
variance P(1 - P)/N; f(v) = |v|^2 (P(v) - b_g) (Z = 1: the input is the renormalised filtered state). With
w = |psi_S|^2, a block's evidence E_S = Re<psi_S|M|psi> has
  standard (Eq. polar):  Var = [s(psi) + w^2 s(psi_S) + (1 - w)^2 s(psi_notS)] / 4,
  sign flip (Eq. flip):  Var = w^2 s(psi_S) + [s(psi) + s(psi_S - psi_notS)] / 16,
with s(v) = P(v)(1 - P(v))/N. A Hadamard test (one ancilla, controlled preparations of psi_S/|psi_S| and psi) reads
E_S = sqrt(w) Re<u|M|psi> from a per-shot value (+-1) o, o = 1 - b_g on class g's outcomes and -b_g elsewhere:
  Var = w (E[o^2] - (E_S/sqrt(w))^2)/N,  E[o^2] = [q(u) + q(psi)]/2,  q(v) = P(1 - b_g)^2 + (1 - P) b_g^2.
Reported for N = 4,000 shots per circuit: the median predicted standard deviation of a block for each, the mean
absolute error that implies (sqrt(2 Var / pi), to compare with the simulated shot floor), and the shots a whole
g x g map needs for the same per-block error: (1 + 2 g^2) N for the two estimators, g^2 N_HT for the Hadamard test
with N_HT = N Var_HT / Var.
Run:  python src/shot_variance.py  -> numbers, results/shot_variance.json
"""
import glob
import json
import pathlib

import numpy as np

from bssg import class_groups

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
N = 4000


def main():
    s = np.load(glob.glob(str(HERE / "results" / "ibm_testset_*" / "random" / "states.npz"))[0])
    y = np.load(HERE / "data" / "caltech10fct.npz")["y_val"]
    groups = class_groups("bhr", 128, 10)
    b = np.array([len(g) / 128 for g in groups])
    out = {}
    for G in (2, 4):
        var = {"standard": [], "sign flip": [], "Hadamard test": []}
        for p in np.unique(s["pic"][s["G"] == G]):
            g = y[p]
            Pg = lambda m: s["P_ex"][m][:, groups[g]].sum(1)
            whole = (s["pic"] == p) & (s["kind"] == "whole")
            if not whole.any():                                  # 2x2 jobs measured the whole picture themselves
                whole = (s["pic"] == p) & (s["G"] == 0)
            P0 = Pg(whole)[0]
            s0 = P0 * (1 - P0) / N
            for blk in range(G * G):
                sel = lambda k: (s["pic"] == p) & (s["G"] == G) & (s["block"] == blk) & (s["kind"] == k)
                w = s["norm"][sel("in")][0]
                Pin, Pfl = Pg(sel("in"))[0], Pg(sel("flip"))[0]
                sin, sfl = (q * (1 - q) / N for q in (Pin, Pfl))
                if sel("out").any():                             # the 4x4 jobs sent only the sign flip's circuits
                    Pout = Pg(sel("out"))[0]
                    var["standard"].append((s0 + w ** 2 * sin + (1 - w) ** 2 * Pout * (1 - Pout) / N) / 4)
                var["sign flip"].append(w ** 2 * sin + (s0 + sfl) / 16)
                E = w * (Pin - b[g]) + 0.25 * ((P0 - b[g]) - (Pfl - b[g]))   # the block's exact evidence (Eq. flip)
                q = lambda P: P * (1 - b[g]) ** 2 + (1 - P) * b[g] ** 2
                var["Hadamard test"].append(w * ((q(Pin) + q(P0)) / 2 - E ** 2 / w) / N)
        res = {}
        var = {k: v for k, v in var.items() if v}
        for k, v in var.items():
            v = np.array(v)
            res[k] = dict(sd_median=float(np.median(np.sqrt(v))), mae=float(np.mean(np.sqrt(2 * v / np.pi))))
        for k in [k for k in ("standard", "Hadamard test") if k in var]:
            res[k]["shots_ratio_vs_sign_flip"] = float(
                (G * G if k == "Hadamard test" else 1 + 2 * G * G) * np.mean(var[k])
                / ((1 + 2 * G * G) * np.mean(var["sign flip"])))
        out[f"{G}x{G}"] = dict(blocks=len(var["sign flip"]), **res)
        print(f"{G}x{G} ({len(var['sign flip'])} blocks, {N} shots a circuit): "
              + "; ".join(f"{k}: sd {r['sd_median']:.4f}, MAE {r['mae']:.4f}" for k, r in res.items())
              + "; shots for the same error, relative to the sign flip: "
              + ", ".join(f"{k} {r['shots_ratio_vs_sign_flip']:.2f}" for k, r in res.items() if "shots_ratio_vs_sign_flip" in r))
    (HERE / "results" / "shot_variance.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
