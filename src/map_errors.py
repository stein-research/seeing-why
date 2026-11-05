#!/usr/bin/env python3
"""Error metrics of the 2x2 hardware maps, beyond correlation (reader's review 2026-11-03): correlation alone says
nothing of size or of the sum rule. For each kind of map (fitted loading, noisy simulation, chip runs, and exact
distributions with only shot noise drawn), against two targets - the exact maps of the ideal masked state, and the
maps of the fitted loading states (what the chip was asked to prepare) - report:
  MAE and RMSE of the block evidence, MAE relative to the mean |exact block|, the slope of a fit through zero (size);
  the sum rule: |sum of the four blocks - P_g - b_g|, against the exact P - b and against the run's own whole-picture
  term (the identity's own closure on that run's data).
Bootstrap over the ten pictures (10,000 resamples) for the chip runs. Shot-noise-only maps draw 4,000 shots per
circuit from the exact and from the fitted-loading distributions (seed 0, 200 draws), the floor a perfect chip would
show at this budget.
Run:  python src/map_errors.py  -> numbers, results/map_errors.json
"""
import glob
import json
import pathlib

import argparse

import numpy as np

from ibm_heatmaps import maps_from, setup, terms

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
SHOTS = 4000
K = 8                                   # 2 G^2 block states per picture, G = 2 (in, out per block)


def metrics(m, ref, whole_ref, whole_own):
    d = m - ref
    return dict(mae=float(np.abs(d).mean()), rmse=float(np.sqrt((d ** 2).mean())),
                rel_mae=float(np.abs(d).mean() / np.abs(ref).mean()),
                slope=float((m * ref).sum() / (ref * ref).sum()),
                sum_vs_ref=float(np.abs(m.sum(1) - whole_ref).mean()),
                sum_vs_own=float(np.abs(m.sum(1) - whole_own).mean()))


def main():
    f = pathlib.Path(glob.glob(str(HERE / "results" / "ibm_testset_*cimp0.5__L8"))[0])
    run_path = HERE / "results" / "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json"
    _, _, Y, _, _, _, no, _, groups = setup(argparse.Namespace(run=str(run_path)))
    h = np.load(f / "heat_g2.npz")
    sel, norms = h["sel"], h["norms"]

    def maps(P, Pf):
        rows, whole = [], []
        for k, i in enumerate(sel):
            Gm = np.isin(np.arange(2 ** no), groups[Y[i]])
            bg = len(groups[Y[i]]) / 2 ** no
            t = terms(P[K * k:K * k + K], norms[K * k:K * k + K], Gm, bg)
            tf = Pf[k][Gm].sum() - bg
            rows.append(maps_from(np.array([tf]), t[0::2][None], t[1::2][None])[0])
            whole.append(tf)
        return np.array(rows), np.array(whole)

    ex, wex = maps(h["P_ex"], h["P_full_ex"])
    pr, wpr = maps(h["P_pr"], h["P_full_pr"])
    saved = json.load(open(f / "heat_g2_maps.json"))
    assert np.allclose(ex, saved["exact"]) and np.allclose(pr, saved["prep"]), "groups or order differ from report()"
    kinds = {"prep": (pr, wpr)}
    for p in sorted(f.glob("heat_g2_noisy_*.npz")) + sorted(f.glob("heat_g2_chip_*.npz")):
        z = np.load(p)
        name = p.stem.replace("heat_g2_", "")
        kinds[name + (" (no shot noise)" if int(z["shots"]) == 0 else "")] = maps(z["P"], z["P_full"])
    out = {}
    print(f"mean |exact block| {np.abs(ex).mean():.4f}, mean exact P - b {wex.mean():.4f}")
    print(f"{'kind':38s} {'target':6s}  MAE    RMSE   relMAE slope | sum-ref sum-own")
    for name, (m, w) in kinds.items():
        for tgt, ref, wref in (("exact", ex, wex), ("prep", pr, wpr)):
            r = metrics(m, ref, wref, w)
            out[f"{name} vs {tgt}"] = r
            print(f"{name:38s} {tgt:6s} {r['mae']:.4f} {r['rmse']:.4f} {r['rel_mae']:.2f}  {r['slope']:.2f} |"
                  f" {r['sum_vs_ref']:.4f}  {r['sum_vs_own']:.4f}")
    rng = np.random.default_rng(0)
    draw = lambda Q: np.array([rng.multinomial(SHOTS, q / q.sum()) / SHOTS for q in Q])
    for src, P, Pf, ref, wref in (("exact", h["P_ex"], h["P_full_ex"], ex, wex),
                                  ("prep", h["P_pr"], h["P_full_pr"], pr, wpr)):
        rs = [metrics(*(lambda mw: (mw[0], ref, wref, mw[1]))(maps(draw(P), draw(Pf)))) for _ in range(200)]
        r = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
        out[f"shot noise only ({src} distributions, {SHOTS} shots) vs {src}"] = r
        print(f"{'shot noise only, ' + src:38s} {src:6s} {r['mae']:.4f} {r['rmse']:.4f} {r['rel_mae']:.2f}  "
              f"{r['slope']:.2f} | {r['sum_vs_ref']:.4f}  {r['sum_vs_own']:.4f}")
    for name in [k for k in kinds if k.startswith("chip")]:
        m, w = kinds[name]
        boot = []
        for _ in range(10000):
            i = rng.integers(0, len(sel), len(sel))
            boot.append([np.abs(m[i] - ex[i]).mean() / np.abs(ex[i]).mean(), (m[i] * ex[i]).sum() / (ex[i] ** 2).sum(),
                         np.abs(m[i].sum(1) - wex[i]).mean()])
        lo, hi = np.percentile(boot, 2.5, 0), np.percentile(boot, 97.5, 0)
        out[f"{name} vs exact"]["ci"] = dict(rel_mae=[lo[0], hi[0]], slope=[lo[1], hi[1]], sum_vs_ref=[lo[2], hi[2]])
        print(f"{name}: 95% CI relMAE [{lo[0]:.2f}, {hi[0]:.2f}], slope [{lo[1]:.2f}, {hi[1]:.2f}], "
              f"sum-ref [{lo[2]:.4f}, {hi[2]:.4f}]")
    (HERE / "results" / "map_errors.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
