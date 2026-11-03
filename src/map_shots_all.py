#!/usr/bin/env python3
"""The 2x2 map measurement on every test picture, not only the ten chosen ones (reader's review 2026-11-03): the
hardware maps came from ten pictures picked among the chip's correct, confident ones; this asks how well the same
measurement (Eq. polar, 1 + 2 G^2 circuits, 4,000 shots each) recovers the exact map on all 300 test pictures,
correct or wrong on the chip, confident or not. Exact block states (no fitted loading, no gate noise): shot noise
only, so it isolates what the budget allows from what the chip adds (map_errors.py has the chip's own errors).

Per picture, evidence for the true class: correlation of the four measured blocks with the exact ones, sign
agreement, strongest block found, and the sum rule |sum - (P - b)|. Pooled over pictures as in Table hwmaps.
Grouped by the chip's answer (results/<run>/chip_ibm_kingston_r0.json) and by the size of the exact P - b.
Shots 4,000 (the hardware budget), 16,000 and 64,000; 20 draws each, seed 0.
Run:  python src/map_shots_all.py  -> numbers, results/map_shots_all.json
"""
import argparse
import glob
import json
import pathlib

import numpy as np
import torch

from ibm_heatmaps import G, maps_from, setup, terms
from ibm_testset import outcome_batch

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
RUN = HERE / "results" / "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json"


def main():
    a = argparse.Namespace(run=str(RUN), prep_depth=8)
    run, b, Y, wc, depth, n, no, block, groups = setup(a)
    f = pathlib.Path(glob.glob(str(HERE / "results" / "ibm_testset_*L8"))[0])
    pz = np.load(f / "preps.npz")
    wct = torch.tensor(pz["wc"])
    chip = json.loads((f / "chip_ibm_kingston_r0.json").read_text())
    N = len(b)
    vecs, norms = [], []
    for i in range(N):
        for j in range(G * G):
            for inside in (True, False):
                v = b[i] * torch.tensor((block == j) == inside)
                nv = float((v ** 2).sum())
                vecs.append(v / nv ** 0.5)
                norms.append(nv)
    norms = np.array(norms)
    K = 2 * G * G
    P_blk = np.concatenate([outcome_batch(torch.stack(vecs[s:s + 400]), wct, depth, no)
                            for s in range(0, len(vecs), 400)])
    P_full = outcome_batch(b, wct, depth, no)
    masks = [np.isin(np.arange(2 ** no), groups[Y[i]]) for i in range(N)]
    bgs = np.array([len(groups[Y[i]]) / 2 ** no for i in range(N)])

    def maps(Pb, Pf):
        out, whole = np.zeros((N, G * G)), np.zeros(N)
        for i in range(N):
            t = terms(Pb[K * i:K * i + K], norms[K * i:K * i + K], masks[i], bgs[i])
            whole[i] = Pf[i][masks[i]].sum() - bgs[i]
            out[i] = maps_from(np.array([whole[i]]), t[0::2][None], t[1::2][None])[0]
        return out, whole

    ex, wex = maps(P_blk, P_full)
    h = np.load(f / "heat_g2.npz")
    assert np.allclose(ex[h["sel"]], json.loads((f / "heat_g2_maps.json").read_text())["exact"], atol=1e-9), \
        "exact maps of the ten pictures differ from ibm_heatmaps.py"
    correct = np.array([chip[str(i)]["pred"] == chip[str(i)]["label"] for i in range(N)])
    q = np.array([chip[str(i)]["q"][chip[str(i)]["label"]] for i in range(N)])
    big = wex > np.median(wex)
    groups_ = {"all 300": np.ones(N, bool), "chip correct": correct, "chip wrong": ~correct,
               "chip correct, q < 0.6": correct & (q < 0.6), "exact P - b above median": big,
               "exact P - b below median": ~big}
    rng = np.random.default_rng(0)
    res = {}
    print(f"{N} pictures; chip correct {correct.sum()}; mean |exact block| {np.abs(ex).mean():.4f}, "
          f"mean exact P - b {wex.mean():.4f}")
    for shots in (4000, 16000, 64000):
        draws = []
        for _ in range(20):
            draw = lambda Q: np.array([rng.multinomial(shots, p / p.sum()) / shots for p in Q])
            draws.append(maps(draw(P_blk), draw(P_full))[0])
        for gname, m_ in groups_.items():
            e = ex[m_]
            st = []
            for m in draws:
                mm = m[m_]
                st.append([np.corrcoef(mm.ravel(), e.ravel())[0, 1], (np.sign(mm) == np.sign(e)).mean(),
                           (mm.argmax(1) == e.argmax(1)).mean(), np.abs(mm.sum(1) - wex[m_]).mean(),
                           np.abs(mm - e).mean() / np.abs(e).mean()])
            st = np.array(st).mean(0)
            res[f"{gname}, {shots} shots"] = dict(n=int(m_.sum()), corr=st[0], signs=st[1], top=st[2],
                                                  sum_resid=st[3], rel_mae=st[4],
                                                  mean_P_minus_b=float(wex[m_].mean()))
            print(f"{shots:6d} shots | {gname:26s} n={m_.sum():3d} | corr {st[0]:.3f} signs {st[1]:.2f} "
                  f"strongest {st[2]:.2f} | sum resid {st[3]:.4f} (P-b {wex[m_].mean():.4f}) relMAE {st[4]:.2f}")
    (HERE / "results" / "map_shots_all.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
