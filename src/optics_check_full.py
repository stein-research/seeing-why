#!/usr/bin/env python3
"""End-to-end check of an all-optical run (`--optical`) against LightPipes (owner 2026-10-31: validate the phase-plate
bench with the outside simulator, not only its propagation step).

Bench simulated in LightPipes with square pixels (`--os` samples per picture pixel): each colour plane through its own
4f system with the learned lens-mode mask (a 2 x 2 lenslet array on a real bench; `optics_check.bench`, pixel-averaged
output), the planes tiled 2 x 2, then free-space propagation (Forvard over d = z dx^2 / lambda) and the phase plates
(each plate pixel r x r picture pixels), to the camera; the light summed over each of the 128 detector regions, BHR on
top. Compared with optnet / fastsim, which sample each pixel at its centre: detector-light overlap and predictions.
--continuous hands the 4f output to the plates as it is (band-limited, smooth), not re-made into square pixels.
--propagator fresnel uses LightPipes' convolution propagator instead of the spectral Forvard between the plates.
--thick N makes each plate a surface-relief glass plate (n = 1.457, depth up to lambda/(n-1)) crossed in N layers by
LightPipes' finite-difference Steps, instead of a thin phase mask (2026-11-01: the owner, LightPipes as the reference).

Run:  python src/optics_check_full.py results/<optical run>.json [--pictures 10] [--os 2]
"""
import argparse
import json

import LightPipes as lp
import numpy as np
import torch

import fastsim
import optnet
import pilot_fmnist as pf
from heatmap import NAMES
from ibm_demo import class_probs, picture_state
from optics_check import bench


def steps_phase(lam=633e-9, n=1.457):
    """The phase LightPipes' Steps (2.1.5) gives a uniform slab, as a share of k (n - 1) t: measured, not assumed
    (it gives half). Glass is then given the index 1 + (n - 1) / share, so its phase is right."""
    M, t = 32, lam / (n - 1) / 4
    F = lp.Steps(lp.Begin(M * 4e-6, lam, M), t, 1, np.full((M, M), n, complex))
    return float(np.angle(F.field[M // 2, M // 2]) / (2 * np.pi / lam * (n - 1) * t))


def glass(F, ph, layers, n, lam, sign=1, share=1.0):
    """A surface-relief plate whose glass height gives the phase ph (mod 2 pi), crossed layer by layer by Steps
    (which takes the index array transposed)."""
    h = np.mod(sign * ph, 2 * np.pi) / (2 * np.pi) * lam / (n - 1)
    dz = lam / (n - 1) / layers
    n_eff = 1 + (n - 1) / share
    for j in range(layers):
        refr = np.where(h > (j + 0.5) * dz, n_eff, 1.0).astype(complex)
        F = lp.Steps(F, dz, 1, refr.T)
    return F


def glass_sign(lam=633e-9, pitch=8e-6, n=1.457, layers=8, share=1.0):
    """Which sign of the glass height gives exp(+i ph) as the thin plate does: a random plate of 16 um pixels, both
    ways; returns (sign, overlap with the thin plate)."""
    M = 128
    rng = np.random.default_rng(0)
    ph = np.zeros((M, M))
    ph[32:96, 32:96] = np.kron(rng.uniform(0, 2 * np.pi, (16, 16)), np.ones((4, 4)))
    best = None
    for sign in (1, -1):
        t = glass(lp.Begin(M * pitch / 2, lam, M), ph, layers, n, lam, sign, share).field[32:96, 32:96]
        ov = abs(np.vdot(np.exp(1j * ph[32:96, 32:96]), t)) / np.linalg.norm(t) / 64
        if best is None or ov > best[1]:
            best = (sign, ov)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pictures", type=int, default=10)
    ap.add_argument("--os", type=int, default=2)
    ap.add_argument("--continuous", action="store_true", help="the 4f output reaches the plates unaveraged")
    ap.add_argument("--propagator", default="forvard", choices=["forvard", "fresnel"])
    ap.add_argument("--thick", type=int, default=0, help="layers per glass plate (0: thin phase masks)")
    ap.add_argument("--n_glass", type=float, default=1.457)
    a = ap.parse_args()
    run = json.loads(open(a.run).read())
    names = NAMES[run["dataset"]]
    lam, pitch, s = 633e-9, 8e-6, 32
    N, r, z, L = 2 * s, run["opt_res"], run["opt_z"], run["optical"]
    d = z * pitch * pitch / lam
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")       # before any permutation is taken
    cam_fn, _, det, os_o = optnet.from_run(run, pf.morton_perm() if pf.permuted() else np.arange(s * s))
    agree, ovl = 0, []
    sign, share = 1, 1.0
    if a.thick:
        share = steps_phase(lam, a.n_glass)
        sign, ov = glass_sign(lam, pitch, a.n_glass, a.thick, share)
        print(f"Steps gives {share:.3f} of k (n-1) t, corrected; glass height sign {sign:+d}: one thick plate against "
              f"its thin phase, overlap {ov:.4f}")
    for c in range(min(a.pictures, len(names))):
        x, perm, idx = picture_state(run, c)
        n = len(x).bit_length() - 1
        planes = 2 ** n // (s * s)
        w = torch.tensor(run["weights"], dtype=torch.float64)
        fr = fastsim.fourier_rows(n, len(run["weights"][0][0]), s, planes, run.get("fourier_phase", False))
        _, kw = fastsim.split_masks(w, 0, 0, -1, (torch.tensor(perm), s, planes, run.get("fourier_phase", False), True))
        D = kw["fmask"][1].numpy()
        phi = w[1 + fr:].reshape(-1)[:L * (N // r) ** 2].reshape(L, N // r, N // r).numpy()
        grid = np.zeros((planes, s * s), complex)
        grid[:, perm] = x.numpy().reshape(planes, s * s)
        o = a.os
        so = o if a.continuous else 1                                       # samples per pixel of the 4f output
        tiles = [bench(grid[p].reshape(s, s), D[p], o if a.continuous else 4, 2, average=not a.continuous)
                 for p in range(planes)]                                    # each plane's own 4f (lenslet)
        g0 = np.zeros((N * so, N * so), complex)
        for p in range(planes):
            g0[(p // 2) * s * so:(p // 2 + 1) * s * so, (p % 2) * s * so:(p % 2 + 1) * s * so] = tiles[p]
        M = 2 * N * o                                                       # padded 2x, as optnet
        F = lp.Begin(M * pitch / o, lam, M)
        fld = np.zeros((M, M), complex)
        q = (M - N * o) // 2
        fld[q:q + N * o, q:q + N * o] = g0 if a.continuous else np.kron(g0, np.ones((o, o)))
        F.field = fld
        ap_mask = np.zeros((M, M))
        ap_mask[q:q + N * o, q:q + N * o] = 1                               # the aperture: light outside is lost
        go = lp.Fresnel if a.propagator == "fresnel" else lp.Forvard
        for l in range(L):
            F = go(F, d)
            ph = np.pad(np.kron(phi[l], np.ones((r * o, r * o))), q)
            if a.thick:                                                     # glass of height h: phase k (n - 1) h
                F = glass(F, ph, a.thick, a.n_glass, lam, sign, share)
                F.field = F.field * ap_mask
            else:
                F.field = F.field * ap_mask * np.exp(1j * ph)
        F = go(F, d)
        I = np.abs(F.field[q:q + N * o, q:q + N * o]) ** 2
        Ipix = I.reshape(N, o, N, o).sum((1, 3))
        p_lp = np.bincount(det.numpy().ravel(), Ipix.ravel(), 128)
        p_ours = optnet.outcome_probs(cam_fn(x[None].to(torch.complex128)), det, os=os_o)[0].numpy()
        ovl.append(np.dot(p_lp, p_ours) / np.linalg.norm(p_lp) / np.linalg.norm(p_ours))
        q_lp, q_ours = class_probs(run, p_lp / p_lp.sum()), class_probs(run, p_ours / p_ours.sum())
        agree += int(q_lp.argmax() == q_ours.argmax())
        print(f"{names[c]:11s} detector-light overlap {ovl[-1]:.4f}   ours -> {names[q_ours.argmax()]} {q_ours.max():.2f}"
              f"   LightPipes -> {names[q_lp.argmax()]} {q_lp.max():.2f}")
    print(f"mean overlap {np.mean(ovl):.4f} (min {np.min(ovl):.4f}); same prediction {agree}/{len(ovl)}; "
          f"d = {d * 1e3:.2f} mm, {o} samples per pixel{', continuous 4f output' if a.continuous else ''}"
          f"{', LightPipes Fresnel' if a.propagator == 'fresnel' else ''}"
          f"{f', glass plates in {a.thick} layers' if a.thick else ''}")


if __name__ == "__main__":
    main()
