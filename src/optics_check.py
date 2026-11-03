#!/usr/bin/env python3
"""Check fastsim's Fourier-plane mask against a physical 4f bench simulated with LightPipes (BSD-3, an established
scalar-optics package; owner 2026-10-31: an outside simulator, in case ours is wrong).

fastsim models the mask as an orthonormal 2-D DFT of each 32x32 colour plane, times the learned multiplier d(k), and
back: a circular convolution on the pixel grid. The bench: coherent light (633 nm) through an SLM of 32x32 square
pixels (8 um), lens - Fourier plane - lens (2f + 2f, free-space propagation by LightPipes' Forvard, thin lenses by
Lens), a second SLM in the Fourier plane carrying d at the position of each spatial frequency and blocking light
outside the band |k| < 16 (one period of the DFT), and a camera / next SLM that averages the field over each image
pixel. The 4f image is inverted; it is flipped back. Each SLM pixel is sampled `--os` times per side and the grid is
padded `--pad` times, so propagation does not wrap. The focal length is chosen so that the Fourier plane falls on the
same grid (f = L^2 / (N lambda)); every number in it is a grid choice, not a claim about a real lens.

Differences to expect: the bench convolves linearly (no wrap at the picture's edge) where the DFT wraps; square SLM
pixels weight the band by sinc(k / 32) per axis, which the mask divides out by default (a mask designed for the pixel
shape; `--no_sinc` keeps it); the Fourier SLM covers |k| <= 16, both band edges (the plane-4 reference tile sits at
k = +-16, a band cut at |k| < 16 blocks half of it). Checked first: without any band cut the bench images the
picture to 2e-6 (LightPipes and the geometry are right). Reported per picture: correlation and relative error of the bench's output
against fastsim's, and the class probabilities of the trained circuit fed with each.

Run:  python src/optics_check.py results/<fourier-mask run>.json [--pictures 10] [--os 4] [--pad 2]
"""
import argparse
import json
import pathlib

import LightPipes as lp
import numpy as np
import torch

import fastsim
import pilot_fmnist as pf
from heatmap import NAMES
from ibm_demo import class_probs, picture_state


def bench(plane, mask, os_, pad, lam=633e-9, pitch=8e-6, sinc=True, average=True):
    """One 32x32 plane (complex) through the 4f bench with the learned multiplier mask (32x32, np.fft frequency order)
    -> the field averaged over each image pixel, 32x32, flipped back (average=False: the continuous field, os_
    samples per pixel side)."""
    s = plane.shape[0]
    N = s * os_ * pad
    dx = pitch / os_
    L = N * dx
    f = L * L / (N * lam)                                   # Fourier plane on the same grid (du = dx)
    F = lp.Begin(L, lam, N)
    field = np.zeros((N, N), complex)
    o = (N - s * os_) // 2
    field[o:o + s * os_, o:o + s * os_] = np.kron(plane, np.ones((os_, os_)))
    F.field = field
    F = lp.Forvard(F, f)
    F = lp.Lens(F, f)
    F = lp.Forvard(F, f)                                    # Fourier plane: u = lambda f nu
    u = (np.arange(N) - N // 2) * dx
    k = u * (s * pitch) / (lam * f)                         # cycles per picture width at each Fourier-plane sample
    K = (np.sign(k) * np.floor(np.abs(k) + 0.5)).astype(int)   # halves away from 0, as fastsim._lens_mask
    inband = np.abs(k) <= s / 2 + 1e-9                      # both edges: the plane-4 tile sits at k = +-16
    m = mask[np.mod(K, s)[:, None], np.mod(K, s)[None, :]]
    if sinc:                                                # undo the square-pixel envelope inside the band
        env = np.sinc(k / s)
        m = m / np.maximum(env[:, None] * env[None, :], 0.2)
    m = m * (inband[:, None] & inband[None, :])
    F.field = F.field * m
    F = lp.Forvard(F, f)
    F = lp.Lens(F, f)
    F = lp.Forvard(F, f)
    out = F.field[::-1, ::-1]                               # the 4f image is inverted
    out = np.roll(out, (1, 1), (0, 1)) if N % 2 == 0 else out   # inversion about the grid centre (even N)
    out = out[o:o + s * os_, o:o + s * os_]
    return out.reshape(s, os_, s, os_).mean((1, 3)) if average else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--pictures", type=int, default=10, help="first test picture of each of the first N classes")
    ap.add_argument("--os", type=int, default=4)
    ap.add_argument("--pad", type=int, default=2)
    ap.add_argument("--no_sinc", action="store_true", help="do not correct the mask for the square-pixel envelope")
    a = ap.parse_args()
    run = json.loads(pathlib.Path(a.run).read_text())
    names = NAMES[run["dataset"]]
    w = torch.tensor(run["weights"], dtype=torch.float64)
    corr, rel, agree = [], [], 0
    ident = []
    for c in range(min(a.pictures, len(names))):
        x, perm, idx = picture_state(run, c)
        n, s, depth = len(x).bit_length() - 1, pf.side(), run["depth"]
        planes = 2 ** n // (s * s)
        wc, kw = fastsim.split_masks(w, depth, 0, -1, (torch.tensor(perm), s, planes, run.get("fourier_phase", False),
                                                       run.get("fourier_linear", False)))
        D = kw["fmask"][1].numpy()                                            # (planes, s, s), np.fft order
        ours = fastsim.fourier_filter(x[None].to(torch.complex128), kw["fmask"][1], kw["fmask"][0], kw["fmask"][2])[0].numpy()
        grid = np.zeros((planes, s * s), complex)                             # wire order -> pixel grids
        grid[:, perm] = x.numpy().reshape(planes, s * s)
        phys = np.zeros_like(grid)
        for p in range(planes):
            phys[p] = bench(grid[p].reshape(s, s), D[p], a.os, a.pad, sinc=not a.no_sinc).ravel()
            if c == 0:                                                        # mask = 1: the bench should image
                im = grid[p].reshape(s, s)
                bi = bench(im, np.ones((s, s)), a.os, a.pad, sinc=not a.no_sinc)
                gi = np.vdot(im, bi) / max(np.vdot(im, im).real, 1e-30)       # the bench's overall gain / phase
                ident.append(np.linalg.norm(bi / gi - im) / (np.linalg.norm(im) + 1e-12))
        phys_w = phys[:, perm].reshape(-1)                                    # pixel grids -> wire order
        g = np.vdot(ours, phys_w) / np.vdot(ours, ours)                       # the bench's overall gain / phase
        corr.append(abs(np.vdot(ours, phys_w)) / (np.linalg.norm(ours) * np.linalg.norm(phys_w)))
        rel.append(np.linalg.norm(phys_w / g - ours) / np.linalg.norm(ours))
        P = []
        for v in (ours, phys_w):
            st = torch.tensor(v)[None]
            pr = fastsim.probs(fastsim.run(st / st.norm(), wc, depth, torch.complex128), run["n_out"])[0].numpy()
            P.append(class_probs(run, pr / pr.sum()))
        agree += int(P[0].argmax() == P[1].argmax())
        print(f"{names[c]:11s} overlap {corr[-1]:.4f}  rel. error {rel[-1]:.3f}  "
              f"fastsim -> {names[P[0].argmax()]} {P[0].max():.2f}   bench -> {names[P[1].argmax()]} {P[1].max():.2f}")
    print(f"identity bench (mask 1), relative error per plane: " + " ".join(f"{v:.3f}" for v in ident))
    print(f"overlap |<ours|bench>| / norms: mean {np.mean(corr):.4f}, min {np.min(corr):.4f}; "
          f"relative error mean {np.mean(rel):.3f}; same prediction {agree}/{len(corr)}"
          f"{'' if a.no_sinc else ' (mask divided by the pixel sinc)'}")


if __name__ == "__main__":
    main()
