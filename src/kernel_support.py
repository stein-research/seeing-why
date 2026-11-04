#!/usr/bin/env python3
"""Spatial support of the qubit model's learned Fourier-plane mask (referee B, second round, Q2). On qubits (D') the
mask is a circular convolution of each plane with a kernel k = ifft2(d) (fastsim.fourier_filter); the 2x2 chip maps are
maps of the filtered state, so how far the kernel reaches decides how much a filtered-state quadrant still stands for
the same quadrant of the picture. D', seeds 0-2:
  - the kernel's energy within a circular distance r of its centre (r = 0, 1, 2, 4, 8 pixels), per colour plane, the
    median over planes and seeds;
  - over the 300 test pictures (encoded as the model encodes them), the share of a filtered-state quadrant's energy
    that comes from the same picture quadrant: ||P_Q K a_Q||^2 / sum_S ||P_Q K a_S||^2 (the parts' own energies; the
    cross terms between them are left out, so the shares add up to one), median over quadrants and pictures.
Run:  python src/kernel_support.py  -> numbers, results/kernel_support.json
"""
import json
import pathlib

import numpy as np
import torch

import fastsim
import pilot_fmnist as pf

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
RUN = "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s{}.json"
RADII = [0, 1, 2, 4, 8]


def main():
    out = {}
    s = 32
    yy, xx = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    dist = np.hypot(np.minimum(yy, s - yy), np.minimum(xx, s - xx))       # circular distance from the kernel's centre
    quad = (yy >= s // 2) * 2 + (xx >= s // 2)
    for seed in range(3):
        run = json.loads((HERE / "results" / RUN.format(seed)).read_text())
        pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
        perm = pf.morton_perm()
        w = torch.tensor(run["weights"], dtype=torch.float32)
        _, kw = fastsim.split_masks(w, run["depth"], 0, -1, (torch.tensor(perm), s, 4, run.get("fourier_phase", False)))
        d = kw["fmask"][1].reshape(4, s, s).to(torch.complex128)
        k = torch.fft.ifft2(d, norm="ortho").numpy() / s                     # the kernel of g -> ifft2(fft2(g) d)
        e = np.abs(k) ** 2
        within = {r: (e * (dist <= r)).sum((1, 2)) / e.sum((1, 2)) for r in RADII}
        # the test pictures as the model encodes them, on the pixel grid: (n, 4 planes, s, s)
        X = np.load(HERE / "data" / f"{run['dataset']}.npz")["x_val"].astype(float)
        v = X.transpose(0, 3, 1, 2).reshape(len(X), 3, -1)[:, :, perm].reshape(len(X), -1)
        enc = pf.centre_planes(v, 0, run["mean_plane"])
        g = np.empty((len(X), 4, s * s))
        g[:, :, perm] = enc.reshape(len(X), 4, s * s)
        g = g.reshape(len(X), 4, s, s)
        part = np.stack([np.where(quad == q, g, 0) for q in range(4)], 1)  # (n, picture quadrant, planes, s, s)
        F = np.fft.ifft2(np.fft.fft2(part, norm="ortho") * d.numpy(), norm="ortho")
        en = np.stack([(np.abs(F) ** 2 * (quad == q)).sum((2, 3, 4)) for q in range(4)], 2)   # (n, from S, into Q)
        own = np.stack([en[:, q, q] / en[:, :, q].sum(1) for q in range(4)], 1)
        out[f"seed {seed}"] = dict(within={str(r): within[r].tolist() for r in RADII},
                                   same_quadrant_median=float(np.median(own)),
                                   same_quadrant_q=np.percentile(own, [25, 75]).tolist())
        print(f"D' seed {seed}: kernel energy within r = " + ", ".join(f"{r}: {100 * np.median(within[r]):.0f} %" for r in RADII)
              + f" (median over planes); a filtered quadrant's energy from the same picture quadrant: median "
              f"{100 * np.median(own):.1f} % (quartiles {100 * np.percentile(own, 25):.1f}-{100 * np.percentile(own, 75):.1f})")
    (HERE / "results" / "kernel_support.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
