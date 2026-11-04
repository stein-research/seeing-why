#!/usr/bin/env python3
"""Evidence maps measured on a simulated lens bench (owner 2026-10-31: the paper says the maps are measurable on a
bench - show it). Only camera intensities are used, as on a real bench: for a block S of pixels (the same pixels in all
four tiles of the field), the input modulator shows the whole picture, the picture inside S only, and the picture
outside S only, and the camera's light per detector region is read each time. With W_r = 1 - b_g on class g's regions
and -b_g elsewhere, the polarization identity gives the block's evidence

    E_S = ( W.I(a) + W.I(a_S) - W.I(a_not S) ) / 2 / sum_r I_r(a)

- the same identity as on the qubit chip (ibm_demo.py heatmap), here with intensities instead of probabilities, so no
phase measurement and no light sent backwards. A g x g map needs 1 + 2 g^2 camera frames (g = 32: 2,049 frames, the
per-pixel map). The bench is the LightPipes model of optics_check_full.py (square pixels, each plane's own 4f with the
learned mask, Fresnel propagation through the plates). Compared with the exact per-pixel evidence of our model
(attention_in_box.maps with the means plane at each pixel's place, as the bench shows it), summed over the same blocks; optionally with Poisson photon noise per frame.

Bench flaws (2026-11-01, a hardware review): --drift laser power off by a random relative error per frame (no
reference detector); --leak an "off" modulator pixel still passes this fraction of its amplitude (finite extinction);
--xtalk each pixel on the block's edge loses this fraction of its amplitude to each off neighbour, which shows it
(fringing between modulator pixels). Each is compared with the flawless bench's map and with our model's.

Run:  python src/bench_maps.py results/<optical run>.json [--grid 8] [--pictures 0,4] [--photons 1e6,1e8]
      [--drift 1e-3,1e-2] [--leak 0.01,0.03] [--xtalk 0.02,0.05] [--no_figure]
      -> numbers on stdout, figures/bench_maps_<tag>.png
"""
import argparse
import json
import pathlib
import time

import LightPipes as lp
import numpy as np
import torch

import fastsim
import optnet
import pilot_fmnist as pf
from attention_in_box import maps as pixel_maps
from bssg import class_groups
from heatmap import NAMES
from optics_check import bench

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root


class Bench:
    """The trained run as a LightPipes bench: regions(state) -> light on the 128 detector regions (not normalised)."""

    def __init__(self, run, os_=2):
        self.run = run
        self.lam, self.pitch, self.s = 633e-9, 8e-6, 32
        s = self.s
        self.N, self.r, self.L = 2 * s, run["opt_res"], run["optical"]
        self.d = run["opt_z"] * self.pitch ** 2 / self.lam
        self.perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
        self.n = len(run["weights"][0])
        self.planes = 2 ** self.n // (s * s)
        w = torch.tensor(run["weights"], dtype=torch.float64)
        fr = fastsim.fourier_rows(self.n, len(run["weights"][0][0]), s, self.planes, run.get("fourier_phase", False))
        _, kw = fastsim.split_masks(w, 0, 0, -1, (torch.tensor(self.perm), s, self.planes,
                                                  run.get("fourier_phase", False), True))
        self.D = kw["fmask"][1].numpy()
        N, r, L = self.N, self.r, self.L
        self.phi = w[1 + fr:].reshape(-1)[:L * (N // r) ** 2].reshape(L, N // r, N // r).numpy()
        self.det = optnet.detector_index(N).numpy()
        self.o = os_

    def regions(self, x):
        s, N, o = self.s, self.N, self.o
        grid = np.zeros((self.planes, s * s), complex)
        grid[:, self.perm] = np.asarray(x).reshape(self.planes, s * s)
        g0 = np.zeros((N, N), complex)
        for p in range(self.planes):
            if np.any(grid[p]):
                g0[(p // 2) * s:(p // 2 + 1) * s, (p % 2) * s:(p % 2 + 1) * s] = bench(grid[p].reshape(s, s),
                                                                                     self.D[p], 4, 2)
        M = 2 * N * o
        F = lp.Begin(M * self.pitch / o, self.lam, M)
        fld = np.zeros((M, M), complex)
        q = (M - N * o) // 2
        fld[q:q + N * o, q:q + N * o] = np.kron(g0, np.ones((o, o)))
        F.field = fld
        ap = np.zeros((M, M))
        ap[q:q + N * o, q:q + N * o] = 1
        for l in range(self.L):
            F = lp.Forvard(F, self.d)
            F.field = F.field * ap * np.exp(1j * np.pad(np.kron(self.phi[l], np.ones((self.r * o, self.r * o))), q))
        F = lp.Forvard(F, self.d)
        I = np.abs(F.field[q:q + N * o, q:q + N * o]) ** 2
        return np.bincount(self.det.ravel(), I.reshape(N, o, N, o).sum((1, 3)).ravel(), 128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--grid", type=int, default=8, help="blocks per side (32 = every pixel)")
    ap.add_argument("--pictures", default="0,4", help="test picture indices")
    ap.add_argument("--photons", default="1e6,1e8", help="photons per camera frame for the noisy versions")
    ap.add_argument("--drift", default="", help="relative laser power error per frame (std), comma-separated")
    ap.add_argument("--leak", default="", help="amplitude an off modulator pixel still passes, comma-separated")
    ap.add_argument("--xtalk", default="", help="amplitude an edge pixel lends each off neighbour, comma-separated")
    ap.add_argument("--no_figure", action="store_true")
    ap.add_argument("--json", default="", help="also write each picture's numbers to this file (results/...)")
    a = ap.parse_args()
    run = json.loads(open(a.run).read())
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")
    pf.N_QUBITS = len(run["weights"][0])
    B = Bench(run)
    s, g, n = B.s, a.grid, B.n
    names = NAMES[run["dataset"]]
    groups = class_groups(run["mode"], 128, len(names))
    d = np.load(HERE / "data" / f"{run['dataset']}.npz")
    pics = [int(v) for v in a.pictures.split(",")]
    X, Y = d["x_val"][pics], d["y_val"][pics]
    ev = pixel_maps(run, X, Y, signed=True, means=True)[2]   # exact per-pixel evidence (our model), the means plane's
    # amplitude at each pixel's place included, as the bench block shows it (until 2026-11-04 it was left out, so the
    # model's map summed to P - b minus the means plane's part and the comparison was not like for like)
    # the encoded pictures, wire order, as the model sees them
    x = X.astype(np.float64).transpose(0, 3, 1, 2).reshape(len(X), -1, s * s)[:, :, B.perm].reshape(len(X), -1)
    if run.get("center_image"):
        x = pf.centre_planes(x, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.pad(x, ((0, 0), (0, 2 ** n - x.shape[1])))
    pixel = np.asarray(B.perm)[np.arange(2 ** n) % (s * s)]
    block = (pixel // s // (s // g)) * g + pixel % s // (s // g)
    photons = [float(v) for v in a.photons.split(",")] if a.photons else []
    rng = np.random.default_rng(0)
    rows, summary = [], []
    drifts = [float(v) for v in a.drift.split(",")] if a.drift else []
    leaks = [float(v) for v in a.leak.split(",")] if a.leak else []
    xtalks = [float(v) for v in a.xtalk.split(",")] if a.xtalk else []
    py, px = np.divmod(np.arange(s * s), s)

    def shown(on, leak=0.0, xtalk=0.0):
        """The amplitude factor the modulator really shows per picture pixel (32 x 32) when `on` pixels are meant on."""
        t = on.astype(float).reshape(s, s)
        if xtalk:
            m = t.copy()
            P, inn = np.pad(m, 1), np.pad(np.ones_like(m), 1)
            near = lambda A: A[:-2, 1:-1] + A[2:, 1:-1] + A[1:-1, :-2] + A[1:-1, 2:]   # 4 neighbours, none past the edge
            on_nb = near(P)
            off_nb = near(inn) - on_nb
            t = m + xtalk * on_nb * (1 - m) - xtalk * off_nb * m
        t = t + leak * (1 - on.reshape(s, s))
        return np.clip(t, 0, 1).ravel()

    def frames_for(k, leak=0.0, xtalk=0.0):
        I_full = B.regions(x[k])
        fr = [(I_full, None)]
        for j in range(g * g):
            inside_pix = (py // (s // g)) * g + px // (s // g) == j          # block j on the 32 x 32 picture
            for on in (inside_pix, ~inside_pix):
                f = shown(on, leak, xtalk)[pixel]                            # every plane shows the same pixels
                fr.append((B.regions(x[k] * f), None))
        return fr

    for k, i in enumerate(pics):
        G = np.isin(np.arange(128), groups[Y[k]])
        W = np.where(G, 1 - G.mean(), -G.mean())
        t0 = time.time()
        frames = frames_for(k)
        I_full = frames[0][0]
        secs = time.time() - t0
        ex = np.zeros((g, g))
        np.add.at(ex, (np.arange(s)[:, None] // (s // g), np.arange(s)[None, :] // (s // g)), ev[k])

        def measured(noise=None, drift=0.0, frames=frames):
            Is = []
            for I, _ in frames:
                if noise:                                                   # photons scale with the frame's light
                    lam = I / I_full.sum() * noise
                    I = rng.poisson(lam) * I_full.sum() / noise
                if drift:                                                   # the laser's power, frame by frame
                    I = I * (1 + drift * rng.standard_normal())
                Is.append(I)
            k_ = Is[0].sum()
            t_full = W @ Is[0]
            m = np.array([(t_full + W @ Is[1 + 2 * j] - W @ Is[2 + 2 * j]) / 2 / k_ for j in range(g * g)])
            return m.reshape(g, g)

        out = {"bench": measured()}
        for ph in photons:
            out[f"{ph:.0e} photons/frame"] = measured(ph)
        for dr in drifts:
            out[f"drift {dr:g}"] = measured(drift=dr)
        for lk in leaks:
            out[f"leak {lk:g}"] = measured(frames=frames_for(k, leak=lk))
        for xt in xtalks:
            out[f"xtalk {xt:g}"] = measured(frames=frames_for(k, xtalk=xt))
        own = W @ I_full / I_full.sum()                                     # the bench's own P - b
        line = (f"picture {i} ({names[Y[k]]}), {g}x{g}, {len(frames)} frames ({secs:.0f} s): exact sum {ex.sum():+.4f}, "
                f"bench's own P - b {own:+.4f} (map sum - it: {out['bench'].sum() - own:.1e})")
        for name, m in out.items():
            c = np.corrcoef(m.ravel(), ex.ravel())[0, 1]
            cb = np.corrcoef(m.ravel(), out["bench"].ravel())[0, 1]
            line += f" | {name}: corr {c:.4f} (with the flawless bench {cb:.4f}), max |diff| {np.abs(m - ex).max():.1e}, sum {m.sum():+.4f}"
        print(line, flush=True)
        rows.append((i, X[k], ex, out))
        summary.append(dict(picture=i, label=names[Y[k]], exact_sum=float(ex.sum()), bench_own=float(own),
                            **{name: dict(corr=float(np.corrcoef(m.ravel(), ex.ravel())[0, 1]),
                                          corr_flawless=float(np.corrcoef(m.ravel(), out["bench"].ravel())[0, 1]),
                                          sum=float(m.sum())) for name, m in out.items()}))
        if a.json:
            (HERE / a.json).write_text(json.dumps(summary, indent=1))
    if a.no_figure:
        return
    np.savez(HERE / "results" / f"bench_maps_g{g}.npz", pictures=np.array(pics), labels=Y,
             exact=np.array([r[2] for r in rows]),
             **{k_.replace(" ", "_").replace("/", "per").replace("+", ""): np.array([r[3][k_] for r in rows])
                for k_ in rows[0][3]})                          # for paper/make_figures.py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    kinds = list(rows[0][3])
    fig, axs = plt.subplots(len(rows), 2 + len(kinds), figsize=(2.3 * (2 + len(kinds)), 2.4 * len(rows)), squeeze=False)
    for r_, (i, pic, ex, out) in enumerate(rows):
        lim = np.abs(ex).max()
        axs[r_, 0].imshow(pic)
        axs[r_, 0].set_title(f"{names[Y[r_]]} (picture {i})", fontsize=8)
        for c_, (m, title) in enumerate([(ex, "exact (our model)")] + [(out[k_], k_) for k_ in kinds]):
            A = axs[r_, 1 + c_]
            A.imshow(m, cmap="RdBu_r", vmin=-lim, vmax=lim)
            A.set_title(title + f", {g}x{g}", fontsize=7)
        for A in axs[r_]:
            A.set_xticks([])
            A.set_yticks([])
    fig.suptitle("evidence for the true class measured on a simulated lens bench from camera intensities only "
                 "(polarization identity)", fontsize=8)
    fig.tight_layout()
    tag = pathlib.Path(a.run).stem[:50]
    outp = HERE / "figures" / f"bench_maps_g{g}_{tag}.png"
    fig.savefig(outp, dpi=300)
    fig.savefig(outp.with_suffix(".pdf"), dpi=600)
    print(outp)


if __name__ == "__main__":
    main()
