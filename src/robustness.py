#!/usr/bin/env python3
"""How exact must a bench be? Test accuracy of a trained all-optical run (`--optical`) under bench flaws, one at a time
(owner 2026-10-31: the work should help whoever builds it). Nothing is retrained; each flaw is applied to the trained
model at test time, with 3 random draws where it is random.

Flaws (each swept from none to large):
  pic_bits    picture SLM: amplitude and phase each quantised to 2^b levels (amplitude on [0, max])
  mask_bits   Fourier-plane SLM: the learned mask's amplitude and phase quantised the same way
  plate_bits  phase plates: phase quantised to 2^b levels
  plate_shift each plate shifted sideways by a random vector of this length (in plate pixels; sub-pixel by the fine grid)
  z_err       every plate spacing off by a random relative error of this standard deviation
  pic_shift   the picture shifted against the system by this many picture pixels (both axes)
  phase_noise a fixed random phase error of this standard deviation (rad) on every plate pixel
  photons     the camera counts Poisson photons, this many per picture in total
Added 2026-11-01 (a hardware review: the quarter-pixel row was a bound, not a measurement, and four flaws were missing):
  plate_shift_fine  plate_shift between 0 and a quarter plate pixel
  plate_rot   each plate turned by a random angle of this size (degrees) about its centre
  mask_shift  the Fourier-plane mask off-centre by this many frequency samples, random direction
  mag_err     the 4f system's image scaled by 1 + this against the plates (magnification error)
  cam_shift   the camera's detector regions shifted by this many field pixels, random direction
Added 2026-11-03 (number audit: the table's "means fixed" row came from a test copy that was not kept):
  pic_shift_means_fixed  pic_shift with the means tile held in place (only the three colour tiles shift)
Run:  python src/robustness.py results/<optical run>.json [flaw,flaw,...]
      -> figures/robustness_<tag>.png, numbers on stdout (only the named flaws: results/robustness_<tag>_extra.json,
         or robustness_<tag>_<flaws>.json if that file exists already, so an earlier extra run is never overwritten)
"""
import json
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import fastsim
import optnet
import pilot_fmnist as pf
from bssg import Readout

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SWEEPS = {"pic_bits": [None, 8, 6, 5, 4, 3, 2], "mask_bits": [None, 8, 6, 5, 4, 3, 2],
          "plate_bits": [None, 8, 6, 5, 4, 3, 2], "plate_shift": [0, 0.25, 0.5, 1, 1.5, 2],
          "z_err": [0, 0.02, 0.05, 0.1, 0.2], "pic_shift": [0, 0.5, 1, 2],
          "phase_noise": [0, 0.05, 0.1, 0.2, 0.3, 0.5], "photons": [None, 1e6, 1e5, 1e4, 3e3, 1e3, 300],
          "plate_shift_fine": [0, 0.05, 0.1, 0.15, 0.2, 0.25], "plate_rot": [0, 0.1, 0.25, 0.5, 1, 2],
          "mask_shift": [0, 0.1, 0.25, 0.5, 1], "mag_err": [0, 0.002, 0.005, 0.01, 0.02],
          "cam_shift": [0, 0.25, 0.5, 1, 2], "pic_shift_means_fixed": [0, 0.5, 1, 1.5, 2, 3, 4]}


def _shift(f, dy, dx):
    """Band-limited sub-sample shift of the last two axes (periodic)."""
    ky = torch.fft.fftfreq(f.shape[-2], device=f.device, dtype=torch.float64)
    kx = torch.fft.fftfreq(f.shape[-1], device=f.device, dtype=torch.float64)
    return torch.fft.ifft2(torch.fft.fft2(f) * torch.exp(-2j * math.pi * (dy * ky[:, None] + dx * kx[None, :])))


def _resample(f, theta=0.0, scale=1.0):
    """A complex field turned by theta (rad) and scaled by `scale` about its centre, bilinear; outside is dark."""
    import torch.nn.functional as F
    B = f.reshape(-1, 1, *f.shape[-2:])
    c, s_ = math.cos(theta) / scale, math.sin(theta) / scale
    A = torch.tensor([[c, -s_, 0.0], [s_, c, 0.0]], dtype=torch.float64, device=f.device)[None].expand(len(B), 2, 3)
    grid = F.affine_grid(A, B.shape, align_corners=False)
    re = F.grid_sample(B.real, grid, align_corners=False)
    im = F.grid_sample(B.imag, grid, align_corners=False)
    return torch.complex(re, im).reshape(f.shape)


def quant_complex(z, bits):
    """Amplitude (relative to the largest) and phase each on 2^bits levels."""
    if bits is None:
        return z
    L = 2 ** bits - 1
    a = z.abs()
    m = a.amax(dim=tuple(range(1, a.dim())), keepdim=True).clamp_min(1e-30)
    aq = torch.round(a / m * L) / L * m
    pq = torch.round(torch.angle(z) / (2 * math.pi) * (L + 1)) / (L + 1) * 2 * math.pi
    return aq * torch.exp(1j * pq)


def main():
    run = json.loads(pathlib.Path(sys.argv[1]).read_text())
    tag = pathlib.Path(sys.argv[1]).stem[:60]
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")
    pf.N_QUBITS = n = len(run["weights"][0])
    s = pf.side()
    planes = 2 ** n // (s * s)
    perm = pf.morton_perm()
    d = np.load(HERE / "data" / f"{pf.DATASET}.npz")
    X, Y = d["x_val"].astype(np.float64), d["y_val"]
    x = X.transpose(0, 3, 1, 2).reshape(len(X), 3, s * s)[:, :, perm].reshape(len(X), -1) if X.ndim == 4 else None
    if run.get("center_image"):
        x = pf.centre_planes(x, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.pad(x, ((0, 0), (0, 2 ** n - x.shape[1])))
    a0 = torch.tensor(x, dtype=torch.complex128, device=DEV)
    w = torch.tensor(run["weights"], dtype=torch.float64, device=DEV)
    pix = torch.tensor(perm, device=DEV)
    nr = w.shape[2]
    fr = fastsim.fourier_rows(n, nr, s, planes, run.get("fourier_phase", False)) if run.get("fourier_mask") else 0
    N, r, L, os_ = 2 * s, run["opt_res"], run["optical"], run.get("opt_os", 1)
    phi0 = w[1 + fr:].reshape(-1)[:L * (N // r) ** 2].reshape(L, N // r, N // r)
    D0 = None
    if run.get("fourier_mask"):
        _, kw = fastsim.split_masks(w, 0, 0, -1, (pix, s, planes, run.get("fourier_phase", False), True))
        D0 = kw["fmask"][1]
    det = optnet.detector_index(N).to(DEV)
    rd = Readout(run["mode"], 128, 10, pt_ratio=run["pt_ratio"]).to(DEV)
    z0, pad = run["opt_z"], run["opt_pad"]

    def accuracy(flaw, v, rng):
        a = a0
        if flaw == "pic_bits":                                              # per plane, wire order -> same values
            a = quant_complex(a.reshape(len(a), planes, -1), v).reshape(len(a), -1)
        D = quant_complex(D0[None], v)[0] if (flaw == "mask_bits" and D0 is not None) else D0
        if flaw == "mask_shift" and v and D is not None:                   # the mask off-centre in the Fourier plane
            ang = rng.uniform(0, 2 * math.pi)                              # D in np.fft order: shift = re-sample
            D = _shift(D, v * math.sin(ang), v * math.cos(ang))
        st = fastsim.fourier_filter(a, D, pix, True) if D is not None else a
        f = optnet.tile(st, pix, s, planes)
        if flaw in ("pic_shift", "pic_shift_means_fixed") and v:            # sub-pixel shift by a phase ramp
            k = torch.fft.fftfreq(N, device=DEV, dtype=torch.float64)
            ramp = torch.exp(-2j * math.pi * v * (k[:, None] + k[None, :]))
            if flaw == "pic_shift":
                f = torch.fft.ifft2(torch.fft.fft2(f) * ramp)
            else:                                                           # the means tile (last plane) stays put
                means = st.reshape(len(st), planes, -1).clone()
                means[:, :planes - 1] = 0
                fm = optnet.tile(means.reshape(len(st), -1), pix, s, planes)
                f = torch.fft.ifft2(torch.fft.fft2(f - fm) * ramp) + fm
        phi = phi0.clone()
        if flaw == "plate_bits" and v is not None:
            q = 2 ** v
            phi = torch.round(phi / (2 * math.pi) * q) / q * 2 * math.pi
        if flaw == "phase_noise" and v:
            phi = phi + v * torch.tensor(rng.standard_normal(phi.shape), device=DEV)
        P = optnet.plate_phase(phi, r * os_).to(f.dtype)
        if flaw in ("plate_shift", "plate_shift_fine") and v:              # shift each plate on the fine grid
            M = P.shape[-1]
            P2 = []
            for l in range(L):
                ang = rng.uniform(0, 2 * math.pi)
                dy, dx = v * math.sin(ang) * r * os_, v * math.cos(ang) * r * os_
                k = torch.fft.fftfreq(M, device=DEV, dtype=torch.float64)
                ramp = torch.exp(-2j * math.pi * (dy * k[:, None] + dx * k[None, :]))
                ph = torch.angle(torch.fft.ifft2(torch.fft.fft2(P[l]) * ramp))  # band-limited shifted phase
                P2.append(torch.exp(1j * ph))
            P = torch.stack(P2)
        if flaw == "plate_rot" and v:                                       # each plate turned about its centre
            P = torch.stack([torch.exp(1j * torch.angle(_resample(P[l], math.radians(v) * rng.choice([-1, 1])) + 1e-30))
                             for l in range(L)])
        if flaw == "mag_err" and v:                                         # the 4f image a little too large
            f = _resample(f, 0.0, 1 + v)
        zs = [z0 * (1 + (v * rng.standard_normal() if flaw == "z_err" and v else 0)) for _ in range(L + 1)]
        g = f.repeat_interleave(os_, -2).repeat_interleave(os_, -1) if os_ > 1 else f
        for l in range(L):
            g = optnet.propagate(g, zs[l] * os_ * os_, pad) * P[l]
        g = optnet.propagate(g, zs[L] * os_ * os_, pad)
        if flaw == "cam_shift" and v:                                       # regions off by v field pixels
            ang = rng.uniform(0, 2 * math.pi)
            g = _shift(g, v * os_ * math.sin(ang), v * os_ * math.cos(ang))
        cnt = optnet.outcome_probs(g, det, os=os_)
        if flaw == "photons" and v:
            lam = cnt / cnt.sum(1, keepdim=True) * v
            cnt = torch.poisson(lam)
        p = cnt / cnt.sum(1, keepdim=True).clamp_min(1e-30)
        pred = rd(p.float()).argmax(1).cpu().numpy()
        return float((pred == Y).mean())

    only = sys.argv[2].split(",") if len(sys.argv) > 2 else list(SWEEPS)
    out = {}
    with torch.no_grad():
        for flaw, vals in ((k, SWEEPS[k]) for k in only):
            out[flaw] = []
            for v in vals:
                accs = [accuracy(flaw, v, np.random.default_rng(seed)) for seed in range(3)]
                out[flaw].append((v, float(np.mean(accs)), float(np.std(accs))))
            print(f"{flaw:12s} " + "  ".join(f"{str(v):>6s}:{m:.3f}" for v, m, _ in out[flaw]), flush=True)
    if len(sys.argv) > 2:
        dst = HERE / "results" / f"robustness_{tag}_extra.json"
        if dst.exists():
            dst = dst.with_name(f"robustness_{tag}_{'+'.join(only)}.json")
        dst.write_text(json.dumps(out))
        print(dst)
        return
    fig, ax = plt.subplots(2, 4, figsize=(16, 6.5))
    for i, (flaw, rows) in enumerate(out.items()):
        A = ax.flat[i]
        labels = ["ideal" if v is None else (f"{v:g}" if not isinstance(v, float) or v >= 1 or v == 0 else f"{v:g}")
                  for v, _, _ in rows]
        A.errorbar(range(len(rows)), [m for _, m, _ in rows], [e for _, _, e in rows], marker="o")
        A.set_xticks(range(len(rows)))
        A.set_xticklabels(labels, fontsize=8)
        A.set_title(flaw, fontsize=10)
        A.set_ylim(0, 1)
        A.axhline(rows[0][1], color="grey", lw=0.6, ls="--")
        A.axhline(0.1, color="k", lw=0.5, ls=":")
        A.grid(alpha=0.3)
    fig.suptitle(f"test accuracy under bench flaws (no retraining): {tag}", fontsize=10)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / f"robustness_{tag}.png", dpi=110)
    (HERE / "results" / f"robustness_{tag}.json").write_text(json.dumps(out))
    print(HERE / "figures" / f"robustness_{tag}.png")


if __name__ == "__main__":
    main()
