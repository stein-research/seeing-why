#!/usr/bin/env python3
"""Optical-native main circuit: phase plates and free-space propagation in place of the brickwall (owner 2026-10-31: train
the main circuit as phase plates from the start, so the bench is built from what was trained).

Light path (all linear in the field, so the exact evidence maps of fastsim carry over with A^dag):
  wire-ordered state (planes x s x s, after the plane-4 encoding and the Fourier-plane mask)
  -> the planes tiled 2 x 2 on one field of N = 2s pixels per side (plane p at block row p // 2, column p % 2)
  -> L times: Fresnel propagation over z, then a phase plate (pixels of r x r field samples, its own phases)
  -> Fresnel propagation over z to the camera
  -> the camera (N x N) cut into 128 detector regions, 16 rows x 8 columns of (N/16) x (N/8) pixels; outcome
     o = row * 8 + column is the light falling on region o. The detectors tile the camera, so the light kept is all
     the light that reaches it; light that leaves the N x N aperture on the way is lost (a lens bench's loss, counted by
     the conditional probabilities as the post-selection is).
Propagation is the paraxial transfer function exp(-i pi z (u^2 + v^2)) on a grid padded `pad` times (no wrap), with
z = lambda d / dx^2 dimensionless (d the plate spacing, dx the field pixel) and u, v in cycles per pixel; the result is
cropped back to N x N (the aperture). Its adjoint is the same with exp(+i ...): embedding and cropping are each
other's adjoints.

Run as a script: checks the adjoint, the energy bookkeeping and the propagation against LightPipes' Forvard.
"""
import math

import torch


def tile(state, pix, s, planes):
    """Wire-ordered states (B, planes * s * s) -> fields (B, 2s, 2s), the planes tiled 2 x 2 (planes <= 4)."""
    B = state.shape[0]
    g = torch.empty(B, planes, s * s, dtype=state.dtype, device=state.device)
    g[:, :, pix] = state.reshape(B, planes, s * s)
    g = g.reshape(B, planes, s, s)
    f = torch.zeros(B, 2 * s, 2 * s, dtype=state.dtype, device=state.device)
    for p in range(planes):
        f[:, (p // 2) * s:(p // 2 + 1) * s, (p % 2) * s:(p % 2 + 1) * s] = g[:, p]
    return f


def untile(field, pix, s, planes):
    """Adjoint of tile: fields (B, 2s, 2s) -> wire-ordered (B, planes * s * s)."""
    B = field.shape[0]
    g = torch.stack([field[:, (p // 2) * s:(p // 2 + 1) * s, (p % 2) * s:(p % 2 + 1) * s] for p in range(planes)], 1)
    return g.reshape(B, planes, s * s)[:, :, pix].reshape(B, -1)


_H = {}


def propagate(field, z, pad=2, conj=False):
    """Fresnel propagation of (B, N, N) fields over dimensionless distance z, padded `pad` times, cropped to N x N."""
    B, N, _ = field.shape
    M = pad * N
    key = (N, pad, float(z), str(field.device), field.dtype, conj)
    if key not in _H:
        u = torch.fft.fftfreq(M, device=field.device, dtype=torch.float64)
        ph = -math.pi * z * (u[:, None] ** 2 + u[None, :] ** 2)
        _H[key] = torch.exp(1j * (-ph if conj else ph)).to(field.dtype)
    o = (M - N) // 2
    big = torch.zeros(B, M, M, dtype=field.dtype, device=field.device)
    big[:, o:o + N, o:o + N] = field
    big = torch.fft.ifft2(torch.fft.fft2(big) * _H[key])
    return big[:, o:o + N, o:o + N]


def plate_phase(phi, r):
    """(L, N/r, N/r) plate angles -> (L, N, N) phase factors, each plate pixel covering r x r field samples."""
    return torch.exp(1j * phi.repeat_interleave(r, -2).repeat_interleave(r, -1))


def _shift(a, dy, dx):
    """Band-limited sideways shift of (..., M, M) arrays by (dy, dx) samples (a phase ramp in the Fourier plane)."""
    M = a.shape[-1]
    k = torch.fft.fftfreq(M, device=a.device, dtype=torch.float64)
    ramp = torch.exp(-2j * math.pi * (dy * k[:, None] + dx * k[None, :])).to(a.dtype if a.is_complex() else torch.complex128)
    return torch.fft.ifft2(torch.fft.fft2(a) * ramp)


def jittered_plates(P, shifts):
    """Plate transmissions (L, M, M) each moved sideways by shifts[l] = (dy, dx) samples, kept phase-only."""
    return torch.stack([torch.exp(1j * torch.angle(_shift(P[l], *shifts[l]))) for l in range(len(P))]).to(P.dtype)


def forward(field, phi, z, r, pad=2, os=1, plate_shifts=None, pic_shift=None):
    """Fields at the SLM (B, N, N) -> fields at the camera through len(phi) plates. os > 1 (pixel-aware): every pixel
    becomes os x os samples of a square pixel (constant across it), plates likewise, propagation on the finer grid
    (z scales by os^2); the camera field comes back on that grid, (B, N os, N os) - use camera_pixels for intensities."""
    if pic_shift is not None:                    # training for a bench: the picture off by (dy, dx) pixels
        field = _shift(field, *pic_shift).to(field.dtype)
    if os > 1:
        field = field.repeat_interleave(os, -2).repeat_interleave(os, -1)
    P = plate_phase(phi, r * os).to(field.dtype)
    if plate_shifts is not None:                 # ... and each plate off by plate_shifts[l] fine-grid samples
        P = jittered_plates(P, plate_shifts)
    zz = z * os * os
    for l in range(len(phi)):
        field = propagate(field, zz, pad) * P[l]
    return propagate(field, zz, pad)


def backward(field, phi, z, r, pad=2, os=1):
    """Adjoint of forward: camera fields (on forward's grid) back to the SLM pixels (the light sent back,
    phase-conjugated; with os > 1 summed over each pixel's samples, the adjoint of spreading it)."""
    P = plate_phase(phi, r * os).to(field.dtype).conj()
    zz = z * os * os
    field = propagate(field, zz, pad, conj=True)
    for l in range(len(phi) - 1, -1, -1):
        field = propagate(field * P[l], zz, pad, conj=True)
    if os > 1:
        B, M, _ = field.shape
        field = field.reshape(B, M // os, os, M // os, os).sum((2, 4))
    return field


def camera_pixels(cam, os=1):
    """Camera fields on forward's grid -> intensity per camera pixel (B, N, N), summed over each pixel's samples."""
    I = cam.abs() ** 2
    if os > 1:
        B, M, _ = I.shape
        I = I.reshape(B, M // os, os, M // os, os).sum((2, 4))
    return I


def detector_index(N, rows=16, cols=8):
    """(N, N) long: the detector region (outcome) of every camera pixel, o = row * cols + column."""
    rr = torch.arange(N) // (N // rows)
    cc = torch.arange(N) // (N // cols)
    return rr[:, None] * cols + cc[None, :]


def outcome_probs(cam, det, n_out=128, os=1):
    """Camera fields (on forward's grid) -> light per detector region (B, n_out), unnormalised."""
    I = camera_pixels(cam, os).reshape(cam.shape[0], -1)
    return torch.zeros(cam.shape[0], n_out, dtype=I.dtype, device=I.device).index_add_(1, det.reshape(-1).to(I.device), I)


def detector_weights(det, group, b, n_out=128):
    """(N, N) real weights (1 - b) on the class's regions and -b elsewhere: Pi_g - b on the camera."""
    w = torch.full((n_out,), -b, dtype=torch.float64)
    w[list(group)] = 1 - b
    return w[det]


def from_run(run, perm, device="cpu", dtype=torch.complex128):
    """(camera(amps) -> (B, N, N) camera fields, back(fields) -> (B, 2^n) the adjoint to the wire-ordered input, det)
    for a trainer result with --optical: the Fourier-plane mask (if any) and the phase plates, as trained."""
    import fastsim
    s, n = 32, len(run["weights"][0])
    planes = 2 ** n // (s * s)
    nr = len(run["weights"][0][0])
    w = torch.tensor(run["weights"], dtype=torch.float64, device=device)
    pix = torch.as_tensor(perm, device=device)
    fr = fastsim.fourier_rows(n, nr, s, planes, run.get("fourier_phase", False)) if run.get("fourier_mask") else 0
    N, r, L, os_ = 2 * s, run["opt_res"], run["optical"], run.get("opt_os", 1)
    phi = w[1 + fr:].reshape(-1)[:L * (N // r) ** 2].reshape(L, N // r, N // r)
    d = None
    if run.get("fourier_mask"):
        _, kw = fastsim.split_masks(w, 0, 0, -1, (pix, s, planes, run.get("fourier_phase", False),
                                                  run.get("fourier_linear", False)))
        d, lin = kw["fmask"][1], kw["fmask"][2]

    def camera(amps):
        st = amps.to(dtype)
        if d is not None:
            st = fastsim.fourier_filter(st, d.to(dtype), pix, lin)
        return forward(tile(st, pix, s, planes), phi, run["opt_z"], r, run["opt_pad"], os_)

    def back(field):
        st = untile(backward(field.to(dtype), phi, run["opt_z"], r, run["opt_pad"], os_), pix, s, planes)
        return fastsim.fourier_filter(st, d.conj().to(dtype), pix, lin) if d is not None else st
    return camera, back, detector_index(N).to(device), os_


if __name__ == "__main__":
    torch.manual_seed(0)
    N, L, r, z = 64, 3, 2, 32.0
    phi = 2 * math.pi * torch.rand(L, N // r, N // r, dtype=torch.float64)
    x = torch.randn(3, N, N, dtype=torch.complex128)
    y = torch.randn(3, N, N, dtype=torch.complex128)
    lhs = (forward(x, phi, z, r).conj() * y).sum((1, 2))
    rhs = (x.conj() * backward(y, phi, z, r)).sum((1, 2))
    print("adjoint <Ax, y> = <x, A^dag y>:", torch.allclose(lhs, rhs, atol=1e-9))
    y2 = torch.randn(3, 2 * N, 2 * N, dtype=torch.complex128)
    lhs2 = (forward(x, phi, z, r, os=2).conj() * y2).sum((1, 2))
    rhs2 = (x.conj() * backward(y2, phi, z, r, os=2)).sum((1, 2))
    print("adjoint, pixel-aware (os 2):", torch.allclose(lhs2, rhs2, atol=1e-9))
    e_in = (x.abs() ** 2).sum((1, 2))
    e_out = (forward(x, phi, z, r).abs() ** 2).sum((1, 2))
    print("energy kept (<= 1, aperture loss):", [round(v, 3) for v in (e_out / e_in).tolist()])
    det = detector_index(N)
    P = outcome_probs(forward(x, phi, z, r), det)
    print("detectors tile the camera:", torch.allclose(P.sum(1), e_out), "regions", int(det.max()) + 1)
    try:                                                     # the propagation against LightPipes (BSD-3)
        import LightPipes as lp
        import numpy as np
        lam, dx = 633e-9, 8e-6
        d = z * dx * dx / lam                                # physical spacing for this z
        f0 = np.zeros((2 * N, 2 * N), complex)
        f0[N // 2:N // 2 + N, N // 2:N // 2 + N] = x[0].numpy()
        F = lp.Begin(2 * N * dx, lam, 2 * N)
        F.field = f0
        F = lp.Forvard(F, d)
        ref = F.field[N // 2:N // 2 + N, N // 2:N // 2 + N]
        ours = propagate(x[:1], z)[0].numpy()
        g = np.vdot(ours, ref) / np.vdot(ours, ours)          # LightPipes adds the overall phase exp(i k d)
        print(f"propagation vs LightPipes Forvard: relative error {np.linalg.norm(ref - g * ours) / np.linalg.norm(ref):.2e}"
              f" (d = {d * 1e3:.2f} mm at 633 nm, 8 um pixels)")
    except ImportError:
        print("LightPipes not installed: propagation check skipped")
