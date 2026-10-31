#!/usr/bin/env python3
"""Pilot: pure-quantum Fashion-MNIST with photonic (coherent-light) encoding and a parameter-free readout.

Pipeline (no trainable classical layer anywhere):
  image 28x28 -> zero-pad to 32x32 = 1024 optical modes -> one photon in |psi> = sum_x a(x)|x>,
  a = pixel / ||pixel|| (coherent illumination through the image as a transmission mask; amplitude
  encoding done by the optics, here qml.AmplitudeEmbedding on 10 qubits)
  -> trainable circuit: RY, RX on every qubit + entangling layer, depth L (brickwall by default)
  -> qml.probs on the first n_out qubits (N = 2^n_out outcomes)
  -> readout (bssg.Readout): raw_renorm | raw_group | bss_renorm | bss_group -> NLL on the true class.

Photon-limited evaluation: each test image yields S photons; the outcome histogram over N outcomes replaces
the exact probabilities before the readout. Direct-imaging baseline ("CCD"): S photons land on the 1024 pixels
with probability |a(x)|^2; a multinomial logistic regression on the normalised pixel histogram, trained on
exact intensities, classifies. (The CCD baseline has 10,250 parameters, far more than the circuit; the
parameter-matched comparison comes later.)

Run:  python src/pilot_fmnist.py --mode bss_group --seed 0
"""
import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

import fastsim
from bssg import Readout


class _LazyPennyLane:
    """PennyLane only for the cross-check paths (make_qnode, make_unitary); training and the maps use fastsim, so the
    Colab demo runs without installing it."""
    def __getattr__(self, name):
        import pennylane
        return getattr(pennylane, name)


qml = _LazyPennyLane()

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
OUT = HERE / "results"
N_QUBITS = 10          # overwritten by --n_qubits
N_CLASSES = 10         # overwritten by --dataset
DATASET = "fmnist"     # fmnist / mnist: 28x28 grey, padded to 32x32 (10 qubits); imagenet10 / imagenette: 64x64 grey (12 qubits); imagenet32: 32x32x3 colour, channel-first,
                       # padded to 4096 amplitudes (12 qubits)
PHOTONS = (16, 64, 256, 1024, 4096)


INPUT = "pad"          # pad: 32x32 image on 10 qubits (+ ancillas in |0> if n_qubits > 10); upsample: 64x64 on 12
MEAN_PLANE = 0.0       # --mean_plane: with --center_image, the share of the energy given to the means in plane 4
DC_ENERGY = 0.0        # --dc_energy: with --center_image, the share of the energy left in the picture's own means
CENTER_IMAGE = False   # --center_image: subtract each picture's own mean per colour plane (optically: block the
                       # zero frequency of each colour in the Fourier plane), so energy = contrast, not brightness
CENTER = False         # --center: subtract the training-mean image before normalising (signed amplitudes; optically
                       # an amplitude-and-phase mask, a negative value = a pi phase shift)
SIM = "pennylane"      # pennylane (reference) | fast (fastsim.py: same circuit, complex64; check_fastsim.py)
HIER = False           # fastsim hier argument: False, True (--ent hier) or (wires, every) (--local)
ORDER = "rowcol"       # rowcol: index = row * 32 + col (wires 0-4 row bits, 5-9 column bits, most significant first)
                       # morton: row and column bits interleaved (r4 c4 r3 c3 ... r0 c0), so neighbouring wires are
                       # the same scale of both axes and the first measured wires are a coarse 2-D patch
DOMAIN = "pixel"       # --domain dct: amplitudes are the picture's orthonormal 2-D DCT (per colour plane) - the light in
                       # the Fourier plane of a lens (a fixed real orthogonal map of the pixel state, owner 2026-10-30);
                       # basis states are frequencies (Morton-ordered (u, v)), so one-wire gates act as smooth cosines
FILTER = "none"        # --filter sobel: light = Sobel gradient magnitude of the picture (edges; background light removed)
ENCODING = "amp"       # amp: amplitude = brightness (coherent light, amplitude mask)
                       # sqrt: amplitude = sqrt(brightness) (coherent light, intensity mask; Fischbacher & Sbaiz)
                       # incoherent: self-luminous image, mixed state rho = sum_x p(x)|x><x|, p = brightness / sum


GREY64 = ("imagenet10", "imagenette", "animals10")     # 64x64 grey sets on 12 qubits (animals10 letterboxed)


def side():
    """Image side in pixels: 64 for the 64x64 grey sets and for --input upsample (32x32 repeated 2x2), else 32."""
    return 64 if DATASET in GREY64 or INPUT == "upsample" else 32


def hilbert_perm(s):
    """perm[m] = row * s + col of the pixel at Hilbert-curve index m: consecutive indices are always neighbouring
    pixels, and each group of 4^k indices is a 2^k x 2^k block (owner 2026-10-30: an order other than Morton)."""
    perm = np.empty(s * s, dtype=np.int64)
    for d in range(s * s):
        x = y = 0
        t, q = d, 1
        while q < s:
            rx = 1 & (t // 2)
            ry = 1 & (t ^ rx)
            if ry == 0:                                  # rotate the quadrant
                if rx == 1:
                    x, y = q - 1 - x, q - 1 - y
                x, y = y, x
            x, y = x + q * rx, y + q * ry
            t //= 4
            q *= 2
        perm[d] = y * s + x
    return perm


def morton_perm(s=None, gray=None):
    """perm[m] = row * s + col of the pixel at Z-order index m (row and column bits interleaved, row first, most
    significant first: r4 c4 ... r0 c0 for s = 32). gray (default: ORDER == "graymorton") interleaves the Gray codes of
    row and column instead, so every pair of neighbouring pixels differs in exactly one wire."""
    s = s or side()
    if ORDER == "hilbert" and gray is None:
        return hilbert_perm(s)
    gray = (ORDER == "graymorton") if gray is None else gray
    nb = s.bit_length() - 1
    g = (lambda v: v ^ (v >> 1)) if gray else (lambda v: v)
    perm = np.empty(s * s, dtype=np.int64)
    for r in range(s):
        for c in range(s):
            m, gr, gc = 0, g(r), g(c)
            for b in range(nb - 1, -1, -1):
                m = (m << 2) | (((gr >> b) & 1) << 1) | ((gc >> b) & 1)
            perm[m] = r * s + c
    return perm


def permuted():
    """True when the pixel grid is reordered on the wires (Morton or Gray-Morton)."""
    return ORDER in ("morton", "graymorton", "hilbert")


def centre_planes(x, dc_energy=0.0, mean_plane=0.0):
    """(B, channels * side^2 [+ padding]) -> each picture's own mean removed from each colour plane (padding kept).
    dc_energy f > 0 keeps the means, scaled by one factor per picture so they hold the fraction f of the energy: a
    phase reference (the sign of a deviation is otherwise lost, P(psi) = P(-psi)) and the picture's colour balance;
    optically a zero-frequency stop that lets part of the light through (phase contrast).
    mean_plane f > 0 (colour sets, 12 wires: the fourth colour plane is empty) puts the means there instead, holding the
    fraction f of the energy: the picture planes keep no DC at all (their energy is pure contrast, no brightness bias)
    while the circuit still gets the colour balance and a phase reference. Plane 4 holds a 2x2 tile repeated over the
    grid - (even row, even column) red mean, (even, odd) green, (odd, even) blue, (odd, odd) dark - so every pixel has
    all three means within one cell of it; optically the blocked DC sent on as a separate reference beam.
    Grey sets (one plane, 11 wires: a second, empty plane): the picture's mean fills plane 2 evenly - every pixel has
    its reference behind it; on a bench a spare tile of the field."""
    s = side()
    d = channels() * s ** 2
    v = x[:, :d].reshape(len(x), channels(), -1)
    mu = v.mean(2, keepdims=True)
    ac = v - mu
    if mean_plane > 0 and channels() == 1:
        if x.shape[1] not in (d, 2 * s * s):
            raise ValueError("mean_plane on a grey set needs a state with a second, empty plane (11 wires)")
        ref = np.broadcast_to(mu[:, 0, :], (len(x), s * s)).copy()
        e_ac = (ac ** 2).sum((1, 2))[:, None]
        e_ref = np.maximum((ref ** 2).sum(1, keepdims=True), 1e-30)
        ref = ref * np.sqrt(mean_plane * e_ac / ((1 - mean_plane) * e_ref))
        return np.concatenate([ac.reshape(len(x), -1), ref], 1)
    if mean_plane > 0:
        if channels() != 3 or x.shape[1] not in (d, 4 * s * s):
            raise ValueError("mean_plane needs a colour set whose state has an empty fourth plane")
        r, c = np.divmod(morton_perm() if permuted() else np.arange(s * s), s)  # pixel of each wire-ordered amplitude
        tile = (r % 2) * 2 + c % 2                                              # 0 red, 1 green, 2 blue, 3 dark
        ref = np.where(tile[None] < 3, np.take_along_axis(np.concatenate([mu[:, :, 0], np.zeros((len(x), 1))], 1),
                                                          np.broadcast_to(tile, (len(x), s * s)), 1), 0.0)
        e_ac = (ac ** 2).sum((1, 2))[:, None]
        e_ref = np.maximum((ref ** 2).sum(1, keepdims=True), 1e-30)
        ref = ref * np.sqrt(mean_plane * e_ac / ((1 - mean_plane) * e_ref))
        return np.concatenate([ac.reshape(len(x), -1), ref], 1)
    if dc_energy > 0:
        e_ac = (ac ** 2).sum((1, 2), keepdims=True)
        e_dc = (mu ** 2).sum((1, 2), keepdims=True) * v.shape[2]
        ac = ac + np.sqrt(dc_energy * e_ac / ((1 - dc_energy) * np.maximum(e_dc, 1e-30))) * mu
    return np.concatenate([ac.reshape(len(x), -1), x[:, d:]], 1)


def channels():
    """Colour channels stored one after another in the state: 3 for the colour sets, else 1."""
    return 3 if DATASET in ("imagenet32", "cifar10c", "wmnistc", "caltech10c", "caltech10f", "caltech10fc", "caltech10fct", "flowers17c") else 1


def _to_grids(x):
    """(B, d) wire-ordered states -> (B * channels, side, side) pixel grids, plus what _from_grids needs."""
    s, nc, B = side(), channels(), len(x)
    perm = morton_perm() if permuted() else np.arange(s * s)
    g = np.empty((B * nc, s * s))
    g[:, perm] = x[:, :nc * s * s].reshape(B * nc, s * s)
    g = g.reshape(-1, s, s)
    if DOMAIN == "dct":                               # transforms act on the picture, not on its spectrum
        import scipy.fft
        g = scipy.fft.idctn(g, axes=(-2, -1), norm="ortho")
    return g, (perm, B, x.shape[1])


def _from_grids(g, info):
    perm, B, d = info
    g = np.clip(g, 0, None)
    if DOMAIN == "dct":
        import scipy.fft
        g = scipy.fft.dctn(g, axes=(-2, -1), norm="ortho")
    out = g.reshape(len(g), -1)[:, perm].reshape(B, -1)
    out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return np.pad(out, ((0, 0), (0, d - out.shape[1]))) if d > out.shape[1] else out


def shift_images(x, shifts):
    """Translate wire-ordered amplitude images by integer (dy, dx) per image, zero fill (no wrap), renormalised.
    x: (B, d); shifts: (B, 2) ints; every colour channel moves the same way."""
    s = side()
    g, info = _to_grids(x)
    out = np.zeros_like(g)
    shifts = np.repeat(np.asarray(shifts), channels(), axis=0)
    for dy, dx in np.unique(shifts, axis=0):                     # one slice per distinct shift, all its images
        b = np.all(shifts == (dy, dx), axis=1)
        ys, yd = (slice(0, s - dy), slice(dy, s)) if dy >= 0 else (slice(-dy, s), slice(0, s + dy))
        xs, xd = (slice(0, s - dx), slice(dx, s)) if dx >= 0 else (slice(-dx, s), slice(0, s + dx))
        out[b, yd, xd] = g[b, ys, xs]
    return _from_grids(out, info)


def affine_images(x, angle, scale, dx, dy):
    """Rotate (degrees, about the centre), scale and translate (pixels) wire-ordered amplitude images with bilinear
    interpolation and zero fill, then renormalise. Per-image arrays of length B (or scalars)."""
    import cv2
    s, B, nc = side(), len(x), channels()
    g, info = _to_grids(x)
    angle, scale, dx, dy = (np.repeat(np.broadcast_to(np.asarray(v, dtype=float), (B,)), nc)
                            for v in (angle, scale, dx, dy))
    c = (s - 1) / 2.0
    out = np.empty_like(g)
    for b in range(len(g)):
        M = cv2.getRotationMatrix2D((c, c), float(angle[b]), float(scale[b]))
        M[:, 2] += (dx[b], dy[b])
        out[b] = cv2.warpAffine(g[b], M, (s, s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return _from_grids(out, info)


def random_affine(B, rng, a):
    """Per-image (angle, scale, dx, dy) for --aug_affine (defaults: scale 0.25, translate 0.1,
    tilt -15 / 0 / +15 degrees)."""
    t = a.aff_translate * side()
    return (rng.choice([-a.aff_rot, 0.0, a.aff_rot], B), rng.uniform(1 - a.aff_scale, 1 + a.aff_scale, B),
            rng.uniform(-t, t, B), rng.uniform(-t, t, B))


def pixel_planes(v):
    """(nc, s, s) picture-space planes of one wire-ordered vector v (amplitudes, evidence factors, ...): undo the pixel
    order and, for --domain dct, the transform (v may be any real vector; no clipping, no renormalising)."""
    s, nc = side(), channels()
    perm = morton_perm() if permuted() else np.arange(s * s)
    g = np.empty((nc, s * s))
    g[:, perm] = np.asarray(v)[:nc * s * s].reshape(nc, s * s)
    g = g.reshape(nc, s, s)
    if DOMAIN == "dct":
        import scipy.fft
        g = scipy.fft.idctn(g, axes=(-2, -1), norm="ortho")
    return g


def sobel(x):
    """Gradient magnitude of (n, ..., s, s) pictures, replicate border, every channel on its own."""
    p = np.pad(x, [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    sm = lambda q, ax: np.take(q, range(0, q.shape[ax] - 2), ax) + 2 * np.take(q, range(1, q.shape[ax] - 1), ax) \
        + np.take(q, range(2, q.shape[ax]), ax)                                       # [1 2 1] along one axis
    d = lambda q, ax: np.take(q, range(2, q.shape[ax]), ax) - np.take(q, range(0, q.shape[ax] - 2), ax)
    return np.hypot(d(sm(p, -2), -1), d(sm(p, -1), -2))


def load(n_train, n_val, seed):
    rng = np.random.default_rng(seed)
    if DATASET in ("imagenet32", "cifar10", "cifar10c", "wmnistc", "caltech10c", "caltech10f", "caltech10fc", "caltech10fct", "flowers17c") + GREY64:
        d = np.load(HERE / "data" / {"cifar10c": "cifar10.npz", "wmnistc": "wmnist.npz"}.get(DATASET, f"{DATASET}.npz"))
        xtr_all, ytr_all = d["x_train"], d["y_train"]
        xte_raw, yte = d["x_val"], d["y_val"]                 # ImageNet validation images = our test set
    else:
        d = np.load(HERE / "data" / ("mnist.npz" if DATASET == "mnist" else "fashion_mnist.npz"))
        xtr_all, ytr_all = d["x_train"], d["y_train"]
        xte_raw, yte = d["x_test"], d["y_test"]

    def prep(x):
        if DATASET in ("imagenet32", "cifar10c", "wmnistc", "caltech10c", "caltech10f", "caltech10fc", "caltech10fct", "flowers17c"):
            x = np.transpose(x.astype(np.float64), (0, 3, 1, 2))   # (n, colour, row, col); 3 x 1024 padded to 4096
        elif DATASET == "cifar10":                                 # 32x32 RGB -> luminance, 10 qubits, no padding
            x = x.astype(np.float64) @ np.array([0.299, 0.587, 0.114])
        elif DATASET in GREY64:
            x = x.astype(np.float64)                               # (n, 64, 64) grey, no padding
        else:
            x = np.pad(x.astype(np.float64), ((0, 0), (2, 2), (2, 2)))
        if FILTER == "sobel":
            x = sobel(x)
        if DOMAIN == "dct":                           # to the frequency plane (norm kept: orthonormal)
            import scipy.fft
            x = scipy.fft.dctn(x, axes=(-2, -1), norm="ortho")
        if INPUT == "upsample":                       # 32x32 -> 64x64 by pixel repetition (no new information)
            x = x.repeat(2, axis=1).repeat(2, axis=2)
        if permuted():                                # permute the 32x32 grid of each channel to Z-order
            x = x.reshape(len(x), -1, side() ** 2)[:, :, morton_perm()]
        return x.reshape(len(x), -1)

    def finish(x, mu):
        if ENCODING == "incoherent":
            x = x / x.sum(1, keepdims=True)           # intensity distribution p(x)
        else:
            if ENCODING == "sqrt":
                x = np.sqrt(x)
            if mu is not None:                        # signed amplitudes: the training-mean image removed
                x = x - mu
            if CENTER_IMAGE:                          # signed amplitudes: the picture's own mean per plane removed
                x = centre_planes(x, DC_ENERGY, MEAN_PLANE)
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        if x.shape[1] < 2 ** N_QUBITS:                # ancilla qubits start in |0>: zero amplitudes (dark modes)
            x = np.pad(x, ((0, 0), (0, 2 ** N_QUBITS - x.shape[1])))
        return x

    idx = rng.permutation(len(xtr_all))
    tr, va = idx[:n_train], idx[n_train:n_train + n_val]
    gtr = prep(xtr_all[tr])
    mu = None
    if CENTER:
        if ENCODING != "amp":
            raise SystemExit("--center needs --encoding amp")
        mu = gtr.mean(0)                              # mean image of the training subset (fixed, not learned)
    xtr, ytr = finish(gtr, mu), ytr_all[tr]
    xva, yva = (finish(prep(xtr_all[va]), mu) if len(va) else np.zeros((0, xtr.shape[1]))), ytr_all[va]   # small sets
    xte = finish(prep(xte_raw), mu)
    RAW.update(train=xtr_all[tr], encode=lambda r: finish(prep(r), mu))   # for augmentations made on raw pictures
    return (xtr, ytr), (xva, yva), (xte, yte)


RAW = {}                                              # load(): raw training pictures and their encoder


def random_crops(x, rng, f):
    """Raw pictures (B, H, W[, C]) -> each cut to a random square of side U(f, 1) x H at a random place and resized back
    to H x W (bilinear). Only the picture's own pixels are used - no black fill, so no new edge appears, and an edge
    that sits in the data (a frame, a background strip) lands at a different place, or out, each time."""
    import cv2
    B, H, W = x.shape[:3]
    out = np.empty_like(x)
    for i in range(B):
        c = int(round(rng.uniform(f, 1.0) * H))
        r0, c0 = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
        out[i] = cv2.resize(x[i, r0:r0 + c, c0:c0 + c], (W, H), interpolation=cv2.INTER_LINEAR).reshape(x[i].shape)
    return out


def entangle(layer, kind):
    if kind == "brickwall":
        start = layer % 2
        for i in range(start, N_QUBITS - 1, 2):
            qml.CNOT(wires=[i, i + 1])
    else:                                            # owner's chain, direction alternating per layer
        if layer % 2:
            for i in range(N_QUBITS - 1):
                qml.CNOT(wires=[i, i + 1])
        else:
            for i in range(N_QUBITS - 1, 0, -1):
                qml.CNOT(wires=[i, i - 1])


def rotate(wk):
    """One rotation layer: RY, RX on every qubit (wk: (n_qubits, 2)), or RZ, RY, RX (wk: (n_qubits, 3)) - a full
    single-qubit rotation; the RZ puts a position-dependent phase on the amplitudes (a phase mask, in optics)."""
    for i in range(N_QUBITS):
        if wk.shape[-1] == 3:
            qml.RZ(wk[i, 2], wires=i)
        qml.RY(wk[i, 0], wires=i)
        qml.RX(wk[i, 1], wires=i)


def tree_angles(amps, levels):
    """Angles of the amplitude-encoding tree (Mottonen), levels 0..levels-1, level order: (B, 2^levels - 1).

    Node j at level l covers the basis states whose first l wires spell j; its angle is
    theta = 2 arccos(sqrt(light in its |0> half / light in the node)), the RY that splits the node's light between its
    two children (0 for a dark node). With the Morton order each node is an image region split alternately top/bottom
    and left/right, so the angles are multi-scale "which way does the light lean" features, scale-free in the image.
    """
    p = amps.abs() ** 2
    out = []
    for l in range(levels):
        q = p.reshape(len(p), 2 ** (l + 1), -1).sum(-1).reshape(len(p), 2 ** l, 2)    # children of each node
        tot = q.sum(-1)
        frac = torch.where(tot > 1e-12, q[..., 0] / tot.clamp_min(1e-12), torch.ones_like(tot))
        out.append(2 * torch.arccos(frac.clamp(1e-9, 1 - 1e-9).sqrt()))
    return torch.cat(out, 1)


def make_qnode(depth, n_out, ent, device, n_blocks=0):
    """n_blocks > 0: after the first n_blocks layers, re-upload 10 tree angles each - RY(c_i) on wire i, then RX on the
    same 10 angles in reverse order (RX(c_{n-1-i}) on wire i); the circuit then takes the angles as a third input."""
    if SIM == "fast" and n_blocks == 0:
        if ent not in ("brickwall", "hier", "qcnn"):
            raise SystemExit("--sim fast simulates the brickwall, hier and qcnn entanglers only")
        return lambda amps, w, ang=None: fastsim.probs(fastsim.run(amps, w, depth, hier=HIER,
                                                                   qcnn=n_out if ent == "qcnn" else 0), n_out)
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(amps, w, ang=None):
        qml.AmplitudeEmbedding(amps, wires=range(N_QUBITS), normalize=False)
        for k in range(depth):
            rotate(w[k])
            entangle(k, ent)
            if k < n_blocks:
                for i in range(N_QUBITS):
                    qml.RY(ang[:, k, i], wires=i)
                for i in range(N_QUBITS):
                    qml.RX(ang[:, k, N_QUBITS - 1 - i], wires=i)
        rotate(w[depth])
        return qml.probs(wires=list(range(n_out)))
    return circuit


def make_unitary(depth, ent):
    """The circuit's unitary U(w) (no embedding), differentiable, for the incoherent encoding."""
    def body(w):
        for k in range(depth):
            rotate(w[k])
            entangle(k, ent)
        rotate(w[depth])
    return lambda w: qml.matrix(body, wire_order=range(N_QUBITS))(w)


def intensity(x):
    """What a camera sees: photon-arrival probability per pixel."""
    return x if ENCODING == "incoherent" else x ** 2


def photon_hist(P, S, rng):
    """P: (B, N) exact outcome probabilities -> empirical frequencies from S photons each."""
    P = np.clip(P.astype(np.float64), 0, None)            # float64: complex64 simulation leaves sums of 1 +- 1e-7
    P = P / P.sum(1, keepdims=True)
    return np.array([rng.multinomial(S, p) for p in P]) / S


def train_ccd(xtr, ytr, steps=400, seed=0):
    torch.manual_seed(seed)
    D = xtr.shape[1]
    X = torch.tensor(intensity(xtr), dtype=torch.float32) * D       # intensity, mean ~1 per pixel
    y = torch.tensor(ytr, dtype=torch.long)
    W = torch.zeros(D, N_CLASSES, requires_grad=True)
    b = torch.zeros(N_CLASSES, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=0.01)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(X @ W + b, y) + 1e-4 * (W ** 2).sum()
        loss.backward()
        opt.step()
    return W.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="bss_group")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--n_out", type=int, default=4)
    ap.add_argument("--ent", default="brickwall", choices=["brickwall", "chain", "hier", "qcnn"],
                    help="hier: brickwall layers on the finest 4 wires first, 2 more wires every 4 layers; qcnn: brickwall "
                         "layers on fewer wires every stage, each wire dropped pooled onto the one above by a controlled "
                         "rotation, down to the n_out measured wires (fast sim only)")
    ap.add_argument("--local", type=int, default=0,
                    help="patch-local circuit (brickwall or qcnn, fast sim): the first layers act only on the finest W "
                         "wires (a 4x4-pixel block for W = 4, 8x8 for 6; the same gates in every block = a shared "
                         "convolution), 2 more wires every --local_every layers, then all")
    ap.add_argument("--local_every", type=int, default=4, help="layers before the --local window grows by 2 wires")
    ap.add_argument("--local_colour", action="store_true",
                    help="colour sets: the --local window holds the colour-plane wires from the start too, so the "
                         "block filter mixes R, G and B of its pixels (as a CNN's first layer)")
    ap.add_argument("--pt_ratio", type=float, default=1.5)
    ap.add_argument("--n_train", type=int, default=10000)
    ap.add_argument("--n_val", type=int, default=2000)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--channels", type=int, default=1,
                    help="C > 1: log2(C) ancilla wires make C channels; each gets its own --bank-layer circuit on the "
                         "picture (filter bank), then the main circuit acts on all wires (fast sim)")
    ap.add_argument("--bank", type=int, default=3, help="filter-bank layers per channel (with --channels)")
    ap.add_argument("--bank_init", type=float, default=1.0, help="std of the filter bank's initial angles")
    ap.add_argument("--init_from", default="",
                    help="start from the weights of a results/*.json run or checkpoint with the same circuit")
    ap.add_argument("--init_scale", type=float, default=0.1, help="std of the initial angles (0.01 = near identity)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (PennyLane 0.45.1 broadcasts on CUDA tensors)")
    ap.add_argument("--n_qubits", type=int, default=10)
    ap.add_argument("--input", default="pad", choices=["pad", "upsample"])
    ap.add_argument("--domain", default="pixel", choices=["pixel", "dct"], help="amplitudes = pixels or their 2-D DCT")
    ap.add_argument("--filter", default="none", choices=["none", "sobel"], help="optical pre-filter of the picture")
    ap.add_argument("--encoding", default="amp", choices=["amp", "sqrt", "incoherent"])
    ap.add_argument("--dataset", default="fmnist", choices=["fmnist", "mnist", "imagenet32", "imagenet10", "imagenette", "animals10", "cifar10", "cifar10c", "wmnistc", "caltech10c", "caltech10f", "caltech10fc", "caltech10fct", "flowers17c"])
    ap.add_argument("--full_eval", action="store_true",
                    help="also test shifted / tilted / scaled pictures, photon budgets and the direct-imaging baseline "
                         "(off by default since 2026-10-30: the owner wants only accuracy, and they doubled the test time)")
    ap.add_argument("--lr_floor", type=float, default=0.0,
                    help="cosine schedule ends at this fraction of lr instead of 0 (owner 2026-10-30: long runs, 0.3)")
    ap.add_argument("--warmup_epochs", type=int, default=1)
    ap.add_argument("--rot", default="ryrx", choices=["ryrx", "rzryrx"], help="rotation layer (rzryrx adds an RZ)")
    ap.add_argument("--order", default="rowcol", choices=["rowcol", "morton", "graymorton", "hilbert"],
                    help="pixel order on the wires")
    ap.add_argument("--reupload_levels", type=int, default=0,
                    help="re-upload the tree angles of levels 0..L-1 (2^L - 1 angles, 10 per block) after the first layers")
    ap.add_argument("--sim", default="fast", choices=["fast", "pennylane"],
                    help="fast: fastsim.py (checked against PennyLane by check_fastsim.py)")
    ap.add_argument("--center_image", action="store_true",
                    help="subtract each picture's own mean per colour plane before normalising (signed amplitudes; "
                         "energy = contrast, so flat backgrounds go dark and a dark object on a bright ground is lit)")
    ap.add_argument("--mean_plane", type=float, default=0.0,
                    help="with --center_image on a colour set: put each picture's colour means in the empty fourth "
                         "plane at this fraction of the energy (picture planes stay pure contrast)")
    ap.add_argument("--dc_energy", type=float, default=0.0,
                    help="with --center_image: keep each picture's colour means at this fraction of the energy (a "
                         "phase reference, so a picture and its negative differ; 0 = means removed)")
    ap.add_argument("--center", action="store_true",
                    help="subtract the training-mean image before normalising (signed amplitudes)")
    ap.add_argument("--patch_wires", type=int, default=0,
                    help="bag of patches: the circuit acts on and measures only the last P wires (the content of a "
                         "2^(P/2) x 2^(P/2) patch under the Morton order); every patch shares it, no interference "
                         "between patches; n_out is set to P")
    ap.add_argument("--pyramid", type=int, default=0,
                    help="spatial pyramid: a 50:50 mixture (beam splitter) of the full circuit and a bag of patches on "
                         "the last K wires; both measure n_out wires and BHR adds their class groups; weights "
                         "(depth+1, n + K, rot) = full level then bag level")
    ap.add_argument("--aug_crop", type=float, default=0.0,
                    help="augment training batches with random square crops of side U(f, 1) of the picture, resized "
                         "back (no black fill; made on the raw pictures, then encoded). 0 = off")
    ap.add_argument("--aug_shift", type=int, default=0,
                    help="augment training batches with random translations of up to K pixels (zero fill)")
    ap.add_argument("--aux_spread", type=float, default=0.0,
                    help="weight of the spatial spread of the true class's positive evidence map (heatmap concentration)")
    ap.add_argument("--aux_smooth", type=float, default=0.0,
                    help="weight of the roughness of the true class's evidence map: sum of squared neighbour differences / "
                         "sum of squares (scale-free; small = smooth)")
    ap.add_argument("--att_spread", type=float, default=0.0,
                    help="with --mask_group: weight of the spatial spread of the attention map (each pixel's share of "
                         "the light that passes the mask; positive part, side^2 units) - asks the mask to pass light "
                         "from one compact region without saying where")
    ap.add_argument("--balanced", action="store_true", help="draw training pictures with class-balanced probability")
    ap.add_argument("--aug_affine", action="store_true",
                    help="augment with random affine transforms; defaults: "
                         "scale 0.25, translate 0.1, tilt -15 / 0 / +15 degrees")
    ap.add_argument("--aff_scale", type=float, default=0.25, help="scale factor drawn from [1 - s, 1 + s]")
    ap.add_argument("--aff_translate", type=float, default=0.1, help="shift up to this fraction of the image side")
    ap.add_argument("--aff_rot", type=float, default=15.0, help="tilt drawn from {-r, 0, +r} degrees")
    ap.add_argument("--mask_group", type=int, default=0,
                    help="soft mask (post-selection) after layer --mask_at: m(x) = prod over groups of G consecutive "
                         "wires of cos(phi[group, bits]); the class probabilities are conditional on the light passing "
                         "(the division is the nonlinearity, as a softmax's denominator). 0 = off")
    ap.add_argument("--fourier_bank", type=int, default=0,
                    help="K > 1 (power of 2): K Fourier masks in superposition - log2(K) register wires in front "
                         "pick the mask; the circuit then acts on register and picture wires (channels mixed by "
                         "interference); --fourier_phase applies to every mask")
    ap.add_argument("--optical", type=int, default=0,
                    help="L > 0: the main circuit is L phase plates with free-space propagation between them (optnet.py; "
                         "the planes tiled 2 x 2 on one field, 128 camera regions as the outcomes) instead of the "
                         "brickwall; --depth is set to 0 and its row is unused")
    ap.add_argument("--opt_z", type=float, default=32.0, help="plate spacing as lambda d / dx^2 (dimensionless)")
    ap.add_argument("--opt_res", type=int, default=2, help="field samples per plate pixel (per side)")
    ap.add_argument("--opt_pad", type=int, default=2, help="padding of the propagation grid (no wrap)")
    ap.add_argument("--opt_keep", type=float, default=0.0,
                    help="with --optical: weight of (1 - share of the light entering the plates that reaches the "
                         "camera) in the loss - rewards keeping the light inside the plates' aperture (the mask's "
                         "own dimming is not counted)")
    ap.add_argument("--opt_jitter", type=float, default=0.0,
                    help="with --optical, training only: each plate moved sideways by a random vector of length "
                         "U(0, this) plate pixels, and the picture by U(0, --opt_pic_jitter) pixels, new each batch - "
                         "a bench that tolerates misalignment (evaluation stays aligned)")
    ap.add_argument("--opt_pic_jitter", type=float, default=0.0, help="see --opt_jitter (picture pixels)")
    ap.add_argument("--opt_os", type=int, default=1,
                    help="pixel-aware: samples per pixel side (square pixels diffract as on a bench; 2 matches "
                         "LightPipes at 2 and 4 samples)")
    ap.add_argument("--fourier_linear", action="store_true",
                    help="lens mode for --fourier_mask / --fourier_bank: linear convolution (picture zero-padded, no "
                         "wrap at its edge), as a 4f lens bench gives; default circular, as a QFT on qubits gives")
    ap.add_argument("--fourier_bank_mix", action="store_true",
                    help="control for --fourier_bank: the register is measured after the masks (no interference "
                         "between filters; same circuit and parameters): P = sum over filters of each branch's P")
    ap.add_argument("--fourier_phase", action="store_true",
                    help="with --fourier_mask: the mask also sets a phase (odd in k, so the kernel stays real), so it "
                         "can learn directional filters such as derivatives / Sobel")
    ap.add_argument("--mask_at", type=int, default=-1, help="layer after whose CNOTs the mask acts (-1: depth // 3)")
    ap.add_argument("--fourier_mask", action="store_true",
                    help="soft mask in the Fourier plane of the input instead (4f system: each colour plane's 2-D "
                         "spatial frequencies times a learned real even mask, then back); a learned convolution that "
                         "cannot favour a place in the picture; conditional probabilities as --mask_group")
    ap.add_argument("--glimpse", type=int, default=0,
                    help="glimpse attention on a gc x gc grid of cells (power of 2): the coarse picture on its own "
                         "wires, a look circuit V on them, and the picture seen through one learned window moved to "
                         "the cell V picks - a mixture over the gc^2 cells weighted by the picture's own content; "
                         "conditional probabilities as --mask_group. 0 = off")
    ap.add_argument("--glimpse_depth", type=int, default=8, help="brickwall depth of the look circuit V")
    ap.add_argument("--micro", type=int, default=0,
                    help="split each batch into micro-batches of this size for the backward pass (same gradient, "
                         "less GPU memory; glimpse runs gc^2 copies of the circuit). 0 = whole batch")
    ap.add_argument("--tag", default="", help="extra word in the result file name (keeps a pilot from overwriting a run "
                    "whose settings differ only in ones the name does not show, such as --n_train)")
    ap.add_argument("--checkpoint_every", type=int, default=0,
                    help="write results/checkpoint_<name>.json (best-so-far weights) every K epochs; removed at the end")
    ap.add_argument("--time_only", action="store_true")
    a = ap.parse_args()
    global N_QUBITS, INPUT, ENCODING, DATASET, N_CLASSES, ORDER, SIM, CENTER, CENTER_IMAGE, DC_ENERGY, MEAN_PLANE, FILTER, DOMAIN, HIER
    N_QUBITS, INPUT, ENCODING, DATASET, ORDER = a.n_qubits, a.input, a.encoding, a.dataset, a.order
    SIM = "pennylane" if a.reupload_levels else a.sim
    colour_wires = N_QUBITS - 2 * (side().bit_length() - 1) if a.local_colour else 0   # wires above the pixel bits
    if a.local_colour and (not a.local or channels() != 3 or colour_wires < 1):
        raise SystemExit("--local_colour needs --local and a colour dataset")
    HIER = (a.local, a.local_every, colour_wires) if a.local else a.ent == "hier"
    if a.local and (a.ent not in ("brickwall", "qcnn") or SIM != "fast" or a.channels > 1 or a.patch_wires):
        raise SystemExit("--local needs --ent brickwall or qcnn, --sim fast, no channels and no bag of patches")
    CENTER = a.center
    CENTER_IMAGE, DC_ENERGY, MEAN_PLANE = a.center_image, a.dc_energy, a.mean_plane
    if MEAN_PLANE and not (CENTER_IMAGE and 0 < MEAN_PLANE < 1 and not DC_ENERGY
                           and ((channels() == 3 and 2 ** N_QUBITS == 4 * side() ** 2)
                                or (channels() == 1 and 2 ** N_QUBITS == 2 * side() ** 2))):
        raise SystemExit("--mean_plane needs --center_image (no --dc_energy) and an empty plane: a colour set on 12 "
                         "wires or a grey set on 11")
    if DC_ENERGY and not (CENTER_IMAGE and 0 < DC_ENERGY < 1):
        raise SystemExit("--dc_energy needs --center_image and a fraction between 0 and 1")
    FILTER = a.filter
    DOMAIN = a.domain
    if CENTER_IMAGE and (ENCODING != "amp" or DOMAIN != "pixel"):
        raise SystemExit("--center_image needs --encoding amp in the pixel domain")
    if DOMAIN == "dct" and (a.aux_spread or a.aux_smooth):
        raise SystemExit("--domain dct with the heatmap aux losses is not implemented")
    N_CLASSES = 32 if DATASET == "imagenet32" else 17 if DATASET == "flowers17c" else 10
    if DATASET in ("imagenet32", "cifar10c", "wmnistc", "caltech10c", "caltech10f", "caltech10fc", "caltech10fct", "flowers17c") + GREY64 and (N_QUBITS != 12 or INPUT != "pad"):
        raise SystemExit(f"{DATASET} needs --n_qubits 12 --input pad")
    if INPUT == "upsample" and N_QUBITS != 12:
        raise SystemExit("--input upsample needs --n_qubits 12")

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    (xtr, ytr), (xva, yva), (xte, yte) = load(a.n_train, a.n_val, a.seed)
    if a.patch_wires:
        if not permuted() or SIM != "fast":
            raise SystemExit("--patch_wires needs the Morton (or Gray-Morton) order and --sim fast")
        a.n_out = a.patch_wires
    N = 2 ** a.n_out
    dev = torch.device(a.device)
    readout = Readout(a.mode, N, N_CLASSES, pt_ratio=a.pt_ratio).to(dev)
    n_ang = 2 ** a.reupload_levels - 1
    n_blocks = math.ceil(n_ang / N_QUBITS)
    if n_blocks > a.depth or (n_blocks and ENCODING == "incoherent"):
        raise SystemExit("--reupload_levels needs ceil((2^L - 1) / n_qubits) <= depth and a coherent encoding")
    circuit = make_qnode(a.depth, a.n_out, a.ent, "cpu", n_blocks)
    n_anc = int(math.log2(a.channels))
    if a.fourier_bank > 1:                             # register wires of the mask bank, in front as channels' are
        if a.channels > 1 or a.fourier_mask or 2 ** int(math.log2(a.fourier_bank)) != a.fourier_bank:
            raise SystemExit("--fourier_bank needs a power of 2 and no --channels / --fourier_mask")
        n_anc = int(math.log2(a.fourier_bank))
    if a.channels > 1:                                 # ancilla channels: filter bank, then the main circuit
        if 2 ** n_anc != a.channels or SIM != "fast" or a.ent not in ("brickwall", "qcnn") or a.pyramid \
                or a.patch_wires or n_blocks or ENCODING == "incoherent" or a.aux_spread or a.aux_smooth:
            raise SystemExit("--channels needs a power of 2, --sim fast, brickwall or qcnn, coherent amplitudes, "
                             "and no pyramid / bag / re-upload / aux loss")
        q_ch = a.n_out if a.ent == "qcnn" else 0
        circuit = lambda amps, w, ang=None: fastsim.probs(
            fastsim.run_channels(amps, w, a.depth, a.bank, n_anc, q_ch), a.n_out)
    if a.patch_wires:                                  # P(b) = sum_patch |<b| V |content of the patch>|^2
        Pw = a.patch_wires
        circuit = lambda amps, w, ang=None: (fastsim.run(amps.reshape(-1, 2 ** Pw), w, a.depth).abs() ** 2
                                             ).reshape(len(amps), -1, 2 ** Pw).sum(1)
    if a.pyramid:                                      # 0.5 P_full(b) + 0.5 sum_patch P_bag(b | patch)
        K, nf = a.pyramid, N_QUBITS
        if not permuted() or SIM != "fast" or a.n_out > K:
            raise SystemExit("--pyramid needs Morton order, --sim fast and n_out <= K")

        def circuit(amps, w, ang=None):
            full = fastsim.probs(fastsim.run(amps, w[:, :nf], a.depth), a.n_out)
            bag = fastsim.probs(fastsim.run(amps.reshape(-1, 2 ** K), w[:, nf:], a.depth), a.n_out)
            return 0.5 * full + 0.5 * bag.reshape(len(amps), -1, 2 ** a.n_out).sum(1)
    n_rot = 3 if a.rot == "rzryrx" else 2
    if (a.mask_group > 0) + a.fourier_mask + (a.glimpse > 0) > 1:
        raise SystemExit("--mask_group, --fourier_mask and --glimpse are different masks: pick one")
    s_m, nc_m = side(), channels()
    planes_m = 2 ** N_QUBITS // (s_m * s_m)                 # colour planes in the state (a padded one included)
    fourier_m = (torch.tensor(morton_perm() if permuted() else np.arange(s_m * s_m), device=dev), s_m, planes_m,
                 a.fourier_phase, a.fourier_linear) if a.fourier_mask else None
    if a.fourier_phase and not (a.fourier_mask or a.fourier_bank > 1):
        raise SystemExit("--fourier_phase needs --fourier_mask or --fourier_bank")
    pix_f = torch.tensor(morton_perm() if permuted() else np.arange(s_m * s_m), device=dev)

    if a.optical:
        import optnet
        if a.n_out != 7 or a.fourier_bank > 1 or a.mask_group or a.glimpse or a.channels > 1 or SIM != "fast":
            raise SystemExit("--optical needs --n_out 7 (128 camera regions), --sim fast, and no bank / group mask / "
                             "glimpse / channels")
        a.depth = 0
        N_opt, n_rot_ = 2 * s_m, 3 if a.rot == "rzryrx" else 2
        Np_opt = N_opt // a.opt_res
        fr_opt = fastsim.fourier_rows(N_QUBITS, n_rot_, s_m, planes_m, a.fourier_phase) if a.fourier_mask else 0
        det_opt = optnet.detector_index(N_opt).to(dev)

        def plates(w):                                 # (L, N/r, N/r) plate angles packed after the mask rows
            return w[1 + fr_opt:].reshape(-1)[:a.optical * Np_opt * Np_opt].reshape(a.optical, Np_opt, Np_opt)

        KEEP = {}                                      # last batch: share of the plates' input light at the camera

        def optical_camera(amps, w):                   # amplitudes -> camera field (B, N, N), unnormalised
            st = amps.to(torch.complex64)
            if a.fourier_mask:
                _, kw = fastsim.split_masks(w, a.depth, 0, -1, fourier_m)
                st = fastsim.fourier_filter(st, kw["fmask"][1].to(st.device), kw["fmask"][0], kw["fmask"][2])
            KEEP["in"] = (st.abs() ** 2).sum(-1)
            ps = qs = None
            if torch.is_grad_enabled() and (a.opt_jitter or a.opt_pic_jitter):   # training: a misaligned bench
                def vec(m):
                    rr, th = np.random.uniform(0, m), np.random.uniform(0, 2 * math.pi)
                    return rr * math.sin(th), rr * math.cos(th)
                if a.opt_jitter:
                    ps = [tuple(v * a.opt_res * a.opt_os for v in vec(a.opt_jitter)) for _ in range(a.optical)]
                if a.opt_pic_jitter:
                    qs = vec(a.opt_pic_jitter)
            return optnet.forward(optnet.tile(st, pix_f, s_m, planes_m), plates(w), a.opt_z, a.opt_res, a.opt_pad,
                                  a.opt_os, ps, qs)

    def masks(w):                                      # circuit rows + run() keyword arguments of the soft mask
        return fastsim.split_masks(w, a.depth, a.mask_group, a.mask_at, fourier_m)
    if a.glimpse:
        pix_g = torch.tensor(morton_perm() if permuted() else np.arange(s_m * s_m), device=dev)
        pixc_g = torch.tensor(morton_perm(a.glimpse) if permuted() else np.arange(a.glimpse ** 2), device=dev)
        qc_g = (planes_m * a.glimpse ** 2).bit_length() - 1   # wires of the look register

        def glimpse(amps, w):                          # (states U D_j a (B, J, 2^n), weights (B, J)), unnormalised
            return fastsim.glimpse_run(amps, w, a.depth, qc_g, a.glimpse_depth, a.glimpse, pix_g, s_m, pixc_g,
                                       hier=HIER, qcnn=a.n_out if a.ent == "qcnn" else 0)
    if a.mask_group or a.fourier_mask or a.glimpse or a.fourier_bank > 1 or a.optical:   # conditional probabilities
        if SIM != "fast" or a.channels > 1 or a.patch_wires or a.pyramid or n_blocks or ENCODING == "incoherent" \
                or a.aux_spread or a.aux_smooth or a.ent not in ("brickwall", "hier", "qcnn") \
                or (a.mask_group and N_QUBITS % a.mask_group) or (a.fourier_mask and 2 ** N_QUBITS % (s_m * s_m)):
            raise SystemExit("--mask_group / --fourier_mask need --sim fast, brickwall / hier / qcnn, coherent amplitudes, no "
                             "channels / bag / pyramid / re-upload / aux loss, and a group size dividing n_qubits")
        if a.mask_at < 0:
            a.mask_at = a.depth // 3
        q_m = a.n_out if a.ent == "qcnn" else 0

        def circuit(amps, w, ang=None):
            if a.optical:                              # phase plates: light per camera region, conditional
                P = optnet.outcome_probs(optical_camera(amps, w), det_opt, os=a.opt_os)
                KEEP["ratio"] = P.sum(-1) / (KEEP["in"].clamp_min(1e-30) * a.opt_os ** 2)   # os^2: the samples
                # of each pixel all carry its amplitude, so the fine grid holds os^2 times the light
                return P / P.sum(-1, keepdim=True).clamp_min(1e-30)
            if a.fourier_bank > 1:
                st = fastsim.run_fbank(amps, w, a.depth, a.fourier_bank, pix_f, s_m, planes_m, a.fourier_phase, q_m,
                                       hier=HIER, mix=a.fourier_bank_mix, linear=a.fourier_linear)
                p2 = (st.abs() ** 2).sum(1) if a.fourier_bank_mix else st.abs() ** 2   # mixture: branches add
                P = p2.reshape(len(amps), 2 ** a.n_out, -1).sum(-1)
                return P / P.sum(-1, keepdim=True).clamp_min(1e-30)
            if a.glimpse:                              # mixture over glimpses: sum_j w_j P(out | U D_j a)
                S, wts = glimpse(amps, w)
                P = (wts[..., None] * fastsim.probs(S.reshape(-1, S.shape[-1]), a.n_out).reshape(*wts.shape, -1)).sum(1)
                return P / P.sum(-1, keepdim=True).clamp_min(1e-30)
            wc, kw = masks(w)
            P = fastsim.probs(fastsim.run(amps, wc, a.depth, hier=HIER, qcnn=q_m, **kw), a.n_out)
            return P / P.sum(-1, keepdim=True).clamp_min(1e-30)

        def mask_pass(x):                              # mean chance the light gets through the mask (shots cost 1/it)
            with torch.no_grad():
                out = []
                for i in range(0, len(x), 250):
                    xt = torch.tensor(x[i:i + 250], dtype=torch.float64, device=dev)
                    if a.optical:
                        out.append(optnet.camera_pixels(optical_camera(xt, w), a.opt_os).sum((1, 2)).cpu()
                                   / a.opt_os ** 2)                 # the fine grid holds os^2 times the light
                    elif a.fourier_bank > 1:
                        st = fastsim.run_fbank(xt, w, a.depth, a.fourier_bank, pix_f, s_m, planes_m, a.fourier_phase,
                                               q_m, hier=HIER, mix=a.fourier_bank_mix, linear=a.fourier_linear)
                        out.append((st.abs() ** 2).reshape(len(xt), -1).sum(-1).cpu())
                    elif a.glimpse:                    # the look register is amplitude-encoded too: divide by its norm
                        S, wts = glimpse(xt, w)
                        c2 = (fastsim.coarse_picture(xt.to(torch.complex64), pix_g, s_m, a.glimpse, pixc_g).abs() ** 2).sum(-1)
                        out.append(((wts * (S.abs() ** 2).sum(-1)).sum(1) / c2).cpu())
                    else:
                        wc, kw = masks(w)
                        out.append((fastsim.run(xt, wc, a.depth, hier=HIER, qcnn=q_m, **kw).abs() ** 2).sum(-1).cpu())
                return float(torch.cat(out).mean())
    w = a.init_scale * torch.randn(a.depth + 1, (a.patch_wires or N_QUBITS) + a.pyramid + n_anc, n_rot)
    if a.mask_group:                                   # mask angles near pi/4: m = 0.71 everywhere (a uniform mask is
        nm = fastsim.mask_rows(N_QUBITS, n_rot, a.mask_group)   # no mask after normalising) with full gradients
        w = torch.cat([w, math.pi / 4 + a.init_scale * torch.randn(nm, N_QUBITS, n_rot)])
    if a.fourier_mask:
        nm = fastsim.fourier_rows(N_QUBITS, n_rot, s_m, planes_m, a.fourier_phase)
        w0 = math.pi / 4 + a.init_scale * torch.randn(nm, N_QUBITS, n_rot)
        if a.fourier_phase:                            # phases start near 0 (the real mask), amplitudes near pi/4
            flat = w0.reshape(-1)
            flat[planes_m + s_m * s_m:] = a.init_scale * torch.randn(len(flat) - planes_m - s_m * s_m)
        w = torch.cat([w, w0])
    if a.fourier_bank > 1:                             # K masks: amplitudes spread around pi/4, phases near 0, so the
        per = planes_m + s_m * s_m * (2 if a.fourier_phase else 1)   # filters start apart
        nm = fastsim.fbank_rows(N_QUBITS + n_anc, n_rot, a.fourier_bank, s_m, planes_m, a.fourier_phase)
        w0 = torch.zeros(nm * (N_QUBITS + n_anc) * n_rot)
        f = w0[:a.fourier_bank * per].view(a.fourier_bank, per)
        f[:, :planes_m + s_m * s_m] = math.pi / 4 + 0.3 * torch.randn(a.fourier_bank, planes_m + s_m * s_m)
        f[:, planes_m + s_m * s_m:] = 0.3 * torch.randn(a.fourier_bank, per - planes_m - s_m * s_m)
        w = torch.cat([w, w0.view(nm, N_QUBITS + n_anc, n_rot)])
    if a.optical:                                      # plate angles N(0, 1) after the mask rows
        nm = -(-a.optical * Np_opt * Np_opt // (w.shape[1] * n_rot))
        w = torch.cat([w, torch.randn(nm, w.shape[1], n_rot)])
    if a.glimpse:                                      # V near the identity (looks where the light is), window flat
        nm = fastsim.glimpse_rows(N_QUBITS, n_rot, a.glimpse_depth, a.glimpse)
        w = torch.cat([w, a.init_scale * torch.randn(a.glimpse_depth + 1, N_QUBITS, n_rot),
                       math.pi / 4 + a.init_scale * torch.randn(nm - a.glimpse_depth - 1, N_QUBITS, n_rot)])
    if a.channels > 1:                                 # bank rows: wide random angles, so the channels start apart
        w = torch.cat([w, a.bank_init * torch.randn(a.channels * (a.bank + 1), N_QUBITS + n_anc, n_rot)])
    if a.init_from:                                    # continue from a finished run's (or checkpoint's) weights
        w0 = torch.tensor(json.loads(pathlib.Path(a.init_from).read_text())["weights"], dtype=w.dtype)
        if w0.shape != w.shape:
            raise SystemExit(f"--init_from weights {tuple(w0.shape)} do not fit this circuit {tuple(w.shape)}")
        w = w0
    w = torch.nn.Parameter(w.to(dev))
    head_params = list(readout.parameters())                 # non-empty only for the linear_head control
    opt = torch.optim.Adam([w] + head_params, lr=a.lr)
    steps_per_epoch = math.ceil(a.n_train / a.batch)
    total = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / (a.warmup_epochs * steps_per_epoch))
        * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * s / total))))

    eye = torch.eye(2 ** N_QUBITS, dtype=torch.float64, device=dev)

    def probs(x):
        xt = torch.tensor(x, dtype=torch.float64, device=dev)
        if n_blocks:
            ang = F.pad(tree_angles(xt, a.reupload_levels), (0, n_blocks * N_QUBITS - n_ang))
            return circuit(xt, w, ang.reshape(len(xt), n_blocks, N_QUBITS)).float()
        if ENCODING != "incoherent":
            return circuit(xt, w).float()
        # P_b = sum_x p(x) P_b(|x>)  (no interference between pixels). P_b(|x>) for every basis input x
        # comes from ONE broadcast circuit call on the d basis states (rows of the identity).
        Mt = circuit(eye, w)                                   # (d, N)
        return (xt @ Mt.to(xt.dtype)).float()

    if a.mode == "bss_group_ref":
        # reference baseline: mean group probability of the UNTRAINED circuit over (up to) 2000 training images
        with torch.no_grad():
            P0 = torch.cat([probs(xtr[i:i + 500]) for i in range(0, min(2000, a.n_train), 500)])
            b = readout.bssg.marginalize(P0).mean(0)
        readout.bssg.set_baseline(b)
        print("reference baseline x G:", [round(v, 2) for v in (b * N_CLASSES).tolist()], flush=True)

    if a.att_spread and not (a.mask_group or a.fourier_mask):
        raise SystemExit("--att_spread needs --mask_group or --fourier_mask")
    if a.aux_spread or a.aux_smooth or a.att_spread:           # exact evidence maps in training (owner 2026-10-30)
        if a.pyramid or a.patch_wires or SIM != "fast" or a.mode not in ("bhr", "bhr_single"):
            raise SystemExit("--aux_spread / --aux_smooth need a plain fast-sim circuit with a bhr readout")
        from bssg import class_groups
        groups_aux = class_groups(a.mode, N, N_CLASSES)
        outc = torch.arange(2 ** N_QUBITS, device=dev) >> (N_QUBITS - a.n_out)
        Wcls = torch.stack([torch.isin(outc, torch.tensor(g, device=dev)) for g in groups_aux]).to(torch.complex64)
        b_aux = torch.tensor([len(g) / N for g in groups_aux], device=dev)
        s_, nc_ = side(), channels()
        pix = torch.tensor(morton_perm() if permuted() else np.arange(s_ * s_), device=dev)
        rr, cc = (pix // s_).float(), (pix % s_).float()       # pixel row / column of each wire-ordered amplitude

    def aux_loss(xb, yb):
        """Heatmap terms on the true class's exact evidence e(x) = a(x) Re((U^dag Pi_g U a)(x)) - b_g a(x)^2, colour
        planes summed: spread = spatial variance (side^2 units) of the normalised positive evidence (small = one compact
        blob); roughness = sum of squared differences between 4-neighbour pixels / sum of e^2 (small = smooth).
        att_spread: the same spread of a soft mask's attention map a(x) Re((A^dag A a)(x)) / keep."""
        at = torch.tensor(xb, dtype=torch.float32, device=dev)
        yt = torch.tensor(yb, dtype=torch.long, device=dev)
        q_ = a.n_out if a.ent == "qcnn" else 0

        def spread(m):                                     # positional variance of the normalised positive part
            m = F.relu(m)
            m = m / m.sum(1, keepdim=True).clamp_min(1e-12)
            cr, ccen = (m * rr).sum(1, keepdim=True), (m * cc).sum(1, keepdim=True)
            return ((m * ((rr - cr) ** 2 + (cc - ccen) ** 2)).sum(1) / s_ ** 2).mean()
        out = 0.0
        if a.att_spread:
            wc, kw = masks(w)
            st = fastsim.run(at, wc, a.depth, hier=HIER, qcnn=q_, **kw)
            keep = (st.abs() ** 2).sum(1, keepdim=True)
            att = at * fastsim.run_inverse(st, wc, a.depth, hier=HIER, qcnn=q_, **kw).real / keep
            out = out + a.att_spread * spread(att[:, :nc_ * s_ * s_].reshape(len(at), nc_, s_ * s_).sum(1))
        if not (a.aux_spread or a.aux_smooth):
            return out
        back = fastsim.run_inverse(fastsim.run(at, w, a.depth, hier=HIER, qcnn=q_) * Wcls[yt], w, a.depth, hier=HIER, qcnn=q_)
        e = at * back.real - b_aux[yt, None] * at ** 2
        e = e[:, :nc_ * s_ * s_].reshape(len(at), nc_, s_ * s_).sum(1)          # (B, pixels in wire order)
        if a.aux_spread:
            out = out + a.aux_spread * spread(e)
        if a.aux_smooth:
            img = torch.zeros_like(e).index_copy(1, pix, e).reshape(len(e), s_, s_)   # pixel grid
            rough = ((img[:, 1:] - img[:, :-1]) ** 2).sum((1, 2)) + ((img[:, :, 1:] - img[:, :, :-1]) ** 2).sum((1, 2))
            out = out + a.aux_smooth * (rough / (img ** 2).sum((1, 2)).clamp_min(1e-20)).mean()
        return out

    def accuracy(x, y, S=None):
        with torch.no_grad():
            out = []
            for i in range(0, len(x), 500):
                P = probs(x[i:i + 500])
                if S is not None:
                    P = torch.tensor(photon_hist(P.cpu().numpy(), S, rng), dtype=torch.float32, device=dev)
                out.append(readout(P).argmax(-1).cpu().numpy())
            return float(np.mean(np.concatenate(out) == y))

    OUT.mkdir(exist_ok=True)
    fname = f"pilot_{'' if DATASET == 'fmnist' else DATASET + '_'}{'' if a.epochs == 10 else f'e{a.epochs}_'}{a.mode}_{ENCODING}_pt{a.pt_ratio:g}_q{N_QUBITS}{'' if INPUT == 'pad' else 'up'}_d{a.depth}{'' if a.rot == 'ryrx' else 'rz'}{dict(rowcol='', morton='mo', graymorton='gm', hilbert='hb')[a.order]}{f'_ru{a.reupload_levels}' if a.reupload_levels else ''}{'_ctr' if a.center else ''}{'_ci' if a.center_image else ''}{f'{a.dc_energy:g}' if a.dc_energy else ''}{f'mp{a.mean_plane:g}' if a.mean_plane else ''}{'_sob' if a.filter == 'sobel' else ''}{'_dct' if a.domain == 'dct' else ''}{f'_aug{a.aug_shift}' if a.aug_shift else ''}{f'_crop{a.aug_crop:g}' if a.aug_crop else ''}{'_aff' if a.aug_affine else ''}{'_bal' if a.balanced else ''}{f'_spread{a.aux_spread:g}' if a.aux_spread else ''}{f'_smooth{a.aux_smooth:g}' if a.aux_smooth else ''}{f'_lrf{a.lr_floor:g}' if a.lr_floor else ''}{f'_bag{a.patch_wires}' if a.patch_wires else ''}{f'_pyr{a.pyramid}' if a.pyramid else ''}{f'_ch{a.channels}b{a.bank}' if a.channels > 1 else ''}{f'_loc{a.local}e{a.local_every}' if a.local else ''}{'c' if a.local_colour else ''}{'_cont' if a.init_from else ''}{f'_b{a.batch}' if a.batch != 64 else ''}{f'_mk{a.mask_group}at{a.mask_at}' if a.mask_group else ''}{f'_opt{a.optical}z{a.opt_z:g}r{a.opt_res}' if a.optical else ''}{f'os{a.opt_os}' if a.optical and a.opt_os > 1 else ''}{f'k{a.opt_keep:g}' if a.optical and a.opt_keep else ''}{f'j{a.opt_jitter:g}p{a.opt_pic_jitter:g}' if a.optical and (a.opt_jitter or a.opt_pic_jitter) else ''}{'_fm' if a.fourier_mask else ''}{f'_fb{a.fourier_bank}' if a.fourier_bank > 1 else ''}{'mix' if a.fourier_bank_mix else ''}{'L' if a.fourier_linear else ''}{'p' if a.fourier_phase else ''}{f'_gl{a.glimpse}v{a.glimpse_depth}' if a.glimpse else ''}{f'_as{a.att_spread:g}' if a.att_spread else ''}{f'_{a.tag}' if a.tag else ''}_o{a.n_out}_{a.ent}_i{a.init_scale:g}_s{a.seed}.json"
    t0 = time.time()
    best = (-1.0, None)
    history = []
    step = 0
    for ep in range(a.epochs):
        perm = rng.permutation(a.n_train)
        if a.balanced:                                         # every class equally likely (unbalanced sets, Caltech)
            wb = 1.0 / np.bincount(ytr)[ytr]
            perm = rng.choice(a.n_train, a.n_train, p=wb / wb.sum())
        for i in range(0, a.n_train, a.batch):
            idx = perm[i:i + a.batch]
            xb = xtr[idx]
            if a.aug_crop:                                     # random crops of the raw pictures, then encoded
                xb = RAW["encode"](random_crops(RAW["train"][idx], rng, a.aug_crop))
            if a.aug_shift:                                    # random translations, zero fill
                xb = shift_images(xb, rng.integers(-a.aug_shift, a.aug_shift + 1, size=(len(idx), 2)))
            if a.aug_affine:
                xb = affine_images(xb, *random_affine(len(idx), rng, a))
            opt.zero_grad()
            mb = a.micro or len(idx)
            for j in range(0, len(idx), mb):                   # micro-batches: gradients add up to the batch's
                q = readout(probs(xb[j:j + mb]))
                loss = F.nll_loss(torch.log(q + 1e-12), torch.tensor(ytr[idx[j:j + mb]], dtype=torch.long, device=dev))
                if a.aux_spread or a.aux_smooth or a.att_spread:
                    loss = loss + aux_loss(xb[j:j + mb], ytr[idx[j:j + mb]])
                if a.optical and a.opt_keep:                   # keep the light inside the plates' aperture
                    loss = loss + a.opt_keep * (1 - KEEP["ratio"].mean())
                (loss * len(q) / len(idx)).backward()
            opt.step()
            sched.step()
            step += 1
            if a.time_only and step == 20:
                print(f"20 steps: {time.time() - t0:.1f} s  -> ~{(time.time() - t0) / 20 * total / 60:.1f} min for {total} steps")
                return 0
        va_acc = accuracy(xva, yva)
        tr_acc = accuracy(xtr[:2000], ytr[:2000])           # train accuracy on a fixed 2,000-image subset
        history.append(dict(epoch=ep + 1, loss=float(loss.item()), train_acc=tr_acc, val_acc=va_acc))
        if va_acc > best[0]:
            best = (va_acc, w.detach().clone(), {k: v.detach().clone() for k, v in readout.state_dict().items()})
        print(f"epoch {ep + 1}: loss {loss.item():.3f} train {tr_acc:.4f} val {va_acc:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if a.checkpoint_every and (ep + 1) % a.checkpoint_every == 0:     # best-so-far weights, for heatmaps mid-run
            ck = dict(mode=a.mode, pt_ratio=a.pt_ratio, seed=a.seed, depth=a.depth, n_out=a.n_out, ent=a.ent,
                      channels=a.channels, bank=a.bank if a.channels > 1 else 0, init_from=a.init_from,
                      local=a.local, local_every=a.local_every if a.local else 0,
                      local_colour=HIER[2] if a.local else 0,
                      mask_group=a.mask_group, mask_at=a.mask_at if a.mask_group else -1,
                      fourier_mask=a.fourier_mask, glimpse=a.glimpse, glimpse_depth=a.glimpse_depth,
                      att_spread=a.att_spread,
                      dataset=DATASET, order=a.order,
                      rot=a.rot, epoch=ep + 1, val_best=best[0], test_exact=float("nan"), history=history,
                      weights=best[1].cpu().tolist())
            (OUT / ("checkpoint_" + fname)).write_text(json.dumps(ck))

    with torch.no_grad():
        w.copy_(best[1])
    readout.load_state_dict(best[2])                       # restores the linear head (control) if there is one
    res = dict(mode=a.mode, seed=a.seed, depth=a.depth, n_out=a.n_out, ent=a.ent, pt_ratio=a.pt_ratio,
               n_qubits=N_QUBITS, input=INPUT,
               init_scale=a.init_scale, device=a.device,
               n_params=int((w[:a.depth + 1].numel() if a.ent not in ("hier", "qcnn") and not a.local else
                             fastsim.n_angles(a.depth, N_QUBITS + n_anc, w.shape[2], HIER, a.n_out if a.ent == "qcnn" else 0))
                            + ((N_QUBITS // a.mask_group) * 2 ** a.mask_group if a.mask_group else 0)
                            + (fastsim.fourier_params(s_m, nc_m, a.fourier_phase) if a.fourier_mask else 0)
                            + (a.optical * Np_opt * Np_opt - w[:1].numel() if a.optical else 0)
                            + (a.fourier_bank * fastsim.fourier_params(s_m, nc_m, a.fourier_phase) if a.fourier_bank > 1 else 0)
                            + (fastsim.glimpse_params(qc_g, n_rot, a.glimpse_depth, a.glimpse) if a.glimpse else 0)
                            + (a.channels * (a.bank + 1) * N_QUBITS * w.shape[2] if a.channels > 1 else 0)),
               channels=a.channels, bank=a.bank if a.channels > 1 else 0, init_from=a.init_from, local=a.local,
               local_every=a.local_every if a.local else 0, local_colour=HIER[2] if a.local else 0,
               mask_group=a.mask_group, mask_at=a.mask_at if a.mask_group else -1,
               fourier_mask=a.fourier_mask, fourier_phase=a.fourier_phase, fourier_bank=a.fourier_bank,
               fourier_bank_mix=a.fourier_bank_mix, fourier_linear=a.fourier_linear, glimpse=a.glimpse,
               glimpse_depth=a.glimpse_depth, att_spread=a.att_spread,
               mask_pass=mask_pass(xte[:2000]) if a.mask_group or a.fourier_mask or a.glimpse or a.fourier_bank > 1 else None, n_head_params=int(sum(p.numel() for p in head_params)), val_best=best[0], test_exact=accuracy(xte, yte),
               test_photons={S: accuracy(xte, yte, S) for S in PHOTONS} if a.full_eval else None,
               train_s=time.time() - t0,
               encoding=ENCODING)
    # class information per photon: mean probability that a photon lands in the true class's group (or state)
    from bssg import make_group_map, class_groups
    gm = class_groups(a.mode, N, N_CLASSES)
    res["dataset"] = DATASET
    res["rot"] = a.rot
    res["order"] = a.order
    res["reupload_levels"] = a.reupload_levels
    res["sim"] = SIM
    from sklearn.metrics import roc_auc_score                  # one-vs-rest ROC-AUC of the readout output p
    from sklearn.preprocessing import label_binarize
    with torch.no_grad():
        pte = torch.cat([readout(probs(xte[i:i + 500])) for i in range(0, len(xte), 500)]).cpu().numpy()
    Yte = label_binarize(yte, classes=list(range(N_CLASSES)))
    res["roc_auc_macro"] = float(roc_auc_score(Yte, pte, average="macro"))
    res["roc_auc_micro"] = float(roc_auc_score(Yte, pte, average="micro"))
    res["center"] = CENTER
    res["center_image"] = CENTER_IMAGE
    res["dc_energy"] = DC_ENERGY
    res["mean_plane"] = MEAN_PLANE
    res["filter"] = FILTER
    res["domain"] = DOMAIN
    res["aug_shift"] = a.aug_shift
    def acc_on(transform):                             # transformed test set, 2,000 pictures at a time (low RAM)
        return float(np.mean([accuracy(transform(xte[i:i + 2000]), yte[i:i + 2000]) * len(yte[i:i + 2000])
                              for i in range(0, len(xte), 2000)]) * len(range(0, len(xte), 2000)) / len(xte))
    res["test_shifted_right"] = {k: acc_on(lambda x, k=k: shift_images(x, np.tile([0, k], (len(x), 1))))
                                 for k in (1, 2, 3, 4)} if a.full_eval else None
    res["aug_affine"] = a.aug_affine
    res["aug_crop"] = a.aug_crop
    res.update(optical=a.optical, opt_z=a.opt_z, opt_res=a.opt_res, opt_pad=a.opt_pad, opt_os=a.opt_os,
               opt_keep=a.opt_keep, opt_jitter=a.opt_jitter, opt_pic_jitter=a.opt_pic_jitter)
    if a.optical:                                      # share of the plates' input light reaching the camera, test set
        with torch.no_grad():
            kr = []
            for i in range(0, len(xte), 500):
                circuit(torch.tensor(xte[i:i + 500], dtype=torch.float64, device=dev), w)
                kr.append(KEEP["ratio"].cpu())
            res["plates_keep"] = float(torch.cat(kr).mean())
    res["patch_wires"] = a.patch_wires
    res["pyramid"] = a.pyramid
    res["test_affine"] = {name: acc_on(lambda x, t=t: affine_images(x, *t)) for name, t in
                          (("rot+15", (15, 1, 0, 0)), ("rot-15", (-15, 1, 0, 0)),
                           ("scale0.75", (0, 0.75, 0, 0)), ("scale1.25", (0, 1.25, 0, 0)))} if a.full_eval else None
    res["history"] = history
    res["weights"] = w.detach().cpu().tolist()                 # best-on-val circuit angles (for hardware runs)
    res["head_state"] = {k: v.cpu().tolist() for k, v in readout.state_dict().items()}
    res["warmup_epochs"], res["epochs"], res["n_train"] = a.warmup_epochs, a.epochs, a.n_train
    with torch.no_grad():
        pt = []
        for i in range(0, len(xte), 500):
            P = probs(xte[i:i + 500]).cpu().numpy()
            yy = yte[i:i + 500]
            if a.mode not in ("raw_renorm", "bss_renorm"):        # every grouped readout (bhr included)
                pt.append(np.array([P[j, gm[c]].sum() for j, c in enumerate(yy)]))
            else:
                pt.append(P[np.arange(len(yy)), yy])
    res["p_true_group"] = float(np.concatenate(pt).mean())
    if a.full_eval:                                    # direct-imaging baseline at the same photon budgets
        Wc, bc = train_ccd(xtr, ytr, seed=a.seed)

        def ccd_acc(S):
            I = intensity(xte)
            H = photon_hist(I, S, rng) if S is not None else I
            return float(((torch.tensor(H * xte.shape[1], dtype=torch.float32) @ Wc + bc).argmax(-1).numpy() == yte).mean())
        res["ccd_exact"] = ccd_acc(None)
        res["ccd_photons"] = {S: ccd_acc(S) for S in PHOTONS}
    print(json.dumps(res, indent=1))
    (OUT / fname).write_text(json.dumps(res, indent=1))
    (OUT / ("checkpoint_" + fname)).unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
