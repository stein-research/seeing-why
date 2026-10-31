#!/usr/bin/env python3
"""Attention and evidence inside the annotated object, Caltech-10 centre crop (`caltech10fc`, 300 test pictures).

Caltech-101 ships one annotated object per picture (Annotations.tar, CC BY 4.0: `box_coord` = [top bottom left right],
`obj_contour` = 2 x N points relative to the box's top-left corner; as show_annotation.m draws them). The test
pictures are found again by replaying prepare_caltech101.py --all --crop (same seed, same order) and checked pixel for
pixel against data/caltech10fc.npz; the box and contour go through the same centre crop and 32x32 resize, as
fractional cover per pixel (8 x 8 samples a pixel).

Maps per picture, colour planes summed per pixel, plane 4 (the reference) left out:
- light: the encoded picture's |a(x)|^2 - what the plain circuit's attention is (no mask: A^dag A = 1);
- attention: a(x) (A^dag A a)(x) / keep, each pixel's share of the light passing the mask(s) (= light without one);
- evidence: the true class's evidence map (x M_g x - b_g x A^dag A x) / keep, positive part.
Reported: share inside the box and inside the contour, each divided by the region's area share (1 = no preference),
and the pointing game (the map's peak pixel inside the box / contour). Signed maps: the share is of the map's sum over
the picture planes; evidence uses its positive part.

Run:  python src/attention_in_box.py results/<run>.json [...] [--draw figures/out.png]
"""
import io
import json
import pathlib
import sys
import tarfile

import numpy as np
import scipy.io
import torch
from matplotlib.path import Path as Poly
from PIL import Image

import fastsim
import optnet
import pilot_fmnist as pf
import prepare_caltech101
from bssg import class_groups
from prepare_caltech101 import CLASSES, SRC, centre_crop

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
TAR = HERE / "data" / "caltech101" / "x" / "caltech-101" / "Annotations.tar"
ANN = {"airplanes": "Airplanes_Side_2", "Motorbikes": "Motorbikes_16", "Faces": "Faces_2"}
DEV = "cpu"                                                  # light work; the GPU stays with the trainings
S, SUB = 32, 8


def test_regions(trim=0.0):
    """(300, 32, 32) box and contour cover in [0, 1], and the raw crops, in caltech10fc's (trim 0) or caltech10fct's
    (trim 0.04: the margin cut first, as prepare_caltech101.py --trim) test order."""
    prepare_caltech101.TRIM = trim
    rng = np.random.default_rng(0)
    tar = tarfile.open(TAR)
    boxes, conts, crops = [], [], []
    g = (np.arange(S * SUB) + 0.5) / SUB                     # sample points in 32-pixel units
    gx, gy = np.meshgrid(g, g)
    for name in CLASSES:
        files = sorted((SRC / name).glob("*.jpg"))
        pick = rng.permutation(len(files))
        for i in pick[:30]:
            f = files[i]
            im = Image.open(f).convert("RGB")
            crops.append(centre_crop(im))
            w, h = im.size
            tx, ty = round(w * trim), round(h * trim)                 # the trimmed margin (0 without --trim)
            w, h = w - 2 * tx, h - 2 * ty
            k = min(w, h)
            ox, oy = (w - k) // 2 + tx, (h - k) // 2 + ty
            m = scipy.io.loadmat(io.BytesIO(tar.extractfile(
                f"Annotations/{ANN.get(name, name)}/annotation_{f.stem.split('_')[1]}.mat").read()))
            top, bot, left, right = m["box_coord"].ravel().astype(float)
            to32 = lambda px, py: ((px - 1 - ox) * S / k, (py - 1 - oy) * S / k)   # MATLAB 1-based -> crop grid
            x0, y0 = to32(left, top)
            x1, y1 = to32(right, bot)
            box = (gx >= x0) & (gx <= x1) & (gy >= y0) & (gy <= y1)
            cx, cy = to32(m["obj_contour"][0] + left, m["obj_contour"][1] + top)
            cont = Poly(np.c_[cx, cy]).contains_points(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
            pool = lambda r: r.reshape(S, SUB, S, SUB).mean((1, 3))
            boxes.append(pool(box))
            conts.append(pool(cont))
    return np.array(boxes), np.array(conts), np.array(crops)


def maps(run, xte, yte, signed=False):
    """light, attention and positive (signed: net) true-class evidence per pixel: (3, n, 32, 32)."""
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")
    fb = run.get("fourier_bank", 0) if run.get("fourier_bank", 0) > 1 else 0
    n_anc = int(np.log2(fb)) if fb else 0
    pf.N_QUBITS = n = len(run["weights"][0]) - n_anc
    nt, no, depth = n + n_anc, run["n_out"], run["depth"]
    s, nc = pf.side(), pf.channels()
    perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
    x = xte.astype(np.float64).transpose(0, 3, 1, 2).reshape(len(xte), -1, s * s)[:, :, perm].reshape(len(xte), -1)
    if run.get("center_image"):
        x = pf.centre_planes(x, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = np.pad(x, ((0, 0), (0, 2 ** n - x.shape[1])))
    groups = class_groups(run["mode"], 2 ** no, 10)
    outc = np.arange(2 ** nt) >> (nt - no)
    W = torch.tensor(np.stack([np.isin(outc, g) for g in groups]), dtype=torch.complex64, device=DEV)
    b = np.array([len(g) / 2 ** no for g in groups])
    w = torch.tensor(run["weights"], dtype=torch.float32, device=DEV)
    a = torch.tensor(x, dtype=torch.complex64, device=DEV)
    if run.get("fourier_bank_mix"):
        raise SystemExit("the mixture control is not mapped")
    with torch.no_grad():
        if run.get("optical"):                                       # phase plates: camera regions are the outcomes
            cam, back, det, os_ = optnet.from_run(run, perm, DEV, torch.complex64)
            det_f = det.repeat_interleave(os_, 0).repeat_interleave(os_, 1).reshape(-1)
            W = torch.stack([torch.isin(det_f, torch.tensor(g)) for g in groups]).to(torch.complex64)
            phi = torch.cat([cam(a[i:i + 50]).reshape(len(a[i:i + 50]), -1) for i in range(0, len(a), 50)])
            Ui = lambda v: torch.cat([back(v[i:i + 50].reshape(len(v[i:i + 50]), *(det.shape[0] * os_,) * 2))
                                      for i in range(0, len(v), 50)])
        elif fb:
            fbk = (fb, torch.tensor(perm, device=DEV), s, 2 ** n // (s * s), run.get("fourier_phase", False))
            phi = fastsim.run_fbank(a, w, depth, *fbk)
            Ui = lambda v: fastsim.run_fbank_inverse(v, w, depth, *fbk)
        else:
            fourier = (torch.tensor(perm, device=DEV), s, 2 ** n // (s * s), run.get("fourier_phase", False),
                       run.get("fourier_linear", False)) if run.get("fourier_mask") else None
            wc, kw = fastsim.split_masks(w, depth, 0, -1, fourier)
            phi = fastsim.run(a, wc, depth, **kw)
            Ui = lambda v: fastsim.run_inverse(v, wc, depth, **kw)
        keep = (phi.abs() ** 2).sum(-1, keepdim=True)
        z_all = Ui(phi).real.numpy()
        z_true = Ui(phi * W[torch.tensor(yte)]).real.numpy()
    keep = keep.numpy().real
    att = x * z_all / keep
    ev = (x * z_true - b[yte, None] * x * z_all) / keep
    ev = ev if signed else np.maximum(ev, 0)

    def per_pixel(v):
        g = np.empty((len(v), nc, s * s))
        g[:, :, perm] = v[:, :nc * s * s].reshape(len(v), nc, s * s)
        return g.sum(1).reshape(len(v), s, s)

    return np.stack([per_pixel(x ** 2), per_pixel(att), per_pixel(ev)])


def scores(M, R):
    """share of each map in region R / R's area share, and the pointing game; means over pictures whose region is at
    least one pixel inside the crop (a few objects lie outside the centre square)."""
    ok = R.reshape(len(R), -1).sum(1) >= 1
    M, R = M[ok], R[ok]
    tot = M.reshape(len(M), -1).sum(1)
    share = (M * R).reshape(len(M), -1).sum(1) / tot
    area = R.reshape(len(R), -1).mean(1)
    peak = M.reshape(len(M), -1).argmax(1)
    hit = R.reshape(len(R), -1)[np.arange(len(R)), peak] > 0.5
    return share.mean(), (share / area).mean(), hit.mean()


def main():
    args = sys.argv[1:]
    draw = None
    if "--draw" in args:
        i = args.index("--draw")
        draw = args[i + 1]
        del args[i:i + 2]
    sets = {json.loads(pathlib.Path(f).read_text())["dataset"] for f in args}
    if len(sets) > 1:
        raise SystemExit("give runs of one set at a time (caltech10fc or caltech10fct)")
    name = sets.pop() if sets else "caltech10fc"
    box, cont, crops = test_regions(0.04 if name == "caltech10fct" else 0.0)
    d = np.load(HERE / "data" / f"{name}.npz")
    xte, yte = d["x_val"], d["y_val"]
    assert np.array_equal(crops, xte), f"replayed crops differ from {name}.npz"
    print(f"300 test pictures replayed and matched; area share: box {box.mean():.3f}, contour {cont.mean():.3f}")
    if draw:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 10, figsize=(20, 4.4))
        for c in range(10):
            j = c * 30
            for r, R in enumerate((box, cont)):
                ax[r, c].imshow(xte[j])
                ax[r, c].contour(R[j], levels=[0.5], colors="yellow" if r == 0 else "red", linewidths=1.5)
                ax[r, c].axis("off")
            ax[0, c].set_title(CLASSES[c], fontsize=9)
        fig.tight_layout()
        fig.savefig(draw, dpi=80)
        print("wrote", draw)
    print("run | map | box: share, /area, pointing | contour: share, /area, pointing")
    for f in args:
        run = json.loads(pathlib.Path(f).read_text())
        out = maps(run, xte, yte)
        for name, M in zip(("light", "attention", "evidence+"), out):
            sb, rb, pb = scores(M, box)
            sc, rc, pc = scores(M, cont)
            print(f"{pathlib.Path(f).stem} | {name} | {sb:.3f} {rb:.2f} {pb:.2f} | {sc:.3f} {rc:.2f} {pc:.2f}")


if __name__ == "__main__":
    main()
