#!/usr/bin/env python3
"""Deletion test with baselines (referee 2026-10-31, item 19): do the exact evidence maps rank pixels better than the
usual alternatives? Same deletion as deletion_meanplane.py (a pixel's contrast set to zero in R, G, B, renormalised,
classified again), with the pixels ranked by:
  evidence   - the exact evidence for the true class (ours; Integrated Gradients from a zero baseline);
  gradient   - the plain gradient of the true class's probability with respect to the pixel's amplitudes (summed
               over the colour planes), highest first;
  |gradient| - its magnitude, highest first (the classic saliency map);
  occlusion  - the drop of the true class's probability when that one pixel alone is deleted, largest first
               (1,024 extra runs per picture);
  grad x input      - the pixel's amplitudes times the gradient of P(true class), summed over the colour planes
                      (added 2026-11-03 after a referee review: it carries the amplitude factor the plain gradient
                      lacks; with the transmission held fixed it is exactly twice the evidence);
  grad x input, log p - the same with the gradient of the readout's log p-hat(true class), the trained output;
  random     - three draws.
Full optical model (row D) and qubit model (D'), seed 0, the 300 Caltech-10 test pictures.
Run:  python src/deletion_baselines.py   -> numbers, results/deletion_baselines.json
"""
import json
import pathlib

import numpy as np
import torch

import fastsim
import optnet
import pilot_fmnist as pf
from attention_in_box import maps as pixel_maps
from bssg import Readout, class_groups

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FRACS = [0.0, 0.02, 0.05, 0.1, 0.2]
C = "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_"
RUNS = {"plates (row D)": C + "d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s0.json",
        "qubits (row D')": C + "d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json"}


def model(run):
    """(encoded pictures, pixel of each wire, P(true class | amplitudes) as a torch function, classify)"""
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    s, n, no = 32, pf.N_QUBITS, run["n_out"]
    perm = pf.morton_perm()
    d = np.load(HERE / "data" / f"{run['dataset']}.npz")
    X, Y = d["x_val"], d["y_val"]
    v = X.astype(float).transpose(0, 3, 1, 2).reshape(len(X), 3, -1)[:, :, perm].reshape(len(X), -1)
    enc = np.pad(pf.centre_planes(v, 0, run["mean_plane"]), ((0, 0), (0, 2 ** n - 4 * s * s)))
    groups = class_groups(run["mode"], 2 ** no, 10)
    G = torch.tensor(np.stack([np.isin(np.arange(2 ** no), g) for g in groups]), dtype=torch.float32, device=DEV)
    readout = Readout(run["mode"], 2 ** no, 10, pt_ratio=run["pt_ratio"]).to(DEV)
    if run.get("optical"):
        cam, _, det, os_ = optnet.from_run(run, perm, DEV, torch.complex64)
        outcome = lambda a: optnet.outcome_probs(cam(a), det, os=os_)
    else:
        w = torch.tensor(run["weights"], dtype=torch.float32, device=DEV)
        fourier = (torch.tensor(perm, device=DEV), s, 2 ** n // (s * s), run.get("fourier_phase", False),
                   run.get("fourier_linear", False))
        wc, kw = fastsim.split_masks(w, run["depth"], 0, -1, fourier)
        outcome = lambda a: fastsim.probs(fastsim.run(a, wc, run["depth"], **kw), no)

    def probs(e):                                              # (B, 2^n) real torch -> group probabilities (B, 10)
        a = e / e.norm(dim=1, keepdim=True)
        P = outcome(a.to(torch.complex64)).float()
        P = P / P.sum(1, keepdim=True)
        return P @ G.T, P

    def classify(e):
        with torch.no_grad():
            out = [readout(probs(torch.tensor(e[i:i + 100], dtype=torch.float32, device=DEV))[1]) for i in range(0, len(e), 100)]
        return torch.cat(out).argmax(1).cpu().numpy()
    return X, Y, enc, np.asarray(perm), probs, classify, readout


def main():
    out = {}
    rng = np.random.default_rng(0)
    for name, f in RUNS.items():
        run = json.loads((HERE / "results" / f).read_text())
        X, Y, enc, pix_of_wire, probs, classify, readout = model(run)
        s2 = 1024
        ranks = {"evidence": pixel_maps(run, X, Y, signed=True)[2].reshape(len(X), -1)}
        grad = np.zeros((len(X), s2))
        gxi = np.zeros((len(X), s2))
        gxl = np.zeros((len(X), s2))
        occ = np.zeros((len(X), s2))
        yt = torch.tensor(Y, device=DEV)
        for i in range(len(X)):
            e = torch.tensor(enc[i:i + 1], dtype=torch.float32, device=DEV, requires_grad=True)
            pg, _ = probs(e)
            pg[0, yt[i]].backward()
            gw = e.grad[0, :3 * s2].reshape(3, s2).sum(0).cpu().numpy()        # wire order within a plane
            grad[i, pix_of_wire] = gw
            ew = e.detach()[0, :3 * s2].reshape(3, s2)
            gxi[i, pix_of_wire] = (ew * e.grad[0, :3 * s2].reshape(3, s2)).sum(0).cpu().numpy()
            e2 = torch.tensor(enc[i:i + 1], dtype=torch.float32, device=DEV, requires_grad=True)
            torch.log(readout(probs(e2)[1])[0, yt[i]]).backward()
            gxl[i, pix_of_wire] = (ew * e2.grad[0, :3 * s2].reshape(3, s2)).sum(0).cpu().numpy()
            with torch.no_grad():                                               # occlusion: one pixel at a time
                base = pg[0, yt[i]].item()
                E = torch.tensor(enc[i], dtype=torch.float32, device=DEV).repeat(s2, 1)
                idx = torch.arange(s2, device=DEV)
                for c in range(3):
                    E[idx, c * s2 + idx] = 0.0                                   # row j deletes wire j of each plane
                drop = np.concatenate([base - probs(E[j:j + 256])[0][:, yt[i]].cpu().numpy() for j in range(0, s2, 256)])
                occ[i, pix_of_wire] = drop
        ranks["gradient"], ranks["|gradient|"], ranks["occlusion"] = grad, np.abs(grad), occ
        ranks["grad x input"], ranks["grad x input, log p"] = gxi, gxl
        res = {}
        base_acc = (classify(enc) == Y).mean()
        for kind in list(ranks) + ["random"]:
            accs = []
            for frac in FRACS:
                k = int(round(frac * s2))
                draws = 3 if kind == "random" else 1
                a = 0.0
                for _ in range(draws):
                    key = rng.random((len(X), s2)) if kind == "random" else -ranks[kind]
                    e = enc.copy()
                    if k:
                        dead = np.zeros((len(X), s2), bool)
                        np.put_along_axis(dead, np.argsort(key, 1)[:, :k], True, 1)
                        dead_w = dead[:, pix_of_wire]
                        for c in range(3):
                            e[:, c * s2:(c + 1) * s2][dead_w] = 0.0
                    a += (classify(e) == Y).mean() / draws
                accs.append(float(a))
            res[kind] = accs
        out[name] = dict(fracs=FRACS, base=float(base_acc), **res)
        print(f"{name} (accuracy {base_acc:.3f}), deleted " + " ".join(f"{100 * f:>4.0f}%" for f in FRACS))
        for kind, accs in res.items():
            print(f"  {kind:11s} " + " ".join(f"{100 * a:5.1f}" for a in accs))
    (HERE / "results" / "deletion_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
