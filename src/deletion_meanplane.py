#!/usr/bin/env python3
"""Deletion test for mean-plane runs, qubit or phase plates (referee 2026-10-31: do the maps point at what the model
uses?). Each test picture's pixels are ranked by the exact evidence for its true class (colour planes summed, from
attention_in_box.maps); a fraction of them is deleted by setting their contrast to zero in all three colour planes
(the pixel becomes the picture's own mean colour; the means plane is kept), the state is renormalised and the model
run again. Orders: highest evidence first, random (3 draws), lowest first. Faithful maps: deleting the highest drops
accuracy and the true class's headroom much faster than random; deleting the lowest raises them.
Run:  python src/deletion_meanplane.py results/<run>.json   -> numbers, results/deletion_mp_<tag>.json
"""
import json
import pathlib
import sys

import numpy as np
import torch

import fastsim
import optnet
import pilot_fmnist as pf
from attention_in_box import maps as pixel_maps
from bssg import Readout, class_groups
from heatmap import NAMES

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FRACS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def main():
    run = json.loads(pathlib.Path(sys.argv[1]).read_text())
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    if not run.get("center_image"):
        raise SystemExit("a mean-plane run is expected (deletion = contrast set to zero)")
    s, n, no = 32, pf.N_QUBITS, run["n_out"]
    perm = pf.morton_perm()
    d = np.load(HERE / "data" / f"{run['dataset']}.npz")
    X, Y = d["x_val"], d["y_val"]
    ev = pixel_maps(run, X, Y, signed=True)[2].reshape(len(X), -1)          # (B, 1024) pixel evidence, row-major
    v = X.astype(float).transpose(0, 3, 1, 2).reshape(len(X), 3, -1)[:, :, perm].reshape(len(X), -1)
    enc = pf.centre_planes(v, 0, run["mean_plane"])                          # (B, 4 * 1024) wire order, not normalised
    enc = np.pad(enc, ((0, 0), (0, 2 ** n - enc.shape[1])))
    pix_of_wire = np.asarray(perm)                                           # wire m (within a plane) -> pixel
    groups = class_groups(run["mode"], 2 ** no, len(NAMES[run["dataset"]]))
    readout = Readout(run["mode"], 2 ** no, len(groups), pt_ratio=run["pt_ratio"]).to(DEV)
    b = torch.tensor([len(g) / 2 ** no for g in groups], device=DEV)
    G = torch.tensor(np.stack([np.isin(np.arange(2 ** no), g) for g in groups]), dtype=torch.float32, device=DEV)
    if run.get("optical"):
        cam, _, det, os_ = optnet.from_run(run, perm, DEV, torch.complex64)
        outcome = lambda a: optnet.outcome_probs(cam(a), det, os=os_)
    else:
        w = torch.tensor(run["weights"], dtype=torch.float32, device=DEV)
        fourier = (torch.tensor(perm, device=DEV), s, 2 ** n // (s * s), run.get("fourier_phase", False),
                   run.get("fourier_linear", False)) if run.get("fourier_mask") else None
        wc, kw = fastsim.split_masks(w, run["depth"], 0, -1, fourier)
        outcome = lambda a: fastsim.probs(fastsim.run(a, wc, run["depth"], **kw), no)

    def score(e):
        a = torch.tensor(e / np.linalg.norm(e, axis=1, keepdims=True), dtype=torch.complex64, device=DEV)
        P = torch.cat([outcome(a[i:i + 100]) for i in range(0, len(a), 100)]).float()
        P = P / P.sum(1, keepdim=True)
        q = readout(P)
        yt = torch.tensor(Y, device=DEV)
        Pg = P @ G.T
        u = (Pg[torch.arange(len(Y)), yt] - b[yt]) / (1 - b[yt])
        return float((q.argmax(1) == yt).float().mean()), float(u.mean())

    rng = np.random.default_rng(0)
    res = {}
    with torch.no_grad():
        for order in ("highest", "random", "lowest"):
            draws = 3 if order == "random" else 1
            accs, us = np.zeros(len(FRACS)), np.zeros(len(FRACS))
            for _ in range(draws):
                key = {"highest": -ev, "lowest": ev, "random": rng.random(ev.shape)}[order]
                rank = np.argsort(key, 1)                                    # pixels, first to delete first
                for j, f in enumerate(FRACS):
                    k = int(round(f * s * s))
                    e = enc.copy()
                    if k:
                        dead = np.zeros((len(X), s * s), bool)
                        np.put_along_axis(dead, rank[:, :k], True, 1)
                        dead_w = dead[:, pix_of_wire]                        # wire order within a plane
                        for c in range(3):                                   # contrast planes only
                            e[:, c * s * s:(c + 1) * s * s][dead_w] = 0.0
                    acc, u = score(e)
                    accs[j] += acc / draws
                    us[j] += u / draws
            res[order] = dict(acc=accs.tolist(), u=us.tolist())
    tag = pathlib.Path(sys.argv[1]).stem
    print(f"{tag}: deleted " + " ".join(f"{f:>5.0%}" for f in FRACS))
    for o, r in res.items():
        print(f"  acc {o:8s} " + " ".join(f"{v:5.3f}" for v in r["acc"]))
    for o, r in res.items():
        print(f"  u   {o:8s} " + " ".join(f"{v:+5.3f}" for v in r["u"]))
    (HERE / "results" / f"deletion_mp_{tag[:80]}.json").write_text(json.dumps(dict(fracs=FRACS, **res), indent=1))


if __name__ == "__main__":
    main()
