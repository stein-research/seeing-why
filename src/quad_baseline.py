#!/usr/bin/env python3
"""A quadratic baseline with no physics (a first-draft review, 2026-11-01: separate the architecture's value from what any
quadratic form could do). Same encoding (trimmed Caltech-10, picture centred on its own means, means plane 0.5), same
128 outcomes read by the same BHR (phi 0.0556), same training (50 epochs, balanced batches of 64, random crops 0.8,
Adam 0.01 with a one-epoch warm-up and cosine decay, the best validation epoch kept). Only the map changes: instead of
the mask and phase plates (5,124 parameters), a free complex matrix A (128 x 4096, 1,048,576 real parameters), and
P_b = |(A a)_b|^2 / sum_b' |(A a)_b'|^2. Every such model is a ratio of quadratic forms of the encoded picture, as ours
is, so it bounds what the quadratic structure alone can reach with this readout and data.
Run:  python src/quad_baseline.py [seed ...]   -> numbers, results/quad_baseline.json
"""
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

import pilot_fmnist as pf
from bssg import Readout

HERE = pf.HERE if hasattr(pf, "HERE") else __import__("pathlib").Path(__file__).resolve().parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def run(seed, epochs=50, batch=64, lr=0.01):
    pf.DATASET, pf.N_QUBITS, pf.INPUT, pf.ENCODING, pf.ORDER = "caltech10fct", 12, "pad", "amp", "morton"
    pf.CENTER, pf.CENTER_IMAGE, pf.DC_ENERGY, pf.MEAN_PLANE = False, True, 0.0, 0.5
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    (xtr, ytr), (xva, yva), (xte, yte) = pf.load(2644, 100, seed)
    readout = Readout("bhr", 128, 10, pt_ratio=0.0556).to(DEV)
    A = torch.nn.Parameter(torch.randn(128, 4096, dtype=torch.complex64, device=DEV) / math.sqrt(4096))
    opt = torch.optim.Adam([A], lr=lr)
    steps = epochs * math.ceil(len(xtr) / batch)
    warm = math.ceil(len(xtr) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))

    def probs(x):
        z = torch.as_tensor(x, dtype=torch.complex64, device=DEV) @ A.T
        P = z.real ** 2 + z.imag ** 2
        return P / P.sum(1, keepdim=True).clamp_min(1e-30)

    def acc(x, y):
        with torch.no_grad():
            return float((readout(probs(x)).argmax(1).cpu().numpy() == y).mean())

    best = (-1, None)
    for ep in range(epochs):
        wb = 1.0 / np.bincount(ytr)[ytr]
        perm = rng.choice(len(xtr), len(xtr), p=wb / wb.sum())
        for i in range(0, len(xtr), batch):
            idx = perm[i:i + batch]
            xb = pf.RAW["encode"](pf.random_crops(pf.RAW["train"][idx], rng, 0.8))
            q = readout(probs(xb))
            loss = F.nll_loss(torch.log(q + 1e-12), torch.tensor(ytr[idx], dtype=torch.long, device=DEV))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        va = acc(xva, yva)
        if va > best[0]:
            best = (va, A.detach().clone(), ep + 1)
    with torch.no_grad():
        A.copy_(best[1])
        P = probs(xte)
        q = readout(P).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    te = float((q.argmax(1) == yte).mean())
    auc = float(roc_auc_score(yte, q, multi_class="ovr", average="macro"))
    print(f"seed {seed}: test {te:.3f}, ROC-AUC {auc:.3f}, best validation {best[0]:.3f} at epoch {best[2]}, "
          f"train {acc(xtr[:2000], ytr[:2000]):.3f}", flush=True)
    return dict(seed=seed, test=te, auc=auc, val=best[0], epoch=best[2])


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or [0, 1, 2]
    rows = [run(s) for s in seeds]
    t = np.array([r["test"] for r in rows]); u = np.array([r["auc"] for r in rows])
    print(f"free quadratic (1,048,576 parameters): test {100 * t.mean():.1f} +- {100 * t.std(ddof=1):.1f} %, "
          f"ROC-AUC {u.mean():.3f} +- {u.std(ddof=1):.3f}  (sample std over seeds, as every other row)")
    (HERE / "results" / "quad_baseline.json").write_text(json.dumps(dict(params=2 * 128 * 4096, runs=rows), indent=1))
