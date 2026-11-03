#!/usr/bin/env python3
"""Deletion by the evidence for the model's own answer (a first-draft review, 2026-11-01: Table 3 ranks by the true
class, so it uses the label; explain the decision instead). Same deletion as deletion_baselines.py (a pixel's contrast
set to zero in R, G, B, the means plane kept, renormalised, classified again). Pixels ranked by the exact evidence for
the predicted class (highest first, and reversed), and for comparison by the true class. Reported over all 300
pictures and separately over those first answered right and wrong: how often the answer stays the same, the accuracy,
and the headroom margin u_target - max over other classes u (u = (P - b) / (1 - b)), mean over pictures.
Run:  python src/deletion_predicted.py   -> numbers, results/deletion_predicted.json
"""
import json

import numpy as np
import torch

from attention_in_box import maps as pixel_maps
from bssg import class_groups
from deletion_baselines import DEV, FRACS, HERE, RUNS, model


def main():
    out = {}
    for name, f in RUNS.items():
        run = json.loads((HERE / "results" / f).read_text())
        X, Y, enc, pix_of_wire, probs, classify, _ = model(run)
        s2 = 1024
        groups = class_groups(run["mode"], 2 ** run["n_out"], 10)
        b = np.array([len(g) for g in groups]) / 2 ** run["n_out"]
        pred0 = classify(enc)
        right = pred0 == Y

        def masses(e):
            with torch.no_grad():
                return np.concatenate([probs(torch.tensor(e[i:i + 100], dtype=torch.float32, device=DEV))[0].cpu().numpy()
                                       for i in range(0, len(e), 100)])

        def margin(m, target):
            u = (m - b) / (1 - b)
            t = u[np.arange(len(u)), target]
            u2 = u.copy()
            u2[np.arange(len(u)), target] = -np.inf
            return t - u2.max(1)

        res = {}
        for by, target in (("predicted", pred0), ("true", Y)):
            ev = pixel_maps(run, X, target, signed=True)[2].reshape(len(X), -1)
            for order in ("highest", "reversed"):
                key = -ev if order == "highest" else ev
                rows = []
                for frac in FRACS:
                    k = int(round(frac * s2))
                    e = enc.copy()
                    if k:
                        dead = np.zeros((len(X), s2), bool)
                        np.put_along_axis(dead, np.argsort(key, 1)[:, :k], True, 1)
                        dw = dead[:, pix_of_wire]
                        for c in range(3):
                            e[:, c * s2:(c + 1) * s2][dw] = 0.0
                    p = classify(e)
                    mg = margin(masses(e), target)
                    rows.append({"frac": frac, "same answer": float((p == pred0).mean()),
                                 "same answer | right": float((p == pred0)[right].mean()),
                                 "same answer | wrong": float((p == pred0)[~right].mean()),
                                 "accuracy": float((p == Y).mean()), "margin": float(mg.mean()),
                                 "margin | right": float(mg[right].mean()), "margin | wrong": float(mg[~right].mean())})
                res[f"{by}, {order}"] = rows
        out[name] = dict(base=float(right.mean()), n_right=int(right.sum()), n_wrong=int((~right).sum()), **res)
        print(f"{name}: accuracy {right.mean():.3f} ({right.sum()} right, {(~right).sum()} wrong); deleted "
              + " ".join(f"{100 * fr:>5.0f}%" for fr in FRACS))
        for k_, rows in res.items():
            for q in ("same answer", "same answer | right", "same answer | wrong", "accuracy", "margin"):
                print(f"  {k_:20s} {q:20s} " + " ".join(f"{(100 if q != 'margin' else 1) * r[q]:6.3g}" for r in rows))
    (HERE / "results" / "deletion_predicted.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
