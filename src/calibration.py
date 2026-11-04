#!/usr/bin/env python3
"""Calibration of the readouts (referee B, second round, items 5 and Q7): BHR fixes the slope of the readout before
training, and the slope matters for the confidence, not for the argmax. Row D (phase plates), seeds 0-2, the 300
Caltech-10 test pictures: the expected calibration error (ECE, 15 equal-width bins of the top-class confidence), the
mean negative log-likelihood and the mean confidence, for
  - BHR at each phi of readout_sweep.sh (slope about 544 to 34; 136 at the paper's phi 0.0556),
  - a softmax of the group sums at slope 136 fixed (fixtemp_group), and with a learned temperature (temp_group),
  - BHR at phi 0.0556 after temperature scaling (one temperature on log p-hat, fitted by NLL on one half of the test
    pictures and applied to the other, both ways: no validation logits are kept in the run files).
Run:  python src/calibration.py  -> table, results/calibration.json
"""
import json
import pathlib

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from deletion_baselines import DEV, model

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
TAIL = "_q12_d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s{}.json"
READOUTS = [("bhr", p) for p in ("0.0139", "0.0278", "0.0556", "0.111", "0.222")] + \
           [("fixtemp_group", "136"), ("temp_group", "0.0556")]


def phat(run):
    X, Y, enc, _, probs, _, readout = model(run)
    if run.get("head_state"):
        readout.load_state_dict({k: torch.tensor(v) for k, v in run["head_state"].items()}, strict=False)
    with torch.no_grad():
        p = torch.cat([readout(probs(torch.tensor(enc[i:i + 100], dtype=torch.float32, device=DEV))[1])
                       for i in range(0, len(enc), 100)])
    return p.double().cpu().numpy(), Y


def ece(p, y, bins=15):
    conf, hit = p.max(1), p.argmax(1) == y
    k = np.minimum((conf * bins).astype(int), bins - 1)
    return float(sum(abs(hit[k == b].mean() - conf[k == b].mean()) * (k == b).mean() for b in range(bins) if (k == b).any()))


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def scaled(p, t):
    z = np.log(np.clip(p, 1e-12, None)) / t
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def cross_fitted(p, y):
    """temperature fitted on one half, applied to the other, both ways (fixed split, numpy default_rng(0))."""
    half = np.random.default_rng(0).permutation(len(y)) < len(y) // 2
    q, ts = np.empty_like(p), []
    for fit in (half, ~half):
        t = minimize_scalar(lambda t: nll(scaled(p[fit], t), y[fit]), bounds=(0.05, 20), method="bounded").x
        q[~fit] = scaled(p[~fit], t)
        ts.append(float(t))
    return q, ts


def main():
    out = {}
    for mode, pt in READOUTS:
        rows, extra = [], []
        for s in range(3):
            run = json.loads((HERE / "results" / f"pilot_caltech10fct_e50_{mode}_amp_pt{pt}{TAIL.format(s)}").read_text())
            p, y = phat(run)
            rows.append([ece(p, y), nll(p, y), p.max(1).mean(), (p.argmax(1) == y).mean()])
            if (mode, pt) == ("bhr", "0.0556"):
                q, ts = cross_fitted(p, y)
                extra.append([ece(q, y), nll(q, y), q.max(1).mean(), (q.argmax(1) == y).mean(), *ts])
        name = f"{mode} {pt}"
        for key, r in [(name, rows)] + ([(name + ", temperature-scaled", extra)] if extra else []):
            r = np.array(r)
            out[key] = dict(ece=r[:, 0].tolist(), nll=r[:, 1].tolist(), confidence=r[:, 2].tolist(), acc=r[:, 3].tolist(),
                            **({"temperatures": r[:, 4:].tolist()} if r.shape[1] > 4 else {}))
            print(f"{key:36s} ECE {r[:, 0].mean():.3f} +- {r[:, 0].std(ddof=1):.3f}  NLL {r[:, 1].mean():.3f}"
                  f"  confidence {r[:, 2].mean():.3f}  accuracy {r[:, 3].mean():.3f}"
                  + (f"  T {r[:, 4:].mean():.2f}" if r.shape[1] > 4 else ""))
    (HERE / "results" / "calibration.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
