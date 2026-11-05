#!/usr/bin/env python3
"""Would a map measured on the chip see a fault in the device? (referee B, second round, Q1: "a deliberately
mis-calibrated layer that the measured map detects but the classically computed map of the nominal model cannot").

Simulated, not run on the chip. The trained circuit of D' (seed 0) gets one deliberate fault, as if mis-calibrated:
  - a whole layer: one rotation (RY or RZ) of every qubit in layer 0, 5 or 11 shifted by 0.1-1.0 rad;
  - one gate: RY or RX of one qubit in layer 0, 1 or 2 shifted by 0.5, 1.0 or 1.5 rad.
For each fault, the exact 2x2 sign-flip maps (Eq. flip) of ten pre-registered pictures (the first of each class's
three in random/prereg.json, drawn before any was measured) are compared with the nominal model's: the mean absolute
change per block, against the chip's own error on such maps (the sign-flip 2x2 MAE against the exact maps over the 30
random pictures at 4,000 shots, read from random/report.json), and the test-set accuracy of the faulty model (300 pictures, exact loading).
A measured map can show a fault only if the fault moves the maps by more than the chip's error; the question is
whether some fault does that while the accuracy, which anyone can check, stays.
Run:  python src/fault_maps.py   -> numbers, results/<test-set folder>/random/fault_sweep.json
"""
import argparse
import json
import pathlib

import numpy as np
import torch

import ibm_maps_random as mr
from ibm_testset import outcome_batch, predictions

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root


def pictures(pr):
    """The first of each class's three pre-registered pictures (they are stored class by class)."""
    return pr["pictures_2x2"][::mr.PER_CLASS]


def faulty(wc, layer, delta, qubits=None, angle=0):
    """The trained weights with rotation `angle` (0: RY, 1: RX, 2: RZ) of `layer` shifted by delta on `qubits` (all)."""
    w = np.array(wc, dtype=float).copy()
    w[layer, slice(None) if qubits is None else qubits, angle] += delta
    return w


def main():
    run, b, Y, wc, depth, n, no, groups, rows = mr.layout(argparse.Namespace())
    f = mr.folder(argparse.Namespace())
    h = np.load(f / "states.npz")
    pics = pictures(json.loads((f / "prereg.json").read_text()))
    chip_err = json.loads((f / "report.json").read_text())["2x2, sign flip, chip_ibm_kingston vs exact"]["mae"]
    S = torch.stack([r["state"] for r in rows])
    W0 = np.asarray(wc)

    def maps(W):
        m, _ = mr.maps(h, outcome_batch(S, torch.tensor(W), depth, no), Y, groups, no, 2, "sign flip")
        return np.array([m[i] for i in pics])

    acc = lambda W: float((predictions(run, outcome_batch(b, torch.tensor(W), depth, no)).argmax(1) == Y).mean())
    m0, a0 = maps(W0), acc(W0)
    print(f"ten pictures {pics}: nominal mean |block| {np.abs(m0).mean():.4f}; the chip's error on these maps "
          f"(MAE, 30 random pictures) {chip_err:.4f}; test accuracy {a0:.3f}")
    names = {0: "RY", 1: "RX", 2: "RZ"}
    faults = [dict(kind="layer", rot=r, layer=l, qubit=None, delta=d)
              for r in (0, 2) for l in (0, depth // 2, depth) for d in (0.1, 0.2, 0.3, 0.5, 1.0)]
    faults += [dict(kind="gate", rot=r, layer=l, qubit=q, delta=d)
               for l in (0, 1, 2) for q in range(n) for r in (0, 1) for d in (0.5, 1.0, 1.5)]
    for x in faults:
        W = faulty(W0, x["layer"], x["delta"], None if x["qubit"] is None else [x["qubit"]], x["rot"])
        x.update(rot=names[x["rot"]], map_change=float(np.abs(maps(W) - m0).mean()), accuracy=acc(W))
        x["accuracy_cost"] = round(a0 - x["accuracy"], 10)              # whole pictures: 5 points = 15 of 300, exactly
        if x["kind"] == "layer":
            print(f"layer fault {x['rot']} layer {x['layer']:2d} +{x['delta']:.1f} rad: mean |map change| "
                  f"{x['map_change']:.4f}, accuracy {x['accuracy']:.3f} ({-x['accuracy_cost']:+.3f})", flush=True)
    seen = [x for x in faults if x["map_change"] > chip_err]
    quiet = [x for x in faults if x["accuracy_cost"] <= 0.05]
    cheapest = min(seen, key=lambda x: x["accuracy_cost"])
    loudest = max(quiet, key=lambda x: x["map_change"])
    summary = dict(chip_error=chip_err, accuracy_nominal=a0, faults_tried=len(faults), faults_seen=len(seen),
                   least_accuracy_cost_when_seen=cheapest, largest_map_change_when_cost_at_most_5_points=loudest)
    print(f"{len(faults)} faults; {len(seen)} move the maps by more than the chip's error, and the cheapest of them "
          f"costs {100 * cheapest['accuracy_cost']:.1f} points ({cheapest['kind']} {cheapest['rot']} layer "
          f"{cheapest['layer']}{'' if cheapest['qubit'] is None else ' qubit ' + str(cheapest['qubit'])} "
          f"+{cheapest['delta']}: change {cheapest['map_change']:.4f}); of the faults that cost at most 5 points the "
          f"largest change is {loudest['map_change']:.4f} ({loudest['kind']} {loudest['rot']} layer {loudest['layer']}"
          f"{'' if loudest['qubit'] is None else ' qubit ' + str(loudest['qubit'])} +{loudest['delta']})")
    (f / "fault_sweep.json").write_text(json.dumps(dict(pictures=pics, summary=summary, faults=faults), indent=1))


if __name__ == "__main__":
    main()
