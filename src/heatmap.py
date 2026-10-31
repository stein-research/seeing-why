#!/usr/bin/env python3
"""Where a trained circuit looks: exact pixel maps from the circuit's unitary (simulation only, no approximation).

Class g is read by the projector Pi_g on its outcome group; seen from the input it is M_g = U^dag Pi_g U, and
P_g(psi) = <psi|M_g|psi>.

Figure 1 - class maps: the diagonal M_g[x, x] = probability that a photon entering at pixel x alone lands in class g.
  Shown as M_g[x, x] - b_g, b_g = |S_g| / N (the no-information anchor): red pushes towards g, blue away. This is the
  incoherent response, so it can be measured on hardware by preparing the basis state |x> (at most n X gates).
Figure 2 - per-image attribution: c_g(x) = a(x) Re[(U|x>)^dag Pi_g U|psi>], with sum_x c_g(x) = P_g exactly (zero on
  dark pixels by construction, since a(x) = 0 there), and the sensitivity dP_g/da(x) on the sphere (row 4), which is
  non-zero on dark pixels: where added light would raise (red) or lower (blue) the true class's probability; the
  off-diagonal part of M_g (interference between pixels) is included.

Run:  python src/heatmap.py results/<run>.json
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

import fastsim
import pilot_fmnist as pf
from bssg import Readout, make_group_map, class_groups

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
DEV = "cuda" if torch.cuda.is_available() else "cpu"     # unitaries on the GPU (12 qubits: seconds, not minutes)
NAMES = {"fmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
         "mnist": [str(d) for d in range(10)],
         "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
         "wmnistc": [str(d) for d in range(10)],
         "caltech10c": ["airplane", "motorbike", "face", "watch", "leopard", "bonsai", "car_side", "ketch",
                        "chandelier", "hawksbill"],
         "flowers17c": ["daffodil", "snowdrop", "lily_valley", "bluebell", "crocus", "iris", "tigerlily", "tulip",
                        "fritillary", "sunflower", "daisy", "colts_foot", "dandelion", "cowslip", "buttercup",
                        "windflower", "pansy"],
         "caltech10fc": ["airplane", "motorbike", "face", "watch", "leopard", "bonsai", "car_side", "ketch",
                         "chandelier", "hawksbill"],
         "caltech10fct": ["airplane", "motorbike", "face", "watch", "leopard", "bonsai", "car_side", "ketch",
                         "chandelier", "hawksbill"],
         "caltech10f": ["airplane", "motorbike", "face", "watch", "leopard", "bonsai", "car_side", "ketch",
                        "chandelier", "hawksbill"],
         "animals10": ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"],
         "cifar10c": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]}


def outcome_of(run, n):
    """Measured outcome of every basis state: the first n_out wires, or the last P wires for a bag of patches."""
    s = np.arange(2 ** n)
    return (s & (2 ** run["n_out"] - 1)) if run.get("patch_wires") else s >> (n - run["n_out"])


def components(run, n):
    """[(V, weight, outcome)] - the circuit(s) a photon may take: one for a plain circuit or a bag of patches; for a
    spatial pyramid the full circuit and the bag of patches on the last K wires, each with weight 0.5. V[x] = U|x>,
    outcome[s] = the measured outcome of final basis state s."""
    if run.get("stack"):                                # two-phase (ssl_two_phase.py): U_total = (V x I_B) U
        no = run["n_out"]
        with torch.no_grad():
            U = fastsim.unitary(torch.tensor(run["stack"]["u"], dtype=torch.float64, device=DEV), run["depth"],
                                torch.complex128)
            V = fastsim.unitary(torch.tensor(run["stack"]["v"], dtype=torch.float64, device=DEV),
                                run["stack"]["depth_v"], torch.complex128)
            Ut = torch.kron(V, torch.eye(2 ** (n - no), dtype=V.dtype, device=DEV)) @ U
        return [(Ut.T.cpu().numpy(), 1.0, np.arange(2 ** n) >> (n - no))]
    if not run.get("pyramid"):
        return [(unitary_columns(run), 1.0, outcome_of(run, n))]
    K, no = run["pyramid"], run["n_out"]
    w = torch.tensor(run["weights"], dtype=torch.float64, device=DEV)
    depth, s = w.shape[0] - 1, np.arange(2 ** n)
    with torch.no_grad():
        Vf = fastsim.unitary(w[:, :n], depth, torch.complex128).T.cpu().numpy()
        Vb = np.kron(np.eye(2 ** (n - K)), fastsim.unitary(w[:, n:], depth, torch.complex128).cpu().numpy()).T
    return [(Vf, 0.5, s >> (n - no)), (Vb, 0.5, (s & (2 ** K - 1)) >> (K - no))]


def unitary_columns(run):
    """V[x] = U|x> for every basis input x (rows), as a (2^n, 2^n) complex array."""
    w = torch.tensor(run["weights"], dtype=torch.float64)
    depth, n = w.shape[0] - 1, pf.N_QUBITS
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev, interface="torch")
    def state(amps):
        qml.AmplitudeEmbedding(amps, wires=range(n), normalize=False)
        for k in range(depth):
            pf.rotate(w[k])
            pf.entangle(k, run["ent"])
        pf.rotate(w[depth])
        return qml.state()

    if run.get("patch_wires"):                          # bag of patches: U = I (x) V on the last P wires
        with torch.no_grad():
            V = fastsim.unitary(w.to(DEV), depth, torch.complex128).cpu().numpy()
        return np.kron(np.eye(2 ** (run["n_qubits"] - run["patch_wires"])), V).T
    if run["ent"] in ("brickwall", "hier"):             # fastsim (checked against PennyLane), lighter at 12 qubits
        with torch.no_grad():
            return fastsim.unitary(w.to(DEV), depth, torch.complex128, hier=run["ent"] == "hier").T.cpu().numpy()
    with torch.no_grad():
        return state(torch.eye(2 ** n, dtype=torch.float64)).numpy()


def to_grid(v):
    """Pixel-ordered side x side image from a wire-ordered vector (undoes the Morton permutation); for the colour
    sets the channels are summed into one map."""
    sz, nc = pf.side() ** 2, pf.channels()
    v = np.asarray(v)[:nc * sz].reshape(nc, sz)
    if pf.permuted():
        out = np.empty_like(v)
        out[:, pf.morton_perm()] = v
        v = out
    return v.sum(0).reshape(pf.side(), pf.side())


def main():
    run = json.loads(pathlib.Path(sys.argv[1]).read_text())
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "rowcol")
    pf.FILTER = run.get("filter", "none")
    pf.N_QUBITS = run.get("n_qubits") if (run.get("patch_wires") or run.get("pyramid")) else len(run["weights"][0])
    pf.INPUT = run.get("input", "pad")
    names = NAMES[pf.DATASET] if pf.DATASET in NAMES else \
        [str(c)[4:] for c in np.load(HERE / "data" / f"{pf.DATASET}.npz")["class_names"]]
    n_out, n = run["n_out"], pf.N_QUBITS
    lss = run["mode"] == "lss"
    # class score = sum_s W[g, s] |<s|U|psi>|^2 over the final basis states s:
    #   group readouts - W = indicator of the class's outcome group, score P_g, anchor |S_g| / N
    #   lss            - W = +1 / -1 for wire g in |0> / |1>, score <Z_g>, anchor 0 (signed by nature)
    s_idx = np.arange(2 ** n)
    if lss:
        W = np.stack([1.0 - 2.0 * ((s_idx >> (n - 1 - g)) & 1) for g in range(len(names))])
        anchors, sizes, score = [0.0] * len(names), ["<Z> of wire %d" % g for g in range(len(names))], "<Z>"
    else:
        groups = class_groups(run["mode"], 2 ** n_out, len(names))
        outcome = (s_idx & (2 ** n_out - 1)) if run.get("patch_wires") else s_idx >> (n - n_out)   # measured outcome
        W = np.stack([np.isin(outcome, g).astype(float) for g in groups])
        anchors, sizes, score = [len(g) / 2 ** n_out for g in groups], [f"{len(g)} outcomes" for g in groups], "P"
    V = unitary_columns(run)

    tag = pathlib.Path(sys.argv[1]).stem.replace("pilot_", "")
    out = HERE / "figures"
    out.mkdir(exist_ok=True)

    # Figure 1: class maps
    diag = (np.abs(V) ** 2) @ W.T - np.array(anchors)               # (2^n, G): score for a photon at pixel x
    diag = diag.T
    lim = np.abs(diag).max()
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.6))
    for g, ax in enumerate(axes.flat):
        im = ax.imshow(to_grid(diag[g]), cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"{names[g]} ({sizes[g]})", fontsize=9)
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8, label=("<Z_g> for a photon at the pixel" if lss
                                                 else "P(class | photon at pixel) - anchor"))
    fig.suptitle((f"Class maps <x|U^dag Z_g U|x>   ({tag}, test {run['test_exact']:.3f})" if lss else
                  f"Class maps M_g[x,x] - |S_g|/N   ({tag}, test {run['test_exact']:.3f})"), fontsize=10)
    fig.savefig(out / f"heatmap_classes_{tag}.png", dpi=120)
    print(out / f"heatmap_classes_{tag}.png")

    # Figure 2: per-image attribution for the first test image of each class; titles give the class's Born probability P
    # (what the map sums to) and p, the readout's output after BHR
    readout = Readout(run["mode"], 2 ** n_out, len(names), pt_ratio=run["pt_ratio"])
    if run.get("head_state"):
        readout.load_state_dict({k: torch.tensor(v) for k, v in run["head_state"].items()})
    _, _, (xte, yte) = pf.load(10, 10, 0)
    fig, axes = plt.subplots(4, 10, figsize=(18, 8.2))
    for c in range(len(names)):
        a = xte[np.argmax(yte == c)]
        phi = V.T @ a                                                  # U|psi>
        P = W @ (np.abs(phi) ** 2)                                     # class scores (P_g, or <Z_g> for lss)
        q = readout(torch.tensor(np.bincount(outcome_of(run, n), np.abs(phi) ** 2, 2 ** n_out)[None], dtype=torch.float32)
                    )[0].detach().numpy()                              # the readout's output (BHR softmax for bhr)
        pred = int(np.argmax(q))
        attr = lambda g: a * np.real(np.conj(V) @ (W[g] * phi))
        axes[0, c].imshow(to_grid(a), cmap="gray")
        axes[0, c].set_title(f"true {names[c]}", fontsize=9)
        # sensitivity: dP_true / da(x) = 2 Re (M_g psi)(x), projected onto the sphere (adding light anywhere also
        # renormalises the state); non-zero on dark pixels too - where light WOULD change the class score
        sens = 2 * np.real(np.conj(V) @ (W[c] * phi))
        sens = sens - (sens @ a) * a
        m = to_grid(sens)
        l = np.abs(m).max()
        axes[3, c].imshow(m, cmap="RdBu_r", vmin=-l, vmax=l)
        axes[3, c].set_title(f"sensitivity, true {names[c]}", fontsize=8)
        for row, g, lab in ((1, c, "true"), (2, pred, "pred")):
            m = to_grid(attr(g))
            l = np.abs(m).max()
            axes[row, c].imshow(m, cmap="RdBu_r", vmin=-l, vmax=l)
            axes[row, c].set_title(f"{lab} {names[g]}: {score}={P[g]:.3f}, p={q[g]:.2f}", fontsize=8,
                                   color="black" if pred == c else "red")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Per-image attribution c_g(x), sum_x c_g(x) = {score}_g "
                 f"({'Pauli-Z expectation of wire g' if lss else 'Born probability of the class group'}); "
                 f"p = readout output after {run['mode'].upper()}   ({tag})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"heatmap_images_{tag}.png", dpi=110)
    print(out / f"heatmap_images_{tag}.png")


if __name__ == "__main__":
    main()
