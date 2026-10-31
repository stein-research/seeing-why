#!/usr/bin/env python3
"""2x2 block evidence maps of test pictures measured on IBM hardware, one picture per class (owner 2026-10-31).

Pictures (a stated rule, not a choice by eye; owner 2026-10-31: confident but with a background to make it hard): for
each class, of the test pictures the chip classified correctly with probability >= 0.6 for their class, the one whose
background (outside the Caltech outline) covers most of the crop and varies most (share x grey std); a class with none
>= 0.6 takes its most confident correct picture (--busy 0: the most confident everywhere). The maps stand for
confident pictures only.

Each block S of the chip's input b gives E_S = (<b|M|b> + <b_S|M|b_S> - <b_~S|M|b_~S>) / 2 (the polarization identity,
as ibm_demo.py heatmap); each term is a class probability of a prepared state times its squared norm. <b|M|b> is the
picture's own classification circuit - its counts come from the test-set job (the ledger's lookup), not a new
circuit - so a picture needs 2 g^2 = 8 new circuits. The block states b_S, b_~S get fitted RY + CNOT preparations
(depth 8), all at once on the GPU.

Compared with the exact map in four ways (pooled over the pictures): correlation of the block values, the strongest
block (same or not), the sum against P - b, and the signs. Steps as for accuracy: exact, prep (fitted states, no
noise), noisy (Aer + fake backend), chip.

Run:  python src/ibm_heatmaps.py fit    results/<run>.json
      python src/ibm_heatmaps.py noisy  results/<run>.json [--noise FakeKingston]
      python src/ibm_heatmaps.py chip   results/<run>.json --keys <file> [--accounts 0] [--dry_run]
      python src/ibm_heatmaps.py report results/<run>.json   (numbers + figures/ibm_heatmaps_<backend>.png)
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

import fastsim
import pilot_fmnist as pf
from bssg import class_groups
from heatmap import NAMES
from ibm_demo import brickwall, outcome_probs
from ibm_testset import fit_batch, folder_for, outcome_batch, ry_brickwall, test_states

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
G, K = 2, 8                                                     # blocks per side; new circuits per picture (2 G^2)


def setup(a):
    run = json.loads(pathlib.Path(a.run).read_text())
    b, Y, wc, depth = test_states(run)
    n, no, s = b.shape[1].bit_length() - 1, run["n_out"], pf.side()
    perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
    pixel = np.asarray(perm)[np.arange(2 ** n) % (s * s)]                   # pixel of every amplitude, all planes
    block = (pixel // s // (s // G)) * G + pixel % s // (s // G)
    groups = class_groups(run["mode"], 2 ** no, len(NAMES[run["dataset"]]))
    return run, b, Y, wc, depth, n, no, block, groups


def pick(f, Y, backend, top=0, b=None, wc=None, depth=None, groups=None, no=None, busy=0.0):
    """One picture per class: chip correct, highest true-class probability; busy > 0: of the correct ones with
    true-class probability >= busy, the one with the most and busiest background (Caltech outline), else the most
    confident; top > 0: of those, the `top` with the
    largest exact P - b (the strongest maps - for finer grids, whose blocks are small)."""
    st = json.loads((f / f"chip_{backend}_r0.json").read_text())
    out = []
    if busy:                                                    # owner 2026-10-31: confident AND a real background
        from attention_in_box import test_regions
        _, cont, crops = test_regions(0.04)
        grey = crops.astype(float).mean(-1)

        def background(i):                                      # background's share of the crop x its grey spread
            out_ = cont[i] < 0.5
            return (1 - cont[i].mean()) * (grey[i][out_].std() if out_.sum() > 20 else 0.0)
    for c in range(int(Y.max()) + 1):
        ok = [(st[k]["q"][c], int(k)) for k in st if st[k]["label"] == c and st[k]["pred"] == c]
        if not ok:
            continue
        sure = [i for q, i in ok if q >= busy] if busy else []
        out.append(max(sure, key=background) if sure else max(ok)[1])
    if top:
        P = outcome_batch(b[out], wc, depth, no)
        head = [P[k][np.isin(np.arange(2 ** no), groups[Y[i]])].sum() - len(groups[Y[i]]) / 2 ** no
                for k, i in enumerate(out)]
        out = [out[k] for k in np.argsort(head)[::-1][:top]]
    return out


def terms(P, norms, G_mask, bg):
    """norm * (P_g - b_g) per state."""
    return norms * (P[:, G_mask].sum(1) - bg)


def maps_from(t_full, t_in, t_out):
    """(pictures, G*G) block evidence from the term of b, of b_S and of b_~S."""
    return (t_full[:, None] + t_in - t_out) / 2


def fit(a):
    run, b, Y, wc, depth, n, no, block, groups = setup(a)
    f = folder_for(a)
    pz = np.load(f / "preps.npz")
    sel = pick(f, Y, a.backend, a.top, b, torch.tensor(pz["wc"]), depth, groups, no, a.busy)
    names = NAMES[run["dataset"]]
    print("pictures:", ", ".join(f"{i} ({names[Y[i]]})" for i in sel))
    vecs, norms = [], []
    for i in sel:
        for j in range(G * G):
            for inside in (True, False):
                v = b[i] * torch.tensor((block == j) == inside)
                nv = float((v ** 2).sum())
                vecs.append(v / nv ** 0.5)
                norms.append(nv)
    V = torch.stack(vecs)
    t0 = time.time()
    ang, fid = fit_batch(V, a.prep_depth, a.steps)
    print(f"fitted {len(V)} block preparations in {time.time() - t0:.0f} s: fidelity median {fid.median():.3f}, "
          f"min {fid.min():.3f}", flush=True)
    P_ex = outcome_batch(V, torch.tensor(pz["wc"]), depth, no)
    P_pr = outcome_batch(ry_brickwall(ang, n), torch.tensor(pz["wc"]), depth, no)
    P_full_ex = outcome_batch(b[sel], torch.tensor(pz["wc"]), depth, no)
    P_full_pr = outcome_batch(ry_brickwall(torch.tensor(pz["ang"][sel]), n), torch.tensor(pz["wc"]), depth, no)
    np.savez(f / f"heat_g{G}.npz", sel=sel, ang=ang.numpy(), fid=fid.numpy(), norms=np.array(norms),
             P_ex=P_ex, P_pr=P_pr, P_full_ex=P_full_ex, P_full_pr=P_full_pr)
    # check: the exact block maps against the per-amplitude evidence summed by block
    wct = torch.tensor(pz["wc"])
    for k, i in enumerate(sel[:3]):                             # (per-picture offset K)
        gm = torch.tensor(np.isin(np.arange(2 ** n) >> (n - no), groups[Y[i]]))
        bg = len(groups[Y[i]]) / 2 ** no
        bi = b[i][None].to(torch.complex128)
        Mb = fastsim.run_inverse(fastsim.run(bi, wct, depth, torch.complex128) * gm, wct, depth,
                                 torch.complex128)[0].real.numpy() - bg * b[i].numpy()
        direct = np.zeros(G * G)
        np.add.at(direct, block, b[i].numpy() * Mb)
        Gm = np.isin(np.arange(2 ** no), groups[Y[i]])
        nr = np.array(norms[K * k:K * k + K])
        t = terms(P_ex[K * k:K * k + K], nr, Gm, bg)
        tf = P_full_ex[k][Gm].sum() - bg
        m = maps_from(np.array([tf]), t[0::2][None], t[1::2][None])[0]
        print(f"check picture {i}: polarization vs per-amplitude sum, max diff {np.abs(m - direct).max():.1e}")
    print(f / f"heat_g{G}.npz")


def circuits_for(run, a, ang):
    from qiskit import QuantumCircuit
    pz = np.load(folder_for(a) / "preps.npz")
    n = len(run["weights"][0])
    wires = list(range(n - 1, -1, -1))
    out = []
    for k in range(len(ang)):
        qc = QuantumCircuit(n)
        brickwall(qc, wires, ang[k], a.prep_depth, "ry")
        brickwall(qc, wires, pz["wc"], run["depth"], "rzryrx")
        qc.measure_all()
        out.append(qc)
    return out


def noisy(a):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    run, b, Y, wc, depth, n, no, block, groups = setup(a)
    f = folder_for(a)
    h = np.load(f / f"heat_g{G}.npz")
    pz = np.load(f / "preps.npz")
    circs = circuits_for(run, a, np.concatenate([h["ang"], pz["ang"][h["sel"]]]))   # blocks, then the pictures
    fake = getattr(fake_provider, a.noise)()
    tq = transpile(circs, fake, optimization_level=3, seed_transpiler=0)
    P, t0 = [], time.time()
    if a.exact_noise:                                           # the noisy distribution itself (density matrix on the
        sim = AerSimulator.from_backend(fake, method="density_matrix")   # 12 used qubits); shots drawn in report
        for tc in tq:
            lay = tc.layout.final_index_layout()
            tc.remove_final_measurements()
            tc.save_probabilities(qubits=lay)                   # bit k = virtual qubit k = our wire n-1-k
        for i in range(0, len(tq), 4):
            res = sim.run(tq[i:i + 4]).result()
            for j in range(len(tq[i:i + 4])):
                full = np.asarray(res.data(j)["probabilities"])
                P.append(np.bincount(np.arange(2 ** n) >> (n - no), weights=full, minlength=2 ** no))
            print(f"  {len(P)}/{len(tq)} circuits, {time.time() - t0:.0f} s", flush=True)
        shots = 0
    else:
        sim = AerSimulator.from_backend(fake)
        for i in range(0, len(tq), 10):
            res = sim.run(tq[i:i + 10], shots=a.shots, seed_simulator=i).result()
            P += [outcome_probs(res.get_counts(j), n, no) for j in range(len(tq[i:i + 10]))]
            print(f"  {len(P)}/{len(tq)} circuits, {time.time() - t0:.0f} s", flush=True)
        shots = a.shots
    P = np.array(P)
    tag = "exactnoise_" if a.exact_noise else ""
    np.savez(f / f"heat_g{G}_noisy_{tag}{a.noise}.npz", P=P[:len(h["ang"])], P_full=P[len(h["ang"]):], shots=shots)


def chip(a):
    from ibm_jobs import QuotaShort, lookup, sampler_counts
    run, b, Y, wc, depth, n, no, block, groups = setup(a)
    f = folder_for(a)
    h = np.load(f / f"heat_g{G}.npz")
    pz = np.load(f / "preps.npz")
    names = NAMES[run["dataset"]]
    full = circuits_for(run, a, pz["ang"][h["sel"]])
    circs = circuits_for(run, a, h["ang"])
    notes = [f"picture {i} ({names[Y[i]]}), block {j} {'inside' if ins else 'outside'}, squared norm "
             f"{h['norms'][K * k + 2 * j + (0 if ins else 1)]:.5f}"
             for k, i in enumerate(h["sel"]) for j in range(G * G) for ins in (True, False)]
    P_full = []
    if a.repeat:                                                # every term in one job: the pictures' circuits too
        circs = circs + full
        notes = notes + [f"picture {i} ({names[Y[i]]}), whole picture" for i in h["sel"]]
    else:
        for i, c in zip(h["sel"], full):                        # the pictures' own circuits: already measured
            cnt = lookup(c, a.backend, a.shots, 0)
            if cnt is None:
                raise SystemExit(f"picture {i}: its classification circuit is not in the ledger for {a.backend}")
            P_full.append(outcome_probs(cnt, n, no))
    accts = [int(v) for v in a.accounts.split(",")]
    for acc in accts:
        try:
            counts, folder = sampler_counts(circs, a.backend, a.shots, a.keys, acc, repeat=a.repeat,
                                            purpose=f"{G}x{G} block heatmaps (repeat {a.repeat}) of {len(h['sel'])} test pictures "
                                                    f"(one per class), {pathlib.Path(a.run).name}",
                                            inputs=[a.run, f / "preps.npz", f / f"heat_g{G}.npz"],
                                            circuit_notes=notes, dry_run=a.dry_run)
            break
        except QuotaShort as e:
            print(e)
    else:
        raise SystemExit("no account has quota for this job")
    if counts is None:
        return
    P = np.array([outcome_probs(c, n, no) for c in counts])
    if a.repeat:
        P, P_full = P[:len(h["ang"])], P[len(h["ang"]):]
    name = f"heat_g{G}_chip_{a.backend}" + (f"_r{a.repeat}" if a.repeat else "")
    np.savez(f / f"{name}.npz", P=P, P_full=np.array(P_full), shots=a.shots, ledger=folder.name)
    print(f / f"{name}.npz")


def report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from attention_in_box import maps as pixel_maps
    run, b, Y, wc, depth, n, no, block, groups = setup(a)
    f = folder_for(a)
    h = np.load(f / f"heat_g{G}.npz")
    sel, norms = h["sel"], h["norms"]
    names = NAMES[run["dataset"]]
    kinds = {"exact": (h["P_ex"], h["P_full_ex"]), "prep": (h["P_pr"], h["P_full_pr"])}
    rng = np.random.default_rng(0)
    for p in sorted(f.glob(f"heat_g{G}_noisy_*.npz")) + sorted(f.glob(f"heat_g{G}_chip_*.npz")):
        d = np.load(p)
        name = p.stem.replace(f"heat_g{G}_", "")
        if int(d["shots"]) == 0:                                # exact noisy distributions: also with shots drawn
            kinds[name + " (no shot noise)"] = (d["P"], d["P_full"])
            for sh in (4000, 16000, 64000):
                draw = lambda Q: np.array([rng.multinomial(sh, q / q.sum()) / sh for q in Q])
                kinds[f"{name}, {sh} shots"] = (draw(d["P"]), draw(d["P_full"]))
        else:
            kinds[name] = (d["P"], d["P_full"])
    M = {}
    for kind, (P, Pf) in kinds.items():
        rows = []
        for k, i in enumerate(sel):
            Gm = np.isin(np.arange(2 ** no), groups[Y[i]])
            bg = len(groups[Y[i]]) / 2 ** no
            t = terms(P[K * k:K * k + K], norms[K * k:K * k + K], Gm, bg)
            rows.append(maps_from(np.array([Pf[k][Gm].sum() - bg]), t[0::2][None], t[1::2][None])[0])
        M[kind] = np.array(rows)
    ex = M["exact"]
    print(f"{len(sel)} pictures, {G}x{G} blocks; block maps against exact:")
    for kind, m in M.items():
        if kind == "exact":
            continue
        corr = np.corrcoef(m.ravel(), ex.ravel())[0, 1]
        top = (m.argmax(1) == ex.argmax(1)).mean()
        sign = (np.sign(m) == np.sign(ex)).mean()
        sums = np.abs(m.sum(1) - ex.sum(1))
        print(f"  {kind:22s} corr {corr:.3f} | strongest block same {top * len(sel):.0f}/{len(sel)} | signs "
              f"{sign:.2f} | sum vs exact P - b: mean |diff| {sums.mean():.4f} (exact sums {ex.sum(1).mean():.4f})")
    chip_kinds = [k for k in M if k.startswith("chip")]       # the figure shows the chip, else 4,000 drawn shots
    last = chip_kinds[-1] if chip_kinds else next((k for k in M if k.endswith(", 4000 shots")), list(M)[-1])
    short = last.replace("noisy_exactnoise_", "sim ").replace("chip_", "").replace(" shots", "")
    # figure: picture, exact per-pixel evidence, exact 2x2, last kind's 2x2
    d = np.load(HERE / "data" / f"{run['dataset']}.npz")
    pics = d["x_val"][sel]
    if a.no_figure:
        return
    mp = pixel_maps(run, pics, Y[sel], signed=True)
    ev = mp[2]
    att = mp[1] / mp[1].mean((1, 2), keepdims=True) - 1        # light kept against the average pixel (exact)
    fig, ax = plt.subplots(len(sel), 5, figsize=(10.5, 2.1 * len(sel)), squeeze=False)
    for r, i in enumerate(sel):
        lim = np.abs(ex[r]).max()
        for c, (img, title) in enumerate([(None, f"{names[Y[i]]} (picture {i})"),
                                          (att[r], "attention (exact)"), (ev[r], "evidence, per pixel"),
                                          (ex[r].reshape(G, G), f"exact, {G}x{G}"),
                                          (M[last][r].reshape(G, G), short + f", {G}x{G}")]):
            A = ax[r, c]
            A.imshow(pics[r].astype(float) / 255, extent=(0, 32, 32, 0))
            if img is not None:
                v = np.abs(img).max() if c <= 2 else lim
                A.imshow(img, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.65, extent=(0, 32, 32, 0),
                         interpolation="nearest")
                if c > 2:
                    for (yy, xx), val in np.ndenumerate(img):
                        A.text(xx * 32 / G + 16 / G, yy * 32 / G + 16 / G, f"{val:+.3f}", ha="center", va="center",
                               fontsize=7 if G == 2 else 4)
            A.set_title(title, fontsize=8)
            A.set_xticks([])
            A.set_yticks([])
    fig.suptitle(f"attention: red more light kept than the average pixel, blue less (the mask, exact)\n"
                 f"evidence for the true class: red for, blue against; right: {last}", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / (2.1 * len(sel))))
    out = HERE / "figures" / f"ibm_heatmaps_g{G}_{last.split(',')[0].replace(' ', '_')}.png"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".pdf"), dpi=600)
    json.dump({k: v.tolist() for k, v in M.items()} | {"pictures": sel.tolist()},
              open(f / f"heat_g{G}_maps.json", "w"))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("fit", "noisy", "chip", "report")}
    for p in ps.values():
        p.add_argument("run")
        p.add_argument("--prep_depth", type=int, default=8)
        p.add_argument("--backend", default="ibm_kingston", help="whose test-set results pick the pictures")
        p.add_argument("--grid", type=int, default=2, help="blocks per side (2, 4, 8)")
    ps["fit"].add_argument("--steps", type=int, default=3000)
    ps["noisy"].add_argument("--noise", default="FakeKingston")
    ps["noisy"].add_argument("--shots", type=int, default=4000)
    ps["chip"].add_argument("--keys", required=True)
    ps["chip"].add_argument("--accounts", default="0")
    ps["chip"].add_argument("--shots", type=int, default=4000)
    ps["chip"].add_argument("--dry_run", action="store_true")
    ps["chip"].add_argument("--repeat", type=int, default=0, help="k > 0: measure every term again in one job, the "
                            "pictures' own circuits included (no reuse from the test set)")
    ps["fit"].add_argument("--top", type=int, default=0, help="only the TOP pictures with the largest P - b")
    ps["fit"].add_argument("--busy", type=float, default=0.6, help="pick confident (>= this) pictures with busy "
                           "backgrounds; 0 = the most confident")
    ps["noisy"].add_argument("--exact_noise", action="store_true", help="noisy distributions by density matrix")
    ps["report"].add_argument("--no_figure", action="store_true")
    a = ap.parse_args()
    global G, K
    G, K = a.grid, 2 * a.grid * a.grid
    {"fit": fit, "noisy": noisy, "chip": chip, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
