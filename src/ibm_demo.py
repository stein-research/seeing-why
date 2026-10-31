#!/usr/bin/env python3
"""One picture through a trained circuit on IBM hardware (owner 2026-10-31: a demo of the d11 model whose maps look good).

The model: picture (own mean removed, colour means in plane 4) -> Fourier-plane mask (amplitude and phase) -> brickwall
circuit, depth 11, 12 wires -> the first 7 wires measured -> BHR. The mask acts on the input before the circuit, so for
one picture its output b = A_mask a is computed here (on a bench this is the 4f system: lenses and an SLM) and the
chip gets b as its input state. b is real (the mask's kernel is real), so it is prepared approximately by a brickwall of
RY + CNOT fitted to b (`--prep_depth`), not by an exact preparation (~4,000 CNOTs for 12 qubits). Both brickwalls are
nearest-neighbour on a line, which maps onto IBM's heavy-hex lattice without SWAPs.

  prepare: exact class probabilities; fit the preparation (fidelity, class probabilities with it); check that the Qiskit
           circuit's statevector equals fastsim's; noisy simulation with a fake IBM backend (local, no account);
           writes results/ibm_demo_<tag>.qpy and .json
  run:     sends the saved circuit to an IBM backend - line --account of a keys file outside the repository (the token
           is read, never printed) - and reads BHR from the counts
Every hardware job goes through ibm_jobs.py: kept in ibm_jobs/<key>/ (job id, circuits, counts, usage, calibration),
never sent twice for the same circuits, backend and shots; --dry_run writes the folder without sending.

Run:  python src/ibm_demo.py prepare results/<run>.json [--picture k] [--prep_depth L]
      python src/ibm_demo.py run results/ibm_demo_<tag>.json [--backend name] [--shots 4000]
"""
import argparse
import json
import pathlib

import numpy as np
import torch

import fastsim
import pilot_fmnist as pf
from bssg import Readout, class_groups
from heatmap import NAMES

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
FILES = {"cifar10c": "cifar10.npz", "cifar10": "cifar10.npz", "wmnistc": "wmnist.npz"}


def picture_state(run, k):
    """Test picture k (the first picture of class k), encoded as the trainer does: (2^n,) wire order, normalised."""
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")
    pf.N_QUBITS = n = len(run["weights"][0])
    d = np.load(HERE / "data" / FILES.get(pf.DATASET, f"{pf.DATASET}.npz"))
    split = "val" if "y_val" in d else "test"
    i = int(np.argmax(d[f"y_{split}"] == k))
    s = pf.side()
    x = d[f"x_{split}"][i:i + 1].astype(np.float64).transpose(0, 3, 1, 2)
    perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
    x = x.reshape(1, -1, s * s)[:, :, perm].reshape(1, -1)
    if run.get("center_image"):
        x = pf.centre_planes(x, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    x = x / np.linalg.norm(x)
    x = np.pad(x, ((0, 0), (0, 2 ** n - x.shape[1])))
    return torch.tensor(x[0], dtype=torch.float64), perm, i


def class_probs(run, pout):
    """BHR class probabilities from the 2^n_out outcome probabilities (1-D, sums to 1)."""
    C, no = len(NAMES[run["dataset"]]), run["n_out"]
    r = Readout(run["mode"], 2 ** no, C, pt_ratio=run["pt_ratio"])
    with torch.no_grad():
        return r(torch.as_tensor(pout, dtype=torch.float32)[None])[0].numpy()


def fit_prep(b, L, steps=3000, seed=0):
    """RY + CNOT brickwall of depth L (L + 1 rotation layers) with |<b| V |0>|^2 as large as Adam gets it; b real."""
    n = int(np.log2(len(b)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    b = b.to(dev)
    torch.manual_seed(seed)
    ang = torch.nn.Parameter(0.1 * torch.randn(L + 1, n, dtype=torch.float64, device=dev))
    zero = torch.zeros(1, 2 ** n, dtype=torch.float64, device=dev)
    zero[0, 0] = 1
    opt = torch.optim.Adam([ang], lr=0.05)
    best = (0.0, None)
    for t in range(steps):
        w = torch.stack([ang, torch.zeros_like(ang)], -1)                    # RY only (RX angle 0): real states
        st = fastsim.run(zero, w, L, torch.complex128)[0]
        f = (st.conj() @ b.to(torch.complex128)).abs() ** 2
        if f.item() > best[0]:
            best = (f.item(), ang.detach().cpu().clone())
        opt.zero_grad()
        (1 - f).backward()
        opt.step()
        if t == steps // 2:
            for g in opt.param_groups:
                g["lr"] = 0.01
    return best


def brickwall(qc, wires, w, depth, rot):
    """Append fastsim's brickwall: depth + 1 rotation layers, CNOT(i, i+1) for i = k mod 2 after layer k. Our wire i
    (wire 0 most significant) is Qiskit qubit wires[i]."""
    n = len(wires)
    for k in range(depth + 1):
        for i in range(n):
            if rot == "ry":
                qc.ry(float(w[k, i]), wires[i])
            else:                                                            # fastsim: RX(w1) RY(w0) RZ(w2), RZ first
                qc.rz(float(w[k, i, 2]), wires[i])
                qc.ry(float(w[k, i, 0]), wires[i])
                qc.rx(float(w[k, i, 1]), wires[i])
        if k < depth:
            for i in range(k % 2, n - 1, 2):
                qc.cx(wires[i], wires[i + 1])


def prepare(a):
    from qiskit import QuantumCircuit, qpy, transpile
    from qiskit.quantum_info import Statevector
    run = json.loads(pathlib.Path(a.run).read_text())
    if run.get("ent") != "brickwall" or run.get("fourier_bank", 0) > 1 or run.get("mask_group", 0) or run.get("glimpse"):
        raise SystemExit("the demo takes a brickwall run with no mask or a single Fourier mask")
    x, perm, idx = picture_state(run, a.picture)
    n, depth, no = len(x).bit_length() - 1, run["depth"], run["n_out"]
    w = torch.tensor(run["weights"], dtype=torch.float64)
    fourier = (torch.tensor(perm), pf.side(), 2 ** n // pf.side() ** 2, run.get("fourier_phase", False),
               run.get("fourier_linear", False)) \
        if run.get("fourier_mask") else None
    wc, kw = fastsim.split_masks(w, depth, 0, -1, fourier)
    b = fastsim.fourier_filter(x[None].to(torch.complex128), kw["fmask"][1], kw["fmask"][0], kw["fmask"][2])[0] if fourier \
        else x.to(torch.complex128)
    keep = float((b.abs() ** 2).sum())
    if b.imag.abs().max() > 1e-9:
        raise SystemExit("the filtered picture is not real; the RY-only preparation cannot reach it")
    b = (b.real / keep ** 0.5)
    exact = fastsim.probs(fastsim.run(b[None], wc, depth, torch.complex128), no)[0].numpy()
    names = NAMES[run["dataset"]]
    q_exact = class_probs(run, exact)
    print(f"picture {idx} ({names[a.picture]}), light through the mask {keep:.3f}")
    print(f"exact: predicted {names[q_exact.argmax()]} p={q_exact.max():.3f}; true class p={q_exact[a.picture]:.3f}")

    fid, ang = fit_prep(b, a.prep_depth)
    w_prep = torch.stack([ang, torch.zeros_like(ang)], -1)
    zero = torch.zeros(1, 2 ** n, dtype=torch.float64)
    zero[0, 0] = 1
    b_apx = fastsim.run(zero, w_prep, a.prep_depth, torch.complex128)[0]
    apx = fastsim.probs(fastsim.run(b_apx[None], wc, depth, torch.complex128), no)[0].numpy()
    q_apx = class_probs(run, apx)
    print(f"preparation depth {a.prep_depth}: fidelity {fid:.4f}; predicted {names[q_apx.argmax()]} "
          f"p={q_apx.max():.3f}; true class p={q_apx[a.picture]:.3f}")

    wires = list(range(n - 1, -1, -1))                                       # our wire i -> Qiskit qubit n-1-i
    qc = QuantumCircuit(n)
    brickwall(qc, wires, ang.numpy(), a.prep_depth, "ry")
    brickwall(qc, wires, wc.numpy(), depth, "rzryrx")
    sv = np.asarray(Statevector(qc).data)                                    # Qiskit index = our index (see wires)
    ours = fastsim.run(b_apx[None], wc, depth, torch.complex128)[0].numpy()
    print(f"Qiskit statevector vs fastsim: max |diff| {np.abs(sv - ours).max():.2e}")
    qc.measure_all()
    n_cx = qc.count_ops().get("cx", 0)
    print(f"circuit: {n} qubits, {n_cx} CNOTs, depth {qc.depth()}")

    tag = f"{pathlib.Path(a.run).stem[:40]}_p{a.picture}_L{a.prep_depth}"
    out = dict(run=a.run, picture=a.picture, index=idx, prep_depth=a.prep_depth, fidelity=fid, keep=keep,
               n_cnot=n_cx, exact_outcomes=exact.tolist(), approx_outcomes=apx.tolist(),
               exact_class=q_exact.tolist(), approx_class=q_apx.tolist())
    if a.noise:
        from qiskit_aer import AerSimulator
        from qiskit_ibm_runtime import fake_provider
        fake = getattr(fake_provider, a.noise)()
        sim = AerSimulator.from_backend(fake)
        tq = transpile(qc, fake, optimization_level=3, seed_transpiler=0)
        ops = tq.count_ops()
        counts = sim.run(tq, shots=a.shots, seed_simulator=0).result().get_counts()
        pn = outcome_probs(counts, n, no)
        q_noisy = class_probs(run, pn)
        print(f"noisy simulation ({a.noise}, {a.shots} shots, transpiled 2-qubit gates "
              f"{sum(v for k, v in ops.items() if k in ('cx', 'ecr', 'cz'))}): predicted {names[q_noisy.argmax()]} "
              f"p={q_noisy.max():.3f}; true class p={q_noisy[a.picture]:.3f}")
        out.update(noise=a.noise, noisy_class=q_noisy.tolist(), noisy_outcomes=pn.tolist())
    with open(HERE / "results" / f"ibm_demo_{tag}.qpy", "wb") as f:
        qpy.dump(qc, f)
    (HERE / "results" / f"ibm_demo_{tag}.json").write_text(json.dumps(out))
    print(HERE / "results" / f"ibm_demo_{tag}.json")


def outcome_probs(counts, n, no):
    """Counts over all n wires (Qiskit bit strings, qubit n-1 first = our wire 0 first) -> 2^no outcome probabilities
    of our first no wires."""
    p = np.zeros(2 ** no)
    for bits, c in counts.items():
        p[int(bits.replace(" ", ""), 2) >> (n - no)] += c
    return p / p.sum()


def run_hw(a):
    from qiskit import qpy
    from ibm_jobs import sampler_counts
    meta = json.loads(pathlib.Path(a.demo).read_text())
    run = json.loads(pathlib.Path(meta["run"]).read_text())
    with open(pathlib.Path(a.demo).with_suffix(".qpy"), "rb") as f:
        qc = qpy.load(f)[0]
    counts, folder = sampler_counts([qc], a.backend, a.shots, a.keys, a.account,
                                    purpose=f"demo: {NAMES[run['dataset']][meta['picture']]} picture, {pathlib.Path(a.demo).name}",
                                    inputs=[a.demo, pathlib.Path(a.demo).with_suffix(".qpy"), meta["run"]],
                                    circuit_notes=["fitted preparation + trained circuit, first n_out wires read"],
                                    dry_run=a.dry_run)
    if counts is None:
        return
    counts = counts[0]
    job = json.loads((folder / "record.json").read_text())["job_id"]
    n, no = qc.num_qubits, run["n_out"]
    p = outcome_probs(counts, n, no)
    q = class_probs(run, p)
    names = NAMES[run["dataset"]]
    print(f"hardware ({a.backend}, {a.shots} shots): predicted {names[q.argmax()]} p={q.max():.3f}; "
          f"true class p={q[meta['picture']]:.3f}")
    meta.update(backend=a.backend, job=job, ledger=folder.name, shots=a.shots, hw_class=q.tolist(), hw_outcomes=p.tolist())
    pathlib.Path(a.demo).write_text(json.dumps(meta))


def chip_input(run, k):
    """(masked picture b normalised, run weights wc, depth, perm) for class k's first test picture."""
    x, perm, idx = picture_state(run, k)
    n, depth = len(x).bit_length() - 1, run["depth"]
    w = torch.tensor(run["weights"], dtype=torch.float64)
    fourier = (torch.tensor(perm), pf.side(), 2 ** n // pf.side() ** 2, run.get("fourier_phase", False),
               run.get("fourier_linear", False)) \
        if run.get("fourier_mask") else None
    wc, kw = fastsim.split_masks(w, depth, 0, -1, fourier)
    b = fastsim.fourier_filter(x[None].to(torch.complex128), kw["fmask"][1], kw["fmask"][0], kw["fmask"][2])[0] if fourier \
        else x.to(torch.complex128)
    return (b.real / (b.abs() ** 2).sum() ** 0.5), wc, depth, perm, idx


def block_heatmap(a):
    """Evidence summed over g x g blocks of the chip's input picture b, measured as a hardware would: by the
    polarization identity E_S = (<b|M|b> + <b_S|M|b_S> - <b_~S|M|b_~S>) / 2 with M = U^dag (Pi_g - b_g) U, b_S the
    picture inside block S (all colour planes), b_~S the rest. Each term is a class probability of a prepared state times
    its squared norm, so 1 + 2 g^2 circuits give a map whose blocks add up to P_g - b_g. Reported three ways: exact
    (states exact, no noise), with the fitted preparations (no noise), and with the preparations under a fake
    backend's noise."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    run = json.loads(pathlib.Path(a.run).read_text())
    b, wc, depth, perm, idx = chip_input(run, a.picture)
    n, no, s, g = len(b).bit_length() - 1, run["n_out"], pf.side(), a.grid
    names = NAMES[run["dataset"]]
    groups = class_groups(run["mode"], 2 ** no, len(names))
    G, bg = np.isin(np.arange(2 ** no), groups[a.picture]), len(groups[a.picture]) / 2 ** no
    pixel = np.asarray(perm)[np.arange(2 ** n) % (s * s)]                    # pixel of every amplitude, all planes
    block = (pixel // s // (s // g)) * g + pixel % s // (s // g)
    vecs = [("full", b)]
    for j in range(g * g):
        inside = torch.tensor(block == j)
        vecs += [(f"in{j}", b * inside), (f"out{j}", b * ~inside)]
    zero = torch.zeros(1, 2 ** n, dtype=torch.float64)
    zero[0, 0] = 1
    wires = list(range(n - 1, -1, -1))
    term = {"exact": {}, "prep": {}, "noisy": {}}
    fids, circs = [], []
    for name, v in vecs:
        nv = float((v ** 2).sum())
        vh = v / nv ** 0.5
        pe = fastsim.probs(fastsim.run(vh[None], wc, depth, torch.complex128), no)[0].numpy()
        fid, ang = fit_prep(vh, a.prep_depth, steps=a.steps)
        fids.append(fid)
        w_prep = torch.stack([ang, torch.zeros_like(ang)], -1)
        va = fastsim.run(zero, w_prep, a.prep_depth, torch.complex128)[0]
        pa = fastsim.probs(fastsim.run(va[None], wc, depth, torch.complex128), no)[0].numpy()
        qc = QuantumCircuit(n)
        brickwall(qc, wires, ang.numpy(), a.prep_depth, "ry")
        brickwall(qc, wires, wc.numpy(), depth, "rzryrx")
        qc.measure_all()
        circs.append(qc)
        term["exact"][name] = nv * (pe[G].sum() - bg)
        term["prep"][name] = nv * (pa[G].sum() - bg)
        term[name] = nv
    ledger = None
    if a.keys:                                   # real hardware: one job, every circuit, through the ledger
        from ibm_jobs import sampler_counts
        counts, folder = sampler_counts(
            circs, a.backend, a.shots, a.keys, a.account,
            purpose=f"heatmap: {names[a.picture]} picture, {g}x{g} blocks, prep depth {a.prep_depth}, "
                    f"{pathlib.Path(a.run).name}",
            inputs=[a.run], circuit_notes=[f"{name}: squared norm {term[name]:.6f}" for name, _ in vecs],
            dry_run=a.dry_run)
        if counts is None:
            return
        ledger = folder.name
        source = f"{a.backend} (hardware)"
    else:
        fake = getattr(fake_provider, a.noise)()
        tq = transpile(circs, fake, optimization_level=3, seed_transpiler=0)
        res = AerSimulator.from_backend(fake).run(tq, shots=a.shots, seed_simulator=0).result()
        counts = [res.get_counts(i) for i in range(len(vecs))]
        source = f"{a.noise} (noisy simulation)"
    for (name, v), cnt in zip(vecs, counts):
        pn = outcome_probs(cnt, n, no)
        term["noisy"][name] = term[name] * (pn[G].sum() - bg)
    print(f"measured on {source}")
    maps = {}
    for kind in ("exact", "prep", "noisy"):
        tk = term[kind]
        maps[kind] = np.array([(tk["full"] + tk[f"in{j}"] - tk[f"out{j}"]) / 2 for j in range(g * g)]).reshape(g, g)
    direct = np.zeros(g * g)                                                 # per-amplitude evidence, summed by block
    Mb = fastsim.run_inverse(fastsim.run(b[None].to(torch.complex128), wc, depth, torch.complex128)
                             * torch.tensor(np.isin(np.arange(2 ** n) >> (n - no), groups[a.picture])), wc, depth,
                             torch.complex128)[0].real.numpy() - bg * b.numpy()
    np.add.at(direct, block, b.numpy() * Mb)
    print(f"{names[a.picture]} picture {idx}, {g}x{g} blocks, {len(vecs)} circuits, preparation depth {a.prep_depth}: "
          f"fidelity min {min(fids):.3f} median {np.median(fids):.3f}")
    print(f"check: polarization (exact) vs per-amplitude sum, max diff {np.abs(maps['exact'].ravel() - direct).max():.2e}")
    for kind in ("exact", "prep", "noisy"):
        m = maps[kind]
        print(f"{kind:5s}: sum {m.sum():+.4f}  blocks " + " ".join(f"{v:+.4f}" for v in m.ravel()))
    c = lambda u, v: float(np.corrcoef(u.ravel(), v.ravel())[0, 1])
    print(f"correlation with exact: prep {c(maps['prep'], maps['exact']):.3f}, noisy {c(maps['noisy'], maps['exact']):.3f}")
    hw = "_hw" if a.keys else ""
    out = HERE / "results" / f"ibm_heatmap_{pathlib.Path(a.run).stem[:40]}_p{a.picture}_g{g}_L{a.prep_depth}{hw}.json"
    out.write_text(json.dumps(dict(run=a.run, picture=a.picture, grid=g, prep_depth=a.prep_depth, shots=a.shots,
                                   measured_on=source, ledger=ledger, fidelities=fids, maps={k: v.tolist() for k, v in maps.items()})))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("run")
    p.add_argument("--picture", type=int, default=0, help="class index; its first test picture is used")
    p.add_argument("--prep_depth", type=int, default=16)
    p.add_argument("--noise", default="FakeFez", help="fake backend for the noisy simulation ('' to skip)")
    p.add_argument("--shots", type=int, default=4000)
    r = sub.add_parser("run")
    r.add_argument("demo")
    r.add_argument("--backend", default="ibm_kingston")
    r.add_argument("--dry_run", action="store_true", help="write the ledger folder, do not send")
    r.add_argument("--shots", type=int, default=4000)
    r.add_argument("--keys", default="", help="file of accounts, one per line: <email> <token> (keep it out of git)")
    r.add_argument("--account", type=int, default=0, help="line of --keys to use")
    h = sub.add_parser("heatmap")
    h.add_argument("run")
    h.add_argument("--picture", type=int, default=0)
    h.add_argument("--grid", type=int, default=2, help="blocks per side")
    h.add_argument("--prep_depth", type=int, default=8)
    h.add_argument("--steps", type=int, default=1500, help="Adam steps per fitted preparation")
    h.add_argument("--noise", default="FakeKingston")
    h.add_argument("--shots", type=int, default=4000)
    h.add_argument("--keys", default="", help="keys file (as for run): measure on hardware instead of --noise")
    h.add_argument("--account", type=int, default=0)
    h.add_argument("--backend", default="ibm_kingston")
    h.add_argument("--dry_run", action="store_true", help="with --keys: write the ledger folder, do not send")
    a = ap.parse_args()
    {"prepare": prepare, "run": run_hw, "heatmap": block_heatmap}[a.cmd](a)


if __name__ == "__main__":
    main()
