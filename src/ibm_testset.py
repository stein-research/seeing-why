#!/usr/bin/env python3
"""The whole test set of a trained circuit on IBM hardware, one circuit per picture (owner 2026-10-31: all 300 Caltech
test pictures, spread over several accounts).

Accuracy is reported in four steps, so whatever is lost on the chip can be told apart:
  exact   the masked picture b as the input state, no noise (= the trainer's test accuracy)
  prep    the fitted RY + CNOT preparation V|0> ~ b instead of b (fidelity |<b|V|0>|^2), no noise
  noisy   the same circuits under a fake IBM backend's noise model (Qiskit Aer), --shots shots
  chip    the same circuits on IBM hardware, through the job ledger (ibm_jobs.py)

  fit:    fits every picture's preparation, all pictures at once on the GPU (a batched RY + CNOT brickwall simulator,
          checked against fastsim), then the exact and prep accuracies; cached in results/ibm_testset_<tag>/
  noisy:  the noisy-simulation accuracy (the first --first pictures of the class-interleaved order, or all)
  chip:   sends positions [--start, --start + --count) of the class-interleaved order (one of each class in turn) in jobs of --chunk circuits, job j from account line
          --accounts[j mod len]; every job goes through the ledger (never sent twice; --dry_run writes, sends nothing).
          Results gather in results/ibm_testset_<tag>/chip_<backend>_r<repeat>.json, picture by picture, so a run can be spread
          over days; --repeat k sends the same circuits again as a new job (run-to-run spread), kept apart.
  report: the four accuracies on the pictures each step has, and the chip's on the pictures it has

Run:  python src/ibm_testset.py fit    results/<run>.json [--prep_depth 8] [--steps 3000]
      python src/ibm_testset.py noisy  results/<run>.json [--first 300] [--noise FakeKingston]
      python src/ibm_testset.py chip   results/<run>.json --keys <keys file> --accounts 0,1
                                                   [--start 0 --count 30 --chunk 30 --backend ibm_kingston] [--dry_run]
      python src/ibm_testset.py report results/<run>.json
      python src/ibm_testset.py backends results/<run>.json --keys <file>   (which machine: reads only)
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

import fastsim
import pilot_fmnist as pf
from heatmap import NAMES
from ibm_demo import FILES, brickwall, class_probs, outcome_probs

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def folder_for(a):
    f = HERE / "results" / f"ibm_testset_{pathlib.Path(a.run).stem[:60]}_L{a.prep_depth}"
    f.mkdir(exist_ok=True)
    return f


def test_states(run):
    """(B, 2^n) real masked pictures b (normalised), labels, the circuit weights wc, depth."""
    pf.DATASET, pf.ORDER = run["dataset"], run.get("order", "morton")
    pf.N_QUBITS = n = len(run["weights"][0])
    if run.get("ent") != "brickwall" or run.get("fourier_bank", 0) > 1 or run.get("optical"):
        raise SystemExit("a brickwall run with no mask or a single Fourier mask")
    d = np.load(HERE / "data" / FILES.get(pf.DATASET, f"{pf.DATASET}.npz"))
    split = "val" if "y_val" in d else "test"
    X, Y = d[f"x_{split}"].astype(np.float64), d[f"y_{split}"]
    s = pf.side()
    perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
    x = X.transpose(0, 3, 1, 2).reshape(len(X), -1, s * s)[:, :, perm].reshape(len(X), -1)
    if run.get("center_image"):
        x = pf.centre_planes(x, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    x = torch.tensor(np.pad(x, ((0, 0), (0, 2 ** n - x.shape[1]))), dtype=torch.complex128)
    w = torch.tensor(run["weights"], dtype=torch.float64)
    fourier = (torch.tensor(perm), s, 2 ** n // s ** 2, run.get("fourier_phase", False),
               run.get("fourier_linear", False)) if run.get("fourier_mask") else None
    wc, kw = fastsim.split_masks(w, run["depth"], 0, -1, fourier)
    b = fastsim.fourier_filter(x, kw["fmask"][1], kw["fmask"][0], kw["fmask"][2]) if fourier else x
    if b.imag.abs().max() > 1e-9:
        raise SystemExit("the filtered pictures are not real; the RY-only preparation cannot reach them")
    b = b.real / (b.abs() ** 2).sum(1, keepdim=True) ** 0.5
    return b, Y, wc, run["depth"]


def ry_brickwall(ang, n):
    """Batched RY + CNOT brickwall from |0>: ang (B, L+1, n) -> real states (B, 2^n), fastsim's wire order and CNOTs
    (wire 0 most significant; CNOT(i, i+1) for i = k mod 2 after rotation layer k)."""
    B, L1, _ = ang.shape
    st = torch.zeros(B, 2 ** n, dtype=ang.dtype, device=ang.device)
    st[:, 0] = 1
    idx = torch.arange(2 ** n, device=ang.device)
    perms = []
    for k in range(2):                                          # CNOT layer k as an index permutation (an involution)
        j = idx.clone()
        for i in range(k, n - 1, 2):
            c, t = n - 1 - i, n - 2 - i                         # bit positions of wires i and i + 1
            j = torch.where((j >> c) & 1 == 1, j ^ (1 << t), j)
        perms.append(j)
    for k in range(L1):
        c, s = torch.cos(ang[:, k] / 2), torch.sin(ang[:, k] / 2)
        for i in range(n):
            v = st.reshape(B, 2 ** i, 2, 2 ** (n - i - 1))
            c_, s_ = c[:, i, None, None], s[:, i, None, None]
            st = torch.stack([c_ * v[:, :, 0] - s_ * v[:, :, 1], s_ * v[:, :, 0] + c_ * v[:, :, 1]], 2).reshape(B, -1)
        if k < L1 - 1:
            st = st[:, perms[k % 2]]
    return st


def fit_batch(b, L, steps, seed=0):
    """Fitted angles (B, L+1, n) and fidelities |<b|V|0>|^2, every picture at once (Adam, best step kept)."""
    B, N = b.shape
    n = N.bit_length() - 1
    torch.manual_seed(seed)
    b = b.to(DEV)
    ang = torch.nn.Parameter(0.1 * torch.randn(B, L + 1, n, dtype=torch.float64, device=DEV))
    opt = torch.optim.Adam([ang], lr=0.05)
    best_f = torch.zeros(B, dtype=torch.float64, device=DEV)
    best_a = ang.detach().clone()
    for t in range(steps):
        f = (ry_brickwall(ang, n) * b).sum(1) ** 2
        better = f.detach() > best_f
        best_f = torch.where(better, f.detach(), best_f)
        best_a[better] = ang.detach()[better]
        opt.zero_grad()
        (1 - f).sum().backward()
        opt.step()
        if t == steps // 2:
            for g in opt.param_groups:
                g["lr"] = 0.01
    return best_a.cpu(), best_f.cpu()


def interleaved(y):
    """Picture indices in the order the noisy and chip steps take them: one of each class in turn (the test set is
    sorted by class, so its first 30 pictures are all airplanes); any first N pictures then mix the classes evenly."""
    y = np.asarray(y)
    rank = np.zeros(len(y), int)
    for c in np.unique(y):
        rank[y == c] = np.arange((y == c).sum())
    return np.lexsort((y, rank))


def outcome_batch(states, wc, depth, no):
    out = []
    for i in range(0, len(states), 50):
        out.append(fastsim.probs(fastsim.run(states[i:i + 50].to(torch.complex128), wc, depth, torch.complex128), no))
    return torch.cat(out).numpy()


def predictions(run, P):
    return np.array([class_probs(run, p / p.sum()) for p in P])


def fit(a):
    run = json.loads(pathlib.Path(a.run).read_text())
    b, Y, wc, depth = test_states(run)
    n, no = b.shape[1].bit_length() - 1, run["n_out"]
    chk = 0.3 * torch.randn(3, a.prep_depth + 1, n, dtype=torch.float64)          # batched simulator vs fastsim
    zero = torch.zeros(1, 2 ** n, dtype=torch.float64)
    zero[0, 0] = 1
    ref = torch.cat([fastsim.run(zero, torch.stack([c, torch.zeros_like(c)], -1), a.prep_depth, torch.complex128)
                     for c in chk])
    err = (ry_brickwall(chk, n) - ref.real).abs().max().item() + ref.imag.abs().max().item()
    print(f"batched RY brickwall vs fastsim: max |diff| {err:.1e}", flush=True)
    assert err < 1e-10
    t0 = time.time()
    ang, fid = fit_batch(b, a.prep_depth, a.steps)
    print(f"fitted {len(b)} preparations (depth {a.prep_depth}, {a.steps} steps) in {time.time() - t0:.0f} s: "
          f"fidelity median {fid.median():.3f}, min {fid.min():.3f}, max {fid.max():.3f}", flush=True)
    va = ry_brickwall(ang, n)
    q_exact = predictions(run, outcome_batch(b, wc, depth, no))
    q_prep = predictions(run, outcome_batch(va, wc, depth, no))
    acc = lambda q: float((q.argmax(1) == Y).mean())
    print(f"accuracy: exact {acc(q_exact):.3f} (trainer: {run.get('test_acc', '?')}), prep {acc(q_prep):.3f}; "
          f"same prediction exact vs prep {(q_exact.argmax(1) == q_prep.argmax(1)).mean():.3f}")
    f = folder_for(a)
    if list(f.glob("chip_*_r*.json")) and not a.force:            # new angles = new circuits = every job sent again
        raise SystemExit(f"{f} has chip results: refitting would change every circuit (their jobs would be new); "
                         f"--force to refit anyway")
    np.savez(f / "preps.npz", ang=ang.numpy(), fid=fid.numpy(), y=Y, wc=wc.numpy(), q_exact=q_exact, q_prep=q_prep)
    (f / "fit.json").write_text(json.dumps(dict(run=a.run, prep_depth=a.prep_depth, steps=a.steps, n=len(Y),
                                                fid_median=float(fid.median()), acc_exact=acc(q_exact),
                                                acc_prep=acc(q_prep))))
    print(f / "preps.npz")


def circuits(run, a, sel):
    from qiskit import QuantumCircuit
    pz = np.load(folder_for(a) / "preps.npz")
    n = len(run["weights"][0])
    wc = pz["wc"]                                               # the circuit after the mask, as fit() split it
    wires = list(range(n - 1, -1, -1))
    out = []
    for i in sel:
        qc = QuantumCircuit(n, name=f"picture{i}")
        brickwall(qc, wires, pz["ang"][i], a.prep_depth, "ry")
        brickwall(qc, wires, wc, run["depth"], "rzryrx")
        qc.measure_all()
        out.append(qc)
    return out, pz


def noisy(a):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    run = json.loads(pathlib.Path(a.run).read_text())
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    pz = np.load(folder_for(a) / "preps.npz")
    sel = interleaved(pz["y"])[:a.first].tolist()
    circs, _ = circuits(run, a, sel)
    fake = getattr(fake_provider, a.noise)()
    sim = AerSimulator.from_backend(fake)
    t0 = time.time()
    if a.seeds > 1:                                             # as ibm_jobs sends them: best of a.seeds by calibration
        from ibm_jobs import best_transpile
        tq, info = best_transpile(circs, fake, a.seeds)
        print(f"transpiled, best of {a.seeds} seeds: expected fidelity median {np.median([i[1] for i in info]):.3f} "
              f"(seed 0 alone {np.median([i[2] for i in info]):.3f}), {time.time() - t0:.0f} s", flush=True)
    else:
        tq = transpile(circs, fake, optimization_level=3, seed_transpiler=0)
    P = []
    for i in range(0, len(tq), 10):
        res = sim.run(tq[i:i + 10], shots=a.shots, seed_simulator=i).result()
        P += [outcome_probs(res.get_counts(j), len(circs[0].qubits), run["n_out"]) for j in range(len(tq[i:i + 10]))]
        print(f"  {i + len(tq[i:i + 10])}/{len(tq)} pictures, {time.time() - t0:.0f} s", flush=True)
    q = predictions(run, np.array(P))
    y = pz["y"][sel]
    acc = float((q.argmax(1) == y).mean())
    print(f"noisy ({a.noise}, {a.shots} shots), {len(sel)} pictures: accuracy {acc:.3f}; on the same pictures exact "
          f"{(pz['q_exact'][sel].argmax(1) == y).mean():.3f}, prep {(pz['q_prep'][sel].argmax(1) == y).mean():.3f}")
    np.savez(folder_for(a) / f"noisy_{a.noise}{'' if a.seeds == 1 else f'_s{a.seeds}'}.npz", sel=sel, q=q, outcomes=np.array(P), shots=a.shots)


def chip(a):
    from ibm_jobs import QuotaShort, sampler_counts
    run = json.loads(pathlib.Path(a.run).read_text())
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    f = folder_for(a)
    out = f / f"chip_{a.backend}_r{a.repeat}.json"                 # one file per machine: never mixed
    store = json.loads(out.read_text()) if out.exists() else {}
    accts = [int(v) for v in a.accounts.split(",")]
    names = NAMES[run["dataset"]]
    order = interleaved(np.load(f / "preps.npz")["y"])          # positions --start.. in the class-interleaved order
    todo = [int(i) for i in order[a.start:a.start + a.count] if str(int(i)) not in store]                             # pictures this repeat already has are never resent
    if todo and not a.dry_run:                                  # jobs sent but never collected (a closed run): collect
        from ibm_jobs import collect_pending
        collect_pending(a.keys)
    if todo:                                                    # measured in an earlier job (any chunking): take them
        from ibm_jobs import lookup
        circs, pz = circuits(run, a, todo)
        for i, c in zip(list(todo), circs):
            cnt = lookup(c, a.backend, a.shots, a.repeat)
            if cnt is not None:
                p = outcome_probs(cnt, len(c.qubits), run["n_out"])
                q = class_probs(run, p)
                store[str(i)] = dict(label=int(pz["y"][i]), pred=int(q.argmax()), q=q.tolist(), outcomes=p.tolist(),
                                     backend=a.backend, ledger="lookup")
                todo.remove(i)
        out.write_text(json.dumps(store))
    if not todo:
        print(f"positions {a.start}-{a.start + a.count - 1}: all already measured (repeat {a.repeat}); nothing to send")
        return
    for j, i0 in enumerate(range(0, len(todo), a.chunk)):
        sel = todo[i0:i0 + a.chunk]
        circs, pz = circuits(run, a, sel)
        counts = None
        for k in range(len(accts)):                             # this chunk's account, then the others in turn
            acc = accts[(j + k) % len(accts)]
            try:
                counts, folder = sampler_counts(
                    circs, a.backend, a.shots, a.keys, acc, repeat=a.repeat,
                    purpose=f"test set (repeat {a.repeat}): pictures {sel} of {pathlib.Path(a.run).name}, "
                            f"prep depth {a.prep_depth}",
                    inputs=[a.run, f / "preps.npz"],
                    circuit_notes=[f"picture {i} ({names[pz['y'][i]]}), prep fidelity {pz['fid'][i]:.3f}" for i in sel],
                    dry_run=a.dry_run)
                break
            except QuotaShort as e:
                print(e, flush=True)
        else:
            print(f"no account has quota for pictures {sel}: stopped; run the same command later and it "
                  f"continues from the pictures still missing ({len(store)} measured so far)")
            return
        if counts is None:
            continue
        for i, c in zip(sel, counts):
            p = outcome_probs(c, len(circs[0].qubits), run["n_out"])
            q = class_probs(run, p)
            store[str(i)] = dict(label=int(pz["y"][i]), pred=int(q.argmax()), q=q.tolist(), outcomes=p.tolist(),
                                 backend=a.backend, ledger=folder.name)
        out.write_text(json.dumps(store))
        got = [v for v in store.values()]
        print(f"chip so far: {len(got)} pictures, accuracy {np.mean([v['pred'] == v['label'] for v in got]):.3f}",
              flush=True)


def backends(a):
    """Scores the first 30 test circuits on each named machine's current calibration (reads it; sends no job): the
    expected fidelity after the best-of-8 transpile, two-qubit gate count, the chip's median errors and its queue."""
    import ibm_jobs
    run = json.loads(pathlib.Path(a.run).read_text())
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    pz = np.load(folder_for(a) / "preps.npz")
    circs, _ = circuits(run, a, interleaved(pz["y"])[:30].tolist())
    service, _, _ = ibm_jobs.service_for(a.keys, line=a.account)
    for name in a.names.split(","):
        b = service.backend(name)
        tq, info = ibm_jobs.best_transpile(circs, b, 8)
        t, st = b.target, b.status()
        cz = [v.error for v in t["cz"].values() if v and v.error is not None] if "cz" in t.operation_names else [np.nan]
        ro = [v.error for v in t["measure"].values() if v and v.error is not None]
        print(f"{name:14s} expected fidelity median {np.median([i[1] for i in info]):.3f} | 2q gates "
              f"{np.median([tc.count_ops().get('cz', 0) for tc in tq]):.0f} | chip median cz err {np.median(cz):.4f}, "
              f"readout err {np.median(ro):.4f} | queue {st.pending_jobs}, {'up' if st.operational else 'DOWN'}")


def scores(q, y):
    """accuracy, macro one-vs-rest ROC-AUC (as auc.py) and the confusion matrix (rows true, columns predicted)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import label_binarize
    q = np.asarray(q, dtype=float)
    pred = q.argmax(1)
    present = np.unique(y)
    auc = roc_auc_score(label_binarize(y, classes=np.arange(q.shape[1]))[:, present], q[:, present], average="macro") \
        if len(present) > 1 else float("nan")
    cm = np.zeros((q.shape[1], q.shape[1]), int)
    np.add.at(cm, (y, pred), 1)
    return float((pred == y).mean()), float(auc), cm


def report(a):
    """Every step on the pictures the chip (or else the noisy simulation) has, so the steps compare like for like:
    accuracy, ROC-AUC and the confusion matrix of the last step."""
    run = json.loads(pathlib.Path(a.run).read_text())
    names = NAMES[run["dataset"]]
    f = folder_for(a)
    pz = np.load(f / "preps.npz")
    y = pz["y"]
    print(f"{len(y)} pictures; preparation fidelity median {np.median(pz['fid']):.3f}")
    steps = {"exact": (np.arange(len(y)), pz["q_exact"]), "prep": (np.arange(len(y)), pz["q_prep"])}
    for nz in sorted(f.glob("noisy_*.npz")):
        d = np.load(nz)
        steps[nz.stem] = (d["sel"], d["q"])
    for cj in sorted(f.glob("chip_*_r*.json")):
        st = json.loads(cj.read_text())
        s = np.array(sorted(int(k) for k in st))
        steps[cj.stem] = (s, np.array([st[str(i)]["q"] for i in s]))
    chips = [k for k in steps if k.startswith("chip")]
    for name in [k for k in steps if k.startswith("noisy")]:   # a partial simulation is shown on its own pictures
        s, q = steps[name]
        if chips and len(s) < len(steps[chips[0]][0]):
            acc, auc, _ = scores(q, y[s])
            acc0, auc0, _ = scores(pz["q_exact"][s], y[s])
            print(f"  {name} on its {len(s)} pictures: accuracy {acc:.3f} ROC-AUC {auc:.3f} (exact there {acc0:.3f} / {auc0:.3f})")
            del steps[name]
    common = set(range(len(y)))
    for s, _ in steps.values():
        common &= set(s.tolist())
    common = np.array(sorted(common))
    print(f"on the {len(common)} pictures every step has:")
    last = None
    for name, (s, q) in steps.items():
        pos = {v: k for k, v in enumerate(s.tolist())}
        qc = q[[pos[i] for i in common]]
        acc, auc, cm = scores(qc, y[common])
        print(f"  {name:20s} accuracy {acc:.3f}  ROC-AUC {auc:.3f}")
        last = (name, cm)
    name, cm = last
    w = max(len(n) for n in names)
    print(f"confusion matrix, {name} (rows true, columns predicted):")
    print(" " * (w + 1) + " ".join(f"{n[:4]:>4s}" for n in names))
    for i, n in enumerate(names):
        print(f"{n:>{w}s} " + " ".join(f"{v:4d}" for v in cm[i]))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("fit", "noisy", "chip", "report", "backends")}
    for p in ps.values():
        p.add_argument("run")
        p.add_argument("--prep_depth", type=int, default=8)
    ps["fit"].add_argument("--steps", type=int, default=3000)
    ps["fit"].add_argument("--force", action="store_true", help="refit even when chip results exist")
    ps["noisy"].add_argument("--first", type=int, default=300)
    ps["noisy"].add_argument("--noise", default="FakeKingston")
    ps["noisy"].add_argument("--shots", type=int, default=4000)
    ps["noisy"].add_argument("--seeds", type=int, default=1, help="transpiler seeds tried, best by calibration kept")
    c = ps["chip"]
    c.add_argument("--keys", required=True, help="keys file, one '<email> <token>' per line (outside git)")
    c.add_argument("--accounts", default="0", help="account lines used in turn, one job each, e.g. 0,1,2")
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--count", type=int, default=30)
    c.add_argument("--chunk", type=int, default=30, help="circuits per job")
    c.add_argument("--shots", type=int, default=4000)
    c.add_argument("--backend", default="ibm_kingston")
    c.add_argument("--repeat", type=int, default=0, help="repeat index: the same circuits sent again as a new job "
                   "(to measure run-to-run spread); 0 = first run")
    c.add_argument("--dry_run", action="store_true")
    ps["backends"].add_argument("--keys", required=True)
    ps["backends"].add_argument("--account", type=int, default=0)
    ps["backends"].add_argument("--names", default="ibm_kingston,ibm_fez,ibm_marrakesh")
    a = ap.parse_args()
    {"backends": backends, "fit": fit, "noisy": noisy, "chip": chip, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
