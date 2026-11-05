#!/usr/bin/env python3
"""Block evidence maps on IBM hardware for pictures drawn at random, fixed before any of them is measured (two referee
reviews, 2026-11-03: the first maps came from ten pictures chosen among the chip's correct, confident answers, and the
whole-picture term of the first run came from the very measurement that chose them).

Pre-registered (step `prereg`, written and committed before anything is fitted or run):
  - 30 test pictures, 3 per class, drawn with numpy default_rng(20261103) among the 30 of each class; correct or not,
    confident or not, on the chip or anywhere else; and 4 of these 30, drawn with the same generator, for 4 x 4 maps;
  - the model: D' seed 0 (the hardware model of the paper), the true class's evidence, 4,000 shots per circuit,
    every term measured in the same job (the whole picture's circuit included, nothing reused).
Two estimators of a block's evidence E_S, from class terms f(v) = ||v||^2 (P_g(v / ||v||) - b_g):
  standard  E_S = [f(psi) + f(psi_S) - f(psi_notS)] / 2                    (the paper's Eq. polar)
  sign flip E_S = f(psi_S) + [f(psi) - f(psi_S - psi_notS)] / 4            (psi_S - psi_notS has unit norm)
The 2 x 2 maps measure psi, psi_S, psi_notS and psi_S - psi_notS for every block (13 circuits a picture), so both
estimators come from the same job; the 4 x 4 maps use the sign flip only (1 + 2 * 16 = 33 circuits a picture).
Compared with three references: the exact maps of the ideal masked state, the maps of the fitted states (what the chip
is asked to prepare, noise-free), and the maps of the noisy simulation of the same circuits.
Reported: pooled and within-picture-centred correlation, slope through zero, MAE, sign agreement, the sum rule against
the measured whole-picture term and against the exact P - b, by group (correct / wrong on the chip's test-set run).

Run:  python src/ibm_maps_random.py prereg            -> results/<testset folder>/random/prereg.json
      python src/ibm_maps_random.py fit [--steps 3000] -> random/states.npz (fitted loading circuits)
      python src/ibm_maps_random.py noisy               -> random/noisy_FakeKingston.npz
      python src/ibm_maps_random.py report [kind ...]   -> numbers, random/report.json
      python src/ibm_maps_random.py chip --keys <file> --accounts 3 [--dry_run]
"""
import argparse
import json
import pathlib
import time

import numpy as np
import torch

from ibm_heatmaps import setup
from ibm_testset import fit_batch, folder_for, outcome_batch, ry_brickwall

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
RUN = HERE / "results" / "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json"
SEED, PER_CLASS, N4, SHOTS = 20261103, 3, 4, 4000


def args_for(a):
    return argparse.Namespace(run=str(RUN), prep_depth=8, **{k: v for k, v in vars(a).items() if k not in ("run",)})


def folder(a):
    f = folder_for(args_for(a)) / "random"
    f.mkdir(exist_ok=True)
    return f


def prereg(a):
    f = folder(a)
    if (f / "prereg.json").exists():
        raise SystemExit(f"{f / 'prereg.json'} exists: the selection is fixed, never drawn again")
    _, _, Y, *_ = setup(args_for(a))
    rng = np.random.default_rng(SEED)
    pics = [int(i) for c in range(10) for i in np.sort(rng.choice(np.flatnonzero(Y == c), PER_CLASS, replace=False))]
    pics4 = [int(i) for i in np.sort(rng.choice(pics, N4, replace=False))]
    rec = dict(seed=SEED, pictures_2x2=pics, pictures_4x4=pics4, model=RUN.name, shots=SHOTS, cls="true class",
               estimators=["standard", "sign flip"], references=["exact", "prep", "noisy"],
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               note="drawn with default_rng(seed): 3 per class with choice() over that class's test pictures in class "
                    "order, then 4 of the 30 for 4x4 maps; nothing about the chip's answers was used")
    (f / "prereg.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


def block_states(b, block, G, flip_only):
    """For one picture's masked state b: the list of (kind, block j, normalised state, squared norm)."""
    out = []
    for j in range(G * G):
        ins = torch.tensor(block[G] == j)
        vs = b * ins
        vo = b * ~ins
        kinds = [("in", vs), ("flip", vs - vo)] if flip_only else [("in", vs), ("out", vo), ("flip", vs - vo)]
        for kind, v in kinds:
            nv = float((v ** 2).sum())
            out.append((kind, j, v / nv ** 0.5, nv))
    return out


def blocks_for(G, pixel, s):
    return (pixel // s // (s // G)) * G + pixel % s // (s // G)


def layout(a):
    """Every state to prepare, in order: for each 2x2 picture its blocks (in, out, flip), then for each 4x4 picture
    its blocks (in, flip), then the whole pictures (2x2 set; the 4x4 pictures are among them)."""
    run, b, Y, wc, depth, n, no, _, groups = setup(args_for(a))
    pr = json.loads((folder(a) / "prereg.json").read_text())
    import pilot_fmnist as pf
    s = pf.side()
    perm = pf.morton_perm() if pf.permuted() else np.arange(s * s)
    pixel = np.asarray(perm)[np.arange(2 ** n) % (s * s)]
    block = {G: blocks_for(G, pixel, s) for G in (2, 4)}
    rows = []
    for G, pics, flip_only in ((2, pr["pictures_2x2"], False), (4, pr["pictures_4x4"], True)):
        for i in pics:
            for kind, j, v, nv in block_states(b[i], block, G, flip_only):
                rows.append(dict(G=G, pic=i, kind=kind, block=j, norm=nv, state=v))
    for i in pr["pictures_2x2"]:
        rows.append(dict(G=0, pic=i, kind="whole", block=-1, norm=1.0, state=b[i]))
    return run, b, Y, wc, depth, n, no, groups, rows


def fit(a):
    run, b, Y, wc, depth, n, no, groups, rows = layout(a)
    f = folder(a)
    pz = np.load(folder_for(args_for(a)) / "preps.npz")
    V = torch.stack([r["state"] for r in rows if r["kind"] != "whole"])
    t0 = time.time()
    ang_b, fid_b = fit_batch(V, 8, a.steps)
    print(f"fitted {len(V)} block states in {time.time() - t0:.0f} s: fidelity median {fid_b.median():.3f}, "
          f"min {fid_b.min():.3f}", flush=True)
    whole = [r["pic"] for r in rows if r["kind"] == "whole"]
    ang = np.concatenate([ang_b.numpy(), pz["ang"][whole]])     # whole pictures: the test set's own fitted circuits
    fid = np.concatenate([fid_b.numpy(), pz["fid"][whole]])
    W = torch.tensor(pz["wc"])
    S = torch.stack([r["state"] for r in rows])
    P_ex = outcome_batch(S, W, depth, no)
    P_pr = outcome_batch(ry_brickwall(torch.tensor(ang), n), W, depth, no)
    meta = {k: np.array([r[k] for r in rows]) for k in ("G", "pic", "kind", "block", "norm")}
    np.savez(f / "states.npz", ang=ang, fid=fid, P_ex=P_ex, P_pr=P_pr, **meta)
    print(f / "states.npz")


def circuits(a, ang):
    from qiskit import QuantumCircuit
    from ibm_demo import brickwall
    run = json.loads(RUN.read_text())
    pz = np.load(folder_for(args_for(a)) / "preps.npz")
    n = len(run["weights"][0])
    wires = list(range(n - 1, -1, -1))
    out = []
    for k in range(len(ang)):
        qc = QuantumCircuit(n)
        brickwall(qc, wires, ang[k], 8, "ry")
        brickwall(qc, wires, pz["wc"], run["depth"], "rzryrx")
        qc.measure_all()
        out.append(qc)
    return out


def noisy(a):
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import fake_provider
    f = folder(a)
    h = np.load(f / "states.npz")
    run = json.loads(RUN.read_text())
    n, no = len(run["weights"][0]), run["n_out"]
    fake = getattr(fake_provider, a.noise)()
    tq = transpile(circuits(a, h["ang"]), fake, optimization_level=3, seed_transpiler=0)
    sim = AerSimulator.from_backend(fake, method="density_matrix")
    for tc in tq:
        lay = tc.layout.final_index_layout()
        tc.remove_final_measurements()
        tc.save_probabilities(qubits=lay)
    P, t0 = [], time.time()
    for i in range(0, len(tq), 4):
        res = sim.run(tq[i:i + 4]).result()
        for j in range(len(tq[i:i + 4])):
            full = np.asarray(res.data(j)["probabilities"])
            P.append(np.bincount(np.arange(2 ** n) >> (n - no), weights=full, minlength=2 ** no))
        if len(P) % 40 == 0:
            print(f"  {len(P)}/{len(tq)} circuits, {time.time() - t0:.0f} s", flush=True)
    np.savez(f / f"noisy_{a.noise}.npz", P=np.array(P), shots=0)
    print(f / f"noisy_{a.noise}.npz")


def chip_jobs(h, per_job):
    """The planned jobs, in sending order: [(G, repeat, pictures, row indices into states.npz)]. A 2x2 job holds its
    pictures' blocks and whole pictures; a 4x4 job its pictures' blocks and their whole pictures once more (every term
    of a map from the same job). repeat 2 / 3: the whole-picture circuits were measured before (the test-set run at
    repeat 0, the first maps' second run at repeat 1), so these jobs are marked as measuring them again on purpose."""
    jobs = []
    for G, repeat in ((2, 2), (4, 3)):
        pics = sorted({int(p) for p in h["pic"][h["G"] == G]})
        for c in range(0, len(pics), per_job[G]):
            part = pics[c:c + per_job[G]]
            rows = [int(k) for k in np.flatnonzero((h["G"] == G) & np.isin(h["pic"], part))]
            rows += [int(k) for k in np.flatnonzero((h["kind"] == "whole") & np.isin(h["pic"], part))]
            jobs.append((G, repeat, part, rows))
    return jobs


def chip(a):
    """Send the planned jobs through the ledger (a job already done is read back, never sent again), each to the first
    account in --accounts with quota for it, and write random/chip_<backend>.npz: P (rows of states.npz, NaN where not
    measured; whole pictures from the 2x2 jobs), P_whole4 / pic_whole4 (the 4x4 jobs' own whole pictures)."""
    from ibm_demo import outcome_probs
    from ibm_jobs import QuotaShort, sampler_counts
    f = folder(a)
    h = np.load(f / "states.npz")
    run = json.loads(RUN.read_text())
    n, no = len(run["weights"][0]), run["n_out"]
    circs = circuits(a, h["ang"])
    P = np.full((len(circs), 2 ** no), np.nan)
    P4, ledger = {}, []
    accts = [int(v) for v in a.accounts.split(",")]
    jobs = [j for j in chip_jobs(h, {2: a.per_job_2x2, 4: a.per_job_4x4}) if str(j[0]) in a.parts]
    for G, repeat, pics, rows in jobs:
        notes = [f"{G}x{G} map, picture {h['pic'][k]}, " + (f"block {h['block'][k]} {h['kind'][k]}, squared norm "
                 f"{h['norm'][k]:.5f}" if h["kind"][k] != "whole" else "whole picture") for k in rows]
        counts = None
        for acc in accts:
            try:
                counts, fl = sampler_counts([circs[k] for k in rows], a.backend, SHOTS, a.keys, acc, repeat=repeat,
                                            purpose=f"pre-registered random pictures: {G}x{G} block maps of test "
                                                    f"pictures {pics}, {RUN.name}",
                                            inputs=[RUN, f / "prereg.json", f / "states.npz"], circuit_notes=notes,
                                            dry_run=a.dry_run)
                break
            except QuotaShort as e:
                print(e, flush=True)
        else:
            print(f"{G}x{G} pictures {pics}: no account has quota for this job - stopping", flush=True)
            break
        if counts is None:                                      # dry run
            continue
        ledger.append(fl.name)
        for k, c in zip(rows, counts):
            p = outcome_probs(c, n, no)
            if G == 4 and h["kind"][k] == "whole":
                P4[int(h["pic"][k])] = p
            else:
                P[k] = p
        print(f"{G}x{G} pictures {pics}: done ({fl.name})", flush=True)
    if a.dry_run:
        return
    out = f / f"chip_{a.backend}.npz"
    if out.exists():                                            # keep what earlier runs of this step measured
        old = np.load(out)
        P = np.where(np.isnan(P), old["P"], P)
        P4 = {**{int(i): p for i, p in zip(old["pic_whole4"], old["P_whole4"])}, **P4}
        ledger = list(old["ledger"]) + [x for x in ledger if x not in old["ledger"]]
    k4 = sorted(P4)
    np.savez(out, P=P, P_whole4=np.array([P4[i] for i in k4]).reshape(-1, 2 ** no), pic_whole4=np.array(k4, int),
             shots=SHOTS, ledger=np.array(ledger))
    print(out)


def maps(h, P, Y, groups, no, G, estimator, whole4=None):
    """{picture: (G*G,) block evidence} from the class terms of the measured or simulated distributions P; pictures
    with a term not measured are left out. whole4: {picture: distribution} of the whole pictures for the 4x4 maps,
    when they were measured on their own (the chip)."""
    def term(p, k, i):
        g = np.isin(np.arange(2 ** no), groups[Y[i]])
        return h["norm"][k] * (p[g].sum() / p.sum() - len(groups[Y[i]]) / 2 ** no)
    f = lambda k, i: term(P[k], k, i)
    whole = {int(h["pic"][k]): f(k, int(h["pic"][k])) for k in np.flatnonzero(h["kind"] == "whole")}
    if G == 4 and whole4 is not None:
        k_whole = {int(h["pic"][k]): k for k in np.flatnonzero(h["kind"] == "whole")}
        whole = {i: term(p, k_whole[i], i) for i, p in whole4.items()}
    out = {}
    for i in sorted({int(p) for p in h["pic"][h["G"] == G]}):
        if i not in whole or np.isnan(whole[i]) or np.isnan(P[(h["G"] == G) & (h["pic"] == i)]).any():
            continue
        E = np.zeros(G * G)
        for j in range(G * G):
            sel = lambda kind: int(np.flatnonzero((h["G"] == G) & (h["pic"] == i) & (h["block"] == j) & (h["kind"] == kind))[0])
            if estimator == "standard":
                E[j] = (whole[i] + f(sel("in"), i) - f(sel("out"), i)) / 2
            else:
                E[j] = f(sel("in"), i) + (whole[i] - f(sel("flip"), i)) / 4
        out[i] = E
    return out, whole


def stats(m, ref, whole, ref_whole):
    pics = sorted(m)
    A, R = np.array([m[i] for i in pics]), np.array([ref[i] for i in pics])
    c = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    return dict(n=len(pics), corr=c(A, R), centred=c(A - A.mean(1, keepdims=True), R - R.mean(1, keepdims=True)),
                slope=float((A * R).sum() / (R * R).sum()), mae=float(np.abs(A - R).mean()),
                rel_mae=float(np.abs(A - R).mean() / np.abs(R).mean()), signs=float((np.sign(A) == np.sign(R)).mean()),
                sum_vs_own_whole=float(np.mean([A[k].sum() - whole[i] for k, i in enumerate(pics)])),          # signed
                sum_vs_own_whole_abs=float(np.mean([abs(A[k].sum() - whole[i]) for k, i in enumerate(pics)])),
                sum_vs_ref=float(np.mean([abs(A[k].sum() - ref_whole[i]) for k, i in enumerate(pics)])))


def baselines(m, ex, wex, h, Y, groups, no, G, est):
    """What the agreement of measured maps m with the exact ex is worth (as map_nulls.py for the first maps):
    a 95 % bootstrap interval over pictures; a map with no spatial information (each picture's exact P - b spread
    evenly over its blocks); the energy-weighted map (P - b spread over the blocks in proportion to their light
    ||psi_S||^2, referee B 2026-11-03: the amplitude weighting is built into the estimator), pooled and centred;
    each measured map's blocks permuted within its picture (95th percentile of 10,000
    draws); the share of exact blocks that are positive; how often the strongest exact block is the measured one;
    and the shot-noise floor: the same estimator from 4,000 shots drawn from the exact distributions (200 draws)."""
    rng = np.random.default_rng(0)
    pics = sorted(m)
    A, R = np.array([m[i] for i in pics]), np.array([ex[i] for i in pics])
    c = lambda x, y: float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
    boot = [c(A[s], R[s]) for s in (rng.integers(0, len(pics), len(pics)) for _ in range(2000))]
    perm = [c(np.array([x[rng.permutation(G * G)] for x in A]), R) for _ in range(10000)]
    energy = np.array([[h["norm"][(h["G"] == G) & (h["pic"] == i) & (h["block"] == j) & (h["kind"] == "in")][0]
                        for j in range(G * G)] for i in pics])
    En = energy * R.sum(1, keepdims=True)
    cen = lambda x: x - x.mean(1, keepdims=True)
    floor_mae, floor_sum = [], []
    for _ in range(200):
        Q = np.array([rng.multinomial(SHOTS, q / q.sum()) / SHOTS for q in h["P_ex"]])
        ms, _ = maps(h, Q, Y, groups, no, G, est)
        S = np.array([ms[i] for i in pics])
        floor_mae.append(np.abs(S - R).mean())
        floor_sum.append(np.mean([abs(S[k].sum() - wex[i]) for k, i in enumerate(pics)]))
    return dict(corr_ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                uniform_split_corr=c(np.repeat(R.sum(1, keepdims=True) / (G * G), G * G, 1), R),
                energy_null_corr=c(En, R), energy_null_centred=c(cen(En), cen(R)),
                within_perm_null_95=float(np.percentile(perm, 95)), exact_positive=float((R > 0).mean()),
                strongest_found=int((A.argmax(1) == R.argmax(1)).sum()), mean_abs_exact=float(np.abs(R).mean()),
                mean_exact_total=float(np.mean([wex[i] for i in pics])),
                shot_floor_mae=float(np.mean(floor_mae)), shot_floor_sum_residual=float(np.mean(floor_sum)))


def report(a):
    f = folder(a)
    h = np.load(f / "states.npz")
    run, b, Y, wc, depth, n, no, _, groups = setup(args_for(a))
    chip = json.loads((folder_for(args_for(a)) / "chip_ibm_kingston_r0.json").read_text())
    kinds, whole4 = {"prep": h["P_pr"]}, {}
    for p in sorted(f.glob("noisy_*.npz")) + sorted(f.glob("chip_*.npz")):
        d = np.load(p)
        kinds[p.stem] = d["P"]
        if "P_whole4" in d.files:
            whole4[p.stem] = {int(i): q for i, q in zip(d["pic_whole4"], d["P_whole4"])}
    rng = np.random.default_rng(0)
    for name in [k for k in list(kinds) if k.startswith("noisy")]:          # the shot noise of the planned budget
        Q = kinds[name]
        kinds[name + f", {SHOTS} shots"] = np.array([rng.multinomial(SHOTS, q / q.sum()) / SHOTS for q in Q])
    out = {}
    for G in (2, 4):
        for est in (("standard", "sign flip") if G == 2 else ("sign flip",)):
            ex, wex = maps(h, h["P_ex"], Y, groups, no, G, est)
            for name, P in kinds.items():
                m, w = maps(h, P, Y, groups, no, G, est, whole4.get(name))
                if not m:
                    continue
                r = stats(m, ex, w, wex)
                key = f"{G}x{G}, {est}, {name} vs exact"
                out[key] = r
                print(f"{key:52s} r {r['corr']:.3f} centred {r['centred']:.3f} slope {r['slope']:.2f} "
                      f"relMAE {r['rel_mae']:.2f} signs {r['signs']:.2f} | sum - own whole {r['sum_vs_own_whole']:+.4f}"
                      f" | sum vs exact {r['sum_vs_ref']:.4f}")
                if name.startswith("chip"):
                    r.update(baselines(m, ex, wex, h, Y, groups, no, G, est))
                    for ref in [k for k in kinds if k.startswith("noisy") and "shots" not in k]:   # the noisy device's
                        mn, wn = maps(h, kinds[ref], Y, groups, no, G, est)                        # own maps
                        s = stats(m, mn, w, wn)
                        r[f"vs {ref}"] = dict(corr=s["corr"], centred=s["centred"], slope=s["slope"])
                    print("   " + ", ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}"
                                            for k, v in r.items() if k not in stats(m, ex, w, wex)))
                if G == 2 and name.startswith(("chip", "noisy")):
                    right = [i for i in m if chip[str(i)]["pred"] == chip[str(i)]["label"]]
                    for grp, keep in (("chip right", right), ("chip wrong", [i for i in m if i not in right])):
                        if len(keep) > 2:
                            rr = stats({i: m[i] for i in keep}, ex, w, wex)
                            out[key + f", {grp}"] = rr
                            print(f"   {grp:12s} n {rr['n']:2d} r {rr['corr']:.3f} centred {rr['centred']:.3f}")
    (f / "report.json").write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("prereg", "fit", "noisy", "report", "chip")}
    ps["fit"].add_argument("--steps", type=int, default=3000)
    ps["noisy"].add_argument("--noise", default="FakeKingston")
    ps["chip"].add_argument("--keys", required=True)
    ps["chip"].add_argument("--accounts", required=True, help="account lines to try, in order, e.g. 3,1,2,0")
    ps["chip"].add_argument("--backend", default="ibm_kingston")
    ps["chip"].add_argument("--parts", default="2,4", help="2: the 2x2 jobs, 4: the 4x4 jobs")
    ps["chip"].add_argument("--per_job_2x2", type=int, default=5, help="pictures in one 2x2 job (13 circuits each)")
    ps["chip"].add_argument("--per_job_4x4", type=int, default=1, help="pictures in one 4x4 job (33 circuits each)")
    ps["chip"].add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    {"prereg": prereg, "fit": fit, "noisy": noisy, "report": report, "chip": chip}[a.cmd](a)


if __name__ == "__main__":
    main()
