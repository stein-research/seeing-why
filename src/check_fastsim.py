#!/usr/bin/env python3
"""fastsim.py against the PennyLane reference: probabilities, states and gradients, then the speed of both.

Run:  python src/check_fastsim.py
"""
import time

import pennylane as qml
import torch

import fastsim
import pilot_fmnist as pf

torch.manual_seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
ok = True
for n, depth, nr, n_out in ((10, 6, 2, 4), (10, 14, 3, 7), (12, 6, 3, 7)):
    pf.N_QUBITS = n
    w = 0.7 * torch.randn(depth + 1, n, nr, dtype=torch.float64)
    x = torch.rand(8, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)
    q = pf.make_qnode(depth, n_out, "brickwall", "cpu")
    P_ref = q(x, w)
    S_ref = qml.QNode(lambda v: (qml.AmplitudeEmbedding(v, wires=range(n), normalize=False),
                                 [(pf.rotate(w[k]), pf.entangle(k, "brickwall")) for k in range(depth)],
                                 pf.rotate(w[depth]), qml.state())[-1], qml.device("default.qubit", wires=n))(x)
    for dt, tol in ((torch.complex128, 1e-10), (torch.complex64, 2e-5)):
        with torch.no_grad():
            S, S2 = fastsim.run(x, w, depth, dt), fastsim.run_dense(x, w, depth, dt)
        P = fastsim.probs(S, n_out)
        good = all(torch.allclose(s_.to(torch.complex128), S_ref, atol=tol) for s_ in (S, S2)) and \
            torch.allclose(P.double(), P_ref, atol=tol)
        ok &= good
        print(f"n={n} depth={depth} rot={nr} n_out={n_out} {str(dt)[6:]}: state and probs match {good}")
    # gradients of a scalar loss w.r.t. the angles
    w1 = w.clone().requires_grad_(True)
    q1 = pf.make_qnode(depth, n_out, "brickwall", "cpu")
    g_ref, = torch.autograd.grad(q1(x, w1)[:, 0].sum(), w1)
    w2 = w.clone().requires_grad_(True)
    f = fastsim.run_dense if n <= 10 else fastsim.run        # dense 4096 x 4096 backward is too big for this check
    g, = torch.autograd.grad(fastsim.probs(f(x, w2, depth, torch.complex128), n_out)[:, 0].sum(), w2)
    good = torch.allclose(g, g_ref, atol=1e-9)
    ok &= good
    print(f"  gradients match {good}")

# QCNN (--ent qcnn) against PennyLane gates written independently: qml.ctrl of RZ, RY, RX for the pooling gate
for n, depth, keep in ((10, 12, 4), (12, 11, 7)):
    pf.N_QUBITS = n
    w = 0.7 * torch.randn(depth + 1, n, 3, dtype=torch.float64)
    x = torch.rand(4, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(n), normalize=False)
        for k in range(depth + 1):
            hi, prev = fastsim.qcnn_active(k, depth, n, keep), fastsim.qcnn_active(max(k - 1, 0), depth, n, keep)
            for j in range(prev - 1, hi - 1, -1):
                qml.ctrl(lambda a=w[k, j], t=j - 1: (qml.RZ(a[2], wires=t), qml.RY(a[0], wires=t), qml.RX(a[1], wires=t)),
                         control=j)()
            for i in range(hi):
                qml.RZ(w[k, i, 2], wires=i)
                qml.RY(w[k, i, 0], wires=i)
                qml.RX(w[k, i, 1], wires=i)
            if k < depth:
                for i in range(k % 2, hi - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return qml.state()
    S_ref = qml.QNode(body, qml.device("default.qubit", wires=n))(x)
    S = fastsim.run(x, w, depth, torch.complex128, qcnn=keep)
    back = fastsim.run_inverse(S, w, depth, torch.complex128, qcnn=keep)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(back, x.to(torch.complex128), atol=1e-10)
    S32 = fastsim.run(x, w, depth, qcnn=keep)
    good &= torch.allclose(S32.to(torch.complex128), S_ref, atol=2e-5)
    ok &= good
    print(f"qcnn n={n} depth={depth} keep={keep}: state, complex64 state and U^dag U = 1 match {good}")

# patch-local (--local W --local_every L [--local_colour: c colour wires in the window from the start]) alone and with
# the QCNN, against PennyLane gates with the window written out here
for n, depth, keep, W, L, c in ((10, 22, 0, 4, 4, 0), (10, 22, 7, 4, 4, 0), (10, 16, 7, 6, 2, 0), (12, 20, 7, 4, 3, 0),
                               (12, 24, 7, 4, 4, 2), (12, 20, 0, 6, 3, 2), (11, 18, 7, 4, 4, 1)):
    w = 0.7 * torch.randn(depth + 1, n, 3, dtype=torch.float64)
    x = torch.rand(4, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(n), normalize=False)
        for k in range(depth + 1):
            lo = max(0, n - W - 2 * (k // L)) if k < depth else 0
            hi = fastsim.qcnn_active(k, depth, n, keep) if keep else n
            prev = fastsim.qcnn_active(max(k - 1, 0), depth, n, keep) if keep else n
            for j in range(prev - 1, hi - 1, -1):
                qml.ctrl(lambda a=w[k, j], t=j - 1: (qml.RZ(a[2], wires=t), qml.RY(a[0], wires=t), qml.RX(a[1], wires=t)),
                         control=j)()
            win = list(range(min(c, lo))) + list(range(lo, hi))
            for i in win:
                qml.RZ(w[k, i, 2], wires=i)
                qml.RY(w[k, i, 0], wires=i)
                qml.RX(w[k, i, 1], wires=i)
            if k < depth:
                used = set()
                for i0, i1 in zip(win, win[1:]):
                    if i0 % 2 == k % 2 and i0 not in used:
                        used |= {i0, i1}
                        qml.CNOT(wires=[i0, i1])
        return qml.state()
    S_ref = qml.QNode(body, qml.device("default.qubit", wires=n))(x)
    S = fastsim.run(x, w, depth, torch.complex128, hier=(W, L, c), qcnn=keep)
    back = fastsim.run_inverse(S, w, depth, torch.complex128, hier=(W, L, c), qcnn=keep)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(back, x.to(torch.complex128), atol=1e-10)
    ok &= good
    print(f"local W={W} every={L} colour={c} n={n} depth={depth} keep={keep}: state and U^dag U = 1 match {good}, "
          f"{fastsim.n_angles(depth, n, 3, (W, L, c), keep)} angles")

# soft mask (--mask_group g --mask_at K): one ancilla per wire group, RY(2 phi[bits]) under control of the group's
# wires reading `bits`, post-selected on every ancilla reading 0; compares the unnormalised post-selected state and
# the adjoint (<A x, y> = <x, A^dag y>)
for n, depth, keep, g, K in ((6, 8, 0, 2, 3), (6, 10, 4, 1, 4), (8, 12, 5, 4, 5)):
    G = n // g
    w = 0.7 * torch.randn(depth + 1, n, 3, dtype=torch.float64)
    phi = 1.2 * torch.randn(G, 2 ** g, dtype=torch.float64)
    x = torch.rand(3, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(n), normalize=False)
        for k in range(depth + 1):
            hi = fastsim.qcnn_active(k, depth, n, keep) if keep else n
            prev = fastsim.qcnn_active(max(k - 1, 0), depth, n, keep) if keep else n
            for j in range(prev - 1, hi - 1, -1):
                qml.ctrl(lambda a=w[k, j], t=j - 1: (qml.RZ(a[2], wires=t), qml.RY(a[0], wires=t), qml.RX(a[1], wires=t)),
                         control=j)()
            for i in range(hi):
                qml.RZ(w[k, i, 2], wires=i)
                qml.RY(w[k, i, 0], wires=i)
                qml.RX(w[k, i, 1], wires=i)
            if k < depth:
                for i in range(k % 2, hi - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
            if k == K:
                for j in range(G):
                    ctrl = list(range(j * g, (j + 1) * g))
                    for bits in range(2 ** g):
                        vals = [(bits >> (g - 1 - t)) & 1 for t in range(g)]
                        qml.ctrl(qml.RY, control=ctrl, control_values=vals)(2 * phi[j, bits], wires=n + j)
        return qml.state()
    full = qml.QNode(body, qml.device("default.qubit", wires=n + G))(x)
    S_ref = full.reshape(len(x), 2 ** n, 2 ** G)[:, :, 0]                  # every ancilla reads 0
    m = (K, fastsim.mask_vector(phi))
    S = fastsim.run(x, w, depth, torch.complex128, qcnn=keep, mask=m)
    y = torch.randn(len(x), 2 ** n, dtype=torch.complex128)
    lhs = (S.conj() * y).sum(-1)
    rhs = (x.to(torch.complex128).conj() * fastsim.run_inverse(y, w, depth, torch.complex128, qcnn=keep, mask=m)).sum(-1)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(lhs, rhs, atol=1e-10)
    ok &= good
    print(f"mask g={g} at={K} n={n} depth={depth} keep={keep}: post-selected state and adjoint match {good}, "
          f"pass {float((S.abs() ** 2).sum(-1).mean()):.3f}")

# Fourier-plane mask (--fourier_mask): inverse QFT on the row wires and on the column wires (= an orthonormal FFT),
# one ancilla RY(2 arccos d[plane, ky, kx]) under control of every wire, post-selected on 0, QFT back, then the circuit
pf.ORDER = "morton"
for n, s, depth, phase in ((5, 4, 4, False), (6, 8, 3, False), (5, 4, 3, True), (6, 8, 2, True)):
    nb = s.bit_length() - 1
    planes = 2 ** n // (s * s)
    top = n - 2 * nb                                                         # plane wires above the pixel bits
    rows, cols = [top + 2 * i for i in range(nb)], [top + 2 * i + 1 for i in range(nb)]   # Morton: r c r c ..., MSB first
    pix = torch.tensor(pf.morton_perm(s, gray=False))
    w = 0.7 * torch.randn(depth + 1, n, 3, dtype=torch.float64)
    d = fastsim.fourier_vector(1.2 * torch.randn(planes + 2 * s * s, dtype=torch.float64), s, planes, phase)
    x = torch.rand(3, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(n), normalize=False)
        qml.adjoint(qml.QFT)(wires=rows)
        qml.adjoint(qml.QFT)(wires=cols)
        for m in range(2 ** n):                                          # basis m: plane, then Morton bits of (ky, kx)
            vals = [(m >> (n - 1 - t)) & 1 for t in range(n)]
            plane, r, c = m >> (2 * nb), 0, 0
            for i in range(nb):
                r |= vals[rows[i]] << (nb - 1 - i)
                c |= vals[cols[i]] << (nb - 1 - i)
            qml.ctrl(qml.RY, control=list(range(n)), control_values=vals)(
                2 * torch.arccos(d[plane, r, c].abs().clamp(-1, 1)), wires=n)
            if phase or d[plane, r, c].real < 0:                         # sign / phase: a controlled phase gate
                qml.ctrl(qml.GlobalPhase, control=list(range(n)), control_values=vals)(
                    -float(torch.angle(d[plane, r, c])))
        qml.QFT(wires=rows)
        qml.QFT(wires=cols)
        for k in range(depth + 1):
            for i in range(n):
                qml.RZ(w[k, i, 2], wires=i)
                qml.RY(w[k, i, 0], wires=i)
                qml.RX(w[k, i, 1], wires=i)
            if k < depth:
                for i in range(k % 2, n - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return qml.state()
    full = qml.QNode(body, qml.device("default.qubit", wires=n + 1))(x)
    S_ref = full.reshape(len(x), 2 ** n, 2)[:, :, 0]
    S = fastsim.run(x, w, depth, torch.complex128, fmask=(pix, d))
    y = torch.randn(len(x), 2 ** n, dtype=torch.complex128)
    lhs = (S.conj() * y).sum(-1)
    rhs = (x.to(torch.complex128).conj() * fastsim.run_inverse(y, w, depth, torch.complex128, fmask=(pix, d))).sum(-1)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(lhs, rhs, atol=1e-10)
    ok &= good
    print(f"fourier mask n={n} s={s} depth={depth} phase={phase}: post-selected state and adjoint match {good}, "
          f"pass {float((S.abs() ** 2).sum(-1).mean()):.3f}")

# Fourier masks in superposition (--fourier_bank K): H on log2(K) register wires in front, inverse QFT on the picture's
# row and column wires, one ancilla RY + phase under control of register and picture wires (filter k on branch k),
# post-selected, QFT back, then the brickwall over register + picture wires
for n, s, r, depth, phase in ((5, 4, 1, 3, True), (6, 8, 2, 2, False)):
    nb, K = s.bit_length() - 1, 2 ** r
    planes = 2 ** n // (s * s)
    top = r + n - 2 * nb
    rows, cols = [top + 2 * i for i in range(nb)], [top + 2 * i + 1 for i in range(nb)]
    pix = torch.tensor(pf.morton_perm(s, gray=False))
    w = 0.7 * torch.randn(depth + 1 + fastsim.fbank_rows(n + r, 3, K, s, planes, phase), n + r, 3, dtype=torch.float64)
    _, D = fastsim.fbank_parts(w, depth, K, s, planes, phase)
    x = torch.rand(3, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(r, n + r), normalize=False)
        for i in range(r):
            qml.Hadamard(wires=i)
        qml.adjoint(qml.QFT)(wires=rows)
        qml.adjoint(qml.QFT)(wires=cols)
        for m in range(2 ** (n + r)):
            vals = [(m >> (n + r - 1 - t)) & 1 for t in range(n + r)]
            k, pm = m >> n, m & (2 ** n - 1)
            plane, ky, kx = pm >> (2 * nb), 0, 0
            for i in range(nb):
                ky |= vals[rows[i]] << (nb - 1 - i)
                kx |= vals[cols[i]] << (nb - 1 - i)
            dv = D[k, plane, ky, kx]
            qml.ctrl(qml.RY, control=list(range(n + r)), control_values=vals)(2 * torch.arccos(dv.abs().clamp(0, 1)),
                                                                              wires=n + r)
            qml.ctrl(qml.GlobalPhase, control=list(range(n + r)), control_values=vals)(-float(torch.angle(dv)))
        qml.QFT(wires=rows)
        qml.QFT(wires=cols)
        for k_ in range(depth + 1):
            for i in range(n + r):
                qml.RZ(w[k_, i, 2], wires=i)
                qml.RY(w[k_, i, 0], wires=i)
                qml.RX(w[k_, i, 1], wires=i)
            if k_ < depth:
                for i in range(k_ % 2, n + r - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return qml.state()
    full = qml.QNode(body, qml.device("default.qubit", wires=n + r + 1))(x)
    S_ref = full.reshape(len(x), 2 ** (n + r), 2)[:, :, 0]
    S = fastsim.run_fbank(x, w, depth, K, pix, s, planes, phase, dtype=torch.complex128)
    y = torch.randn(len(x), 2 ** (n + r), dtype=torch.complex128)
    lhs = (S.conj() * y).sum(-1)
    rhs = (x.to(torch.complex128).conj() * fastsim.run_fbank_inverse(y, w, depth, K, pix, s, planes, phase,
                                                                    dtype=torch.complex128)).sum(-1)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(lhs, rhs, atol=1e-10)
    ok &= good
    print(f"fourier bank K={K} n={n} s={s} depth={depth} phase={phase}: post-selected state and adjoint match {good}")

# glimpse attention (--glimpse gc): coarse picture amplitude-encoded on its own wires, look circuit V on them, one
# ancilla RY(2 phi[offset]) under control of V's cell wires and the picture's cell wires, post-selected on 0, then the
# circuit on the picture; the look register is traced out (summed over)
for n, s, gc, depth, depth_v, n_out in ((6, 8, 2, 3, 2, 3), (7, 8, 4, 4, 3, 3)):
    planes, nb, nbc = 2 ** n // (s * s), s.bit_length() - 1, gc.bit_length() - 1
    top = n - 2 * nb
    pix, pixc = torch.tensor(pf.morton_perm(s, gray=False)), torch.tensor(pf.morton_perm(gc, gray=False))
    qc = (planes * gc * gc).bit_length() - 1
    w = 0.7 * torch.randn(depth + 1 + fastsim.glimpse_rows(n, 3, depth_v, gc), n, 3, dtype=torch.float64)
    x = torch.rand(3, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)
    cp = fastsim.coarse_picture(x, pix, s, gc, pixc)
    cp = cp / cp.norm(dim=1, keepdim=True)
    _, wv, phi = fastsim.split_glimpse(w, depth, qc, depth_v, gc)
    cells = list(range(top, top + 2 * nbc))                                  # picture's cell bits: top Morton bits
    look = list(range(n + qc - 2 * nbc, n + qc))                             # V's cell bits: its lowest wires

    def body(v, c):
        qml.AmplitudeEmbedding(v, wires=range(n), normalize=False)
        qml.AmplitudeEmbedding(c, wires=range(n, n + qc), normalize=False)
        for k in range(depth_v + 1):
            for i in range(qc):
                qml.RZ(wv[k, i, 2], wires=n + i)
                qml.RY(wv[k, i, 0], wires=n + i)
                qml.RX(wv[k, i, 1], wires=n + i)
            if k < depth_v:
                for i in range(k % 2, qc - 1, 2):
                    qml.CNOT(wires=[n + i, n + i + 1])
        for jj in range(gc * gc):                                            # jj, cc: Morton indices on the gc grid
            for cc in range(gc * gc):
                jy, jx, by, bx = int(pixc[jj]) // gc, int(pixc[jj]) % gc, int(pixc[cc]) // gc, int(pixc[cc]) % gc
                vals = [(jj >> (2 * nbc - 1 - t)) & 1 for t in range(2 * nbc)] + \
                       [(cc >> (2 * nbc - 1 - t)) & 1 for t in range(2 * nbc)]
                qml.ctrl(qml.RY, control=look + cells, control_values=vals)(
                    2 * phi[by - jy + gc - 1, bx - jx + gc - 1], wires=n + qc)
        for k in range(depth + 1):
            for i in range(n):
                qml.RZ(w[k, i, 2], wires=i)
                qml.RY(w[k, i, 0], wires=i)
                qml.RX(w[k, i, 1], wires=i)
            if k < depth:
                for i in range(k % 2, n - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return qml.state()
    full = torch.stack([torch.as_tensor(qml.QNode(body, qml.device("default.qubit", wires=n + qc + 1))(x[i], cp[i]))
                        for i in range(len(x))])
    p_ref = (full.reshape(len(x), 2 ** n_out, -1, 2)[..., 0].abs() ** 2).sum(-1)   # ancilla 0, look register summed
    S, wts = fastsim.glimpse_run(x, w, depth, qc, depth_v, gc, pix, s, pixc, torch.complex128)
    scale = fastsim.coarse_picture(x, pix, s, gc, pixc).norm(dim=1) ** 2
    p_fast = (wts[..., None] * fastsim.probs(S.reshape(-1, 2 ** n), n_out).reshape(len(x), gc * gc, -1)).sum(1)
    good = torch.allclose(p_fast / scale[:, None], p_ref, atol=1e-10)
    ok &= good
    print(f"glimpse n={n} s={s} gc={gc} depth={depth} V depth={depth_v}: kept outcome probabilities match {good}, "
          f"pass {float(p_ref.sum(-1).mean()):.3f}")

# channels (--channels): H on the ancillas, qml.ctrl of each channel's bank circuit, then the QCNN over all wires
for n, n_anc, bank, depth, keep in ((8, 2, 3, 8, 7), (10, 2, 3, 8, 7)):
    C, nt = 2 ** n_anc, n + n_anc
    w = 0.7 * torch.randn(depth + 1 + C * (bank + 1), nt, 3, dtype=torch.float64)
    x = torch.rand(3, 2 ** n, dtype=torch.float64)
    x = x / x.norm(dim=1, keepdim=True)

    def rot(a, t):
        qml.RZ(a[2], wires=t)
        qml.RY(a[0], wires=t)
        qml.RX(a[1], wires=t)

    def bank_c(c):
        wb = w[depth + 1 + c * (bank + 1):depth + 1 + (c + 1) * (bank + 1)]
        for k in range(bank + 1):
            for i in range(n):
                rot(wb[k, n_anc + i], n_anc + i)
            if k < bank:
                for i in range(k % 2, n - 1, 2):
                    qml.CNOT(wires=[n_anc + i, n_anc + i + 1])

    def body(v):
        qml.AmplitudeEmbedding(v, wires=range(n_anc, nt), normalize=False)
        for i in range(n_anc):
            qml.Hadamard(wires=i)
        for c in range(C):
            qml.ctrl(bank_c, control=list(range(n_anc)), control_values=[(c >> (n_anc - 1 - i)) & 1 for i in range(n_anc)])(c)
        for k in range(depth + 1):
            hi, prev = fastsim.qcnn_active(k, depth, nt, keep), fastsim.qcnn_active(max(k - 1, 0), depth, nt, keep)
            for j in range(prev - 1, hi - 1, -1):
                qml.ctrl(rot, control=j)(w[k, j], j - 1)
            for i in range(hi):
                rot(w[k, i], i)
            if k < depth:
                for i in range(k % 2, hi - 1, 2):
                    qml.CNOT(wires=[i, i + 1])
        return qml.state()
    S_ref = qml.QNode(body, qml.device("default.qubit", wires=nt))(x)
    S = fastsim.run_channels(x, w, depth, bank, n_anc, keep, torch.complex128)
    back = fastsim.run_channels_inverse(S, w, depth, bank, n_anc, keep, torch.complex128)
    good = torch.allclose(S, S_ref, atol=1e-10) and torch.allclose(back, x.to(torch.complex128), atol=1e-10)
    ok &= good
    print(f"channels n={n} anc={n_anc} bank={bank} depth={depth} keep={keep}: state and <0|U^dag U|0> = 1 match {good}")
print("ALL MATCH" if ok else "MISMATCH")

# speed: one training step (forward + backward) on 64 images and on the 1024 basis states, depth 6, RZ-RY-RX
pf.N_QUBITS = 10
for B in (64, 1024):
    x = torch.rand(B, 1024, dtype=torch.float64, device=dev)
    x = x / x.norm(dim=1, keepdim=True)
    w = (0.1 * torch.randn(7, 10, 3, device=dev)).requires_grad_(True)
    q = pf.make_qnode(6, 7, "brickwall", "cpu")
    for name, f in (("pennylane", lambda: q(x, w)), ("fast layers", lambda: fastsim.probs(fastsim.run(x, w, 6), 7)),
                    ("fast unitary", lambda: fastsim.probs(fastsim.run_dense(x, w, 6), 7))):
        for rep in range(2):                                  # first round warms up
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                f()[:, 0].sum().backward()
            if dev == "cuda":
                torch.cuda.synchronize()
        print(f"batch {B:4d} {name:12s}: {(time.time() - t0) / 5 * 1000:7.1f} ms per step")
