#!/usr/bin/env python3
"""Fast statevector simulator for the brickwall circuits of pilot_fmnist.py / two_copy.py (torch, any device).

Same circuit as the PennyLane qnodes, written for speed: each qubit's rotations (RZ then RY then RX, or RY then RX)
are fused into one 2x2 matrix, a whole CNOT layer is one index permutation, and the state is complex64 by default
(consumer GPUs run float64 at a small fraction of float32 speed). PennyLane stays the reference: check_fastsim.py
compares the two.

Wire 0 is the most significant bit of the basis index, as in PennyLane.
"""
import torch

_PERMS = {}


def _cnot_perm(n, start, device, lo=0, hi=None, extra=0):
    """Index permutation of one brickwall CNOT layer over the active wires 0..extra-1 (the colour wires of a patch-local
    window, when it does not reach them yet) and lo..hi-1: CNOT(a, b) for each pair of neighbours a < b in that list
    with a = start mod 2, skipping a wire already used. With extra = 0: CNOT(i, i+1) for i = start, start+2, ... with
    lo <= i < i+1 < hi."""
    hi = n if hi is None else hi
    key = (n, start, str(device), lo, hi, extra)
    if key not in _PERMS:
        x = torch.arange(2 ** n, device=device)
        wires = list(range(min(extra, lo))) + list(range(lo, hi))
        used = set()
        for a, b in zip(wires, wires[1:]):
            if a % 2 != start or a in used:
                continue
            used |= {a, b}
            c = (x >> (n - 1 - a)) & 1
            x = x ^ (c << (n - 1 - b))                   # flip wire b where wire a is 1
        _PERMS[key] = x
    return _PERMS[key]


def rot_mats(w, dtype=torch.complex64):
    """(..., 2) or (..., 3) angles -> (..., 2, 2) matrices RX(w1) RY(w0) [RZ(w2)], the order pilot_fmnist.rotate uses."""
    rdt = torch.float32 if dtype == torch.complex64 else torch.float64
    w = w.to(rdt)
    c0, s0 = torch.cos(w[..., 0] / 2), torch.sin(w[..., 0] / 2)
    c1, s1 = torch.cos(w[..., 1] / 2), torch.sin(w[..., 1] / 2)
    zero = torch.zeros_like(c0)
    ry = torch.complex(torch.stack([torch.stack([c0, -s0], -1), torch.stack([s0, c0], -1)], -2),
                       torch.zeros(c0.shape + (2, 2), dtype=rdt, device=w.device))
    rx = torch.complex(torch.stack([torch.stack([c1, zero], -1), torch.stack([zero, c1], -1)], -2),
                       torch.stack([torch.stack([zero, -s1], -1), torch.stack([-s1, zero], -1)], -2))
    m = rx @ ry
    if w.shape[-1] == 3:
        h = w[..., 2] / 2
        rz = torch.complex(torch.stack([torch.stack([torch.cos(h), zero], -1), torch.stack([zero, torch.cos(h)], -1)], -2),
                           torch.stack([torch.stack([-torch.sin(h), zero], -1), torch.stack([zero, torch.sin(h)], -1)], -2))
        m = m @ rz
    return m


def _apply_layer(state, mats):
    """state (B, 2^n); mats (n, 2, 2): one 2x2 on every wire."""
    B, d = state.shape
    n = mats.shape[0]
    for i in range(n):
        st = state.reshape(B, 2 ** i, 2, d >> (i + 1))
        state = torch.einsum("jk,bakc->bajc", mats[i], st).reshape(B, d)
    return state


def _apply_controlled(state, mat, ctrl):
    """state (B, 2^n); mat (2, 2) on wire ctrl - 1 where wire ctrl is 1 (the QCNN pooling gate)."""
    B, d = state.shape
    st = state.reshape(B, 2 ** (ctrl - 1), 2, 2, d >> (ctrl + 1))          # (..., target, control, ...)
    on = torch.einsum("jk,bakc->bajc", mat, st[:, :, :, 1])
    return torch.stack([st[:, :, :, 0], on], 3).reshape(B, d)


def _layer_matrix(mats):
    """(n, 2, 2) -> the 2^n x 2^n tensor product mats[0] (x) ... (x) mats[n-1] (wire 0 most significant)."""
    m = mats[0]
    for i in range(1, mats.shape[0]):
        m = torch.kron(m, mats[i])
    return m


def hier_parts(hier):
    """hier argument -> (wires, every, colour): True = (4, 4, 0); (wires, every) or (wires, every, colour)."""
    h = (4, 4) if hier is True else tuple(hier)
    return h + (0,) * (3 - len(h))


def hier_lo(k, n, hier=True):
    """Lowest active wire of layer k in the hierarchical (patch-local) circuit. hier = (wires, every[, colour]), True =
    (4, 4): the finest `wires` wires first (under the Morton order 4 wires = a 2^2 x 2^2-pixel block, 6 = 8 x 8; the
    gates act on the block's pixels only and identically in every block - a shared convolution), then 2 more wires (the
    block doubles in each direction) every `every` layers, then all wires. colour > 0: wires 0..colour-1 (the colour
    planes, most significant) are in the window from the start too, so the block's filter mixes the colours of its
    pixels as a CNN's first layer does."""
    wires, every, _ = hier_parts(hier)
    return max(0, n - wires - 2 * (k // every))


def hier_mask(depth, n, nr, dtype=torch.float64, device="cpu", hier=True):
    """(depth + 1, n, 1) mask: rotations on wires below the active window are switched off (angle 0 = identity)."""
    m = torch.ones(depth + 1, n, 1, dtype=dtype, device=device)
    colour = hier_parts(hier)[2]
    for k in range(depth):
        m[k, colour:hier_lo(k, n, hier)] = 0
    return m


def qcnn_active(k, depth, n, keep):
    """Active wires 0..qcnn_active-1 of layer k (k = depth: the last rotation layer) in the QCNN: all n at first, one
    fewer every stage, `keep` (the measured wires) in the last stage - n - keep + 1 stages of about equal length. Under
    the Morton order the highest wires are the finest pixel bits, so the circuit pools fine scales away first."""
    return n - min(k * (n - keep + 1) // depth, n - keep)


def qcnn_mask(depth, n, keep, dtype=torch.float64, device="cpu"):
    """(depth + 1, n, 1) masks (rotations, pooling): a rotation acts on an active wire; layer k's pooling gate, a
    controlled rotation from each wire dropped at k onto the wire above it, takes that dropped wire's angles."""
    rot = torch.zeros(depth + 1, n, 1, dtype=dtype, device=device)
    pool = torch.zeros_like(rot)
    for k in range(depth + 1):
        hi = qcnn_active(k, depth, n, keep)
        rot[k, :hi] = 1
        if k:
            pool[k, hi:qcnn_active(k - 1, depth, n, keep)] = 1
    return rot, pool


def mask_rows(n, nr, group):
    """Extra rows of the packed weights that hold the soft mask's angles: (n / group) groups x 2^group angles."""
    return -(-(n // group) * 2 ** group // (n * nr))


def split_mask(w, depth, group):
    """Packed weights (depth + 1 + mask_rows, n, nr) -> (circuit rows, mask angles (n / group, 2^group))."""
    n = w.shape[1]
    G = n // group
    return w[:depth + 1], w[depth + 1:].reshape(-1)[:G * 2 ** group].reshape(G, 2 ** group)


def mask_vector(phi):
    """Soft mask over the basis states: m(x) = prod over wire groups of cos(phi[group, bits of x in that group]),
    groups of consecutive wires from wire 0 (the most significant). Physically: per group an ancilla rotated by
    RY(2 phi[bits]) under control of the group's wires, kept only when it reads 0 (post-selection); in optics an
    amplitude mask (attenuator / SLM) - the light it blocks just does not arrive."""
    m = torch.cos(phi[0])
    for j in range(1, len(phi)):
        m = torch.kron(m, torch.cos(phi[j]))
    return m


def fourier_rows(n, nr, s, planes, phase=False):
    """Extra rows of the packed weights for the Fourier-plane mask: `planes` colour angles + s x s frequency angles
    (+ s x s phase angles with phase)."""
    return -(-(planes + s * s * (2 if phase else 1)) // (n * nr))


def fourier_params(s, planes, phase=False):
    """Free angles of the Fourier-plane mask: one per colour plane in use + one per frequency pair {k, -k} (+ one
    phase per pair that is not its own mirror: DC and the Nyquist points have none)."""
    return planes + (s * s + 4) // 2 + ((s * s - 4) // 2 if phase else 0)


def _mirror(a):
    """a[-k mod s] for an (s, s) array over 2-D frequencies (flip + roll)."""
    return torch.roll(torch.flip(a, (0, 1)), (1, 1), (0, 1))


def fourier_vector(phi, s, planes, phase=False):
    """(planes, s, s) multiplier on each colour plane's 2-D spatial frequencies:
    cos(phi_c[plane]) * cos(phi_k[ky, kx]), phi_k symmetrised under k -> -k, so the filter's kernel is real and even.
    phase: times exp(i theta[ky, kx]) with theta made odd under k -> -k, so the kernel stays real but need not be even
    - directional (derivative, Sobel-like) filters become possible; optically an SLM that sets phase as well."""
    pc, pk = phi[:planes], phi[planes:planes + s * s].reshape(s, s)
    d = torch.cos(pc)[:, None, None] * torch.cos(0.5 * (pk + _mirror(pk)))[None]
    if not phase:
        return d
    th = phi[planes + s * s:planes + 2 * s * s].reshape(s, s)
    return d * torch.exp(0.5j * (th - _mirror(th)))[None]


def _lens_mask(d):
    """(planes, s, s) multiplier on the s x s DFT frequencies -> (planes, 2s, 2s) on the frequencies of the picture
    zero-padded to 2s: each fine frequency k2 (cycles per 2s pixels) takes the mask pixel of k = k2 / 2 rounded away
    from 0 (the Fourier-plane SLM's pixel it falls on); symmetric in k, so a Hermitian mask stays Hermitian."""
    s = d.shape[-1]
    k2 = torch.fft.fftfreq(2 * s, 1 / (2 * s), device=d.device).long()
    k = torch.sign(k2) * ((k2.abs() + 1) // 2)
    k = torch.remainder(k, s)
    return d[..., k[:, None], k[None, :]]


def fourier_filter(state, d, pix, linear=False):
    """Fourier-plane soft mask (a 4f system: lens, attenuator in the focal plane, lens): every colour plane goes to
    its 2-D spatial frequencies (on qubits a QFT on the row wires and one on the column wires), is multiplied by the
    real mask d (planes, s, s) (fourier_vector; post-selected ancillas, as mask_vector) and comes back. pix[m] =
    row * s + col of wire-ordered amplitude m. The map is a circular convolution of each plane with an even real
    kernel: shift-equivariant, so it cannot favour a place in the picture; d real and even makes it self-adjoint.
    linear: the lens version - no wrap at the picture's edge (zero-padded to 2s, the s x s frame kept; _lens_mask).
    Its adjoint is the same map with the conjugate mask (embedding and cropping are each other's adjoints)."""
    B, (planes, s, _) = state.shape[0], d.shape                              # complex d: not self-adjoint (conj)
    v = state.reshape(B, planes, s * s)
    g = torch.empty_like(v)
    g[:, :, pix] = v                                                         # wire order -> pixel grid
    g = g.reshape(B, planes, s, s)
    if linear:                                  # a lens: linear convolution - the picture zero-padded to 2s x 2s, the
        z = torch.zeros(B, planes, 2 * s, 2 * s, dtype=g.dtype, device=g.device)   # light leaving the s x s frame is
        z[:, :, :s, :s] = g                                                  # lost (counted by the post-selection)
        z = torch.fft.ifft2(torch.fft.fft2(z, norm="ortho") * _lens_mask(d), norm="ortho")
        g = z[:, :, :s, :s]
    else:                                       # a QFT on the row / column wires: circular convolution
        g = torch.fft.ifft2(torch.fft.fft2(g, norm="ortho") * d, norm="ortho")
    return g.reshape(B, planes, s * s)[:, :, pix].reshape(B, -1)


def split_masks(w, depth, mask_group=0, mask_at=-1, fourier=None):
    """Packed weights -> (circuit rows, keyword arguments for run / run_inverse): the wire-group mask (mask_group > 0)
    or, with fourier = (pix, s, planes), the Fourier-plane mask on the input; neither: the weights unchanged."""
    if fourier is not None:
        pix, s, planes, *opt = fourier                                       # opt: phase, linear
        d = fourier_vector(w[depth + 1:].reshape(-1), s, planes, bool(opt and opt[0]))
        return w[:depth + 1], dict(fmask=(pix, d, bool(len(opt) > 1 and opt[1])))
    if mask_group:
        wc, phi = split_mask(w, depth, mask_group)
        return wc, dict(mask=(mask_at, mask_vector(phi)))
    return w, {}


def glimpse_rows(n, nr, depth_v, gc):
    """Extra rows of the packed weights for glimpse attention: depth_v + 1 rows of the look circuit V (its first qc
    columns used), then rows holding the (2 gc - 1)^2 window angles."""
    return depth_v + 1 + -(-(2 * gc - 1) ** 2 // (n * nr))


def glimpse_params(qc, nr, depth_v, gc):
    """Free angles of glimpse attention: V's rotations on qc wires + the window over relative cell offsets."""
    return (depth_v + 1) * qc * nr + (2 * gc - 1) ** 2


def split_glimpse(w, depth, qc, depth_v, gc):
    """Packed weights -> (circuit rows, V rows (depth_v + 1, qc, nr), window angles (2 gc - 1, 2 gc - 1))."""
    wv = w[depth + 1:depth + 2 + depth_v, :qc]
    phi = w[depth + 2 + depth_v:].reshape(-1)[:(2 * gc - 1) ** 2].reshape(2 * gc - 1, 2 * gc - 1)
    return w[:depth + 1], wv, phi


def coarse_picture(amps, pix, s, gc, pixc):
    """Wire-ordered amplitudes (B, planes * s^2) -> the picture summed over (s/gc) x (s/gc) cells, per colour plane,
    in Morton order of the gc x gc grid (pixc = its permutation): (B, planes * gc^2). Linear and not renormalised, so
    the class probabilities stay a ratio of two degree-4 forms in the pixels (exact maps by Euler's theorem)."""
    B = len(amps)
    v = amps.reshape(B, -1, s * s)
    g = torch.empty_like(v)
    g[:, :, pix] = v
    g = g.reshape(B, v.shape[1], gc, s // gc, gc, s // gc).sum((3, 5)).reshape(B, v.shape[1], gc * gc)
    return g[:, :, pixc].reshape(B, -1)


def glimpse_windows(phi, pix, s, gc, pixc, planes):
    """(J = gc^2, planes * s^2) real windows in wire order: glimpse j (Morton index on the gc grid) multiplies a pixel
    in cell (by, bx) by cos(phi[by - jy + gc - 1, bx - jx + gc - 1]) - one learned window shape, moved to each cell."""
    r, c = pix // s // (s // gc), pix % s // (s // gc)                      # cell of each wire-ordered pixel
    jy, jx = pixc // gc, pixc % gc                                           # cell of each glimpse
    D = torch.cos(phi[r[None] - jy[:, None] + gc - 1, c[None] - jx[:, None] + gc - 1])
    return D.repeat(1, planes)


def glimpse_run(amps, w, depth, qc, depth_v, gc, pix, s, pixc, dtype=torch.complex64, hier=False, qcnn=0):
    """Glimpse attention: a second register holds the coarse picture (coarse_picture, amplitude-encoded on qc wires);
    the look circuit V acts on it; an ancilla rotated under control of V's cell wires and the picture's cell wires
    applies glimpse j's window (glimpse_windows) to the picture and is kept only when it reads 0; then the circuit.
    The look register is never measured, so the result is the mixture over glimpses
    P(out) = sum_j w_j |<out| U D_j a>|^2 with w_j = sum over colour planes of |<plane, j| V c>|^2: content picks where
    to look. Returns (states U D_j a (B, J, 2^n), weights w (B, J)), both unnormalised."""
    B, J = len(amps), gc * gc
    _, wv, phi = split_glimpse(w, depth, qc, depth_v, gc)
    v = run(coarse_picture(amps.to(dtype), pix, s, gc, pixc), wv, depth_v, dtype)
    wts = (v.abs() ** 2).reshape(B, -1, J).sum(1)
    D = glimpse_windows(phi, pix, s, gc, pixc, amps.shape[1] // (s * s)).to(dtype)
    S = run((amps.to(dtype)[:, None] * D[None]).reshape(B * J, -1), w[:depth + 1], depth, dtype, hier, qcnn)
    return S.reshape(B, J, -1), wts


def fbank_rows(n, nr, K, s, planes, phase=False):
    """Extra rows of the packed weights (n wires wide) for K Fourier masks."""
    return -(-(K * (planes + s * s * (2 if phase else 1))) // (n * nr))


def fbank_parts(w, depth, K, s, planes, phase=False):
    """Packed weights -> (main circuit rows, K multipliers (K, planes, s, s), one per filter)."""
    per = planes + s * s * (2 if phase else 1)
    flat = w[depth + 1:].reshape(-1)[:K * per].reshape(K, per)
    return w[:depth + 1], torch.stack([fourier_vector(flat[k], s, planes, phase) for k in range(K)])


def run_fbank(amps, w, depth, K, pix, s, planes, phase=False, qcnn=0, dtype=torch.complex64, hier=False, mix=False,
              linear=False):
    """Fourier masks in superposition: log2(K) register wires (the most significant, H from |0>) pick which of K
    Fourier-plane masks the picture passes (each post-selected, as fourier_filter); the main circuit then acts on
    register and picture wires together, so later layers mix the filtered versions by interference, as a CNN mixes
    channels. State (B, K * 2^n), unnormalised: its squared norm is the mean chance the light passes.
    mix (the control): the register is measured after the masks, so the branches never interfere - returns the K branch
    states run separately, stacked as (B * K, K * 2^n): their probabilities summed per picture give the mixture."""
    wq, D = fbank_parts(w, depth, K, s, planes, phase)
    a = amps.to(dtype) / K ** 0.5
    br = torch.stack([fourier_filter(a, D[k], pix, linear) for k in range(K)], 1)  # (B, K, 2^n)
    if mix:                                                                  # branch k alone: |k> (x) D_k a
        z = torch.zeros(len(amps), K, K, br.shape[-1], dtype=br.dtype, device=br.device)
        z[:, torch.arange(K), torch.arange(K)] = br
        out = run(z.reshape(len(amps) * K, -1), wq, depth, dtype, hier, qcnn)
        return out.reshape(len(amps), K, -1)
    return run(br.reshape(len(amps), -1), wq, depth, dtype, hier, qcnn)


def run_fbank_inverse(state, w, depth, K, pix, s, planes, phase=False, qcnn=0, dtype=torch.complex64, hier=False,
                      linear=False):
    """Adjoint of run_fbank restricted to the register's |0> input: picture-space part (B, 2^n), for evidence maps."""
    wq, D = fbank_parts(w, depth, K, s, planes, phase)
    back = run_inverse(state, wq, depth, dtype, hier, qcnn).reshape(len(state), K, -1)
    return sum(fourier_filter(back[:, k], D[k].conj(), pix, linear) for k in range(K)) / K ** 0.5


def n_angles(depth, n, nr, hier=False, qcnn=0):
    """Trainable angles of the brickwall / hierarchical / QCNN circuit: rotations on active wires + pooling gates."""
    rot, pool = qcnn_mask(depth, n, qcnn) if qcnn else (torch.ones(depth + 1, n, 1, dtype=torch.float64), None)
    if hier:
        rot = rot * hier_mask(depth, n, 1, hier=hier)
    return int((rot.sum() + (pool.sum() if qcnn else 0)).item()) * nr


def _schedule(n, depth, hier, qcnn):
    """Per layer k = 0..depth: (lowest active wire, active wires end, wires pooled before the layer's rotations, colour
    wires in the window besides lo..hi-1)."""
    out = []
    colour = hier_parts(hier)[2] if hier else 0
    for k in range(depth + 1):
        lo = hier_lo(k, n, hier) if hier and k < depth else 0
        hi = qcnn_active(k, depth, n, qcnn) if qcnn else n
        prev = qcnn_active(k - 1, depth, n, qcnn) if qcnn and k else hi
        if prev > hi and hi - 1 <= lo:                   # a pooling gate's target would sit outside the window
            raise ValueError(f"layer {k}: pools wires {hi}..{prev - 1} onto {hi - 1} below the window's lowest wire {lo}")
        out.append((lo, hi, range(prev - 1, hi - 1, -1), colour))
    return out


def unitary(w, depth, dtype=torch.complex64, hier=False):
    """The circuit's 2^n x 2^n unitary: depth+1 dense layer matrices, the CNOT layers folded in as row permutations."""
    n = w.shape[1]
    if hier:
        w = w * hier_mask(depth, n, w.shape[2], w.dtype, w.device, hier)
    mats = rot_mats(w, dtype)
    U = _layer_matrix(mats[0])
    for k in range(depth):
        U = U[_cnot_perm(n, k % 2, U.device, hier_lo(k, n, hier) if hier else 0,
                         extra=hier_parts(hier)[2] if hier else 0)]            # CNOT layer: permute rows
        U = _layer_matrix(mats[k + 1]) @ U
    return U


def run(amps, w, depth, dtype=torch.complex64, hier=False, qcnn=0, mask=None, fmask=None):
    """Final state of embedding amps (B, 2^n) -> [rotation layer, brickwall CNOTs] x depth -> rotation layer,
    applied layer by layer (fastest for a training batch; use unitary() when the circuit acts on ~2^n states).
    hier: hierarchical (patch-local) circuit - layer k acts only on wires >= hier_lo(k, n, hier) (fine scales first,
    as a CNN); True or (wires, every). Combines with qcnn: the window grows from the fine wires while the QCNN pools them.
    qcnn = keep > 0: QCNN - the active wires shrink to the first `keep` (qcnn_active); each wire dropped is pooled by a
    controlled rotation onto the wire above it (no mid-circuit measurement: deferred measurement) and never touched
    again.
    mask = (K, m): after layer K's CNOTs the state is multiplied by the real mask m (2^n,) (mask_vector) - not unitary:
    the result is the post-selected state, unnormalised (its squared norm is the chance the light gets through).
    fmask = (pix, d): the Fourier-plane mask (fourier_filter) on the input, before the first layer; also unnormalised."""
    n = w.shape[1]
    if hier:
        w = w * hier_mask(depth, n, w.shape[2], w.dtype, w.device, hier)
    state = amps.to(dtype)
    if fmask is not None:
        state = fourier_filter(state, fmask[1], fmask[0], fmask[2] if len(fmask) > 2 else False)
    mats = rot_mats(w, dtype)
    sched = _schedule(n, depth, hier, qcnn)
    for k in range(depth + 1):
        lo, hi, pooled, colour = sched[k]
        for j in pooled:
            state = _apply_controlled(state, mats[k, j], j)
        if hi < n:
            mats_k = torch.cat([mats[k, :hi], torch.eye(2, dtype=mats.dtype, device=mats.device).expand(n - hi, 2, 2)])
        else:
            mats_k = mats[k]
        state = _apply_layer(state, mats_k)
        if k < depth:
            state = state[:, _cnot_perm(n, k % 2, state.device, lo, hi, colour)]
        if mask is not None and k == mask[0]:
            state = state * mask[1]
    return state


def run_inverse(state, w, depth, dtype=torch.complex64, hier=False, qcnn=0, mask=None, fmask=None):
    """U^dagger applied to states (B, 2^n): the circuit of run() backwards, every rotation and pooling gate
    conjugate-transposed and every CNOT permutation inverted (used for M_g psi = U^dag Pi_g U psi, the evidence maps).
    With a mask (real, diagonal: its own adjoint) this is the adjoint A^dag of the masked map A, not an inverse."""
    n = w.shape[1]
    if hier:
        w = w * hier_mask(depth, n, w.shape[2], w.dtype, w.device, hier)
    mats = rot_mats(w, dtype).conj().transpose(-1, -2)
    eye = torch.eye(2, dtype=mats.dtype, device=mats.device)
    sched = _schedule(n, depth, hier, qcnn)
    state = state.to(dtype)
    for k in range(depth, -1, -1):
        lo, hi, pooled, colour = sched[k]
        if mask is not None and k == mask[0]:
            state = state * mask[1]
        if k < depth:
            perm = _cnot_perm(n, k % 2, state.device, lo, hi, colour)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm), device=perm.device)
            state = state[:, inv]
        state = _apply_layer(state, torch.cat([mats[k, :hi], eye.expand(n - hi, 2, 2)]) if hi < n else mats[k])
        for j in reversed(pooled):
            state = _apply_controlled(state, mats[k, j], j)
    if fmask is not None:                                                    # adjoint: the conjugate mask
        state = fourier_filter(state, fmask[1].conj(), fmask[0], fmask[2] if len(fmask) > 2 else False)
    return state


def _channel_parts(w, depth, bank, n_anc):
    """Packed weights -> (main circuit rows, per-channel bank weights (C, bank + 1, picture wires, rot))."""
    C = 2 ** n_anc
    wb = w[depth + 1:].reshape(C, bank + 1, w.shape[1], w.shape[2])[:, :, n_anc:]
    return w[:depth + 1], wb


def run_channels(amps, w, depth, bank, n_anc, qcnn=0, dtype=torch.complex64):
    """Channel circuit: n_anc ancilla wires (the most significant, starting in |0>) give C = 2^n_anc channels.
    H on every ancilla splits the picture evenly over the channels; the filter bank runs channel c's own brickwall
    circuit U_c (bank layers, its own angles) on the picture wires - controlled on the ancillas reading c; then the
    main circuit (brickwall or QCNN, depth layers) acts on all wires, so it can mix the channels.
    w packs both: rows 0..depth are the main circuit over all wires, then C x (bank + 1) rows of bank angles (the
    picture columns are used). amps (B, 2^n_picture) -> state (B, 2^(n_anc + n_picture))."""
    wq, wb = _channel_parts(w, depth, bank, n_anc)
    a = amps.to(dtype) / (2 ** n_anc) ** 0.5
    state = torch.stack([run(a, wb[c], bank, dtype) for c in range(len(wb))], 1).reshape(len(amps), -1)
    return run(state, wq, depth, dtype, qcnn=qcnn)


def run_channels_inverse(state, w, depth, bank, n_anc, qcnn=0, dtype=torch.complex64):
    """<0_anc| H U_bank^dag V^dag applied to states (B, 2^(n_anc + n)): the picture-space part (B, 2^n) of U^dag state,
    for the evidence maps (the input is |0_anc> (x) picture)."""
    wq, wb = _channel_parts(w, depth, bank, n_anc)
    back = run_inverse(state, wq, depth, dtype, qcnn=qcnn).reshape(len(state), len(wb), -1)
    return sum(run_inverse(back[:, c], wb[c], bank, dtype) for c in range(len(wb))) / (2 ** n_anc) ** 0.5


def run_dense(amps, w, depth, dtype=torch.complex64, hier=False):
    """Same as run(), through the dense unitary (one batched matmul)."""
    return amps.to(dtype) @ unitary(w, depth, dtype, hier).T


def probs(state, n_out):
    """Born probabilities of the first n_out wires."""
    return (state.abs() ** 2).reshape(len(state), 2 ** n_out, -1).sum(-1)
