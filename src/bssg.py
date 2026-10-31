"""BornScaledSoftmax (BSS) and its grouped form (BSSG) — parameter-free Born readouts.

Re-implemented for this project from the owner's definitions (BSS manuscript and the BSSG design used in
his RL work). Nothing here is imported or copied from the PhD folder.

BSS   (N basis states, classes = basis states):
    z_b = P_b - 1/N;  z~ = LeakyReLU_alpha(z);  p_hat = softmax(kappa z~)
    kappa = ln[(1 - eps)(N - 1)/eps] / ((r - 1)/N)            (r = pt-ratio)
BSSG  (N basis states marginalised onto G groups S_g, contiguous, larger groups first):
    P_g = sum_{b in S_g} P_b;   baseline_g = |S_g|/N;   r_g = P_g / baseline_g
    z_g = r_g - 1;  z~ = LeakyReLU_alpha(z);  p_hat = softmax(kappa z~)
    kappa = ln[(1 - eps)(G - 1)/eps] / (r - 1)
    With G = N it equals BSS exactly (both are N(P - 1/N) up to the factor that kappa absorbs).

Readout modes (two independent axes, as in the owner's RL v1.9):
    raw_renorm  keep the first C basis states, renormalise           (no BSS)
    raw_group   marginalise onto C groups                             (no BSS)
    bss_renorm  BSS over all N states (dummies in the denominator), output restricted to the first C
                and renormalised
    bss_group   BSSG with G = C
    bss_group_hr   headroom-normalised BSSG: z_g = (P_g - b_g)/(1 - b_g), b_g = |S_g|/N, so every group has ceiling 1;
                pt_ratio is read as phi = target fraction of headroom (kappa = ln[(1-eps)(G-1)/eps] / phi)
    linear_head CONTROL ONLY, not part of the method: a trainable linear layer N -> C on the Born probabilities
                (N*C + C parameters, e.g. 16*10 + 10 = 170, as in Shen et al. 2024 and DAQC 2025)
    bss_group_ref  BSSG with a reference baseline b_g instead of |S_g|/N: b_g = the mean group probability of
                the untrained circuit over the training images (class-agnostic, no label used, computed once
                and frozen, not trained). r_g = P_g / b_g. Call set_baseline() before use.
All modes return class probabilities of shape (..., C) that sum to 1. None has a trainable parameter.
"""
import math

import torch
import torch.nn.functional as F

READOUT_MODES = ("raw_renorm", "raw_group", "bss_renorm", "bss_group", "bss_group_ref", "bss_group_hr", "bssg_phi", "bhr",
                 "bhr_single", "linear_head", "lss", "temp_group")


def make_group_map(n, g):
    """Contiguous, balanced partition of 0..n-1 into g groups, larger groups first."""
    if not 1 <= g <= n:
        raise ValueError(f"n_groups={g} must be in [1, {n}]")
    base, rem = divmod(n, g)
    sizes = [base + 1] * rem + [base] * (g - rem)
    groups, start = [], 0
    for s in sizes:
        groups.append(list(range(start, start + s)))
        start += s
    return groups


def class_groups(mode, n, g):
    """Outcome sets of the classes: contiguous balanced groups, or for bhr_single one outcome per class, evenly spaced
    (owner 2026-10-30: one basis state per class, every other outcome a distractor that counts for no class)."""
    if mode == "bhr_single":
        return [[round(k * n / g)] for k in range(g)]
    return make_group_map(n, g)


class BSS(torch.nn.Module):
    def __init__(self, n_states, pt_ratio=1.25, eps=0.01, leaky_slope=0.01):
        super().__init__()
        if not 1.0 < pt_ratio < n_states:
            raise ValueError("pt_ratio must be in (1, N)")
        self.n, self.pt_ratio, self.eps, self.alpha = n_states, pt_ratio, eps, leaky_slope
        self.kappa = math.log((1 - eps) * (n_states - 1) / eps) / ((pt_ratio - 1) / n_states)

    def forward(self, probs):
        z = F.leaky_relu(probs - 1.0 / self.n, negative_slope=self.alpha)
        return F.softmax(self.kappa * z, dim=-1)


class BSSG(torch.nn.Module):
    def __init__(self, n_states, n_groups, pt_ratio=1.25, eps=0.01, leaky_slope=0.01):
        super().__init__()
        self.n, self.g = n_states, n_groups
        groups = make_group_map(n_states, n_groups)
        self.group_map = groups
        sizes = torch.tensor([len(s) for s in groups], dtype=torch.float32)
        self.register_buffer("baseline", sizes / n_states)
        g_of_b = torch.empty(n_states, dtype=torch.long)
        for gi, s in enumerate(groups):
            g_of_b[s] = gi
        self.register_buffer("group_of_basis", g_of_b)
        ceiling = n_states / max(len(s) for s in groups)
        if not 1.0 < pt_ratio < ceiling:
            raise ValueError(f"pt_ratio must be in (1, {ceiling:.4f}) for these groups")
        self.pt_ratio, self.eps, self.alpha = pt_ratio, eps, leaky_slope
        self.kappa = math.log((1 - eps) * (n_groups - 1) / eps) / (pt_ratio - 1)

    def set_baseline(self, b):
        """Replace |S_g|/N by a reference baseline (G,), e.g. the mean P_g of the untrained circuit."""
        b = torch.as_tensor(b, dtype=torch.float32).clamp_min(1e-6)
        b = b / b.sum()
        if not self.pt_ratio < 1.0 / b.max().item():
            raise ValueError(f"pt_ratio {self.pt_ratio} not reachable: ceiling 1/max b = {1 / b.max().item():.3f}")
        self.baseline.copy_(b)

    def marginalize(self, probs):
        out = probs.new_zeros(*probs.shape[:-1], self.g)
        return out.scatter_add(-1, self.group_of_basis.expand(*probs.shape[:-1], self.n), probs)

    def forward(self, probs):
        r = self.marginalize(probs) / self.baseline
        z = F.leaky_relu(r - 1.0, negative_slope=self.alpha)
        return F.softmax(self.kappa * z, dim=-1)


class BornHeadroomReadout(torch.nn.Module):
    """Born Headroom Readout, BHR (see docs/bhr_spec.md; formerly "BSSG-phi"): one free variable phi in (0, 1),
    the target headroom fraction.

    u_g = (P_g - b_g)/(1 - b_g), b_g = |S_g|/N (no-information anchor), p_hat = softmax(kappa LeakyReLU_alpha(u)),
    kappa = ln[(1 - eps)(G - 1)/eps] / phi. eps and alpha are fixed constants, not tuned.
    """

    EPS, ALPHA = 0.01, 0.01

    def __init__(self, n_states, n_groups, phi, groups=None):
        super().__init__()
        if not 0.0 < phi < 1.0:
            raise ValueError("phi must be in (0, 1)")
        self.n, self.g, self.phi = n_states, n_groups, phi
        groups = groups or make_group_map(n_states, n_groups)
        self.group_map = groups
        self.register_buffer("baseline", torch.tensor([len(s) / n_states for s in groups]))
        g_of_b = torch.full((n_states,), n_groups, dtype=torch.long)    # index n_groups = distractor bin (dropped)
        for gi, s in enumerate(groups):
            g_of_b[s] = gi
        self.register_buffer("group_of_basis", g_of_b)
        self.kappa = math.log((1 - self.EPS) * (n_groups - 1) / self.EPS) / phi

    def headroom(self, probs):
        pg = probs.new_zeros(*probs.shape[:-1], self.g + 1).scatter_add(
            -1, self.group_of_basis.expand(*probs.shape[:-1], self.n), probs)[..., :self.g]
        return (pg - self.baseline) / (1 - self.baseline)

    def forward(self, probs):
        return F.softmax(self.kappa * F.leaky_relu(self.headroom(probs), negative_slope=self.ALPHA), dim=-1)


BSSGPhi = BornHeadroomReadout          # old name, kept so earlier runs and scripts still work


class Readout(torch.nn.Module):
    """Class probabilities (..., C) from Born probabilities (..., N) under one of READOUT_MODES."""

    def __init__(self, mode, n_states, n_classes, pt_ratio=1.25, eps=0.01, leaky_slope=0.01):
        super().__init__()
        if mode not in READOUT_MODES:
            raise ValueError(mode)
        self.mode, self.n, self.c = mode, n_states, n_classes
        if mode == "linear_head":
            self.head = torch.nn.Linear(n_states, n_classes)
        elif mode in ("bssg_phi", "bhr", "bhr_single"):
            self.phi_head = BornHeadroomReadout(n_states, n_classes, pt_ratio,      # the pt_ratio argument carries phi here
                                                class_groups(mode, n_states, n_classes))
        elif mode == "temp_group":
            # control for BHR (referee 2026-10-31): softmax of the class group sums with one learned temperature,
            # the readout of diffractive classifiers; same groups as BHR, no anchor, no headroom units, no kink
            self.phi_head = BornHeadroomReadout(n_states, n_classes, 0.5, class_groups("bhr", n_states, n_classes))
            self.log_tau = torch.nn.Parameter(torch.tensor(math.log(10.0)))
        elif mode == "bss_renorm":
            self.bss = BSS(n_states, pt_ratio, eps, leaky_slope)
        elif mode == "lss":
            # Leaky Scaled Softmax on Pauli-Z expectations (the owner's LSS): class c reads <Z_c> on measured wire c,
            # anchor 0 (maximally mixed state), ceiling 1; pt_ratio carries the effective maximum z_eff, the <Z> at
            # which the class reaches probability 1 - eps against C - 1 classes at 0
            n_wires = n_states.bit_length() - 1
            if n_classes > n_wires:
                raise ValueError(f"lss needs one measured wire per class ({n_classes} > {n_wires})")
            b = torch.arange(n_states)
            bits = torch.stack([(b >> (n_wires - 1 - c)) & 1 for c in range(n_classes)], 1)   # wire 0 = MSB
            self.register_buffer("zsign", (1 - 2 * bits).float())                         # (N, C): +1 for |0>
            self.alpha = leaky_slope
            self.kappa_z = math.log((1 - eps) * (n_classes - 1) / eps) / pt_ratio
        else:
            # raw_group uses only the marginalisation; pt_ratio is irrelevant there
            if mode == "bss_group_hr":
                if not 0.0 < pt_ratio < 1.0:
                    raise ValueError("bss_group_hr takes phi in (0, 1) as pt_ratio")
                self.kappa_hr = math.log((1 - eps) * (n_classes - 1) / eps) / pt_ratio
            self.bssg = BSSG(n_states, n_classes,
                             pt_ratio if mode in ("bss_group", "bss_group_ref")
                             else min(1.25, n_states / math.ceil(n_states / n_classes) - 1e-3),
                             eps, leaky_slope)

    def forward(self, probs):
        if self.mode == "linear_head":
            return F.softmax(self.head(probs * self.n), dim=-1)     # inputs scaled to ~1 per outcome
        if self.mode in ("bssg_phi", "bhr", "bhr_single"):
            return self.phi_head(probs)
        if self.mode == "temp_group":
            h = self.phi_head
            pg = probs.new_zeros(*probs.shape[:-1], h.g + 1).scatter_add(
                -1, h.group_of_basis.expand(*probs.shape[:-1], h.n), probs)[..., :h.g]
            return F.softmax(self.log_tau.exp() * pg, dim=-1)
        if self.mode == "lss":
            z = probs @ self.zsign.to(probs.dtype)                   # <Z_c>
            return F.softmax(self.kappa_z * F.leaky_relu(z, negative_slope=self.alpha), dim=-1)
        if self.mode == "raw_renorm":
            p = probs[..., :self.c]
            return p / p.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.mode == "raw_group":
            return self.bssg.marginalize(probs)
        if self.mode == "bss_renorm":
            # BSS over all N, restricted to the first C and renormalised = softmax of the first C BSS logits
            # (computed that way for numerical stability: the earlier divide-by-sum gave 0/0 when dummies dominated)
            z = F.leaky_relu(probs - 1.0 / self.n, negative_slope=self.bss.alpha)
            return F.softmax(self.bss.kappa * z[..., :self.c], dim=-1)
        if self.mode == "bss_group_hr":
            b = self.bssg.baseline
            z = (self.bssg.marginalize(probs) - b) / (1 - b)
            z = F.leaky_relu(z, negative_slope=self.bssg.alpha)
            return F.softmax(self.kappa_hr * z, dim=-1)
        return self.bssg(probs)                       # bss_group and bss_group_ref
