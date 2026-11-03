#!/usr/bin/env python3
"""Stronger baselines for the 2x2 hardware maps (referee review 2026-11-03). The cross-picture shuffle of
map_intervals.py keeps each map's structure but breaks the picture; it says nothing about the pattern within a
picture, and most exact blocks are positive. Here, for each chip run against the exact maps:
  - sign baseline: the share of exact blocks that are positive (what 'same sign' gets by guessing +);
  - uniform split: each picture's exact P - b spread evenly over its four blocks, correlated with the exact maps
    (what a map with no spatial information scores, pooled);
  - within-picture permutation null: each chip map's four blocks permuted within the picture (10,000 draws), the
    pooled correlation with the exact maps, 95th percentile;
  - within-picture-centred correlation: each picture's mean block removed from both maps before correlating (the
    spatial pattern alone; a drift of the shared whole-picture term shifts all four blocks equally and drops out);
  - energy-weighted split (referee B, second round): each picture's exact P - b spread over its blocks in proportion
    to their light ||psi_S||^2 (heat_g2.npz norms), pooled and centred.
Run:  python src/map_nulls.py  -> numbers, results/map_nulls.json
"""
import glob
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root


def corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    m = json.load(open(glob.glob(str(HERE / "results" / "ibm_testset_*L8" / "heat_g2_maps.json"))[0]))
    ex = np.array(m["exact"])
    out = {"sign_baseline": float((ex > 0).mean()),
           "uniform_split_corr": corr(np.repeat(ex.sum(1, keepdims=True) / 4, 4, 1), ex)}
    h = np.load(glob.glob(str(HERE / "results" / "ibm_testset_*L8" / "heat_g2.npz"))[0])
    en = h["norms"].reshape(len(ex), 4, 2)[:, :, 0] * ex.sum(1, keepdims=True)    # (in, out) per block
    cen = lambda x: x - x.mean(1, keepdims=True)
    out.update(energy_split_corr=corr(en, ex), energy_split_centred=corr(cen(en), cen(ex)))
    print(f"exact blocks positive: {out['sign_baseline']:.3f}; uniform split vs exact: r = {out['uniform_split_corr']:.3f}; "
          f"energy split r = {out['energy_split_corr']:.3f}, centred {out['energy_split_centred']:.3f}")
    rng = np.random.default_rng(0)
    for kind in [k for k in m if k.startswith("chip")]:
        ch = np.array(m[kind])
        null = [corr(np.array([c[rng.permutation(4)] for c in ch]), ex) for _ in range(10000)]
        r = dict(corr=corr(ch, ex), within_perm_null_95=float(np.percentile(null, 95)),
                 p_within_perm=float((np.array(null) >= corr(ch, ex)).mean()),
                 centred_corr=corr(ch - ch.mean(1, keepdims=True), ex - ex.mean(1, keepdims=True)),
                 signs=float((np.sign(ch) == np.sign(ex)).mean()))
        out[kind] = r
        print(f"{kind}: r {r['corr']:.3f}; within-picture permutation null 95th pct {r['within_perm_null_95']:.3f} "
              f"(p {r['p_within_perm']:.4f}); centred r {r['centred_corr']:.3f}; same sign {r['signs']:.3f}")
    (HERE / "results" / "map_nulls.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
