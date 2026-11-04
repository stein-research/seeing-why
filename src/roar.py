#!/usr/bin/env python3
"""ROAR (Hooker et al. 2019; referees, second round): does deleting the pixels the evidence ranks highest still hurt
when the model is retrained on the deleted pictures? The deletion test of deletion_baselines.py deletes from the
pictures of a fixed model, so the drop can come from pictures unlike the training ones; ROAR retrains instead.
Row D (phase plates), the seed-0 model ranks the pixels of every Caltech-10 picture (training and test) by
  evidence  - the exact true-class evidence (signed), highest first;
  intensity - the encoded light of the pixel (sum over the colour planes of its amplitude squared), highest first;
  random    - a fixed draw per picture.
A deleted pixel takes the colour means of the pixels kept, in R, G, B: after centring it has zero contrast, as in the
deletion test (there the means plane keeps the whole picture's means; here it holds the kept pixels'). Row D is trained again from scratch on each deleted set, seeds 0-2, and
tested on the test pictures deleted the same way.
Run:  python src/roar.py prepare   -> data/roar/<ranking>_<percent>.npz
      python src/roar.py train     -> results/pilot_..._roar<ranking><percent>_..._s<seed>.json (a done run
                                                  is skipped; about a minute each on a GPU)
      python src/roar.py report    -> table, results/roar.json
The report also asks how much the deletion pattern alone gives away: a logistic regression (C = 0.1) trained on the
training pictures' masks of deleted pixels only (1,024 zeros and ones, no picture), tested on the test pictures'
masks. A ranking by true-class evidence chooses the pixels with the class's label, so its masks can carry the label
into the retrained model (a known weakness of ROAR); a ranking that knows nothing of the class (intensity, random)
cannot carry it.
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
RUN = "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s0.json"
FRACS = [0.1, 0.3, 0.5, 0.7]
RANKINGS = ["evidence", "intensity", "random"]
SEEDS = [0, 1, 2]
ROW_D = ("--mode bhr --pt_ratio 0.0556 --rot rzryrx --order morton --n_out 7 --sim fast --ent brickwall "
         "--dataset caltech10fct --n_qubits 12 --n_train 2644 --balanced --aug_crop 0.8 --epochs 50 --device cuda "
         "--optical 4 --center_image --mean_plane 0.5 --fourier_mask --fourier_linear --fourier_phase").split()


def result(rank, frac, seed):
    return HERE / "results" / RUN.replace("fmLp_o7", f"fmLp_roar{rank}{round(100 * frac)}_o7").replace("_s0.", f"_s{seed}.")


def prepare():
    from attention_in_box import maps
    run = json.loads((HERE / "results" / RUN).read_text())
    d = dict(np.load(HERE / "data" / f"{run['dataset']}.npz"))
    rng = np.random.default_rng(0)
    score = {}
    for split, ys in (("x_train", "y_train"), ("x_val", "y_val")):
        X, Y = d[split], d[ys]
        m = np.concatenate([maps(run, X[i:i + 500], Y[i:i + 500], signed=True) for i in range(0, len(X), 500)], 1)
        score[split] = dict(evidence=m[2].reshape(len(X), -1), intensity=m[0].reshape(len(X), -1),
                            random=rng.random((len(X), 32 * 32)))
    (HERE / "data" / "roar").mkdir(exist_ok=True)
    for rank in RANKINGS:
        for frac in FRACS:
            k = round(frac * 32 * 32)
            out = dict(d)
            for split in ("x_train", "x_val"):
                x = d[split].astype(np.float32)
                dead = np.zeros(score[split][rank].shape, bool)
                np.put_along_axis(dead, np.argsort(-score[split][rank], 1)[:, :k], True, 1)
                dead = dead.reshape(len(x), 32, 32)
                keep = ~dead[..., None]                              # the kept pixels' colour means: after
                mu = (x * keep).sum((1, 2), keepdims=True) / keep.sum((1, 2), keepdims=True)   # centring, zero contrast
                out[split] = np.where(dead[..., None], mu, x)
            np.savez_compressed(HERE / "data" / "roar" / f"{rank}_{round(100 * frac)}.npz", **out)
            print(f"data/roar/{rank}_{round(100 * frac)}.npz")


def train():
    py = sys.executable
    for seed in SEEDS:
        for frac in FRACS:
            for rank in RANKINGS:
                f = result(rank, frac, seed)
                if f.exists():
                    print("done:", f.name)
                    continue
                tag = f"roar{rank}{round(100 * frac)}"
                (HERE / "logs").mkdir(exist_ok=True)
                log = HERE / "logs" / f"roar_{rank}{round(100 * frac)}_s{seed}.log"
                with open(log, "w") as fh:
                    subprocess.run([py, "-u", str(pathlib.Path(__file__).resolve().parent / "pilot_fmnist.py"), *ROW_D, "--seed", str(seed), "--tag", tag,
                                    "--data_npz", str(HERE / "data" / "roar" / f"{rank}_{round(100 * frac)}.npz")],
                                   cwd=HERE, stdout=fh, stderr=subprocess.STDOUT, check=True)
                print(tag, seed, json.loads(f.read_text())["test_exact"], flush=True)


def report():
    base = [json.loads((HERE / "results" / RUN.replace("_s0.", f"_s{s}.")).read_text())["test_exact"] for s in SEEDS]
    out = dict(fracs=FRACS, seeds=SEEDS, none=dict(mean=float(np.mean(base)), sd=float(np.std(base, ddof=1))))
    print(f"no pixel deleted: {100 * np.mean(base):.1f} +- {100 * np.std(base, ddof=1):.1f} %")
    print("deleted, retrained " + "".join(f"{100 * f:>14.0f}%" for f in FRACS))
    for rank in RANKINGS:
        acc = np.array([[json.loads(result(rank, f, s).read_text())["test_exact"] for s in SEEDS] for f in FRACS])
        out[rank] = dict(mean=acc.mean(1).tolist(), sd=acc.std(1, ddof=1).tolist(), runs=acc.tolist())
        print(f"  {rank:16s}" + "".join(f"{100 * m:>9.1f} +- {100 * s:3.1f}" for m, s in zip(acc.mean(1), acc.std(1, ddof=1))))
    from sklearn.linear_model import LogisticRegression
    d = np.load(HERE / "data" / "caltech10fct.npz")
    print("the deleted pixels' pattern alone, logistic regression, test accuracy")
    for rank in RANKINGS:
        acc = []
        for f in FRACS:
            r = np.load(HERE / "data" / "roar" / f"{rank}_{round(100 * f)}.npz")
            mask = lambda s: (np.abs(r[s] - d[s].astype(np.float32)).sum(-1) > 1e-4).reshape(len(d[s]), -1).astype(float)
            clf = LogisticRegression(max_iter=2000, C=0.1).fit(mask("x_train"), d["y_train"])
            acc.append(float(clf.score(mask("x_val"), d["y_val"])))
        out[rank]["mask_only_acc"] = acc
        print(f"  {rank:16s}" + "".join(f"{100 * a:>16.1f}" for a in acc))
    (HERE / "results" / "roar.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    {"prepare": prepare, "train": train, "report": report}[sys.argv[1]]()
