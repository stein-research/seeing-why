#!/usr/bin/env python3
"""Classical references on trimmed Caltech-10 (referee 2026-10-31: put the accuracies in context). Same split as the
trainer (seed 0: 2,644 training pictures, the same 300 test pictures):
  logistic regression on the raw RGB pixels (standardised);
  logistic regression on the mean-plane amplitudes the circuits receive (normalised, 4,096 values);
  a small CNN (two 3x3 conv layers of 16 / 32 channels with pooling, one linear layer; ~23k parameters), with the same
  random crops as the trainer, 50 epochs, Adam 1e-3, 3 seeds.
Run:  python src/classical_refs.py  -> numbers on stdout, results/classical_refs_caltech10fct.json
"""
import json
import pathlib

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, label_binarize

import pilot_fmnist as pf

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root


def auc(p, y):
    return float(roc_auc_score(label_binarize(y, classes=np.arange(10)), p, average="macro"))


def main():
    d = np.load(HERE / "data" / "caltech10fct.npz")
    rng = np.random.default_rng(0)                              # the trainer's split (load(n_train, n_val, seed=0))
    idx = rng.permutation(len(d["x_train"]))
    tr = idx[:2644]
    xtr, ytr, xte, yte = d["x_train"][tr], d["y_train"][tr], d["x_val"], d["y_val"]
    out = {}
    sc = StandardScaler().fit(xtr.reshape(len(xtr), -1).astype(float))
    lr = LogisticRegression(max_iter=3000, C=0.01).fit(sc.transform(xtr.reshape(len(xtr), -1).astype(float)), ytr)
    p = lr.predict_proba(sc.transform(xte.reshape(len(xte), -1).astype(float)))
    out["logreg_pixels"] = dict(acc=float((p.argmax(1) == yte).mean()), auc=auc(p, yte), params=int(lr.coef_.size + 10))
    pf.DATASET, pf.ORDER, pf.N_QUBITS = "caltech10fct", "morton", 12
    perm = pf.morton_perm()

    def enc(x):
        v = x.astype(float).transpose(0, 3, 1, 2).reshape(len(x), 3, -1)[:, :, perm].reshape(len(x), -1)
        v = pf.centre_planes(v, 0, 0.5)
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    lr2 = LogisticRegression(max_iter=3000, C=1.0).fit(enc(xtr) * 64, ytr)
    p = lr2.predict_proba(enc(xte) * 64)
    out["logreg_meanplane_amplitudes"] = dict(acc=float((p.argmax(1) == yte).mean()), auc=auc(p, yte),
                                              params=int(lr2.coef_.size + 10))
    torch.set_num_threads(8)
    accs, aucs = [], []
    for seed in range(3):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Conv2d(3, 16, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                                  torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
                                  torch.nn.Flatten(), torch.nn.Linear(32 * 8 * 8, 10))
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        r = np.random.default_rng(seed)
        to_t = lambda x: torch.tensor(x.astype(np.float32).transpose(0, 3, 1, 2) / 255.0)
        for ep in range(50):
            xa = pf.random_crops(xtr, r, 0.8)
            order = r.permutation(len(xa))
            for i in range(0, len(xa), 64):
                b = order[i:i + 64]
                loss = torch.nn.functional.cross_entropy(net(to_t(xa[b])), torch.tensor(ytr[b], dtype=torch.long))
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            p = torch.softmax(net(to_t(xte)), 1).numpy()
        accs.append(float((p.argmax(1) == yte).mean()))
        aucs.append(auc(p, yte))
        print(f"cnn seed {seed}: {accs[-1]:.3f} / {aucs[-1]:.3f}", flush=True)
    out["small_cnn"] = dict(acc=float(np.mean(accs)), acc_std=float(np.std(accs, ddof=1)), auc=float(np.mean(aucs)),
                            params=int(sum(p_.numel() for p_ in net.parameters())), seeds=accs)
    for k, v in out.items():
        print(f"{k:30s} accuracy {v['acc']:.3f}  ROC-AUC {v['auc']:.3f}  params {v['params']:,}")
    (HERE / "results" / "classical_refs_caltech10fct.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
