#!/usr/bin/env python3
"""Seeing Why in plain Python: classify the test pictures, draw one evidence map, and run the deletion test on it.
The same code is in docs/run-locally.md. CPU, a few seconds.

Run (from anywhere in the clone):  python examples/quickstart.py   -> numbers on screen, examples/quickstart.png
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "demo"))

import matplotlib                                              # noqa: E402
matplotlib.use("Agg")                                          # draws to a file, no window needed
import matplotlib.pyplot as plt                                # noqa: E402

import seeing_why as sw                                        # noqa: E402

x, y = sw.test_pictures()                      # the 300 Caltech-10 test pictures, 32 x 32 colour
run = sw.load("phase plates (row D)")          # or "qubit circuit (row D')", or a results/<run>.json path
p = sw.classify(run, x)                        # BHR class probabilities, (300, 10)
print(f"test accuracy {100 * (p.argmax(1) == y).mean():.1f} %")

i = 210                                        # one test picture (a ketch)
cls = int(y[i])
e = sw.evidence(run, x[i:i + 1], [cls])[0]     # 32 x 32 evidence map for the true class
print(f"picture {i}: {sw.NAMES[cls]}, answer {sw.NAMES[p[i].argmax()]} (p = {p[i].max():.2f}); "
      f"the map sums to {e.sum():+.4f}")

for kind in sw.RANKINGS:                       # the deletion test: the top 5 % of pixels by each ranking
    q = sw.delete(run, x[i:i + 1], sw.ranking(run, x[i], cls, kind)[None], 0.05)[0]
    print(f"  delete 5 % by {kind:18s} -> {sw.NAMES[q.argmax()]:10s} p({sw.NAMES[cls]}) = {q[cls]:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(x[i])
ax[0].set_title(f"picture: {sw.NAMES[cls]}", loc="left")
sw.overlay(ax[1], x[i], e)
ax[1].set_title("evidence (red for, blue against)", loc="left")
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
out = pathlib.Path(__file__).resolve().parent / "quickstart.png"
fig.savefig(out, dpi=150)
print(out)
