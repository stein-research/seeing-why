#!/usr/bin/env python3
"""Caltech-101, the 10 largest classes, balanced to 100 pictures each, 32x32 colour letterboxed (12 qubits).

Input:  data/caltech101/caltech-101.zip from CaltechDATA (record mzrjq-6wc02, CC BY 4.0; MD5 in data/caltech101/MD5SUMS,
        matched), extracted to data/caltech101/x/101_ObjectCategories/<class>/*.jpg.
Classes: the 10 largest object classes (BACKGROUND_Google and Faces_easy, a duplicate of Faces, left out): airplanes,
        Motorbikes, Faces, watch, Leopards, bonsai, car_side, ketch, chandelier, hawksbill - 800 down to 100 pictures.
Balance (owner 2026-10-30): 100 pictures a class drawn at random (seed 0), 80 train / 20 test each.
Resize: RGB, longer side to 32 with a box filter, centred on black (letterbox, as Animals-10).
Output: data/caltech10c.npz with x_train (800, 32, 32, 3) uint8, y_train, x_val (200, ...), y_val, class_names.
--all (owner 2026-10-30, more data): every picture of the 10 classes (3,044); the test set is still balanced, 30 per
class (300), and every other picture trains (2,744, unbalanced 770 ... 70) - the trainer balances it with
--balanced. Output data/caltech10f.npz.
--crop (owner 2026-10-30): centre-crop to a square, then resize to 32 - no black bars, so the aspect ratio is no longer
visible (the letterbox bars were a shortcut: airplane / car_side pictures are wide). Output caltech10fc.npz with --all.
--trim F (2026-10-31): first cut the fraction F off every side of the original picture - the Leopards pictures carry a
black frame 2-3 px wide (top and left), which the circuits used as a shortcut (README "Shortcut audit"). Output name
gets a "t": caltech10fct.npz with --all --crop --trim 0.04. Same pictures and split as without it.

Run:  python -I src/prepare_caltech101.py [--all]
"""
import pathlib
import sys

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
SRC = HERE / "data" / "caltech101" / "x" / "101_ObjectCategories"
CLASSES = ["airplanes", "Motorbikes", "Faces", "watch", "Leopards", "bonsai", "car_side", "ketch", "chandelier",
           "hawksbill"]
SIDE, PER_CLASS, TEST = 32, 100, 20


def letterbox(im):
    w, h = im.size
    k = SIDE / max(w, h)
    im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.BOX)
    out = Image.new("RGB", (SIDE, SIDE), (0, 0, 0))
    out.paste(im, ((SIDE - im.size[0]) // 2, (SIDE - im.size[1]) // 2))
    return np.asarray(out, dtype=np.uint8)


TRIM = 0.0


def trim(im):
    w, h = im.size
    dx, dy = round(w * TRIM), round(h * TRIM)
    return im.crop((dx, dy, w - dx, h - dy)) if TRIM else im


def centre_crop(im):
    im = trim(im)
    w, h = im.size
    k = min(w, h)
    im = im.crop(((w - k) // 2, (h - k) // 2, (w - k) // 2 + k, (h - k) // 2 + k))
    return np.asarray(im.resize((SIDE, SIDE), Image.BOX), dtype=np.uint8)


def main():
    rng = np.random.default_rng(0)
    every = "--all" in sys.argv
    crop = "--crop" in sys.argv
    global TRIM
    if "--trim" in sys.argv:
        TRIM = float(sys.argv[sys.argv.index("--trim") + 1])
    shape = centre_crop if crop else letterbox
    out = {k: [] for k in ("x_train", "y_train", "x_val", "y_val")}
    for c, name in enumerate(CLASSES):
        files = sorted((SRC / name).glob("*.jpg"))
        pick = rng.permutation(len(files)) if every else rng.choice(len(files), PER_CLASS, replace=False)
        for j, i in enumerate(pick):
            split = "val" if j < (30 if every else TEST) else "train"
            out[f"x_{split}"].append(shape(Image.open(files[i]).convert("RGB")))
            out[f"y_{split}"].append(c)
    out = {k: np.array(v, dtype=np.uint8 if k.startswith("x") else np.int64) for k, v in out.items()}
    for s in ("train", "val"):
        print(s, out[f"x_{s}"].shape, "per class", np.bincount(out[f"y_{s}"]).tolist())
    name = ("caltech10f" if every else "caltech10c") + ("c" if crop else "") + ("t" if TRIM else "") + ".npz"
    np.savez_compressed(HERE / "data" / name, **out,
                        class_names=np.array([f"{k:03d}_{c}" for k, c in enumerate(CLASSES)]))
    print("wrote data/" + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
