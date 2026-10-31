"""Helpers for the notebooks: load a trained model, classify a picture with the Born headroom readout, and draw the
exact evidence of every pixel. Everything runs on a CPU in a few seconds per picture.

    import seeing_why as sw
    run = sw.load("phase plates (row D)")
    x, y = sw.test_pictures()
    p = sw.classify(run, x[:1])[0]             # class probabilities (BHR)
    e = sw.evidence(run, x[:1], [7])[0]        # 32 x 32 map: evidence for class 7 (ketch)
"""
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import attention_in_box as aib                                 # noqa: E402
import fastsim                                                 # noqa: E402
import optnet                                                  # noqa: E402
import pilot_fmnist as pf                                      # noqa: E402
import prepare_caltech101 as pc                                # noqa: E402
from bssg import Readout, class_groups                         # noqa: E402

aib.DEV = "cpu"
NAMES = ["airplane", "motorbike", "face", "watch", "leopard", "bonsai", "car side", "ketch", "chandelier", "hawksbill"]
_C = "pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_"
MODELS = {                                                     # Table 1 of the paper, seed 0
    "phase plates (row D)": _C + "d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2_fmLp_o7_brickwall_i0.1_s0.json",
    "phase plates, plain amplitudes (row A)": _C + "d0rzmo_crop0.8_bal_opt4z32r2_o7_brickwall_i0.1_s0.json",
    "qubit circuit (row D')": _C + "d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json",
    "qubit circuit, plain amplitudes (row A')": _C + "d11rzmo_crop0.8_bal_o7_brickwall_i0.1_s0.json",
}


def load(name_or_path):
    """A trained model: one of MODELS, or the path of a trainer result (results/<run>.json)."""
    p = ROOT / "results" / MODELS[name_or_path] if name_or_path in MODELS else pathlib.Path(name_or_path)
    return json.loads(p.read_text())


def test_pictures():
    """The 300 Caltech-10 test pictures (32 x 32 colour, uint8) and their classes."""
    d = np.load(ROOT / "data" / "caltech10fct.npz")
    return d["x_val"], d["y_val"]


def from_photo(path_or_bytes):
    """Any photo -> a 32 x 32 picture prepared as the test set was (4 % off every side, centre square, box filter)."""
    import io
    im = Image.open(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes)
    pc.TRIM = 0.04
    return pc.centre_crop(im.convert("RGB"))


def _setup(run):
    pf.DATASET, pf.ORDER, pf.N_QUBITS = run["dataset"], run.get("order", "morton"), len(run["weights"][0])
    s = pf.side()
    return s, pf.morton_perm() if pf.permuted() else np.arange(s * s)


def encode(run, x):
    """Pictures (B, 32, 32, 3) -> the amplitudes the model receives (B, 2^n), wire order, not yet normalised."""
    s, perm = _setup(run)
    v = x.astype(np.float64).transpose(0, 3, 1, 2).reshape(len(x), -1, s * s)[:, :, perm].reshape(len(x), -1)
    if run.get("center_image"):
        v = pf.centre_planes(v, run.get("dc_energy", 0.0), run.get("mean_plane", 0.0))
    return np.pad(v, ((0, 0), (0, 2 ** pf.N_QUBITS - v.shape[1])))


def classify_amplitudes(run, enc):
    """Encoded inputs -> (BHR class probabilities (B, 10), class masses P_g (B, 10))."""
    s, perm = _setup(run)
    no = run["n_out"]
    a = torch.tensor(enc / np.linalg.norm(enc, axis=1, keepdims=True), dtype=torch.complex64)
    with torch.no_grad():
        if run.get("optical"):
            cam, _, det, os_ = optnet.from_run(run, perm, "cpu", torch.complex64)
            P = optnet.outcome_probs(cam(a), det, os=os_).float()
        else:
            w = torch.tensor(run["weights"], dtype=torch.float32)
            fourier = (torch.tensor(perm), s, 2 ** pf.N_QUBITS // (s * s), run.get("fourier_phase", False),
                       run.get("fourier_linear", False)) if run.get("fourier_mask") else None
            wc, kw = fastsim.split_masks(w, run["depth"], 0, -1, fourier)
            P = fastsim.probs(fastsim.run(a, wc, run["depth"], **kw), no).float()
        P = P / P.sum(1, keepdim=True)                        # conditional on the light that arrives
        groups = class_groups(run["mode"], 2 ** no, len(NAMES))
        G = torch.tensor(np.stack([np.isin(np.arange(2 ** no), g) for g in groups]), dtype=torch.float32)
        q = Readout(run["mode"], 2 ** no, len(groups), pt_ratio=run["pt_ratio"])(P)
    return q.numpy(), (P @ G.T).numpy()


def classify(run, x):
    """Pictures (B, 32, 32, 3) -> BHR class probabilities (B, 10)."""
    return classify_amplitudes(run, encode(run, x))[0]


def evidence(run, x, classes):
    """Exact evidence per pixel (B, 32, 32) for the given class of each picture (colour planes summed; red for,
    blue against). It adds up to P_g - b_g less the means plane's part, which belongs to no pixel."""
    return aib.maps(run, x, np.asarray(classes), signed=True)[2]


def delete(run, x, ev, frac):
    """The deletion test: the fraction frac of pixels with the highest evidence get zero contrast in R, G, B (each
    becomes the picture's own mean colour), then the model is run again. Returns the new BHR probabilities."""
    s, perm = _setup(run)
    enc = encode(run, x)
    k = int(round(frac * s * s))
    if k and run.get("center_image"):
        rank = np.argsort(-ev.reshape(len(x), -1), 1)[:, :k]
        dead = np.zeros((len(x), s * s), bool)
        np.put_along_axis(dead, rank, True, 1)
        dead_w = dead[:, np.asarray(perm)]
        for c in range(3):
            enc[:, c * s * s:(c + 1) * s * s][dead_w] = 0.0
    elif k:                                                   # plain amplitudes: the pixel takes the mean colour
        x = x.copy()
        rank = np.argsort(-ev.reshape(len(x), -1), 1)[:, :k]
        for i in range(len(x)):
            r, c = np.unravel_index(rank[i], (s, s))
            x[i, r, c] = x[i].reshape(-1, 3).mean(0)
        enc = encode(run, x)
    return classify_amplitudes(run, enc)[0]


def overlay(ax, x, e, clip=98):
    """Evidence over the colour picture, opacity following |e| (as Figure 1 of the paper)."""
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "evidence", ["#08306b", "#1f78d1", "#9cc9f5", "#f7f7f7", "#f9a77e", "#e8431c", "#7f0d0d"])
    t = np.clip(e / max(np.percentile(np.abs(e), clip), 1e-12), -1, 1)
    rgba = cmap(t / 2 + 0.5)
    rgba[..., 3] = np.clip(1.5 * (np.abs(t) - 0.1), 0, 0.95)
    ax.imshow(x)
    ax.imshow(rgba)
    return cmap


def chip_maps():
    """The 2 x 2 evidence maps measured on ibm_kingston and the exact ones (ten pictures, one per class)."""
    f = next((ROOT / "results").glob("ibm_testset_*L8")) / "heat_g2_maps.json"
    m = json.loads(f.read_text())
    return m


CALTECH_URL = "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"   # CC BY 4.0


def prepare_caltech10(zip_path=None):
    """Optional check of the shipped data/caltech10fct.npz: download Caltech-101 from CaltechDATA (about 130 MB; MD5
    checked), unpack the picture folders, prepare Caltech-10 again as in the paper (prepare_caltech101.py --all --crop
    --trim 0.04) and compare every picture with the shipped set."""
    import hashlib
    import subprocess
    import tarfile
    import urllib.request
    import zipfile
    d = ROOT / "data" / "caltech101"
    d.mkdir(parents=True, exist_ok=True)
    z = pathlib.Path(zip_path) if zip_path else d / "caltech-101.zip"
    if not z.exists():
        print("downloading Caltech-101 (CaltechDATA, CC BY 4.0) ...")
        urllib.request.urlretrieve(CALTECH_URL, z)
    want = (d / "MD5SUMS").read_text().split()[0]
    h = hashlib.md5()
    with open(z, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != want:
        raise SystemExit(f"MD5 {h.hexdigest()} is not {want}: the download is damaged or changed")
    x = d / "x"
    if not (x / "101_ObjectCategories").exists():
        with zipfile.ZipFile(z) as zf:
            zf.extract("caltech-101/101_ObjectCategories.tar.gz", x)
        with tarfile.open(x / "caltech-101" / "101_ObjectCategories.tar.gz") as tf:
            tf.extractall(x, filter="data") if sys.version_info >= (3, 12) else tf.extractall(x)
    f = ROOT / "data" / "caltech10fct.npz"
    ship = {k: v for k, v in np.load(f).items()} if f.exists() else None
    subprocess.run([sys.executable, "-I", "src/prepare_caltech101.py", "--all", "--crop", "--trim", "0.04"], cwd=ROOT,
                   check=True, capture_output=True)
    new = np.load(f)
    same = ship is not None and all(np.array_equal(new[k], ship[k]) for k in ("x_train", "y_train", "x_val", "y_val"))
    print(f"Caltech-10: {len(new['x_train'])} training and {len(new['x_val'])} test pictures; "
          f"{'identical to' if same else 'DIFFERENT from'} the shipped set")
    return same
