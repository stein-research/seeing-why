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


def photo(source, zoom=1.0, across=0.0, down=0.0):
    """Any photograph -> a 32 x 32 colour picture made as Caltech-10's were: turned upright (EXIF), on white where it
    is transparent, 4 % trimmed off every side, a square crop, area-averaged down to 32 x 32. zoom (0.8-1): the crop's
    side against the shorter side, as the random crops of training; across / down (-1..1): where the crop sits when
    zoomed in or when the photo is not square. source: a path, bytes or a PIL image."""
    import io
    from PIL import ImageOps
    im = source if isinstance(source, Image.Image) else Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray, memoryview)) else source)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if min(im.size) > 1024:                                      # a large photo: reduce first, the same averaging
        im = im.reduce(min(im.size) // 512)
    w, h = im.size
    dx, dy = round(w * 0.04), round(h * 0.04)
    im = im.crop((dx, dy, w - dx, h - dy))
    w, h = im.size
    k = max(1, round(min(w, h) * min(1.0, max(0.5, zoom))))
    x0 = round((w - k) / 2 * (1 + max(-1.0, min(1.0, across))))
    y0 = round((h - k) / 2 * (1 + max(-1.0, min(1.0, down))))
    im = im.crop((x0, y0, x0 + k, y0 + k))
    return np.asarray(im.resize((32, 32), Image.BOX), dtype=np.uint8)


def block(x, row, col, size):
    """The picture with a size x size square at (row, col) set to the picture's own mean colour: zero contrast there,
    the colour means unchanged - the deletion of the paper, for a square chosen by hand."""
    y = x.copy()
    y[row:row + size, col:col + size] = np.round(x.reshape(-1, 3).mean(0)).astype(x.dtype)
    return y


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


def planes(run, x):
    """One picture through the model, step by step: a dict of the fields and numbers the plane-by-plane view draws.
    Phase plates: modulator field, each plane's spectrum, the learned mask, the filtered field, the plates, the field
    after each plate with the share of light still in the aperture, the camera, the light per region. Qubits: the
    amplitudes, spectra, mask, filtered amplitudes and the 128 outcome probabilities. Both: class probabilities."""
    s, perm = _setup(run)
    n = pf.N_QUBITS
    pix = torch.as_tensor(perm)
    n_planes = 2 ** n // (s * s)
    v = encode(run, x[None])
    a = torch.tensor(v / np.linalg.norm(v), dtype=torch.complex128)
    out = {"x": x, "optical": bool(run.get("optical")), "planes": n_planes}
    out["field"] = optnet.tile(a, pix, s, n_planes)[0].numpy()
    linear = bool(run.get("fourier_linear"))
    if run.get("fourier_mask"):
        w = torch.tensor(run["weights"], dtype=torch.float64)
        depth = 0 if run.get("optical") else run["depth"]           # the mask rows follow the circuit's rows
        _, kw = fastsim.split_masks(w, depth, 0, -1, (pix, s, n_planes, run.get("fourier_phase", False), linear))
        D = kw["fmask"][1]
        out["mask"] = np.fft.fftshift(D.numpy()[0], axes=(-2, -1))
        st = fastsim.fourier_filter(a, D, pix, linear)
    else:
        st = a
    out["passed"] = float((st.abs() ** 2).sum())                    # share of the light through the mask
    out["filtered"] = optnet.tile(st, pix, s, n_planes)[0].numpy()
    spec = []                                                        # each plane's spectrum, as the first lens sees it
    for p_ in range(n_planes):
        ch = out["field"][(p_ // 2) * s:(p_ // 2 + 1) * s, (p_ % 2) * s:(p_ % 2 + 1) * s]
        z = np.zeros((2 * s, 2 * s), complex) if linear else np.zeros((s, s), complex)
        z[:s, :s] = ch
        spec.append(np.fft.fftshift(np.log10(np.abs(np.fft.fft2(z)) + 1e-4)))
    out["spectra"] = spec
    no = run["n_out"]
    if run.get("optical"):
        N, r, L, os_ = 2 * s, run["opt_res"], run["optical"], run.get("opt_os", 1)
        w = torch.tensor(run["weights"], dtype=torch.float64)
        fr = fastsim.fourier_rows(n, w.shape[2], s, n_planes, run.get("fourier_phase", False)) if run.get("fourier_mask") else 0
        phi = w[1 + fr:].reshape(-1)[:L * (N // r) ** 2].reshape(L, N // r, N // r)
        out["plates"] = phi.numpy()
        P = optnet.plate_phase(phi, r * os_).to(torch.complex128)
        g = optnet.tile(st, pix, s, n_planes)
        g = g.repeat_interleave(os_, -2).repeat_interleave(os_, -1) if os_ > 1 else g
        e0 = float((g.abs() ** 2).sum())
        after, kept = [], []
        for l in range(L):
            g = optnet.propagate(g, run["opt_z"] * os_ * os_, run["opt_pad"]) * P[l]
            after.append(g[0].numpy())
            kept.append(float((g.abs() ** 2).sum()) / e0)
        cam = optnet.propagate(g, run["opt_z"] * os_ * os_, run["opt_pad"])
        out["after"], out["kept"] = after, kept
        out["camera"] = optnet.camera_pixels(cam, os_)[0].numpy()
        out["camera_kept"] = float(out["camera"].sum()) / e0
        P_out = optnet.outcome_probs(cam, optnet.detector_index(N), os=os_)[0].numpy()
    else:
        w = torch.tensor(run["weights"], dtype=torch.float64)
        fourier = (pix, s, n_planes, run.get("fourier_phase", False), linear) if run.get("fourier_mask") else None
        wc, kw = fastsim.split_masks(w, run["depth"], 0, -1, fourier)
        P_out = fastsim.probs(fastsim.run(a, wc, run["depth"], torch.complex128, **kw), no)[0].numpy()
    out["outcomes"] = P_out / P_out.sum()
    groups = class_groups(run["mode"], 2 ** no, len(NAMES))
    out["group_of"] = np.zeros(2 ** no, int)
    for gi, members in enumerate(groups):
        out["group_of"][list(members)] = gi
    out["p"] = classify(run, x[None])[0]
    return out
