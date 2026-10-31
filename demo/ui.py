"""The interactive demo (ipywidgets). The drawing functions take plain arguments and return a matplotlib figure, so
they also work without widgets; demo() wires them to controls.

    import ui; ui.demo()
"""
import sys

import matplotlib.pyplot as plt
import numpy as np

import seeing_why as sw

INK, MUTED, ACCENT, LINE = "#2b2f36", "#6b7280", "#d9480f", "#c9ced6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LINE})


def _bars(ax, p, title, mark=None, colour=ACCENT):
    order = np.argsort(p)[::-1][:5][::-1]
    ax.barh(range(5), p[order], color=[colour if i == order[-1] else "#adb5bd" for i in order], height=0.62)
    ax.set_yticks(range(5))
    ax.set_yticklabels([sw.NAMES[i] + (" *" if i == mark else "") for i in order])
    ax.set_xlim(0, 1)
    ax.set_title(title, loc="left", fontsize=9, color=INK)
    ax.tick_params(axis="y", length=0)


def explain(run, x, true=None, cls=None, frac=0.0, side_by_side=False):
    """One picture: the picture, the evidence for class cls (default: the predicted class), and the class
    probabilities, before and after deleting the fraction frac of pixels with the highest evidence."""
    p = sw.classify(run, x[None])[0]
    cls = int(np.argmax(p)) if cls is None else int(cls)
    e = sw.evidence(run, x[None], [cls])[0]
    fig, ax = plt.subplots(1, 4 if frac else 3, figsize=(13 if frac else 10, 3.2),
                           gridspec_kw=dict(width_ratios=[1, 1, 1.1, 1.1] if frac else [1, 1, 1.1]))
    ax[0].imshow(x)
    ax[0].set_title("picture" + (f" (true: {sw.NAMES[true]})" if true is not None else ""), loc="left")
    if side_by_side:
        v = max(np.percentile(np.abs(e), 98), 1e-12)
        cmap = sw.overlay(plt.figure().add_subplot(), x, e)
        plt.close()
        ax[1].imshow(e, cmap=cmap, vmin=-v, vmax=v)
    else:
        sw.overlay(ax[1], x, e)
    ax[1].set_title(f"evidence for {sw.NAMES[cls]}\nred for, blue against", loc="left")
    for a in ax[:2]:
        a.set_xticks([])
        a.set_yticks([])
    _bars(ax[2], p, "class probabilities (BHR)", true)
    if frac:
        q = sw.delete(run, x[None], e[None], frac)[0]
        _bars(ax[3], q, f"after deleting the top {100 * frac:.0f} %\nof pixels for {sw.NAMES[cls]}", true, "#1f78d1")
    fig.tight_layout()
    return fig


def compare(x, cls=None):
    """The same picture through the four models of the paper: plain amplitudes against the mean plane, plates and
    qubits. With the mean plane the evidence moves onto the object, dark parts included."""
    fig, ax = plt.subplots(1, 5, figsize=(14, 3.2))
    ax[0].imshow(x)
    ax[0].set_title("picture", loc="left")
    short = ["plates, mean plane (D)", "plates, plain (A)", "qubits, mean plane (D')", "qubits, plain (A')"]
    for a, name, s in zip(ax[1:], sw.MODELS, short):
        run = sw.load(name)
        p = sw.classify(run, x[None])[0]
        c = int(np.argmax(p)) if cls is None else int(cls)
        sw.overlay(a, x, sw.evidence(run, x[None], [c])[0])
        a.set_title(f"{s}\n{sw.NAMES[c]} {p[c]:.2f}", loc="left", fontsize=8)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    return fig


def strip(x, chosen):
    """Thumbnails of the class's test pictures, the chosen one framed."""
    fig, ax = plt.subplots(1, len(x), figsize=(min(14, 0.47 * len(x)), 0.6))
    for i, a in enumerate(np.atleast_1d(ax)):
        a.imshow(x[i])
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(True)
            s.set_color(ACCENT if i == chosen else "white")
            s.set_linewidth(2.5 if i == chosen else 0.5)
    fig.subplots_adjust(0, 0, 1, 1, 0.05, 0)
    return fig


def chip(k):
    """Picture k of the ten measured on ibm_kingston: its exact 2 x 2 evidence and the chip's."""
    m = sw.chip_maps()
    x, y = sw.test_pictures()
    i = m["pictures"][k]
    chips = [key for key in m if key.startswith("chip")]
    ex = np.array(m["exact"][k]).reshape(2, 2)
    ch = np.array(m[chips[-1]][k]).reshape(2, 2)
    v = max(np.abs(ex).max(), np.abs(ch).max())
    cmap = sw.overlay(plt.figure().add_subplot(), x[i], np.zeros((32, 32)))
    plt.close()
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(x[i])
    ax[0].set_title(f"test picture {i} ({sw.NAMES[y[i]]})", loc="left")
    for a, mm, t in ((ax[1], ex, "exact (simulation)"), (ax[2], ch, "measured on ibm_kingston")):
        a.imshow(mm, cmap=cmap, vmin=-v, vmax=v)
        for (r, c), val in np.ndenumerate(mm):
            a.text(c, r, f"{1e3 * val:+.0f}", ha="center", va="center", fontsize=11)
        a.set_title(t + "\n(units of 1e-3)", loc="left")
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    r = np.corrcoef(np.array(m["exact"]).ravel(), np.array(m[chips[-1]]).ravel())[0, 1]
    fig.suptitle(f"all ten pictures: correlation of the 40 blocks r = {r:.2f}", x=0.01, ha="left", fontsize=9,
                 color=MUTED)
    fig.tight_layout()
    return fig


def demo(model=None):
    """The controls: model, class and picture (or your own photo), the class to explain, deletion, layout. model: a
    path to a trainer result (from the training notebook) to add to the list."""
    import ipywidgets as W
    from IPython.display import display
    x_all, y_all = sw.test_pictures()
    state = {"photo": None}
    options = list(sw.MODELS) + ([str(model)] if model else [])
    model = W.Dropdown(options=options, value=options[-1] if model else options[0], description="model", style={"description_width": "70px"},
                       layout=W.Layout(width="360px"))
    klass = W.Dropdown(options=[(n, i) for i, n in enumerate(sw.NAMES)], value=7, description="class",
                       style={"description_width": "70px"})
    pic = W.IntSlider(min=1, max=30, value=1, description="picture", style={"description_width": "70px"})
    rand = W.Button(description="random", layout=W.Layout(width="90px"))
    upload = W.FileUpload(accept="image/*", multiple=False, description="your photo")
    why = W.Dropdown(options=[("predicted class", -1)] + [(n, i) for i, n in enumerate(sw.NAMES)],
                     description="explain", style={"description_width": "70px"})
    frac = W.SelectionSlider(options=[("none", 0.0), ("2 %", 0.02), ("5 %", 0.05), ("10 %", 0.1), ("20 %", 0.2)],
                             description="delete top", style={"description_width": "70px"})
    layout = W.ToggleButtons(options=["over the picture", "side by side"])
    thumbs, out = W.Output(), W.Output()

    def picture():
        if state["photo"] is not None:
            return state["photo"], None
        idx = np.flatnonzero(y_all == klass.value)
        return x_all[idx[pic.value - 1]], int(klass.value)

    def draw(*_):
        x, true = picture()
        thumbs.clear_output(wait=True)
        out.clear_output(wait=True)
        with thumbs:
            if state["photo"] is None:
                f = strip(x_all[y_all == klass.value], pic.value - 1)
                plt.show(f)
        with out:
            f = explain(sw.load(model.value), x, true, None if why.value < 0 else why.value, frac.value,
                        layout.value == "side by side")
            plt.show(f)

    def on_rand(_):
        state["photo"] = None
        klass.value = int(np.random.randint(10))
        pic.value = int(np.random.randint(1, 31))

    def on_upload(change):
        files = upload.value
        if files:
            f = files[0] if isinstance(files, (list, tuple)) else next(iter(files.values()))
            state["photo"] = sw.from_photo(bytes(f["content"]))
            draw()

    def back_to_test(*_):
        state["photo"] = None

    for w in (model, why, frac, layout):
        w.observe(draw, "value")
    for w in (klass, pic):
        w.observe(back_to_test, "value")
        w.observe(draw, "value")
    rand.on_click(on_rand)
    upload.observe(on_upload, "value")
    display(W.VBox([W.HBox([model, why]), W.HBox([klass, pic, rand, upload]), thumbs, W.HBox([frac, layout]), out]))
    draw()


def demo_compare():
    import ipywidgets as W
    from IPython.display import display
    x_all, y_all = sw.test_pictures()
    klass = W.Dropdown(options=[(n, i) for i, n in enumerate(sw.NAMES)], value=7, description="class")
    pic = W.IntSlider(min=1, max=30, value=1, description="picture")
    out = W.Output()

    def draw(*_):
        out.clear_output(wait=True)
        with out:
            plt.show(compare(x_all[np.flatnonzero(y_all == klass.value)[pic.value - 1]]))
    klass.observe(draw, "value")
    pic.observe(draw, "value")
    display(W.VBox([W.HBox([klass, pic]), out]))
    draw()


def demo_chip():
    import ipywidgets as W
    from IPython.display import display
    m = sw.chip_maps()
    y = sw.test_pictures()[1]
    k = W.Dropdown(options=[(sw.NAMES[y[i]], j) for j, i in enumerate(m["pictures"])], description="picture")
    out = W.Output()

    def draw(*_):
        out.clear_output(wait=True)
        with out:
            plt.show(chip(k.value))
    k.observe(draw, "value")
    display(W.VBox([k, out]))
    draw()


# ---------------------------------------------------------------------------------------------------------- training
COMMON = "--mode bhr --pt_ratio 0.0556 --rot rzryrx --order morton --n_out 7 --sim fast --ent brickwall"
CAL = "--dataset caltech10fct --n_qubits 12 --n_train 2644 --balanced --aug_crop 0.8"
RECIPES = {                                                    # the paper's rows (run_ablation_v2.sh)
    "phase plates, mean plane (row D)": "--optical 4 --center_image --mean_plane 0.5 --fourier_mask --fourier_linear --fourier_phase",
    "phase plates, plain amplitudes (row A)": "--optical 4",
    "qubit circuit, mean plane (row D')": "--depth 11 --fourier_mask --fourier_phase --center_image --mean_plane 0.5",
    "qubit circuit, plain amplitudes (row A')": "--depth 11",
}


def train_command(recipe, epochs, seed, device):
    return (f"{sys.executable} -u -W ignore src/pilot_fmnist.py {COMMON} {CAL} --epochs {epochs} --seed {seed} "
            f"--device {device} {RECIPES[recipe]}").split()


def demo_train():
    """Train one of the paper's models in the simulator, watch the validation accuracy, then explain its answers."""
    import re
    import subprocess
    import torch
    import ipywidgets as W
    from IPython.display import display
    if not (sw.ROOT / "data" / "caltech10fct.npz").exists():
        print("Prepare the data first (the cell above).")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    recipe = W.Dropdown(options=list(RECIPES), description="model", layout=W.Layout(width="420px"))
    epochs = W.IntSlider(min=1, max=50, value=10, description="epochs")
    seed = W.BoundedIntText(min=3, max=999, value=3, description="seed",
                            tooltip="0-2 are the paper's runs, kept in results/")
    go = W.Button(description="train", button_style="warning")
    log, chart, after = W.Output(), W.Output(), W.Output()

    def run(_):
        go.disabled = True
        log.clear_output()
        chart.clear_output()
        after.clear_output()
        val = []
        cmd = train_command(recipe.value, epochs.value, seed.value, dev)
        proc = subprocess.Popen(cmd, cwd=sw.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        before = set((sw.ROOT / "results").glob("pilot_*.json"))
        for line in proc.stdout:
            m = re.match(r"epoch (\d+): .* val ([0-9.]+) \((\d+) s\)", line)
            if m:
                val.append(float(m.group(2)))
                with log:
                    print(line.rstrip())
                chart.clear_output(wait=True)
                with chart:
                    fig, ax = plt.subplots(figsize=(5, 2.2))
                    ax.plot(range(1, len(val) + 1), [100 * v for v in val], "o-", color=ACCENT, ms=3)
                    ax.set_xlabel("epoch")
                    ax.set_ylabel("validation accuracy (%)")
                    ax.set_xlim(0.5, epochs.value + 0.5)
                    plt.show(fig)
        proc.wait()
        go.disabled = False
        new = sorted(set((sw.ROOT / "results").glob("pilot_*.json")) - before, key=lambda f: f.stat().st_mtime)
        if proc.returncode or not new:
            with log:
                print(f"training stopped (exit {proc.returncode})")
            return
        res = sw.load(new[-1])
        x, y = sw.test_pictures()
        acc = (sw.classify(res, x).argmax(1) == y).mean()
        with after:
            print(f"test accuracy {100 * acc:.1f} % ({new[-1].name})")
            i = int(np.flatnonzero(y == 7)[0])
            plt.show(explain(res, x[i], int(y[i])))
            print("Explore this model in the demo: ui.demo(model=" + repr(str(new[-1])) + ")")

    go.on_click(run)
    print(f"device: {dev}" + ("" if dev == "cuda" else " (slow: in Colab choose Runtime > Change runtime type > GPU)"))
    display(W.VBox([recipe, W.HBox([epochs, seed, go]), chart, log, after]))
