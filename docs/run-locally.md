# Running Seeing Why on your own computer

The notebooks run in Colab with no install (links in the [README](../README.md)). This page is for running them, and
the paper's scripts, from a clone.

## What you need

- Python 3.10 or newer (the paper used 3.12) and git.
- About 200 MB of disk for the clone and a few hundred MB for the packages.
- A CPU is enough for the demo and for every analysis script below. A GPU (CUDA) only speeds up training.

## Install

```bash
git clone https://github.com/stein-research/seeing-why
cd seeing-why
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt jupyterlab
```

`requirements.txt` lists what the demo needs first, then what training and the IBM scripts add. For the demo alone,
the first block and `jupyterlab` are enough.

## The notebooks

```bash
jupyter lab demo/demo.ipynb        # pick a picture, the deletion test, your own photo, the chip maps
jupyter lab demo/train.ipynb       # train one of the paper's models
```

Run the cells in order. The setup cell finds the clone, so nothing is downloaded. VS Code's notebook editor works too
(it needs the `ipywidgets` package, which `requirements.txt` installs). If the controls do not show, restart the
kernel and run the setup cell again.

## In plain Python

[`examples/quickstart.py`](../examples/quickstart.py) does the following and a little more (one evidence map saved as a
picture, and the deletion test by every ranking); run it from anywhere in the clone:

```bash
python examples/quickstart.py
```

The core of it, from the repository's top folder:

```python
import sys; sys.path.insert(0, "demo")
import seeing_why as sw

x, y = sw.test_pictures()                      # the 300 Caltech-10 test pictures, 32 x 32 colour
run = sw.load("phase plates (row D)")          # or "qubit circuit (row D')", or a results/<run>.json path
p = sw.classify(run, x)                        # BHR class probabilities
print(f"test accuracy {100 * (p.argmax(1) == y).mean():.1f} %")      # 73.0 %
e = sw.evidence(run, x[:1], [int(y[0])])[0]     # 32 x 32 evidence map for the true class
q = sw.delete(run, x[:1], e[None], 0.05)[0]     # the answer after deleting the top 5 % of pixels
```

## Train a model

The command the training notebook runs, here for the phase plates of row D (50 epochs: a few minutes on a GPU):

```bash
python src/pilot_fmnist.py --mode bhr --pt_ratio 0.0556 --rot rzryrx --order morton --n_out 7 --sim fast \
  --ent brickwall --dataset caltech10fct --n_qubits 12 --n_train 2644 --balanced --aug_crop 0.8 \
  --epochs 50 --seed 3 --device cuda --optical 4 --center_image --mean_plane 0.5 --fourier_mask \
  --fourier_linear --fourier_phase
```

Use `--device cpu` without a GPU. The other rows' options are in `RECIPES` in [`demo/ui.py`](../demo/ui.py). The run
is written to `results/` as a JSON file holding the weights; `sw.load("results/<that file>.json")` opens it. Seeds
0-2 are the paper's runs, so pick 3 or more for your own.

## The paper's analyses

Each script prints its numbers and writes them to `results/`; the outputs already there are the paper's.

| script | what it gives (paper section) | rough time on a CPU |
|---|---|---|
| `python src/map_nulls.py` | baselines for the chip's 2 x 2 maps (3.7) | seconds |
| `python src/map_errors.py` | the chip maps' error, slope and sum rule against the shot-noise floor (3.7) | seconds |
| `python src/map_shots_all.py` | the map measurement simulated on all 300 test pictures (3.7) | a few minutes |
| `python src/deletion_baselines.py` | the deletion test, every ranking (3.4) | tens of minutes |
| `python src/deletion_predicted.py` | the deletion test by the model's own answer (3.4) | tens of minutes |
| `python src/classical_refs.py` | logistic regression and a small CNN on the same pictures (3.2) | minutes |
| `python src/quad_baseline.py 0 1 2` | the unconstrained quadratic model, row Q (3.2) | long; use a GPU |
| `python src/robustness.py results/<row E run>.json` | the bench tolerances (3.9) | an hour or more; a GPU helps |
| `python src/optics_check_full.py results/<optical run>.json` | the same model rebuilt in LightPipes (3.8); `pip install LightPipes` first | minutes |
| `python src/check_fastsim.py` | the simulator against PennyLane | minutes |
| `python src/hw_mcnemar.py` | paired tests of the hardware accuracy steps (3.7) | seconds |
| `python src/model_randomisation.py` | the maps of models with their trained values permuted (3.5) | minutes |
| `bash src/readout_sweep.sh`, then `python src/readout_sweep_report.py` | BHR at five values of phi and a softmax at a fixed slope (3.2); the runs are already in `results/` | the sweep: about an hour on a GPU; the report: seconds |
| `python src/calibration.py` | the readouts' calibration (ECE), with temperature scaling of BHR (3.2) | a minute |
| `python src/means_share.py` | the means plane's share of P - b, which pixel maps leave out (2.5) | a minute |
| `python src/kernel_support.py` | how far the qubit model's Fourier-plane kernel reaches (3.7) | seconds |
| `bash src/mean_plane_sweep.sh` | row D at other values of the means plane's energy share f (2.1); the runs are already in `results/` | about ten minutes on a GPU |
| `python src/roar.py prepare`, `train`, then `report` | retraining on deleted pictures (ROAR), and what the deleted pixels' pattern alone gives away (3.4); the runs are already in `results/` | prepare: minutes; train: about 40 minutes on a GPU; report: minutes |
| `python src/bench_maps.py results/<row E run>.json --grid 32 --pictures 10,135,239 --drift 0.001,0.01 --leak 0.01,0.03 --xtalk 0.02,0.05` | per-pixel maps measured on the simulated lens bench, and bench flaws (3.8); `pip install LightPipes` first | about 20 minutes a picture |

The row E run is
`results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d0rzmo_cimp0.5_crop0.8_bal_opt4z32r2os2_fmLp_o7_brickwall_i0.1_s0.json`.

## Data

Caltech-10 as the paper used it is in the repository (`data/caltech10fct.npz`, 7.7 MB), so the notebooks, training
and every script above need no download. The original Caltech-101 archive (about 130 MB, CC BY 4.0) is needed only
for two optional things, and one call fetches it into `data/caltech101/` with its MD5 checked:

```python
import sys; sys.path.insert(0, "demo")
import seeing_why as sw
sw.prepare_caltech10()        # downloads Caltech-101, rebuilds Caltech-10 and compares it with the shipped set
```

The hand-drawn object outlines also come with the archive. To measure where the maps sit against them (3.5), take
them out once, then run `src/attention_in_box.py`:

```python
import zipfile
zipfile.ZipFile("data/caltech101/caltech-101.zip").extract("caltech-101/Annotations.tar", "data/caltech101/x")
```

## The IBM scripts

`src/ibm_*.py` sent the paper's jobs. Running them again needs your own IBM Quantum account, passed as a keys file
(`--keys`); no account details are in this repository. Every job the paper used is recorded in
[`ibm_jobs/`](../ibm_jobs/), with its circuits, calibration and raw counts, so the hardware numbers can be checked
without an account: `src/map_errors.py`, `src/map_nulls.py`, `src/ibm_heatmaps.py report` and
`src/ibm_maps_random.py report` (the maps of 30 pictures drawn at random, 3.7) read those records.
