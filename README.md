# Seeing Why: Exact Evidence Maps for Quantum and Optical Image Classifiers

Interactive companion to the paper by Teerawat Tongloy (KMITL) and Rangsan Jomtarak (Suan Dusit University).

| | |
|---|---|
| Try it: pick a picture and see why the model decides (CPU, no install) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stein-research/seeing-why/blob/main/demo/demo.ipynb) |
| Train a model in the simulator (GPU) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/stein-research/seeing-why/blob/main/demo/train.ipynb) |
| Paper | arXiv link to follow |

![The two classifiers, step by step: (a) all-optical, (b) qubit chip](docs/overview.png)

*(a) All-optical: the picture on a modulator (three contrast planes and a plane of the colour means), a 4f system
with a learned mask in its Fourier plane, four phase plates and a camera of 128 regions, read with the Born headroom
readout (BHR); the exact evidence for "ketch" lies on the dark hull. (b) Qubit chip: the same steps with a quantum
Fourier transform, a brickwall circuit and 128 measured bit strings.*

The classifiers have no trained classical network before or after the quantum or optical part: a 32 x 32 colour
picture is amplitude-encoded, filtered by a learned mask in the Fourier plane, and passed through four phase plates
(light) or a brickwall circuit on twelve qubits; the measured probabilities are read with the Born headroom readout.
Because the model is linear in the input amplitudes up to the measurement, the evidence of every pixel has a closed
form and adds up to each class's probability above its share, and it can be measured on the hardware itself.

## Contents

| | |
|---|---|
| `demo/` | the two notebooks and their helpers (`seeing_why.py`, `ui.py`) |
| `src/` | the simulator, the optics, the readout, the evidence maps, the trainer, and the scripts that ran the IBM jobs |
| `results/` | the four trained models of the demo (weights inside each JSON) and the IBM test-set and map results |
| `data/` | Caltech-10 (32 x 32, from Caltech-101, CC BY 4.0); see `data/README.md` |
| `ibm_jobs/` | the record of every job run on IBM's ibm_kingston: job id, times, calibration, qubits, circuits, raw counts (account details removed) |

## Use of AI tools

An AI assistant (Claude, Anthropic) was used throughout this work: to write and run simulation and analysis code, to
prepare and check the hardware jobs, to draw the figures, and to draft and revise the text. All research directions,
decisions and claims are the authors'.

## Licence and citation

Code: MIT (`LICENSE`). Caltech-10 is derived from Caltech-101 (CC BY 4.0, see `data/README.md`). Please cite the
paper (`CITATION.cff`).
