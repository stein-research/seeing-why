# The 300 Caltech-10 test pictures on ibm_kingston

Model D' (qubit circuit with the Fourier-plane mask and the mean plane, depth 11, seed 0:
`../pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json`).
Each picture's masked state is loaded by a fitted RY-CNOT circuit of depth 8, followed by the trained brickwall;
4,000 shots per circuit, thirty pictures per job (`../../ibm_jobs/`).

**Readable tables** (written from the files below):

| file | one row per | columns |
|---|---|---|
| `test_set.csv` | test picture (300) | true class; predicted class and the true class's probability in three steps: exact simulation, the fitted loading circuit without noise, ibm_kingston; the loading's state fidelity; the ledger folder of the chip job |
| `chip_class_probabilities.csv` | test picture (300) | the chip's Born-headroom probability for each of the ten classes |
| `maps_2x2.csv` | block of a 2 x 2 evidence map (10 pictures x 4 blocks) | evidence for the true class: exact, fitted preparations, noisy simulations, and the two chip runs (`chip_ibm_kingston_r1` is the one in the paper: every term measured in one job) |

Accuracy: exact 56.3 %, fitted loading 41.3 %, chip 37.7 % (chance 10 %). Maps: chip against exact r = 0.77.

**Files the scripts read** (`src/ibm_testset.py`, `src/ibm_heatmaps.py`; numpy `.npz` opens with `numpy.load`):

| file | content |
|---|---|
| `fit.json` | summary of the loading fits (depth, steps, median fidelity, exact and loading accuracy) |
| `preps.npz` | `ang` (300, 9, 12) fitted RY angles per picture, `fid` their state fidelity, `y` true classes, `q_exact` / `q_prep` class probabilities (300, 10), `wc` the circuit's angles |
| `chip_ibm_kingston_r0.json` | per picture: true class (`label`), prediction (`pred`), class probabilities (`q`), the 128 outcome probabilities (`outcomes`), backend, ledger folder |
| `noisy_FakeKingston*.npz` | a noisy simulation of the first 30 pictures (`sel`, `q`, `outcomes`) |
| `heat_g2.npz` | the 2 x 2 map preparations: pictures (`sel`), fitted angles, fidelities, norms, exact and prepared outcome probabilities |
| `heat_g2_chip_ibm_kingston*.npz`, `heat_g2_noisy_*.npz` | measured outcome probabilities of every map circuit (`P`, `P_full`), shots, ledger folder |
| `heat_g2_maps.json` | the 2 x 2 maps of every source (the same numbers as `maps_2x2.csv`) |
| `heat_g4.npz` | 4 x 4 preparations of three pictures (simulation only; 4 x 4 needs more shots than were run) |
