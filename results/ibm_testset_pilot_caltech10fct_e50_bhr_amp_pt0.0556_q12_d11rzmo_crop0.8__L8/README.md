# Model A' on ibm_kingston: the same 300 test pictures, no mask

Model A' (the qubit circuit on plain amplitudes, no Fourier-plane mask, depth 11, seed 0:
`../pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_crop0.8_bal_o7_brickwall_i0.1_s0.json`), run as model D' was
(`../ibm_testset_pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5__L8/`): each picture loaded by a fitted
RY-CNOT circuit of depth 8, then the trained brickwall; 4,000 shots, thirty pictures per job (`../../ibm_jobs/`).

Accuracy: exact 51.3 %, fitted loading 54.7 %, chip 32.3 % (chance 10 %). Paired tests against D':
`src/hw_mcnemar.py`.

| file | content |
|---|---|
| `test_set.csv`, `chip_class_probabilities.csv` | as in D''s folder |
| `fit.json`, `preps.npz`, `chip_ibm_kingston_r0.json` | as in D''s folder |
