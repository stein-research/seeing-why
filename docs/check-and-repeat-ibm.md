# Checking and repeating the IBM jobs

Every hardware number in the paper comes from jobs run on IBM's ibm_kingston, and every job is recorded in
[`ibm_jobs/`](../ibm_jobs/): the circuits as built and as sent, the backend's calibration at the time and the raw
counts (what each file holds: [`ibm_jobs/README.md`](../ibm_jobs/README.md); the list of jobs:
[`ibm_jobs/index.md`](../ibm_jobs/index.md)). This page shows three things, from the least to the most work:

1. check the paper's hardware numbers from those records, with no IBM account;
2. run the same circuits again on IBM hardware with your own account;
3. send the recorded circuits with plain Qiskit, without our scripts.

Install first as in [`run-locally.md`](run-locally.md) (the full `requirements.txt`). Every command runs from the root
of the clone. `RUN` is the qubit model the hardware jobs used:

```bash
RUN=results/pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5_crop0.8_bal_fmp_o7_brickwall_i0.1_s0.json
```

## 1. Check the numbers, no account

These read only the files in the repository and send nothing. Each takes seconds on a CPU
(`ibm_maps_random.py report` about a minute).

| in the paper | command | prints |
|---|---|---|
| the test set on the chip (3.7): 56.3 % exact, 41.3 % with the fitted loading, 37.7 % on ibm_kingston | `python src/ibm_testset.py report $RUN` | the three accuracies, ROC-AUC, the chip's confusion matrix |
| maps of the 30 pictures drawn at random (3.7): 0.88 sign flip, 0.82 standard, 4x4 0.60 | `python src/ibm_maps_random.py report` | every estimator and step against the exact maps, with intervals and nulls |
| the earlier maps of ten chosen pictures: 0.63 and 0.77 (two runs) | `python src/ibm_heatmaps.py report $RUN --no_figure` | each step against the exact maps |
| their errors and nulls | `python src/map_errors.py`, `python src/map_nulls.py` | errors, intervals; permutation nulls |
| the shot floor over all 300 pictures (simulation) | `python src/map_shots_all.py` | maps at 4,000 / 16,000 / 64,000 shots |

The reports read the class probabilities kept beside the results (for the test set,
`results/ibm_testset_*/chip_ibm_kingston_r0.json`). To rebuild those from the raw counts themselves: move the file
aside and let the chip step fill it again from the ledger. It rebuilds every circuit from the fitted loadings, finds
each one in `ibm_jobs/` by its key and reads its counts; with every circuit found it logs in nowhere and sends
nothing (`--keys` is required but not read then):

```bash
D=results/ibm_testset_pilot_caltech10fct_e50_bhr_amp_pt0.0556_q12_d11rzmo_cimp0.5__L8
mv $D/chip_ibm_kingston_r0.json /tmp/chip_r0_kept.json
python src/ibm_testset.py chip $RUN --keys none --count 300 --dry_run     # "all already measured ... nothing to send"
python src/ibm_testset.py report $RUN                                        # 37.7 % again, from the counts
```

The rebuilt class probabilities equal the kept ones exactly (`git diff` shows only the `ledger` field, which then says
`lookup`). `git checkout -- results` puts the original back.

## 2. Repeat the jobs with your own account

### The keys file

The scripts take your IBM Quantum API key from a small text file that you keep outside the repository and never
commit: one account per line, `<name> <API key>` (the name is only a label, an e-mail address or anything without
spaces). The examples below call it `~/ibm_keys.txt` and use its first line, `--accounts 0`.

```bash
python src/ibm_jobs.py check --keys ~/ibm_keys.txt     # logs in (sends no job): usable or not, quota left
```

`check` writes `ibmq_accounts_status.json` next to the keys file, never into the repository, and keeps no key in it.
On IBM's free (Open) plan an account has 10 minutes of quantum time in any 28 days. On ibm_kingston one circuit at
4,000 shots used 1.1 to 1.6 quantum seconds in our jobs.

### Why running the same command again sends nothing

The ledger names each job by a key over its circuits, the backend and the shots. If that key is already in
`ibm_jobs/` with its counts, the scripts take the counts from there and send nothing, so that no job is ever paid for
twice by mistake. To measure the same circuits again on purpose, give a new repeat number: `--repeat 1` for the test
set (ours is repeat 0), `--repeat 2` for the ten-picture maps (ours are 0 and 1). A repeat is a new key, a new job,
and its results are kept in their own files beside ours.

### Dry run, then send

Always start with `--dry_run`: it logs in, checks your quota, transpiles the circuits for the backend's current
calibration and writes the job folder in `ibm_jobs/` (status "prepared"), but sends nothing. Then the same command
without `--dry_run` sends it. Ten test pictures, about 12 quantum seconds:

```bash
python src/ibm_testset.py chip $RUN --keys ~/ibm_keys.txt --accounts 0 --count 10 --chunk 10 --repeat 1 --dry_run
python src/ibm_testset.py chip $RUN --keys ~/ibm_keys.txt --accounts 0 --count 10 --chunk 10 --repeat 1
python src/ibm_testset.py report $RUN
```

The report then shows your repeat on its own pictures beside ours on the same pictures, for example
`chip_ibm_kingston_r1 on its 10 pictures: accuracy ... (exact there ..., chip_ibm_kingston_r0 there ...)`.
`--count 300 --chunk 30` repeats the whole test set (300 circuits, about 6 minutes of quantum time, in ten jobs).

The ten-picture maps again (90 circuits in one job, about 1.5 minutes):

```bash
python src/ibm_heatmaps.py chip $RUN --keys ~/ibm_keys.txt --accounts 0 --repeat 2 --dry_run
python src/ibm_heatmaps.py chip $RUN --keys ~/ibm_keys.txt --accounts 0 --repeat 2
python src/ibm_heatmaps.py report $RUN --no_figure
```

The scripts stop before sending when the account has too little quota left for the job. A job that was sent but whose
results were not collected (the script was closed while it waited) is fetched, not sent again, the next time the same
command runs.

### Another machine

`python src/ibm_testset.py backends $RUN --keys ~/ibm_keys.txt` lists the machines your account can use (it only
reads). `--backend <name>` on the chip commands sends there instead; a different backend is a different key, so the
circuits are transpiled anew for it and nothing of ours is reused.

### What to expect

A repeat will not give our numbers to the digit. The chip's errors change from day to day: compare the
`properties.json` in your job folder (qubit T1/T2, gate and readout errors when it was sent) with ours, and the
qubits each circuit ran on (`record.json`). Our own two runs of the ten-picture maps correlated with the exact maps
at 0.63 and 0.77, which gives an idea of the run-to-run spread. To add your job to a copy of this repository, commit
its `ibm_jobs/<key>/` folder and the results file; `python src/ibm_jobs.py list` rewrites `ibm_jobs/index.md`. Your
folder records which of your keys file's lines sent the job, and a fingerprint of the key (the first 12 hex digits
of its SHA-256), never the key itself.

## 3. Plain Qiskit, without our scripts

Each job folder holds the circuits as built (`circuits.qpy`) and as sent to ibm_kingston (`transpiled.qpy`). The
sketch below runs a recorded job again with IBM's own Sampler; it uses the transpiled circuits, so it works only on
ibm_kingston (for another machine transpile `circuits.qpy` for it first):

```python
import json
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

job_dir = "ibm_jobs/112bcae203fb5551"                    # any folder of ibm_jobs/
with open(f"{job_dir}/transpiled.qpy", "rb") as f:
    circuits = qpy.load(f)
ours = json.load(open(f"{job_dir}/counts.json"))         # our counts, one dictionary per circuit

service = QiskitRuntimeService(channel="ibm_quantum_platform", token="<your API key>")
job = SamplerV2(mode=service.backend("ibm_kingston")).run(circuits, shots=4000)
yours = [r.join_data().get_counts() for r in job.result()]
```

`yours[i]` and `ours[i]` are then the counts of the same circuit, bit strings with qubit 0 rightmost. To turn counts
into class probabilities, use `outcome_probs` and `class_probs` in `src/ibm_demo.py`, as the scripts do.
This is written for qiskit-ibm-runtime 0.51, which the paper used; it warns that `SamplerV2` will be renamed in a
later release.
