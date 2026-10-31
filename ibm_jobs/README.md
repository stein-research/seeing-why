# IBM hardware jobs

`index.md` lists every job: when, on which backend, the job id, circuits x shots, quantum seconds, and what it was for.
Each folder (named by a key over the circuits, backend and shots) holds one job:

| file | content |
|---|---|
| `record.json` | job id, backend, shots, times, quantum seconds, transpile settings, dynamical decoupling, software versions, the input files and their sha256, and per circuit: the picture, its qubits, depth and gate counts |
| `counts.json` | the raw counts: one dictionary per circuit, bit string (qubit 0 rightmost) -> number of shots |
| `circuits.txt` | the circuits as built, one gate per line (readable) |
| `circuits.qpy`, `transpiled.qpy` | the same circuits as built and as sent to the chip, in Qiskit's format (`qiskit.qpy.load`) |
| `properties.json` | the backend's calibration when the job was sent (qubit T1/T2, gate and readout errors) |

Account details are not included. The counts here are the record; reading a job back from IBM needs the account
that ran it.
