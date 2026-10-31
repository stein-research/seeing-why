#!/usr/bin/env python3
"""A ledger of IBM hardware jobs: every job sent is kept, so its results are never paid for twice and anyone who asks
can be shown what ran (owner 2026-10-31).

`sampler_counts(...)` is the one way the scripts reach hardware. Before sending it hashes the logical circuits with the
backend and the shots into a key, and looks in ibm_jobs/<key>/:
  - done:      the counts are read from the folder; nothing is sent, no quota is used;
  - submitted: the job was sent but its result never came back (the script died, the network dropped): the same job is
               fetched again by its id, not sent again;
  - none:      the job is sent, and the folder is written BEFORE waiting for the result.
`dry_run=True` writes the folder (circuits, transpiled circuits, what would be sent) and stops before sending.

Each folder holds:
  record.json      job id, backend, status, shots, times (created, finished), quantum seconds used (job.usage()),
                   IBM's own metrics (job.metrics(): timestamps, usage, qiskit version on their side), the qubits each
                   circuit landed on, the command line, the code version (git commit, uncommitted changes or not), the
                   versions of Python / Qiskit / qiskit-ibm-runtime, the inputs (file name + sha256) and the caller's
                   description of every circuit; the account as its line number in the keys file and a fingerprint
                   (the first 12 hex digits of the token's sha256: names the account, cannot give the token back)
  circuits.qpy     the logical circuits as built
  transpiled.qpy   the circuits as sent (after transpiling for the backend)
  counts.json      the raw counts of every circuit, bitstring -> count
  properties.json  the backend's calibration at sending time (T1, T2, gate and readout errors), when IBM gives it
ibm_jobs/index.md lists every job, one line each (rewritten after every change).

Run:  python src/ibm_jobs.py list                       every job
      python src/ibm_jobs.py check --keys <file>        log in with every account, mark the unusable
      python src/ibm_jobs.py show <key or job id>       one job's record and counts (local, no network)
      python src/ibm_jobs.py fetch <job id> --keys <file> [--account k]
                                                                    (without --account every line is tried)
IBM shows a job only to the account that ran it, so reading one back needs that account's token: the keys file stays
the only place tokens live; the fingerprint finds the right line even after lines are added or reordered.
                                                                    bring a job run before the ledger in (reads, sends nothing)
"""
import datetime
import hashlib
import io
import json
import os
import pathlib
import platform
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent                # the repository root
LEDGER = HERE / "ibm_jobs"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _qpy_bytes(circs):
    from qiskit import qpy
    buf = io.BytesIO()
    qpy.dump(list(circs), buf)
    return buf.getvalue()


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _code_version():
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=HERE,
                                    capture_output=True, text=True).stdout.strip())
        return {"git_commit": commit, "uncommitted_changes": dirty}
    except OSError:
        return {}


def _versions():
    import qiskit
    import qiskit_ibm_runtime
    return {"python": platform.python_version(), "qiskit": qiskit.__version__,
            "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__}


def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def accounts(keys, usable_only=True):
    """[(line, email, token)] of a keys file: either a Python/JSON dict {"<email>": "<token>", ...} (optionally after
    'NAME =', '#' comments allowed) or one '<email> <token>' per line. line = position in the file. usable_only: leave
    out the accounts `check` found unusable (ibmq_accounts_status.json next to the keys file)."""
    import ast
    import re
    text = pathlib.Path(keys).expanduser().read_text()
    if "{" in text:
        body = re.sub(r"#[^\n]*", "", text[text.index("{"):])            # drop comments
        rows = list(ast.literal_eval(body).items())
    else:
        rows = [(r[0] if len(r) > 1 else "", r[-1]) for r in (l.split() for l in text.splitlines() if l.strip())]
    acc = [(i, e, tok) for i, (e, tok) in enumerate(rows)]
    status = status_file(keys)
    if usable_only and status.exists():
        bad = {s["fingerprint"] for s in json.loads(status.read_text()) if not s["ok"]}
        acc = [a for a in acc if fingerprint(a[2]) not in bad]
    return acc


def status_file(keys):
    return pathlib.Path(keys).expanduser().parent / "ibmq_accounts_status.json"


def check(keys):
    """Log in with every account (no job is sent) and record which work and their quota left, by fingerprint, in
    ibmq_accounts_status.json next to the keys file (outside git; no token in it). accounts() then skips the bad ones."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    out = []
    for i, email, token in accounts(keys, usable_only=False):
        row = dict(line=i, email=email, fingerprint=fingerprint(token), checked=_now())
        try:
            service = QiskitRuntimeService(channel="ibm_quantum_platform", token=token)
            row["backends"] = [b.name for b in service.backends(simulator=False, operational=True)]
            try:
                u = service.usage()
                row["usage"] = {k: v for k, v in u.items() if "second" in k or "limit" in k or "period" in k} \
                    if isinstance(u, dict) else str(u)
            except Exception as e:
                row["usage"] = f"unavailable ({type(e).__name__})"
            row["ok"] = bool(row["backends"])
        except Exception as e:
            row.update(ok=False, error=f"{type(e).__name__}: {str(e)[:120]}".replace(token, "<token>"))
        out.append(row)
        print(f"line {i:2d} {email:28s} {'ok ' if row['ok'] else 'BAD'} "
              f"{row.get('usage', row.get('error', ''))}", flush=True)
    status_file(keys).write_text(json.dumps(out, indent=1))
    print(f"{sum(r['ok'] for r in out)} of {len(out)} accounts usable -> {status_file(keys)}")


def fingerprint(token):
    return hashlib.sha256(token.encode()).hexdigest()[:12]


def service_for(keys, line=None, fp=None):
    """(service, line, fingerprint): by fingerprint when given (the account a job was sent from), else by line."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    acc = accounts(keys, usable_only=False)                     # by its own line or fingerprint, never by position
    hit = [a for a in acc if fingerprint(a[2]) == fp] if fp else [a for a in acc if a[0] == line]
    if not hit:
        raise SystemExit(f"no account in {keys} with fingerprint {fp} / line {line}")
    i, _, token = hit[0]
    return QiskitRuntimeService(channel="ibm_quantum_platform", token=token), i, fingerprint(token)


def circuit_text(c):
    """What a circuit does, independent of the Qiskit version and of float noise: one line per gate (name, qubit
    indices, parameters rounded to 1e-9 rad), the measured qubits. QPY bytes would change with a Qiskit upgrade or a
    circuit name and send the same job again."""
    lines = [f"q{c.num_qubits} c{c.num_clbits}"]
    for ins in c.data:
        op = ins.operation
        if op.name == "barrier":
            continue
        qs = ",".join(str(c.find_bit(q).index) for q in ins.qubits)
        cs = ",".join(str(c.find_bit(b).index) for b in ins.clbits)
        ps = ",".join(f"{float(v):.9f}".replace("-0.000000000", "0.000000000") for v in op.params)
        lines.append(f"{op.name}({ps}) {qs}" + (f" -> {cs}" if cs else ""))
    return "\n".join(lines)


def circuit_key(c, backend_name, shots, repeat=0):
    """One circuit's key: the same circuit, backend, shots and repeat is the same measurement, in whatever job."""
    h = hashlib.sha256(circuit_text(c).encode())
    h.update(f"|{backend_name}|{shots}|repeat{repeat}".encode())
    return h.hexdigest()[:16]


def measured(backend_name, shots, repeat=0):
    """{circuit key: (ledger folder, index in its job, status)} over every job in the ledger that was sent."""
    out = {}
    for rec in LEDGER.glob("*/record.json"):
        r = json.loads(rec.read_text())
        if r.get("backend") != backend_name or r.get("shots") != shots or r.get("repeat", 0) != repeat:
            continue
        if not (r["status"] == "done" or r["status"] == "submitted"):
            continue
        for i, k in enumerate(r.get("circuit_keys", [])):
            out[k] = (rec.parent, i, r["status"])
    return out


def lookup(c, backend_name, shots, repeat=0):
    """The counts of circuit c if a done job in the ledger measured it (same backend, shots, repeat), else None."""
    hit = measured(backend_name, shots, repeat).get(circuit_key(c, backend_name, shots, repeat))
    if hit and hit[2] == "done":
        return json.loads((hit[0] / "counts.json").read_text())[hit[1]]
    return None


def expected_fidelity(tc, target):
    """Calibration score of a transpiled circuit: the product of (1 - error) over its gates and measurements, from the
    backend's own error rates (an estimate of the chance that nothing went wrong)."""
    import math
    logf = 0.0
    for ins in tc.data:
        name = ins.operation.name
        if name in ("barrier", "delay") or name not in target.operation_names:
            continue
        qs = tuple(tc.find_bit(q).index for q in ins.qubits)
        props = target[name].get(qs)
        if props is not None and props.error is not None:
            logf += math.log(max(1e-12, 1 - props.error))
    return math.exp(logf)


def best_transpile(circs, backend, seeds=8, optimization_level=3):
    """Every circuit transpiled with seeds 0..seeds-1 (level 3), the one with the highest expected_fidelity kept.
    Returns (circuits, [(seed, score, score of seed 0)])."""
    from qiskit import transpile
    tries = [transpile(list(circs), backend, optimization_level=optimization_level, seed_transpiler=s)
             for s in range(seeds)]
    out, info = [], []
    for i in range(len(circs)):
        sc = [expected_fidelity(tries[s][i], backend.target) for s in range(seeds)]
        b = int(max(range(seeds), key=lambda s: sc[s]))
        out.append(tries[b][i])
        info.append((b, sc[b], sc[0]))
    return out, info


class QuotaShort(Exception):
    """The account has too little quota left for this job; nothing was sent."""


def seconds_per_circuit(backend_name, default=2.0):
    """Quantum seconds per circuit seen in this ledger's done jobs on the backend (the default before any)."""
    v = []
    for rec in LEDGER.glob("*/record.json"):
        r = json.loads(rec.read_text())
        if r.get("backend") == backend_name and r["status"] == "done" and isinstance(r.get("quantum_seconds"), (int, float)) \
                and r["quantum_seconds"] > 0:
            v.append(r["quantum_seconds"] / r["n_circuits"])
    return max(v) if v else default


def collect_pending(keys):
    """Fetch the results of every job in the ledger that was sent but never collected (the run was closed while
    waiting), from the account that sent it. Jobs still queued at IBM are waited for."""
    for rec in sorted(LEDGER.glob("*/record.json")):
        r = json.loads(rec.read_text())
        if r["status"] != "submitted":
            continue
        service, _, _ = service_for(keys, fp=r.get("account_fingerprint")) if r.get("account_fingerprint") \
            else service_for(keys, line=r.get("account_line", 0))
        print(f"ledger {rec.parent.name}: collecting job {r['job_id']} sent {r.get('created')}", flush=True)
        _finish(rec.parent, r, service.job(r["job_id"]), r["n_circuits"])


def job_key(circs, backend_name, shots, repeat=0):
    h = hashlib.sha256("\n==\n".join(circuit_text(c) for c in circs).encode())
    h.update(f"|{backend_name}|{shots}".encode() + (f"|repeat{repeat}".encode() if repeat else b""))
    return h.hexdigest()[:16]


def write_index():
    rows = []
    for rec in sorted(LEDGER.glob("*/record.json")):
        r = json.loads(rec.read_text())
        rows.append(f"| {r.get('created', '')} | {r.get('backend', '')} | {r.get('job_id', '-')} | {r.get('status')} | "
                    f"{r.get('n_circuits')} x {r.get('shots')} | {r.get('quantum_seconds', '-')} | "
                    f"{r.get('purpose', '')} | [{rec.parent.name}]({rec.parent.name}/record.json) |")
    LEDGER.mkdir(exist_ok=True)
    (LEDGER / "index.md").write_text(
        "# IBM hardware jobs\n\nEvery job sent from this repository (written by `ibm_jobs.py`). Quantum seconds as "
        "IBM reports them.\n\n| created (UTC) | backend | job id | status | circuits x shots | quantum s | purpose | "
        "record |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def _save(folder, rec):
    (folder / "record.json").write_text(json.dumps(_jsonable(rec), indent=1))
    write_index()


def _finish(folder, rec, job, n):
    """Wait for the job, write its counts and usage; returns the counts. A job IBM failed or cancelled is recorded as
    failed (it may still have used quota) and is never sent again by itself."""
    try:
        result = job.result()
    except Exception as e:
        rec["status"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
        rec["finished"] = _now()
        try:
            rec["quantum_seconds"] = job.usage()
        except Exception:
            pass
        _save(folder, rec)
        raise
    counts = [result[i].data.meas.get_counts() for i in range(n)]
    (folder / "counts.json").write_text(json.dumps(counts))
    rec["status"] = "done"
    if not rec.get("shots") and counts:                         # a fetched job whose inputs did not say
        rec["shots"] = sum(counts[0].values())
    rec["finished"] = _now()
    try:
        rec["quantum_seconds"] = job.usage()
    except Exception as e:                                      # older jobs or plans may not report it
        rec["quantum_seconds"] = f"unavailable ({type(e).__name__})"
    try:
        rec["ibm_metrics"] = job.metrics()
    except Exception as e:
        rec["ibm_metrics"] = f"unavailable ({type(e).__name__})"
    try:
        rec["result_metadata"] = result.metadata
    except Exception:
        pass
    _save(folder, rec)
    return counts


def sampler_counts(circs, backend_name, shots, keys, account, purpose, inputs=(), circuit_notes=None,
                   dry_run=False, optimization_level=3, seeds=8, repeat=0, retry_failed=False, dd=True):
    """Counts of every circuit on IBM backend `backend_name` (one SamplerV2 job), through the ledger (see the module
    docstring). inputs: files the circuits were built from (recorded with their sha256). circuit_notes: one short text
    per circuit (what it measures). retry_failed: send again a job IBM failed (otherwise it stops). seeds: transpiler
    seeds tried per circuit, the best by calibration kept. dd: IBM's dynamical decoupling (XY4) on idle qubits. Returns (counts or
    None on a dry run, the ledger folder)."""
    from qiskit import qpy, transpile
    key = job_key(circs, backend_name, shots, repeat)       # repeat k > 0: the same circuits again, on purpose
    if not dd:                                                  # a job without decoupling is a different job
        key = hashlib.sha256((key + "|nodd").encode()).hexdigest()[:16]
    folder = LEDGER / key
    rec_path = folder / "record.json"
    rec = json.loads(rec_path.read_text()) if rec_path.exists() else None
    if rec and rec["status"] == "done":
        print(f"ledger {key}: job {rec['job_id']} already done on {rec['backend']} - counts from the ledger, "
              f"nothing sent", flush=True)
        return json.loads((folder / "counts.json").read_text()), folder

    earlier = []
    if rec and rec["status"].startswith("failed") and retry_failed:      # keep the failed attempt as evidence
        earlier = rec.get("earlier_attempts", []) + [{k: rec.get(k) for k in ("job_id", "status", "created", "finished",
                                                                               "quantum_seconds", "account_fingerprint")}]
        rec = None
    if rec and rec["status"].startswith("failed") and not retry_failed:
        raise SystemExit(f"ledger {key}: job {rec.get('job_id')} {rec['status']} - not sent again "
                         f"(look at it with: ibm_jobs.py show {key}; retry_failed=True to send anew)")
    if not rec or rec["status"].startswith("prepared"):        # a new job: none of its circuits may be measured yet
        seen = measured(backend_name, shots, repeat)
        dup = [(i, seen[k]) for i, k in enumerate(circuit_key(c, backend_name, shots, repeat) for c in circs) if k in seen]
        if dup:
            raise SystemExit(f"ledger {key}: {len(dup)} of these {len(circs)} circuits were already sent in another job "
                             f"(e.g. circuit {dup[0][0]} in {dup[0][1][0].name}, {dup[0][1][2]}) - not sending; take "
                             f"their counts with ibm_jobs.lookup, send only the rest")
    from qiskit_ibm_runtime import SamplerV2
    if rec and rec["status"] == "submitted":                    # from the account that sent it, wherever its line is now
        service, _, _ = service_for(keys, fp=rec.get("account_fingerprint")) if rec.get("account_fingerprint") \
            else service_for(keys, line=rec.get("account_line", account))
        print(f"ledger {key}: job {rec['job_id']} was sent before - fetching it, not sending again", flush=True)
        return _finish(folder, rec, service.job(rec["job_id"]), len(circs)), folder
    service, account, fp = service_for(keys, line=account)      # the token is never written
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "sending.lock"
    if lock.exists():
        raise SystemExit(f"ledger {key}: {lock} exists - another process is sending this job, or one died while "
                         f"sending: check the IBM job list for it before removing the lock")

    try:                                                        # enough quota left? (stop before sending, not after)
        left = service.usage().get("usage_remaining_seconds")
    except Exception:
        left = None
    need = len(circs) * seconds_per_circuit(backend_name)
    if left is not None and left < need:
        raise QuotaShort(f"account line {account}: {left} s left, this job needs about {need:.0f} s - not sent")
    backend = service.backend(backend_name)
    tq, tinfo = best_transpile(circs, backend, seeds, optimization_level)
    (folder / "circuits.qpy").write_bytes(_qpy_bytes(circs))
    (folder / "circuits.txt").write_text("\n==\n".join(circuit_text(c) for c in circs))   # what the key hashes
    with open(folder / "transpiled.qpy", "wb") as f:
        qpy.dump(tq, f)
    try:
        props = backend.properties()
        if props is not None:
            (folder / "properties.json").write_text(json.dumps(_jsonable(props.to_dict())))
    except Exception:
        pass
    rec = dict(purpose=purpose, backend=backend.name, shots=shots, n_circuits=len(circs), repeat=repeat,
               circuit_keys=[circuit_key(c, backend_name, shots, repeat) for c in circs], account_line=account,
               quota_left_before=left, earlier_attempts=earlier,
               account_fingerprint=fp,
               command=" ".join(sys.argv), code=_code_version(), versions=_versions(),
               inputs=[{"file": str(pathlib.Path(p).resolve().relative_to(HERE)) if pathlib.Path(p).resolve()
                        .is_relative_to(HERE) else str(p), "sha256": _sha256(p)} for p in inputs],
               transpile=dict(optimization_level=optimization_level, seeds_tried=seeds,
                              per_circuit=[dict(seed=s, expected_fidelity=f, seed0=f0) for s, f, f0 in tinfo]),
               dynamical_decoupling="XY4" if dd else None,
               circuits=[dict(note=(circuit_notes[i] if circuit_notes else ""), logical_qubits=c.num_qubits,
                              physical_qubits=t.layout.final_index_layout() if t.layout else None,
                              depth=t.depth(), ops={k: int(v) for k, v in t.count_ops().items()})
                         for i, (c, t) in enumerate(zip(circs, tq))])
    if dry_run:
        rec.update(status="prepared (dry run, not sent)", created=_now())
        _save(folder, rec)
        print(f"ledger {key}: dry run - {len(circs)} circuits x {shots} shots for {backend.name} written, not sent",
              flush=True)
        return None, folder
    try:                                                        # two processes, same job: only one may send
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()} {_now()}".encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"ledger {key}: another process is sending this job")
    sampler = SamplerV2(mode=backend)
    if dd:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    job = sampler.run(tq, shots=shots)                          # if this raises, the lock stays: it may have been sent
    rec.update(status="submitted", job_id=job.job_id(), created=_now())
    _save(folder, rec)                                          # written before waiting: a crash cannot lose the job
    lock.unlink()
    print(f"ledger {key}: account line {account}, backend {backend.name}, job {job.job_id()}, waiting", flush=True)
    return _finish(folder, rec, job, len(circs)), folder


def find(ref):
    """Ledger folder by its key, or by a job id."""
    if (LEDGER / ref / "record.json").exists():
        return LEDGER / ref
    for rec in LEDGER.glob("*/record.json"):
        if json.loads(rec.read_text()).get("job_id") == ref:
            return rec.parent
    raise SystemExit(f"{ref}: not in the ledger (ibm_jobs.py list; ibm_jobs.py fetch <job id> to bring an old job in)")


def show(ref, top=8):
    """Print a job's record and, per circuit, its note, qubits, shots counted and most frequent outcomes."""
    folder = find(ref)
    r = json.loads((folder / "record.json").read_text())
    skip = {"circuits", "ibm_metrics", "result_metadata", "inputs"}
    for k, v in r.items():
        if k not in skip:
            print(f"{k:16s} {v}")
    for i in r.get("inputs", []):
        print(f"input            {i['file']}  sha256 {i['sha256'][:16]}...")
    counts = json.loads((folder / "counts.json").read_text()) if (folder / "counts.json").exists() else None
    for i, c in enumerate(r.get("circuits") or [{}] * len(counts or [])):
        line = f"circuit {i:3d}  {c.get('note', '')}  qubits {c.get('physical_qubits')}"
        if counts:
            cnt = counts[i]
            tot = sum(cnt.values())
            best = sorted(cnt.items(), key=lambda kv: -kv[1])[:top]
            line += f"\n             {tot} shots, top: " + ", ".join(f"{k} {v / tot:.3f}" for k, v in best)
        print(line)
    print(f"files: {', '.join(sorted(p.name for p in folder.iterdir()))}  ({folder})")


def fetch(job_id, keys, account=None, purpose=""):
    """Bring a job already run on IBM into the ledger by its id (reads it; sends nothing): status, backend, times,
    usage, IBM's metrics, the circuits as sent and the counts. account None: every line of the keys file is tried
    until one can see the job (IBM shows a job only to the account that ran it)."""
    from qiskit import qpy
    for rec in LEDGER.glob("*/record.json"):
        if json.loads(rec.read_text()).get("job_id") == job_id:
            print(f"{job_id} is already in the ledger: {rec.parent.name}")
            return rec.parent
    job = None
    for line in ([account] if account is not None else [a[0] for a in accounts(keys)]):
        service, line, fp = service_for(keys, line=line)
        try:
            job = service.job(job_id)
            account = line
            break
        except Exception as e:                                  # not this account's job (or not reachable)
            print(f"account line {line}: cannot see {job_id} ({type(e).__name__})")
    if job is None:
        raise SystemExit(f"{job_id}: no account in {keys} can see this job")
    circs, shots = [], None
    try:                                                        # SamplerV2 pubs: (circuit, parameters, shots) or so
        for pub in job.inputs["pubs"]:
            circ = pub[0] if isinstance(pub, (tuple, list)) else getattr(pub, "circuit", pub)
            circs.append(circ)
            if isinstance(pub, (tuple, list)) and len(pub) > 2:
                shots = pub[2]
        shots = shots or job.inputs.get("shots")
    except Exception as e:
        print(f"circuits not readable from the job ({type(e).__name__}); counts and metadata only")
    folder = LEDGER / f"fetched_{job_id[:12]}"
    folder.mkdir(parents=True, exist_ok=True)
    if circs:
        with open(folder / "transpiled.qpy", "wb") as f:
            qpy.dump(circs, f)
    rec = dict(purpose=purpose or "fetched from IBM by job id", backend=job.backend().name, shots=shots,
               n_circuits=len(circs) or None, account_line=account,
               account_fingerprint=fp, job_id=job_id, status="submitted",
               created=str(job.creation_date), fetched=_now(), command=" ".join(sys.argv), versions=_versions(),
               circuits=[dict(note="", logical_qubits=None,
                              physical_qubits=c.layout.final_index_layout() if getattr(c, "layout", None) else None,
                              depth=c.depth(), ops={k: int(v) for k, v in c.count_ops().items()}) for c in circs])
    n = len(circs) or len(job.result())
    rec["n_circuits"] = n
    _finish(folder, rec, job, n)
    print(f"{job_id} -> {folder}")
    return folder


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="the IBM job ledger")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="every job in the ledger (rewrites ibm_jobs/index.md)")
    k = sub.add_parser("check", help="log in with every account of the keys file (sends no job); mark the bad ones")
    k.add_argument("--keys", required=True)
    s = sub.add_parser("show", help="one job's record and results")
    s.add_argument("ref", help="ledger key or job id")
    s.add_argument("--top", type=int, default=8, help="most frequent outcomes shown per circuit")
    f = sub.add_parser("fetch", help="bring a job already run on IBM into the ledger by its id (sends nothing)")
    f.add_argument("job_id")
    f.add_argument("--keys", required=True, help="keys file, one '<email> <token>' per line (outside git)")
    f.add_argument("--account", type=int, default=None, help="line of --keys (default: try every line)")
    f.add_argument("--purpose", default="")
    a = ap.parse_args()
    if a.cmd == "list":
        write_index()
        print((LEDGER / "index.md").read_text())
    elif a.cmd == "check":
        check(a.keys)
    elif a.cmd == "show":
        show(a.ref, a.top)
    else:
        fetch(a.job_id, a.keys, a.account, a.purpose)
