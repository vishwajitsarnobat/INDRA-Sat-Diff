"""Single-GPU job queue. One job runs at a time; events are kept in memory and mirrored to a log file.
Each job keeps job.json (request + status), log.txt and result.json in artifacts/jobs/<id>/, and past jobs
are loaded from there when the server starts."""
import glob
import itertools
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field

from . import registry

_ids = itertools.count(1)


@dataclass
class Job:
    id: str
    type: str
    params: dict
    status: str = "queued"        # queued | running | done | failed | cancelled
    progress: float = 0.0
    events: list = field(default_factory=list)
    result: dict = None
    error: str = None
    created: float = field(default_factory=time.time)
    finished: float = None
    proc: subprocess.Popen = None
    cancel_flag: bool = False
    restored: bool = False        # loaded from disk; events are rebuilt from log.txt on demand

    @property
    def dir(self):
        d = os.path.join(registry.JOB_DIR, self.id)
        os.makedirs(d, exist_ok=True)
        return d

    def emit(self, kind, **data):
        ev = {"seq": len(self.events), "t": time.time(), "kind": kind, **data}
        self.events.append(ev)
        if kind == "log":
            with open(os.path.join(self.dir, "log.txt"), "a") as f:
                f.write(data.get("line", "") + "\n")

    def save(self):
        with open(os.path.join(self.dir, "job.json"), "w") as f:
            json.dump({"type": self.type, "params": self.params, "status": self.status, "created": self.created,
                       "finished": self.finished, "error": self.error}, f, indent=1, default=str)

    def all_events(self):
        """Events for the stream; for a job restored from disk they are rebuilt once from log.txt."""
        if self.restored and not self.events:
            log = os.path.join(self.dir, "log.txt")
            if os.path.exists(log):
                with open(log, errors="replace") as f:
                    for line in f.read().splitlines()[-5000:]:
                        self.events.append({"seq": len(self.events), "t": self.created, "kind": "log", "line": line})
            self.events.append({"seq": len(self.events), "t": self.finished or self.created, "kind": "status",
                                "status": self.status, "error": self.error, "result": self.result})
        return self.events

    def public(self, full=True):
        out = {"id": self.id, "type": self.type, "params": self.params, "status": self.status,
               "progress": round(self.progress, 3), "error": self.error, "created": self.created,
               "finished": self.finished}
        if full:
            out["result"] = self.result
        return out


JOBS = {}
_q = queue.Queue()
HANDLERS = {}


def handler(name):
    def wrap(fn):
        HANDLERS[name] = fn
        return fn
    return wrap


def submit(jtype, params):
    if jtype not in HANDLERS:
        raise ValueError(f"unknown job type {jtype}")
    job = Job(id=f"{time.strftime('%Y%m%d-%H%M%S')}-{next(_ids)}-{jtype}", type=jtype, params=params)
    JOBS[job.id] = job
    job.save()
    _q.put(job)
    job.emit("status", status="queued", position=_q.qsize())
    return job


def cancel(job_id):
    job = JOBS[job_id]
    job.cancel_flag = True
    if job.proc and job.proc.poll() is None:
        job.proc.send_signal(signal.SIGINT)
    if job.status == "queued":
        job.status, job.finished = "cancelled", time.time()
        job.emit("status", status="cancelled")
        job.save()


def run_cli(job, args, parse_progress=None):
    """Run `climate-forecast <args>` as a subprocess (frees the warm model first), streaming its output."""
    from . import predictor
    predictor.unload()
    cmd = [sys.executable, "-u", "-m", "climate_forecast.cli", *args]
    job.emit("log", line="$ climate-forecast " + " ".join(args))
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "TQDM_MININTERVAL": "2"}
    job.proc = subprocess.Popen(cmd, cwd=registry.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
    tail = []
    for raw in job.proc.stdout:
        for line in raw.replace("\r", "\n").splitlines():
            line = line.rstrip()
            if not line:
                continue
            tail = (tail + [line])[-200:]
            job.emit("log", line=line)
            if parse_progress:
                p = parse_progress(line)
                if p is not None:
                    job.progress = p
                    job.emit("progress", value=p)
    code = job.proc.wait()
    if job.cancel_flag:
        raise InterruptedError("cancelled")
    if code != 0:
        raise RuntimeError(f"command exited with code {code}")
    return tail


def _worker():
    while True:
        job = _q.get()
        if job.status == "cancelled":
            continue
        job.status = "running"
        job.emit("status", status="running")
        job.save()
        try:
            job.result = HANDLERS[job.type](job) or {}
            job.status, job.progress = "done", 1.0
            json.dump(job.result, open(os.path.join(job.dir, "result.json"), "w"), indent=1, default=str)
        except InterruptedError:
            job.status = "cancelled"
        except Exception as e:
            job.status, job.error = "failed", f"{type(e).__name__}: {e}"
            job.emit("log", line=traceback.format_exc())
        job.finished = time.time()
        try:
            job.save()
        except Exception:
            pass
        job.emit("status", status=job.status, error=job.error, result=job.result)


def load_persisted():
    """Register jobs found in artifacts/jobs/ (from earlier server runs) so their results stay browsable."""
    for d in sorted(glob.glob(os.path.join(registry.JOB_DIR, "*"))):
        jid, meta_path = os.path.basename(d), os.path.join(d, "job.json")
        if jid in JOBS or not os.path.exists(meta_path):
            continue
        try:
            meta = json.load(open(meta_path))
            result = None
            if os.path.exists(os.path.join(d, "result.json")):
                result = json.load(open(os.path.join(d, "result.json")))
            try:      # ids start with the local start time: YYYYMMDD-HHMMSS-n-type
                created = time.mktime(time.strptime(jid[:15], "%Y%m%d-%H%M%S"))
            except ValueError:
                created = os.path.getmtime(meta_path)
            status = meta.get("status")
            error = meta.get("error")
            if result is not None:
                status = "done"
            elif status not in ("failed", "cancelled"):
                status, error = "failed", error or "interrupted: the server stopped before this job finished"
            finished = meta.get("finished") or max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
            JOBS[jid] = Job(id=jid, type=meta.get("type", jid.rsplit("-", 1)[-1]), params=meta.get("params", {}),
                            status=status, progress=1.0 if status == "done" else 0.0, result=result, error=error,
                            created=meta.get("created") or created, finished=finished, restored=True)
        except Exception:
            continue


load_persisted()

threading.Thread(target=_worker, daemon=True, name="gpu-jobs").start()
