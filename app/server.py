"""INDRA-Sat-Diff web app.  Run:  .venv/bin/python -m app  (then open http://127.0.0.1:8765)"""
import asyncio
import gzip
import json
import math
import os
import re
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from . import configs, datasources, jobs, registry


def clean(o):
    """Replace NaN / infinity by None (JSON has no such numbers) and numpy scalars by Python ones."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


class SafeJSON(JSONResponse):
    def render(self, content):
        return json.dumps(clean(content), ensure_ascii=False, allow_nan=False, separators=(",", ":"),
                          default=str).encode("utf-8")


app = FastAPI(title="INDRA-Sat-Diff", default_response_class=SafeJSON)


@app.middleware("http")
async def same_origin_writes(request: Request, call_next):
    """Reject state-changing requests from other sites (a web page could otherwise start jobs on localhost)."""
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1] != request.headers.get("host", ""):
            return JSONResponse({"detail": "cross-origin request refused"}, status_code=403)
    return await call_next(request)
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
SAFE = re.compile(r"^(?!\.)[A-Za-z0-9_.-]+$")   # no leading dot: blocks "." and ".."


def _safe(name):
    if not SAFE.match(name or ""):
        raise HTTPException(400, f"invalid name: {name!r}")
    return name


# ----------------------------------------------------------------------------- data
@app.get("/api/datasets")
def list_datasets(refresh: bool = False):
    return [d.summary() for d in registry.datasets(refresh).values()]


# ----------------------------------------------------------------------------- bring your own data
@app.post("/api/sources/inspect")
async def sources_inspect(req: Request):
    """Inspect a folder of HDF5/netCDF4 files: variables, coordinates, grid, times, suggestions."""
    body = await req.json()
    try:
        return await asyncio.to_thread(datasources.inspect, body.get("path", ""))
    except (ValueError, PermissionError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/sources/preview")
async def sources_preview(req: Request):
    """One frame of the source, converted to mm/h, as uint8 (same encoding as dataset frames)."""
    body = await req.json()
    try:
        u8, ext, fname, stats = await asyncio.to_thread(datasources.preview, body)
    except (ValueError, PermissionError, FileNotFoundError, KeyError) as e:
        raise HTTPException(400, str(e))
    except OSError as e:
        raise HTTPException(400, f"Could not read the file: {e}")
    return Response(u8.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": f"{u8.shape[0]},{u8.shape[1]}", "X-Extent": json.dumps(ext),
                             "X-File": fname, "X-Stats": json.dumps(stats), "Cache-Control": "no-store"})


@app.delete("/api/datasets/{ds_id}")
def delete_dataset(ds_id: str):
    """Remove a dataset folder (frames, index, metadata). Raw source files elsewhere are not touched."""
    ds = _ds(_safe(ds_id))
    base = os.path.dirname(ds.path)
    if any(j.status in ("queued", "running") and ds_id in json.dumps(j.params) for j in jobs.JOBS.values()):
        raise HTTPException(409, "A job is using this dataset; wait for it or cancel it first.")
    import shutil
    shutil.rmtree(base)
    registry.datasets(refresh=True)
    return {"deleted": ds_id}


@app.get("/api/datasets/{ds_id}/timeline")
def timeline(ds_id: str):
    ds = _ds(ds_id)
    return {"times": ds.times, "interval": ds.interval, "gaps": ds.gaps()}


@app.get("/api/datasets/{ds_id}/frame")
def frame(ds_id: str, i: int):
    ds = _ds(ds_id)
    if not 0 <= i < len(ds.files):
        raise HTTPException(404, "frame index out of range")
    a = registry.cube(ds)[i]
    return Response(memoryview(a), media_type="application/octet-stream",
                    headers={"X-Shape": f"{a.shape[0]},{a.shape[1]}", "X-Time": registry.iso(ds.times[i]),
                             "Cache-Control": "max-age=86400"})


@lru_cache(maxsize=256)
def _gz_range(ds_id, version, start, count):
    return gzip.compress(registry.cube(registry.datasets()[ds_id])[start:start + count].tobytes(), compresslevel=1)


@app.get("/api/datasets/{ds_id}/frames")
def frames(ds_id: str, request: Request, start: int = 0, count: int = 16, v: str = ""):
    """`count` consecutive frames from index `start` as raw uint8 (count*n0*n1 bytes, value*255), a slice of
    the dataset's memory-mapped cube. X-Shape: count,n0,n1. Pass v=<dataset version> for immutable caching."""
    ds = _ds(ds_id)
    n = len(ds.files)
    if not 0 <= start < n or count < 1:
        raise HTTPException(404, "frame range out of bounds")
    count = min(count, 256, n - start)
    etag = f'"{ds.version}-{start}-{count}"'
    headers = {"ETag": etag, "Vary": "Accept-Encoding",
               "Cache-Control": "public, max-age=31536000, immutable" if v == ds.version else "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    cube = registry.cube(ds)
    headers |= {"X-Shape": f"{count},{cube.shape[1]},{cube.shape[2]}", "X-Start": str(start),
                "X-Times": f"{ds.times[start]:.0f},{ds.times[start + count - 1]:.0f}"}
    if "gzip" in request.headers.get("accept-encoding", "") and request.client and \
            request.client.host not in ("127.0.0.1", "::1", "localhost"):
        headers["Content-Encoding"] = "gzip"      # worth it over a network; skipped on loopback
        return Response(_gz_range(ds.id, ds.version, start, count), media_type="application/octet-stream",
                        headers=headers)
    return Response(memoryview(cube[start:start + count]), media_type="application/octet-stream", headers=headers)


@app.get("/api/geo")
def geo(lon0: float, lat0: float, lon1: float, lat1: float):
    """Natural Earth 50m coastlines and country borders clipped to a box (flat lon,lat polylines)."""
    box = [round(max(-180.0, lon0)), round(max(-90.0, lat0)), round(min(180.0, lon1)), round(min(90.0, lat1))]
    return SafeJSON(registry.geo_lines(*box), headers={"Cache-Control": "public, max-age=604800"})


def _ds(ds_id):
    try:
        return registry.datasets()[ds_id]
    except KeyError:
        raise HTTPException(404, f"dataset {ds_id} not found")


# ----------------------------------------------------------------------------- models
@app.get("/api/models")
def list_models(all: bool = False):
    return registry.models(include_partial=all)


@app.get("/api/models/{mid}/curves")
def model_curves(mid: str):
    """Training history per stage, flattened to plottable series: {stage: {epochs, series: {"val/loss": [...]}}}."""
    try:
        return registry.curves(registry.model_by_id(_safe(mid)))
    except KeyError:
        raise HTTPException(404, f"model {mid} not found")


@app.post("/api/train/preview")
async def train_preview(req: Request):
    """The YAML a train job with these params would write (nothing is written or run)."""
    try:
        return {"yaml": configs.preview(await req.json())}
    except (KeyError, ValueError, TypeError) as e:
        raise HTTPException(400, str(e))


@app.get("/api/models/{mid}/metrics")
def model_metrics(mid: str):
    m = registry.model_by_id(_safe(mid))
    out = {}
    for p in m["metrics"]:
        try:
            out[p] = json.load(open(os.path.join(registry.ROOT, p)))
        except Exception:
            pass
    return out


# ----------------------------------------------------------------------------- jobs
@app.post("/api/jobs")
async def create_job(req: Request):
    body = await req.json()
    try:
        job = jobs.submit(body.get("type"), body.get("params", {}))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return job.public()


@app.get("/api/jobs")
def list_jobs(limit: int = 50, type: str = None, brief: bool = False):
    js = sorted((j for j in jobs.JOBS.values() if not type or j.type == type), key=lambda j: -j.created)
    return [j.public(full=not brief) for j in js[:max(1, min(limit, 500))]]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    return _job(job_id).public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    jobs.cancel(_job(job_id).id)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request, since: int = 0):
    job = _job(job_id)

    try:   # an EventSource reconnect sends the last id it saw: resume after it instead of replaying
        resume = int(request.headers.get("last-event-id", -1)) + 1
    except ValueError:
        resume = 0

    async def gen():
        i = max(since, resume)
        events = job.all_events()
        while True:
            if await request.is_disconnected():
                return
            while i < len(events):
                ev = events[i]
                i += 1
                yield {"id": str(i - 1), "event": ev["kind"],
                       "data": json.dumps(clean(ev), default=str, allow_nan=False)}
            if job.status in ("done", "failed", "cancelled") and i >= len(events):
                return
            await asyncio.sleep(0.25)
    return EventSourceResponse(gen())


@lru_cache(maxsize=8)
def _npz(path, mtime):
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _u8(a):
    return np.clip(np.rint(np.nan_to_num(a) * 255), 0, 255).astype(np.uint8)


@app.get("/api/jobs/{job_id}/array/{name}")
def job_array(job_id: str, name: str, member: int = 0):
    """Arrays of a finished predict job as uint8 (T*H*W), normalised value * 255 (exceedance: probability * 255).
    `spread` is the ensemble standard deviation in normalised units, also * 255."""
    job = _job(job_id)
    path = os.path.join(job.dir, "result.npz")
    if not os.path.exists(path):
        raise HTTPException(404, "no arrays for this job")
    z = _npz(path, os.path.getmtime(path))
    key = _safe(name)
    if key not in z:
        raise HTTPException(404, f"no array {key}")
    a = z[key]
    if key == "ensemble":
        if not 0 <= member < a.shape[0]:
            raise HTTPException(404, "no such member")
        a = a[member]
    a = _u8(a)
    return Response(a.tobytes(), media_type="application/octet-stream",
                    headers={"X-Shape": ",".join(map(str, a.shape)), "Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{job_id}/arrays")
def job_arrays(job_id: str, names: str):
    """Several arrays of a finished predict job in one response: the uint8 arrays concatenated in the order of
    the comma-separated `names`; X-Layout is JSON [[name, [T, n0, n1]], ...]. `ensemble` is skipped."""
    job = _job(job_id)
    path = os.path.join(job.dir, "result.npz")
    if not os.path.exists(path):
        raise HTTPException(404, "no arrays for this job")
    z = _npz(path, os.path.getmtime(path))
    parts, layout = [], []
    for key in names.split(","):
        if _safe(key) in z and key != "ensemble":
            a = _u8(z[key])
            parts.append(a.tobytes())
            layout.append([key, list(a.shape)])
    return Response(b"".join(parts), media_type="application/octet-stream",
                    headers={"X-Layout": json.dumps(layout), "Cache-Control": "private, max-age=3600"})


@app.get("/api/jobs/{job_id}/file/{name}")
def job_file(job_id: str, name: str):
    path = os.path.join(_job(job_id).dir, _safe(name))
    if not os.path.exists(path):
        raise HTTPException(404)
    return FileResponse(path)


def _job(job_id):
    if job_id not in jobs.JOBS:
        raise HTTPException(404, f"job {job_id} not found")
    return jobs.JOBS[job_id]


# ----------------------------------------------------------------------------- static
@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"), headers={"Cache-Control": "no-cache"})


@app.middleware("http")
async def static_cache(request: Request, call_next):
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):       # revalidate app code (ETag/304); vendor files rarely change
        resp.headers["Cache-Control"] = "public, max-age=604800" if "/vendor/" in request.url.path else "no-cache"
    return resp


app.mount("/static", StaticFiles(directory=STATIC), name="static")

from . import handlers  # noqa: E402,F401  (registers job handlers)


def _warm():
    """Parse the coastline files and open the dataset indexes in the background so the first page is fast."""
    try:
        registry.datasets()
        registry.geo_lines(35, -20, 120, 40)
    except Exception:
        pass


import threading  # noqa: E402

threading.Thread(target=_warm, daemon=True, name="warmup").start()
