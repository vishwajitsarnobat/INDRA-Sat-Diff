"""Bring-your-own-data: inspect a folder of gridded files, preview one frame, import it as a dataset.

Supports HDF5 and netCDF4 files (one time step per file, or the first time step of each file) on a regular
latitude/longitude grid. Only folders under the allowed roots are read (default: the user's home directory;
override with the environment variable INDRA_DATA_ROOTS, separated by ':').
"""
import math
import os
import re
from datetime import datetime, timezone

import h5py
import numpy as np

from climate_forecast.datasets.preprocess_logic import _epoch_from_units
from climate_forecast.datasets.timeutil import time_from_name

EXTS = (".h5", ".hdf5", ".he5", ".nc", ".nc4", ".hdf")
MAX_FILES = 200_000
LAT_NAMES = ("lat", "latitude", "nav_lat", "y", "lats")
LON_NAMES = ("lon", "longitude", "nav_lon", "x", "lons")
TIME_NAMES = ("time", "t", "valid_time")
# multiply raw values by this to get mm/h
UNITS = {
    "mm/h": 1.0, "mm/hr": 1.0, "mm h-1": 1.0, "mm hr-1": 1.0, "mm/hour": 1.0,
    "kg m-2 s-1": 3600.0, "kg/m2/s": 3600.0, "kg m**-2 s**-1": 3600.0, "mm/s": 3600.0, "mm s-1": 3600.0,
    "m/h": 1000.0, "m": None, "mm": None, "mm/30min": 2.0, "mm/day": 1 / 24.0, "mm d-1": 1 / 24.0,
}


def allowed_roots():
    env = os.environ.get("INDRA_DATA_ROOTS")
    roots = env.split(":") if env else [os.path.expanduser("~")]
    return [os.path.realpath(r) for r in roots if r]


def safe_path(path):
    """Resolve a user-typed path; refuse anything outside the allowed roots."""
    if not path:
        raise ValueError("Enter the folder (or file) that holds your data.")
    real = os.path.realpath(os.path.expanduser(path.strip()))
    if not any(real == r or real.startswith(r + os.sep) for r in allowed_roots()):
        raise PermissionError(f"Only folders under {', '.join(allowed_roots())} can be read.")
    if not os.path.exists(real):
        raise FileNotFoundError(f"Not found: {real}")
    return real


def list_files(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(EXTS):
                out.append(os.path.join(root, f))
                if len(out) > MAX_FILES:
                    raise ValueError(f"More than {MAX_FILES:,} files; point to a smaller folder.")
    return sorted(out)


def _attr(obj, *names):
    for n in names:
        if n in obj.attrs:
            v = obj.attrs[n]
            if isinstance(v, (bytes, np.bytes_)):
                v = v.decode(errors="replace")
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
                if isinstance(v, bytes):
                    v = v.decode(errors="replace")
            return v
    return None


def _datasets(h):
    out = {}
    h.visititems(lambda name, obj: out.__setitem__(name, obj) if isinstance(obj, h5py.Dataset) else None)
    return out


def _pick(names, dsets, candidates, units_hint):
    for c in candidates:
        for n in names:
            if n.split("/")[-1].lower() == c and dsets[n].ndim == 1:
                return n
    for n in names:
        u = str(_attr(dsets[n], "units") or "").lower()
        if dsets[n].ndim == 1 and units_hint in u:
            return n
    return None


def inspect_file(path):
    """Variables, coordinates and grid of one file."""
    with h5py.File(path, "r") as h:
        d = _datasets(h)
        names = sorted(d)
        lat = _pick(names, d, LAT_NAMES, "degrees_north")
        lon = _pick(names, d, LON_NAMES, "degrees_east")
        tvar = next((n for n in names if n.split("/")[-1].lower() in TIME_NAMES and d[n].size >= 1), None)
        variables = []
        for n in names:
            ds = d[n]
            if n in (lat, lon, tvar) or ds.ndim < 2 or not np.issubdtype(ds.dtype, np.number):
                continue
            variables.append({"name": n, "shape": list(ds.shape), "dtype": str(ds.dtype),
                              "units": str(_attr(ds, "units", "Units") or ""),
                              "long_name": str(_attr(ds, "long_name", "LongName", "standard_name") or "").strip()[:120],
                              "fill": _num(_attr(ds, "_FillValue", "missing_value", "CodeMissingValue"))})
        info = {"variables": variables, "lat_var": lat, "lon_var": lon, "time_var": tvar}
        if lat and lon:
            la, lo = np.asarray(d[lat][:], dtype=np.float64), np.asarray(d[lon][:], dtype=np.float64)
            info.update(nlat=int(la.size), nlon=int(lo.size), lat_range=[float(la.min()), float(la.max())],
                        lon_range=[float(lo.min()), float(lo.max())],
                        dlat=float(abs(np.median(np.diff(la)))) if la.size > 1 else None,
                        dlon=float(abs(np.median(np.diff(lo)))) if lo.size > 1 else None,
                        lat_ascending=bool(la[-1] > la[0]), lon_ascending=bool(lo[-1] > lo[0]))
            for v in variables:
                v["order"] = _order(v["shape"], info["nlat"], info["nlon"])
        if tvar:
            info["time_units"] = str(_attr(d[tvar], "units") or "")
    return info


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _order(shape, nlat, nlon):
    s = tuple(shape[-2:])
    if s == (nlat, nlon):
        return "lat_lon"
    if s == (nlon, nlat):
        return "lon_lat"
    return None


def file_time(path, time_var, time_units):
    """UTC seconds: embedded time with CF units if available, else from the file name."""
    if time_var and time_units:
        epoch, unit_s = _epoch_from_units(time_units)
        if epoch:
            with h5py.File(path, "r") as h:
                v = float(np.asarray(h[time_var][()]).reshape(-1)[0])
            return (epoch.timestamp() + v * (unit_s or 1.0)), "file"
    t = time_from_name(path)
    return t, "filename" if t is not None else None


def inspect(path, max_time_scan=2000):
    """Summary of a folder: files, variables, grid, times (scans up to max_time_scan files for timing)."""
    real = safe_path(path)
    files = list_files(real)
    if not files:
        raise FileNotFoundError("No HDF5 or netCDF4 files found there (.h5 .hdf5 .nc .nc4 .he5).")
    try:
        info = inspect_file(files[0])
    except OSError:
        raise ValueError("The files could not be opened as HDF5/netCDF4 (netCDF3 classic is not supported; "
                         "convert with `nccopy -k nc4 in.nc out.nc`).")
    step = max(1, len(files) // max_time_scan)
    sample = files[::step]
    head = files[:200]                                          # consecutive files give the time step
    times, source = [], None
    for f in sample:
        try:
            t, source = file_time(f, info.get("time_var"), info.get("time_units"))
        except Exception:
            t = None
        if t is not None:
            times.append(t)
    info.update(path=real, n_files=len(files), first_file=os.path.relpath(files[0], real) if os.path.isdir(real) else os.path.basename(real),
                examples=[os.path.basename(f) for f in files[:3]], time_source=source)
    if len(times) >= 2 and step == 1:
        ts = np.sort(np.array(times))
        dt = np.diff(ts)
        interval = float(np.median(dt))
        info.update(start=_iso(ts[0]), end=_iso(ts[-1]), interval_minutes=round(interval / 60, 2),
                    gaps=int(np.sum(dt > interval * 1.5)), duplicates=int(np.sum(dt == 0)))
    elif times:
        ts = np.sort(np.array(times))
        ht = []
        for f in head:
            try:
                t, _ = file_time(f, info.get("time_var"), info.get("time_units"))
                if t is not None:
                    ht.append(t)
            except Exception:
                pass
        dts = np.diff(np.sort(np.array(ht))) if len(ht) > 1 else np.array([])
        interval = float(np.median(dts[dts > 0])) if (dts > 0).any() else None
        info.update(start=_iso(ts[0]), end=_iso(ts[-1]), interval_minutes=round(interval / 60, 2) if interval else None,
                    gaps=None, note=f"Period from {len(times)} of {len(files):,} files; gaps are checked during import.")
    questions = []
    for v in info["variables"]:
        u = v["units"].strip().lower()
        v["units_factor"] = UNITS.get(u) if u else None
        v["units_known"] = v["units_factor"] is not None
    info["suggested"] = _suggest(info)
    sv = next((v for v in info["variables"] if v["name"] == info["suggested"]), None)
    if sv and not sv["units_known"]:
        questions.append(f"Units of {sv['name']} are " + (f"'{sv['units']}' (an amount, not a rate)" if sv["units"] else "not stated")
                         + ": choose them under Advanced so values become mm/h.")
    if sv and not sv.get("order") and info.get("lat_var"):
        questions.append(f"The grid of {sv['name']} does not match the latitude/longitude sizes; check the preview.")
    if not info.get("lat_var"):
        questions.append("The files have no latitude/longitude: enter the area they cover.")
    if not times:
        questions.append("No times found inside the files or in their names; frames will be ordered by file name.")
    info["questions"] = questions
    return info


def _suggest(info):
    rain = [v for v in info["variables"] if v.get("order") and re.search(r"precip|rain|rr|tp|hem|prcp", v["name"].lower())]
    pool = rain or [v for v in info["variables"] if v.get("order")]
    return pool[0]["name"] if pool else None


def _iso(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def read_field(path, variable, lat_var, lon_var, order, factor=1.0, fill=None, crop=None, downscale=1, box=None):
    """One frame as mm/h on an ascending-lat, ascending-lon grid, returned as [lon, lat] plus its extent.
    Without coordinate variables, `box` = {lat_range, lon_range} (outer edges, south->north / west->east)
    and `order` describe the grid."""
    with h5py.File(path, "r") as h:
        v = h[variable]
        a = np.asarray(v[(0,) * (v.ndim - 2)] if v.ndim > 2 else v[()], dtype=np.float64)   # first time step only
        raw_fill = _num(_attr(v, "_FillValue", "missing_value", "CodeMissingValue"))
        scale, offset = _num(_attr(v, "scale_factor")), _num(_attr(v, "add_offset"))
        if lat_var and lon_var:
            la, lo = np.asarray(h[lat_var][:], dtype=np.float64), np.asarray(h[lon_var][:], dtype=np.float64)
    if fill is None:
        fill = raw_fill
    if fill is not None:
        a[a == fill] = np.nan                                   # before unpacking: fill is in packed units
    if scale is not None or offset is not None:                 # CF packed values
        a = a * (scale if scale is not None else 1.0) + (offset or 0.0)
    fill = None
    if not (lat_var and lon_var):
        if not box:
            raise ValueError("These files have no latitude/longitude variables: enter the region they cover.")
        n_lat, n_lon = (a.shape[0], a.shape[1]) if order == "lat_lon" else (a.shape[1], a.shape[0])
        (s_, n_), (w_, e_) = box["lat_range"], box["lon_range"]
        la = s_ + (np.arange(n_lat) + 0.5) * (n_ - s_) / n_lat
        lo = w_ + (np.arange(n_lon) + 0.5) * (e_ - w_) / n_lon
    if order == "lat_lon":
        a = a.T                                   # -> [lon, lat]
    if la[0] > la[-1]:
        la, a = la[::-1], a[:, ::-1]
    if lat_var and lon_var and lo.max() > 180:                  # 0..360 -> -180..180
        lo = np.where(lo > 180, lo - 360, lo)
        idx = np.argsort(lo, kind="stable"); lo, a = lo[idx], a[idx, :]
    if lo[0] > lo[-1]:
        lo, a = lo[::-1], a[::-1, :]
    if fill is not None:
        a[a == fill] = np.nan
    a[~np.isfinite(a)] = 0.0
    a = np.clip(a * float(factor), 0, None)
    if crop:
        (la0, la1), (lo0, lo1) = crop
        ja, io = (la >= la0) & (la <= la1), (lo >= lo0) & (lo <= lo1)
        if not ja.any() or not io.any():
            raise ValueError("The chosen region is outside the data.")
        a, la, lo = a[np.ix_(io, ja)], la[ja], lo[io]
    if downscale > 1:
        k = int(downscale)
        n0, n1 = (a.shape[0] // k) * k, (a.shape[1] // k) * k
        a = a[:n0, :n1].reshape(n0 // k, k, n1 // k, k).mean((1, 3))
        la, lo = la[:n1].reshape(-1, k).mean(1), lo[:n0].reshape(-1, k).mean(1)
    half_lat = abs(la[1] - la[0]) / 2 if la.size > 1 else 0.05
    half_lon = abs(lo[1] - lo[0]) / 2 if lo.size > 1 else 0.05
    ext = {"lat_range": [float(la[0] - half_lat), float(la[-1] + half_lat)],
           "lon_range": [float(lo[0] - half_lon), float(lo[-1] + half_lon)]}
    return a.astype(np.float32), ext


def to_u8(mm, clip=100.0):
    return np.clip(np.rint(np.log1p(np.clip(mm, 0, clip)) / math.log1p(clip) * 255), 0, 255).astype(np.uint8)


def preview(params):
    real = safe_path(params["path"])
    files = list_files(real)
    i = min(max(int(params.get("index", 0)), 0), len(files) - 1)
    mm, ext = read_field(files[i], params["variable"], params.get("lat_var"), params.get("lon_var"), params["order"],
                         float(params.get("factor", 1.0)), _num(params.get("fill")), box=params.get("box"))
    stats = {"max_mm_h": float(mm.max()), "mean_mm_h": float(mm.mean()), "wet_fraction": float((mm >= 0.1).mean()),
             "p99_mm_h": float(np.percentile(mm, 99))}
    return to_u8(mm), ext, os.path.basename(files[i]), stats


def _import_one(args):
    path, out_dir, p = args
    try:
        return _import_one_unsafe(path, out_dir, p)
    except Exception as e:                                  # one bad file must not stop the import
        return ("error", os.path.basename(path), str(e)[:200])


def _import_one_unsafe(path, out_dir, p):
    t, _ = file_time(path, p.get("time_var"), p.get("time_units"))
    if t is None:
        return None
    mm, ext = read_field(path, p["variable"], p.get("lat_var"), p.get("lon_var"), p["order"], p["factor"], p.get("fill"),
                         p.get("crop"), int(p.get("downscale", 1)), box=p.get("box"))
    norm = np.log1p(np.clip(mm, 0, p["clip"])) / math.log1p(p["clip"])
    name = datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d%H%M") + ".h5"
    with h5py.File(os.path.join(out_dir, name), "w") as h:
        h.create_dataset("precipitation", data=norm.astype(np.float32), compression="gzip", compression_opts=4)
        ds = h.create_dataset("time", data=float(t))
        ds.attrs["units"] = "seconds since 1970-01-01 00:00:00 UTC"
    return t, ext, list(norm.shape)


def import_dataset(p, base_dir, progress=None, cancelled=lambda: False):
    """Write prepared frames (normalised rain, ascending lat/lon, [lon, lat]) and dataset.yaml."""
    from concurrent.futures import ProcessPoolExecutor
    import yaml
    real = safe_path(p["path"])
    files = list_files(real)
    out_dir = os.path.join(base_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(f, out_dir, p) for f in files]
    done, ext, shape, times, bad = 0, None, None, [], []
    with ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) - 1))) as ex:
        for r in ex.map(_import_one, jobs, chunksize=16):
            done += 1
            if r and r[0] == "error":
                bad.append(r[1:])
            elif r:
                times.append(r[0]); ext, shape = r[1], r[2]
            if progress and done % 25 == 0:
                progress(done / len(jobs))
            if cancelled():
                raise InterruptedError("cancelled")
    if not times:
        raise ValueError("No frame could be imported: no time stamps found in the files or their names.")
    meta = {"name": p.get("title") or os.path.basename(base_dir), "lat_range": [round(v, 4) for v in ext["lat_range"]],
            "lon_range": [round(v, 4) for v in ext["lon_range"]], "grid": shape, "channel": "precipitation",
            "time_variable": "time", "clip_value": float(p["clip"]), "interval_minutes": int(round(p.get("interval_minutes") or 30)),
            "array_order": "lon_lat", "time_source": "file", "role": "user",
            "source": f"imported from {real} (variable {p['variable']}, x{p['factor']} to mm/h)"}
    yaml.safe_dump(meta, open(os.path.join(base_dir, "dataset.yaml"), "w"), sort_keys=False)
    if bad:
        with open(os.path.join(base_dir, "import_skipped.txt"), "w") as f:
            f.writelines(f"{n}\t{e}\n" for n, e in bad)
    return {"frames": len(times), "start": _iso(min(times)), "end": _iso(max(times)), "grid": shape,
            "skipped": len(bad), "skipped_examples": [f"{n}: {e}" for n, e in bad[:3]]}
