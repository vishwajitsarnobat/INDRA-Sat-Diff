"""Discovery of datasets and models under artifacts/, with cached time indexes."""
import glob
import json
import math
import re
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import h5py
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")
DATA_DIR = os.path.join(ART, "data")
MODEL_DIR = os.path.join(ART, "models")
JOB_DIR = os.path.join(ART, "jobs")
for d in (DATA_DIR, MODEL_DIR, JOB_DIR):
    os.makedirs(d, exist_ok=True)


def rel(path):
    return os.path.relpath(path, ROOT)


# ----------------------------------------------------------------------------- datasets
@dataclass
class Dataset:
    id: str
    path: str                     # processed directory
    meta: dict
    files: list = field(default_factory=list)
    times: list = field(default_factory=list)   # unix seconds, same order as files
    sig: list = field(default_factory=list)     # index signature (file count, newest mtime)

    @property
    def interval(self):
        return int(self.meta.get("interval_minutes", 30)) * 60

    def summary(self):
        gaps = self.gaps()
        return {
            "id": self.id, "name": self.meta.get("name", self.id),
            "lat_range": self.meta.get("lat_range"), "lon_range": self.meta.get("lon_range"),
            "grid": self.meta.get("grid"), "count": len(self.files),
            "start": iso(self.times[0]) if self.times else None,
            "end": iso(self.times[-1]) if self.times else None,
            "interval_minutes": self.interval // 60, "gaps": len(gaps),
            "clip_value": self.meta.get("clip_value", 100.0),
            "array_order": self.meta.get("array_order", "lon_lat"),
            "source": self.meta.get("source"), "note": self.meta.get("note"),
            "version": self.version, "has_raw": os.path.isdir(os.path.join(os.path.dirname(self.path), "raw")),
        }

    @property
    def version(self):
        """Short token that changes whenever the files change (used in cache keys / ETags)."""
        return f"{int(self.sig[0])}-{int(self.sig[1])}" if self.sig else "0"

    def gaps(self):
        out = []
        for a, b in zip(self.times, self.times[1:]):
            if b - a > self.interval * 1.5:
                out.append([iso(a), iso(b)])
        return out

    def index_of(self, t):
        """Index of the frame at unix time t (exact match within half an interval)."""
        if not self.times:
            raise KeyError("empty dataset")
        i = int(np.searchsorted(self.times, t))
        best = min((j for j in (i - 1, i) if 0 <= j < len(self.times)), key=lambda j: abs(self.times[j] - t))
        if abs(self.times[best] - t) > self.interval / 2:
            raise KeyError(f"no frame at {iso(t)}")
        return best

    def contiguous(self, i0, n):
        """True if frames i0..i0+n-1 exist and are evenly spaced."""
        if i0 < 0 or i0 + n > len(self.times):
            return False
        seg = self.times[i0:i0 + n]
        return all(abs((b - a) - self.interval) < 1 for a, b in zip(seg, seg[1:]))


def iso(t):
    return datetime.fromtimestamp(float(t), timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()


def _time_from_name(name):
    """Unix time from a file name holding YYYYMMDDHHMM (SAC files) or IMERG's YYYYMMDD-SHHMMSS."""
    m = re.search(r"(\d{8})-S(\d{4})", name) or re.search(r"(\d{8})(\d{4})", name)
    if not m:
        return None
    return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp()


def _build_index(path, time_var, time_source="file"):
    files = sorted(f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)
                   and f.lower().endswith((".h5", ".hdf5", ".nc", ".nc4")))
    pairs = []
    if time_source == "filename":
        for f in files:
            t = _time_from_name(os.path.basename(f))
            if t is not None:
                pairs.append((t, os.path.basename(f)))
        return sorted(pairs)
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                pairs.append((float(np.asarray(h[time_var][()]).reshape(-1)[0]), os.path.basename(f)))
        except Exception:
            continue
    pairs.sort()
    return pairs


def load_dataset(ds_dir):
    meta_path = os.path.join(ds_dir, "dataset.yaml")
    meta = yaml.safe_load(open(meta_path)) if os.path.exists(meta_path) else {}
    path = os.path.join(ds_dir, "processed")
    if not os.path.isdir(path):
        return None
    time_var = meta.get("time_variable", "Grid/time")
    idx_path = os.path.join(ds_dir, ".index.json")
    sig = [len(os.listdir(path)), max((os.path.getmtime(os.path.join(path, f)) for f in os.listdir(path)), default=0)]
    pairs = None
    if os.path.exists(idx_path):
        cached = json.load(open(idx_path))
        if cached.get("sig") == sig:
            pairs = cached["pairs"]
    if pairs is None:
        pairs = _build_index(path, time_var, meta.get("time_source", "file"))
        json.dump({"sig": sig, "pairs": pairs}, open(idx_path, "w"))
    ds = Dataset(id=os.path.basename(ds_dir), path=path, meta=meta, sig=sig)
    ds.times = [p[0] for p in pairs]
    ds.files = [os.path.join(path, p[1]) for p in pairs]
    if not meta.get("grid") and ds.files:
        with h5py.File(ds.files[0], "r") as h:
            meta["grid"] = list(h[meta.get("channel", "Grid/precipitation")].shape[-2:])
    return ds


_datasets = {}


def datasets(refresh=False):
    if refresh or not _datasets:
        _datasets.clear()
        for d in sorted(glob.glob(os.path.join(DATA_DIR, "*"))):
            if os.path.isdir(d):
                ds = load_dataset(d)
                if ds:
                    _datasets[ds.id] = ds
    return _datasets


@lru_cache(maxsize=512)
def read_frame(path, channel):
    with h5py.File(path, "r") as h:
        return np.asarray(h[channel][:], dtype=np.float32).squeeze()


def frame_u8(ds: Dataset, i):
    a = read_frame(ds.files[i], ds.meta.get("channel", "Grid/precipitation"))
    return np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- frame cube
# All frames of a dataset as one uint8 file (N*H*W, value*255) so a range of frames is a slice of a memmap.
_cubes = {}
_cube_lock = threading.Lock()


def cube(ds: Dataset):
    """Read-only memmap (N, n0, n1) uint8 for the dataset; built on first use, rebuilt when the index changes."""
    hit = _cubes.get(ds.id)
    if hit and hit[0] == ds.sig:
        return hit[1]
    with _cube_lock:
        hit = _cubes.get(ds.id)
        if hit and hit[0] == ds.sig:
            return hit[1]
        base = os.path.dirname(ds.path)
        bin_path, meta_path = os.path.join(base, "cube.u8"), os.path.join(base, "cube.json")
        n = len(ds.files)
        if n == 0:
            raise KeyError("empty dataset")
        meta = None
        if os.path.exists(meta_path) and os.path.exists(bin_path):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                meta = None
        ok = (meta and meta.get("sig") == ds.sig and meta.get("count") == n
              and os.path.getsize(bin_path) == n * meta["shape"][0] * meta["shape"][1])
        if not ok:
            ch = ds.meta.get("channel", "Grid/precipitation")
            with h5py.File(ds.files[0], "r") as h:
                shape = list(np.asarray(h[ch]).squeeze().shape)
            tmp = bin_path + ".tmp"
            mm = np.memmap(tmp, dtype=np.uint8, mode="w+", shape=(n, *shape))
            for i, f in enumerate(ds.files):
                with h5py.File(f, "r") as h:
                    a = np.asarray(h[ch][:], dtype=np.float32).squeeze()
                mm[i] = np.clip(np.rint(np.nan_to_num(a) * 255.0), 0, 255).astype(np.uint8)
            mm.flush()
            del mm
            os.replace(tmp, bin_path)
            meta = {"sig": ds.sig, "count": n, "shape": shape, "dtype": "uint8", "scale": 255}
            json.dump(meta, open(meta_path, "w"))
        mm = np.memmap(bin_path, dtype=np.uint8, mode="r", shape=(n, *meta["shape"]))
        _cubes[ds.id] = (ds.sig, mm)
        return mm


def sequence(ds: Dataset, i_last, n_before, n_after=0):
    """Normalised frames [i_last-n_before+1 .. i_last+n_after] as (T, H, W) float32."""
    ch = ds.meta.get("channel", "Grid/precipitation")
    return np.stack([read_frame(ds.files[j], ch) for j in range(i_last - n_before + 1, i_last + n_after + 1)])


# ----------------------------------------------------------------------------- models
def _metric_files(d):
    return sorted(rel(p) for p in glob.glob(os.path.join(d, "**", "*metrics*.json"), recursive=True))


def models(include_partial=False):
    out = []
    for d in sorted(glob.glob(os.path.join(MODEL_DIR, "*"))):
        cfg_path = os.path.join(d, "indra_sat_diff", "resolved_config.yaml")
        if not os.path.exists(cfg_path):
            card_path = os.path.join(d, "model_card.yaml")
            if include_partial and os.path.isdir(d) and os.path.exists(card_path):
                card = yaml.safe_load(open(card_path)) or {}
                out.append({"id": os.path.basename(d), "dir": rel(d), "name": card.get("name", os.path.basename(d)),
                            "usable": False, "card": card, "metrics": _metric_files(d),
                            "compute_level": card.get("compute_level"),
                            "size_mb": round(sum(os.path.getsize(f) for f in glob.glob(
                                os.path.join(d, "**", "*.pt"), recursive=True)) / 2**20)})
            continue
        cfg = yaml.safe_load(open(cfg_path))
        card_path = os.path.join(d, "model_card.yaml")
        card = yaml.safe_load(open(card_path)) if os.path.exists(card_path) else {}
        weights = cfg["pipeline"].get("model_checkpoint_path")
        size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
        lay = cfg["layout"]
        out.append({
            "id": os.path.basename(d), "dir": rel(d), "config": rel(cfg_path), "weights": weights,
            "name": card.get("name", os.path.basename(d)),
            "lat_range": cfg.get("preprocess", {}).get("lat_range"),
            "lon_range": cfg.get("preprocess", {}).get("lon_range"),
            "grid": [lay["img_height"], lay["img_width"]], "in_len": lay["in_len"], "out_len": lay["out_len"],
            "compute_level": cfg.get("compute_level", card.get("compute_level", 1)),
            "latent_channels": cfg["model"]["vae"]["latent_channels"],
            "timesteps": cfg["model"]["diffusion"]["timesteps"],
            "size_mb": round(size / 2**20), "card": card, "usable": True,
            "metrics": _metric_files(d),
        })
    return out


def curves(model):
    """Training curves per stage. metrics_history.json is a list of {epoch, val: {...}, train: {...}};
    returns {stage: {"epochs": [...], "series": {"val/loss": [...], ...}}} with gaps as None."""
    out = {}
    for p in model["metrics"]:
        try:
            hist = json.load(open(os.path.join(ROOT, p)))
        except Exception:
            continue
        if isinstance(hist, dict):
            hist = hist.get("history") or hist.get("epochs") or []
        rows = {}
        for k, r in enumerate(hist if isinstance(hist, list) else []):
            if not isinstance(r, dict):
                continue
            flat = rows.setdefault(r.get("epoch", k), {})
            for key, v in r.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, (int, float)):
                            flat[f"{key}/{kk}"] = vv
                elif key != "epoch" and isinstance(v, (int, float)):
                    flat[key] = v
        if not rows:
            continue
        epochs = sorted(rows)
        names = sorted({k for r in rows.values() for k in r})
        stage = os.path.basename(os.path.dirname(p)) or p
        out[stage] = {"path": p, "epochs": epochs, "series": {n: [rows[e].get(n) for e in epochs] for n in names}}
    return out


def model_by_id(mid):
    for m in models(include_partial=True):
        if m["id"] == mid:
            return m
    raise KeyError(mid)


def to_mm(v, clip=100.0):
    return np.expm1(np.asarray(v) * math.log1p(clip))


def from_mm(mm, clip=100.0):
    return np.log1p(np.clip(mm, 0, clip)) / math.log1p(clip)


# ----------------------------------------------------------------------------- coastlines
_geo = {}


def _geo_lines(name):
    if name not in _geo:
        fc = json.load(open(os.path.join(ROOT, "app", "static", "vendor", f"{name}.geojson")))
        lines = []
        for f in fc["features"]:
            g = f["geometry"]
            for c in ([g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]):
                lines.append(np.asarray(c, dtype=np.float32)[:, :2])
        _geo[name] = lines
    return _geo[name]


@lru_cache(maxsize=32)
def geo_lines(lon0, lat0, lon1, lat1):
    """Coastlines and borders inside the box, as lists of flat [lon, lat, lon, lat, ...] polylines."""
    out = {}
    for key, name in (("coast", "ne_50m_coastline"), ("border", "ne_50m_admin_0_boundary_lines_land"),
                      ("india", "india_states")):
        parts = []
        for line in _geo_lines(name):
            inside = (line[:, 0] >= lon0) & (line[:, 0] <= lon1) & (line[:, 1] >= lat0) & (line[:, 1] <= lat1)
            if not inside.any():
                continue
            keep = inside.copy()          # keep one point either side so lines reach the box edge
            keep[1:] |= inside[:-1]
            keep[:-1] |= inside[1:]
            idx = np.flatnonzero(keep)
            for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
                if len(run) > 1:
                    parts.append(np.round(line[run].reshape(-1).astype(float), 3).tolist())
        out[key] = parts
    return out
