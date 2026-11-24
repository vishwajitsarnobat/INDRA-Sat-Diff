# climate_forecast/pipelines/download.py
"""Download NASA GPM IMERG half-hourly files for a region via OPeNDAP server-side subsetting.

Only the requested lat/lon box is transferred, so a 128x128 region costs ~10-30 KB per frame
instead of ~8 MB for the global file. Credentials are read from ~/.netrc
(machine urs.earthdata.nasa.gov login <user> password <pass>).
"""
import math
import netrc
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import requests
from tqdm import tqdm

URS = "urs.earthdata.nasa.gov"
PRODUCTS = {
    # name: (opendap dir, file pattern, grid origin lat, grid origin lon, resolution)
    "IMERG_FINAL_HH_V07": ("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHH.07",
                           "3B-HHR.MS.MRG.3IMERG.{d}-S{s}-E{e}.{m:04d}.V07B.HDF5", -89.95, -179.95, 0.1),
    "IMERG_LATE_HH_V07": ("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHHL.07",
                          "3B-HHR-L.MS.MRG.3IMERG.{d}-S{s}-E{e}.{m:04d}.V07B.RT-H5", -89.95, -179.95, 0.1),
    "IMERG_EARLY_HH_V07": ("https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGHHE.07",
                           "3B-HHR-E.MS.MRG.3IMERG.{d}-S{s}-E{e}.{m:04d}.V07B.RT-H5", -89.95, -179.95, 0.1),
}
NLAT, NLON = 1800, 3600


def _earthdata_session():
    """requests re-applies ~/.netrc credentials when redirected to the Earthdata login host,
    so the password is only ever sent to urs.earthdata.nasa.gov."""
    try:
        auth = netrc.netrc().authenticators(URS)
    except (FileNotFoundError, netrc.NetrcParseError):
        auth = None
    if not auth:
        raise RuntimeError(f"No credentials for {URS} in ~/.netrc. Add the line "
                           f"'machine {URS} login <user> password <pass>' and run chmod 600 ~/.netrc")
    return requests.Session()


def _index_range(lo, hi, origin, res, n):
    """Grid-cell indices whose centres fall inside [lo, hi] (same rule as preprocess_logic)."""
    i0 = math.ceil(round((lo - origin) / res, 6))
    i1 = math.floor(round((hi - origin) / res, 6))
    i0, i1 = max(i0, 0), min(i1, n - 1)
    if i1 < i0:
        raise ValueError(f"Empty index range for [{lo}, {hi}]")
    return i0, i1


def _timesteps(start, end, minutes=30):
    t = start
    while t <= end:
        yield t
        t += timedelta(minutes=minutes)


def _parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _url_and_name(product, t, lat_idx, lon_idx):
    base, pattern, *_ = PRODUCTS[product]
    s = t.strftime("%H%M%S")
    e = (t + timedelta(minutes=29, seconds=59)).strftime("%H%M%S")
    fname = pattern.format(d=t.strftime("%Y%m%d"), s=s, e=e, m=t.hour * 60 + t.minute)
    path = f"{base}/{t:%Y}/{t.timetuple().tm_yday:03d}/{fname}"
    (a0, a1), (o0, o1) = lat_idx, lon_idx
    ce = (f"precipitation[0:0][{o0}:{o1}][{a0}:{a1}],"
          f"lat[{a0}:{a1}],lon[{o0}:{o1}],time[0:0]")
    return f"{path}.nc4?{ce}", fname + ".nc4"


def _fetch(session, url, out_path, retries=4):
    tmp = out_path + ".part"
    for attempt in range(retries):
        try:
            with session.get(url, timeout=120, stream=True) as r:
                if r.status_code in (401, 403):
                    raise PermissionError(
                        f"HTTP {r.status_code}: check ~/.netrc and that 'NASA GESDISC DATA ARCHIVE' is "
                        f"approved under Applications > Authorized Apps in your Earthdata profile.")
                if r.status_code == 404:
                    return "missing"
                r.raise_for_status()
                if "html" in r.headers.get("content-type", ""):
                    raise RuntimeError("Got an HTML page instead of data (login redirect failed).")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 16):
                        f.write(chunk)
            os.replace(tmp, out_path)
            return "ok"
        except PermissionError:
            raise
        except Exception:
            if attempt == retries - 1:
                if os.path.exists(tmp):
                    os.remove(tmp)
                return "failed"
            time.sleep(2 ** attempt)


def run(config: dict):
    d = config["download"]
    product = d.get("product", "IMERG_FINAL_HH_V07")
    if product not in PRODUCTS:
        raise ValueError(f"Unknown product '{product}'. Choose from {list(PRODUCTS)}")
    _, _, lat0, lon0, res = PRODUCTS[product]
    lat_range = d.get("lat_range") or config["preprocess"]["lat_range"]
    lon_range = d.get("lon_range") or config["preprocess"]["lon_range"]
    out_dir = d.get("output_dir") or config["preprocess"]["input_dir"]
    lat_idx = _index_range(lat_range[0], lat_range[1], lat0, res, NLAT)
    lon_idx = _index_range(lon_range[0], lon_range[1], lon0, res, NLON)

    periods = d.get("periods") or [{"start": d["start"], "end": d["end"]}]
    times = sorted({t for p in periods for t in _timesteps(_parse_time(p["start"]), _parse_time(p["end"]))})

    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for t in times:
        url, name = _url_and_name(product, t, lat_idx, lon_idx)
        out_path = os.path.join(out_dir, name)
        if not (os.path.exists(out_path) and os.path.getsize(out_path) > 0):
            jobs.append((url, out_path))

    print(f"{product}: {len(times)} timesteps, region lat {lat_range} lon {lon_range} "
          f"-> {lat_idx[1] - lat_idx[0] + 1}x{lon_idx[1] - lon_idx[0] + 1} cells; "
          f"{len(times) - len(jobs)} already present, {len(jobs)} to download into '{out_dir}'.")
    if not jobs:
        return

    session = _earthdata_session()
    stats = {"ok": 0, "missing": 0, "failed": 0}
    failed = []
    with ThreadPoolExecutor(max_workers=int(d.get("workers", 6))) as ex:
        futs = {ex.submit(_fetch, session, url, path): path for url, path in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Downloading", unit="file"):
            status = fut.result()
            stats[status] += 1
            if status != "ok":
                failed.append((status, os.path.basename(futs[fut])))
    print(f"Done: {stats['ok']} downloaded, {stats['missing']} not on server, {stats['failed']} failed.")
    if failed:
        log = os.path.join(out_dir, "download_failures.txt")
        with open(log, "w") as f:
            f.writelines(f"{s}\t{n}\n" for s, n in failed)
        print(f"See {log}; re-run the same command to retry.")

