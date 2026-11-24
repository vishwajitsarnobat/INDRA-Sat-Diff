"""PySTEPS baseline on exactly the sequences cached by tools/gen_validation.py (run with .venv-steps).

Reproduces artifacts/archive/sac_code_drive/scripts/test/pysteps_test.py: Lucas-Kanade motion from the 7 input frames,
STEPS with 50 members, precip_thr 1.0 mm/h, rain rates in mm/h, seed 42, vel_pert_method None.
Saves the ensemble mean (what SAC scored) and P(>=1 mm/h) so the same post-processing can be tuned for it.

usage: .venv-steps/bin/python tools/run_pysteps.py MODEL DATASET [--workers 4] [--members 50]
Output: artifacts/results/val_cache/<MODEL>/<DATASET>/pysteps/<T0>.npz
"""
import argparse, glob, math, os, re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone, timedelta
import numpy as np
import h5py

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP, THR = 100.0, 1.0


def frame(path):
    with h5py.File(path, "r") as h:
        key = "precipitationCal" if "precipitationCal" in h else "Grid/precipitation"
        a = np.asarray(h[key][:], dtype=np.float64).squeeze()
    a[~np.isfinite(a)] = 0.0
    return np.expm1(a * math.log1p(CLIP))          # mm/h, array order [lon, lat] (same for all inputs)


def run_one(args):
    files, out, members = args
    if os.path.exists(out):
        return out, "cached"
    from pysteps import motion, nowcasts
    seq = np.stack([frame(f) for f in files])     # 13 frames
    train, truth = seq[:7], seq[7:]
    v = motion.get_method("LK")(train)
    ens = nowcasts.get_method("steps")(train, v, 6, members, kmperpixel=0.1 * 111.0, timestep=30,
                                       precip_thr=THR, seed=42, vel_pert_method=None)
    ens = np.nan_to_num(np.asarray(ens), nan=0.0)
    u8 = lambda mm: np.clip(np.rint(np.log1p(np.clip(mm, 0, CLIP)) / math.log1p(CLIP) * 255), 0, 255).astype(np.uint8)
    np.savez_compressed(out, mean=u8(ens.mean(0)), prob=np.rint((ens >= THR).mean(0) * 255).astype(np.uint8),
                        truth=u8(truth), last=u8(train[-1]))
    return out, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dataset")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--members", type=int, default=50)
    a = ap.parse_args()
    cache = os.path.join(ROOT, "artifacts", "results", "val_cache", a.model, a.dataset)
    tags = sorted({os.path.basename(f)[:-4] for f in glob.glob(os.path.join(cache, "*", "*.npz"))
                   if "/pysteps/" not in f})
    src = os.path.join(ROOT, "artifacts", "data", a.dataset, "processed")
    by_time = {}
    for f in os.listdir(src):
        m = re.search(r"(\d{12})", f)
        if m:
            by_time[datetime.strptime(m.group(1), "%Y%m%d%H%M")] = os.path.join(src, f)
    jobs = []
    os.makedirs(os.path.join(cache, "pysteps"), exist_ok=True)
    for tag in tags:
        t0 = datetime.strptime(tag, "%Y-%m-%dT%H%MZ")
        times = [t0 + timedelta(minutes=30 * k) for k in range(-6, 7)]
        if all(t in by_time for t in times):
            jobs.append(([by_time[t] for t in times], os.path.join(cache, "pysteps", tag + ".npz"), a.members))
    print(f"{a.dataset}: {len(jobs)} sequences", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (out, st) in enumerate(ex.map(run_one, jobs), 1):
            if st == "done" or n % 20 == 0:
                print(f"[{n}/{len(jobs)}] {os.path.basename(out)} {st}", flush=True)


if __name__ == "__main__":
    main()
