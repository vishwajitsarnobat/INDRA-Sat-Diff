"""Generate and cache ensemble forecasts for every complete window of a dataset (for post-processing research).

usage: python tools/gen_ensembles.py MODEL DATASET [--members 6] [--steps 25] [--stride 2] [--no-ka]
Writes artifacts/results/ens_cache/<model>[_noka]/<dataset>/<t0>.npz with uint8 ensemble (M,T,H,W),
truth (T,H,W) and the last context frame. Existing files are skipped, so it can be resumed.
"""
import argparse, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import predictor, registry

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("dataset")
ap.add_argument("--members", type=int, default=6); ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--stride", type=int, default=2); ap.add_argument("--no-ka", action="store_true")
a = ap.parse_args()
ds = registry.datasets()[a.dataset]
m = registry.model_by_id(a.model)
out = os.path.join(registry.ART, "results", "ens_cache", a.model + ("_noka" if a.no_ka else ""), a.dataset)
os.makedirs(out, exist_ok=True)
starts = [i for i in range(m["in_len"] - 1, len(ds.times) - m["out_len"], a.stride)
          if ds.contiguous(i - m["in_len"] + 1, m["in_len"] + m["out_len"])]
print(f"{len(starts)} windows -> {out}", flush=True)
u8 = lambda x: np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
for n, i in enumerate(starts):
    path = os.path.join(out, registry.iso(ds.times[i]).replace(":", "") + ".npz")
    if os.path.exists(path):
        continue
    t = time.time()
    r = predictor.predict(a.model, ds.id, ds.times[i], use_alignment=not a.no_ka, members=a.members,
                          steps=a.steps, precision="fp16", seed=n)
    np.savez_compressed(path, ensemble=u8(r["ensemble"]), truth=u8(r["truth"]), last=u8(r["context"][-1]))
    print(f"[{n + 1}/{len(starts)}] {registry.iso(ds.times[i])} {time.time() - t:.0f}s", flush=True)
