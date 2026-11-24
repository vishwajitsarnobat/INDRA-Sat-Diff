"""Cache ensemble forecasts on SAC-style validation sequences, for several Knowledge Alignment settings.

Sequences follow the SAC evaluation (archive/sac_code_drive/scripts/test/prediff_test.py): non-overlapping
runs of 13 contiguous frames (7 inputs + 6 targets). A subset is taken with --every, per --year.

usage: python tools/gen_validation.py MODEL DATASET --year 2022 --every 10 \
           [--variants noka ka1 ka2] [--members 4] [--steps 20]
Output: artifacts/results/val_cache/<model>/<dataset>/<variant>/<T0>.npz  (resumable)
  variant noka: no alignment; kaX: alignment target = X * mean context intensity (SAC used 2.0)
"""
import argparse, os, sys, time
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import predictor, registry


def sac_sequences(ds, in_len=7, out_len=6):
    """Start indices (first input frame) of non-overlapping contiguous sequences, as in the SAC script."""
    n, i, out = in_len + out_len, 0, []
    while i + n <= len(ds.times):
        if ds.contiguous(i, n):
            out.append(i)
            i += n
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dataset")
    ap.add_argument("--year", type=int, nargs="*")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--variants", nargs="+", default=["noka", "ka1", "ka2"])
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    ds = registry.datasets()[a.dataset]
    m = registry.model_by_id(a.model)
    seqs = sac_sequences(ds, m["in_len"], m["out_len"])
    if a.year:
        seqs = [i for i in seqs if datetime.fromtimestamp(ds.times[i], timezone.utc).year in a.year]
    seqs = seqs[a.offset::a.every]
    base = os.path.join(registry.ART, "results", "val_cache", a.model, a.dataset)
    print(f"{a.dataset}: {len(seqs)} sequences x {a.variants} -> {base}", flush=True)
    u8 = lambda x: np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)
    for n, i in enumerate(seqs):
        t0 = ds.times[i + m["in_len"] - 1]
        tag = registry.iso(t0).replace(":", "")
        for v in a.variants:
            path = os.path.join(base, v, tag + ".npz")
            if os.path.exists(path):
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            st = time.time()
            r = predictor.predict(a.model, ds.id, t0, use_alignment=(v != "noka"), members=a.members,
                                  steps=a.steps, precision="fp16", seed=1000 + n,
                                  ka_multiplier=float(v[2:]) if v != "noka" else 1.0)
            np.savez_compressed(path, ensemble=u8(r["ensemble"]), truth=u8(r["truth"]), last=u8(r["context"][-1]))
            print(f"[{n + 1}/{len(seqs)}] {tag} {v} {time.time() - st:.0f}s", flush=True)


if __name__ == "__main__":
    main()
