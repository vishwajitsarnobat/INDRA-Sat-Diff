"""Inference-time post-processing study on cached ensembles (see tools/gen_ensembles.py).

Tunes on one dataset, reports on another (no peeking). Methods, all from the same diffusion samples:
  member      a single sample (what the SAC evaluation used)
  ens_mean    ensemble mean, thresholded
  prob@p      rain where P(member >= thr) >= p   (p tuned per lead on the tuning set)
  pm_cov      probability matching: keep the ensemble-mean ranking, but set the rain area so it
              matches the observed rain area of the last input frame (removes the area over-forecast)
Scores: POD / FAR / CSI pooled over all pixels and windows, per lead time, at each threshold.

usage: python tools/postprocess_study.py CACHE_DIR_TUNE CACHE_DIR_TEST [--thr 1.0 2.5] [--out DIR]
"""
import argparse, glob, json, math, os
import numpy as np

CLIP = 100.0
to_mm = lambda u: np.expm1(u.astype(np.float32) / 255.0 * math.log1p(CLIP))


def load(cache_dir):
    items = []
    for f in sorted(glob.glob(os.path.join(cache_dir, "*.npz"))):
        z = np.load(f)
        items.append({"ens": to_mm(z["ensemble"]), "truth": to_mm(z["truth"]), "last": to_mm(z["last"]),
                      "t0": os.path.basename(f)[:-4]})
    return items


def counts(pred, obs):
    return np.array([np.sum(pred & obs), np.sum(pred & ~obs), np.sum(~pred & obs)], dtype=np.float64)


def scores(c):
    h, fa, mi = c
    f = lambda a, b: float(a / b) if b else float("nan")
    return {"pod": f(h, h + mi), "far": f(fa, h + fa), "csi": f(h, h + mi + fa), "bias": f(h + fa, h + mi)}


def predict_masks(it, thr, method, p=None):
    ens, last = it["ens"], it["last"]          # ens: M T H W
    if method == "member":
        return ens[0] >= thr
    if method == "ens_mean":
        return ens.mean(0) >= thr
    if method == "prob":
        return (ens >= thr).mean(0) >= p
    if method == "pm_cov":
        mean = ens.mean(0)
        frac = float((last >= thr).mean())    # observed rain fraction at T0
        out = np.zeros_like(mean, dtype=bool)
        for k in range(mean.shape[0]):
            n = int(round(frac * mean[k].size))
            if n > 0:
                cut = np.partition(mean[k].ravel(), -n)[-n]
                out[k] = (mean[k] >= cut) & (mean[k] > 0)
        return out
    raise ValueError(method)


def evaluate(items, thr, method, p_per_lead=None):
    T = items[0]["truth"].shape[0]
    c = np.zeros((T, 3))
    for it in items:
        obs = it["truth"] >= thr
        for k in range(T):
            p = None if p_per_lead is None else p_per_lead[k]
            pred = predict_masks(it, thr, method, p) if method != "prob" else ((it["ens"][:, k] >= thr).mean(0) >= p)
            c[k] += counts(pred if method == "prob" else pred[k], obs[k])
    return c


def persistence(items, thr):
    T = items[0]["truth"].shape[0]
    c = np.zeros((T, 3))
    for it in items:
        for k in range(T):
            c[k] += counts(it["last"] >= thr, it["truth"][k] >= thr)
    return c


def tune_prob(items, thr):
    M = items[0]["ens"].shape[0]
    grid = [(j + 0.5) / M for j in range(M)]          # thresholds between member counts
    T = items[0]["truth"].shape[0]
    best = []
    for k in range(T):
        sc = []
        for p in grid:
            c = np.zeros(3)
            for it in items:
                c += counts((it["ens"][:, k] >= thr).mean(0) >= p, it["truth"][k] >= thr)
            sc.append((scores(c)["csi"], p))
        best.append(max(sc)[1])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tune"); ap.add_argument("test")
    ap.add_argument("--thr", type=float, nargs="+", default=[1.0, 2.5])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tune, test = load(a.tune), load(a.test)
    print(f"tune: {len(tune)} windows ({a.tune})\ntest: {len(test)} windows ({a.test}), members={test[0]['ens'].shape[0]}")
    report = {}
    for thr in a.thr:
        p_best = tune_prob(tune, thr)
        rows = {"persistence": persistence(test, thr), "member": evaluate(test, thr, "member"),
                "ens_mean": evaluate(test, thr, "ens_mean"), "pm_cov": evaluate(test, thr, "pm_cov"),
                "prob_tuned": evaluate(test, thr, "prob", p_best)}
        report[str(thr)] = {"p_tuned": p_best, **{k: [scores(c) for c in v] for k, v in rows.items()}}
        print(f"\n>= {thr} mm/h   tuned probability per lead: {[round(p, 2) for p in p_best]}")
        print(f"{'method':12s} " + " ".join(f"+{30 * (k + 1):>3d}" for k in range(6)) + "   (CSI)   | POD | FAR at +90")
        for name, c in rows.items():
            s = [scores(x) for x in c]
            print(f"{name:12s} " + " ".join(f"{x['csi']:.3f}" for x in s) + f"   | {s[2]['pod']:.2f} | {s[2]['far']:.2f}")
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(report, open(os.path.join(a.out, "postprocess_report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
