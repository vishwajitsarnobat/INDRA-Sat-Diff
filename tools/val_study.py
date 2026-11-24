"""Validation study on the cached ensembles of tools/gen_validation.py, using the SAC protocol.

Protocol (as artifacts/archive/sac_code_drive/scripts/test/prediff_test.py): threshold 1.0 mm/h, POD/FAR/CSI and
FSS (1/2/4/8 px) computed per sequence and lead time, then averaged over sequences (nanmean).
Configurations are chosen on the tuning year only and reported on the test year, with bootstrap 95 % CIs
of the difference to persistence (resampling sequences).

Methods per KA variant (noka, ka1, ka2):
  member     first ensemble member (what the SAC evaluation scored)
  mean       ensemble mean
  prob       rain where at least a fraction p of members has rain; p chosen per lead on the tuning year
  pm         ensemble-mean ranking, rain area set to the observed rain area at T0 (probability matching)

usage: python tools/val_study.py MODEL DATASET --tune-year 2022 --test-year 2023 [--out DIR]
"""
import argparse, glob, json, math, os
from datetime import datetime, timezone
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from climate_forecast.diffusion.postprocess import match_intensities  # noqa: E402

CLIP, THR, SCALES = 100.0, 1.0, (1, 2, 4, 8)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mm = lambda u: np.expm1(u.astype(np.float32) / 255.0 * math.log1p(CLIP))


def load(model, dataset, variant, year):
    d = os.path.join(ROOT, "artifacts", "results", "val_cache", model, dataset, variant)
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        tag = os.path.basename(f)[:-4]
        if int(tag[:4]) != year:
            continue
        z = np.load(f)
        out[tag] = {"ens": mm(z["ensemble"]), "truth": mm(z["truth"]), "last": mm(z["last"])}
    return out


def load_steps(model, dataset, year):
    d = os.path.join(ROOT, "artifacts", "results", "val_cache", model, dataset, "pysteps")
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        tag = os.path.basename(f)[:-4]
        if int(tag[:4]) == year:
            z = np.load(f)
            out[tag] = {"mean": mm(z["mean"]), "prob": z["prob"].astype(np.float32) / 255.0,
                        "truth": mm(z["truth"]), "last": mm(z["last"])}
    return out


def steps_seq(items, method, p=None):
    rows = []
    for it in items.values():
        ob = it["truth"] >= THR
        if method == "mean":
            pr = it["mean"] >= THR
        else:
            pr = np.stack([it["prob"][k] >= p[k] for k in range(ob.shape[0])])
        rows.append([[*cat(pr[k], ob[k]), *[fss(pr[k], ob[k], s) for s in SCALES]] for k in range(ob.shape[0])])
    return np.array(rows, dtype=np.float64)


def cat(pred, obs):
    h = np.sum(pred & obs); fa = np.sum(pred & ~obs); mi = np.sum(~pred & obs)
    nan = np.nan
    return (h / (h + mi) if h + mi else nan, fa / (h + fa) if h + fa else nan, h / (h + mi + fa) if h + mi + fa else nan)


def box_mean(x, s):
    if s == 1:
        return x
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    k = s
    out = (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)
    return out


def fss(pred, obs, s):
    pf, of = box_mean(pred.astype(np.float32), s), box_mean(obs.astype(np.float32), s)
    mse = np.mean((pf - of) ** 2); ref = np.mean(pf ** 2) + np.mean(of ** 2)
    return 1.0 if ref == 0 else 1.0 - mse / ref


def masks(it, method, p=None):
    ens = it["ens"]
    if method == "member":
        return ens[0] >= THR
    if method == "mean":
        return ens.mean(0) >= THR
    if method == "prob":
        frac = (ens >= THR).mean(0)
        return np.stack([frac[k] >= p[k] for k in range(frac.shape[0])])
    if method == "pm":
        mean, n = ens.mean(0), int((it["last"] >= THR).sum())
        out = np.zeros_like(mean, dtype=bool)
        for k in range(mean.shape[0]):
            if n > 0:
                cut = np.partition(mean[k].ravel(), -n)[-n]
                out[k] = (mean[k] >= cut) & (mean[k] > 0)
        return out
    if method == "persistence":
        return np.repeat((it["last"] >= THR)[None], it["truth"].shape[0], 0)
    raise ValueError(method)


def per_sequence(items, method, p=None):
    """array (n_seq, T, 3 + len(SCALES)): POD, FAR, CSI, FSS@scales"""
    rows = []
    for it in items.values():
        pr, ob = masks(it, method, p), it["truth"] >= THR
        rows.append([[*cat(pr[k], ob[k]), *[fss(pr[k], ob[k], s) for s in SCALES]] for k in range(ob.shape[0])])
    return np.array(rows, dtype=np.float64)


def csi_only(preds, obss):
    """Mean per-sequence CSI for one lead time (fast path used when tuning thresholds)."""
    v = [cat(p, o)[2] for p, o in zip(preds, obss)]
    return float(np.nanmean(v))


def tune_p(items):
    M = next(iter(items.values()))["ens"].shape[0]
    grid = [(j + 0.5) / M for j in range(M)]
    T = next(iter(items.values()))["truth"].shape[0]
    best = []
    for k in range(T):
        obs = [it["truth"][k] >= THR for it in items.values()]
        fr = [(it["ens"][:, k] >= THR).mean(0) for it in items.values()]
        sc = [(csi_only([f >= p for f in fr], obs), p) for p in grid]
        best.append(max(sc)[1])
    return best


def extra_scores(items, steps, method, p):
    """Heavier-rain CSI, Brier score (1 mm/h) and amount errors on the test sequences, per lead time.
    Heavier thresholds use the method's field (ensemble-mean ranking for pm/mean, member fractions for prob);
    PySTEPS only has its ensemble mean and P(>=1 mm/h) saved, so it is scored on those."""
    T = next(iter(items.values()))["truth"].shape[0]
    out = {}
    for thr in (2.5, 7.6):
        rows = {"model": [], "model_pmi": [], "persistence": [], "pysteps_mean": []}
        for tag, it in items.items():
            ob = it["truth"] >= thr
            mean = it["ens"].mean(0)
            if method == "prob":
                pr = (it["ens"] >= thr).mean(0) >= np.asarray(p).reshape(-1, 1, 1)
            elif method == "member":
                pr = it["ens"][0] >= thr
            else:
                pr = mean >= thr
            rows["model"].append([cat(pr[k], ob[k])[2] for k in range(T)])
            pmi = match_intensities(mean, it["last"], THR) >= thr
            rows.setdefault("model_pmi", []).append([cat(pmi[k], ob[k])[2] for k in range(T)])
            rows["persistence"].append([cat(it["last"] >= thr, ob[k])[2] for k in range(T)])
            if steps:
                rows["pysteps_mean"].append([cat(steps[tag]["mean"][k] >= thr, ob[k])[2] for k in range(T)])
        out[f"csi_{thr}"] = {k: np.nanmean(np.array(v), 0).tolist() for k, v in rows.items() if v}
    brier = {"model": [], "persistence": [], "pysteps": []}
    err = {"model_mean": [], "persistence": [], "pysteps_mean": []}
    for tag, it in items.items():
        ob = (it["truth"] >= THR).astype(np.float32)
        brier["model"].append(((((it["ens"] >= THR).mean(0)) - ob) ** 2).mean((1, 2)))
        brier["persistence"].append((((it["last"] >= THR).astype(np.float32)[None] - ob) ** 2).mean((1, 2)))
        mean = it["ens"].mean(0)
        err["model_mean"].append(np.stack([np.abs(mean - it["truth"]).mean((1, 2)), ((mean - it["truth"]) ** 2).mean((1, 2))]))
        err["persistence"].append(np.stack([np.abs(it["last"][None] - it["truth"]).mean((1, 2)),
                                            ((it["last"][None] - it["truth"]) ** 2).mean((1, 2))]))
        if steps:
            st = steps[tag]
            brier["pysteps"].append(((st["prob"] - ob) ** 2).mean((1, 2)))
            err["pysteps_mean"].append(np.stack([np.abs(st["mean"] - it["truth"]).mean((1, 2)),
                                                 ((st["mean"] - it["truth"]) ** 2).mean((1, 2))]))
    bs = {k: np.mean(v, 0) for k, v in brier.items() if v}
    out["brier_1mm"] = {k: v.tolist() for k, v in bs.items()}
    out["brier_skill_vs_persistence"] = {k: (1 - v / bs["persistence"]).tolist() for k, v in bs.items() if k != "persistence"}
    out["mae_mm_h"] = {k: np.mean(v, 0)[0].tolist() for k, v in err.items() if v}
    out["rmse_mm_h"] = {k: np.sqrt(np.mean(v, 0)[1]).tolist() for k, v in err.items() if v}
    return out


def bootstrap_diff(a, b, n=2000, seed=0):
    """95 % CI of mean(a - b) over sequences (paired), per lead."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), (n, len(d)))
    means = np.nanmean(d[idx], axis=1)
    return np.nanpercentile(means, 2.5, axis=0), np.nanpercentile(means, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("dataset")
    ap.add_argument("--tune-year", type=int, default=2022); ap.add_argument("--test-year", type=int, default=2023)
    ap.add_argument("--variants", nargs="+", default=["noka", "ka1", "ka2"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    report = {"model": a.model, "dataset": a.dataset, "threshold_mm_h": THR, "fss_scales_px": SCALES,
              "tune_year": a.tune_year, "test_year": a.test_year, "configs": {}}
    tune_scores = {}
    tune_all = {v: load(a.model, a.dataset, v, a.tune_year) for v in a.variants}
    tune_all = {v: t for v, t in tune_all.items() if t}
    shared = set.intersection(*[set(t) for t in tune_all.values()]) if tune_all else set()
    for v, tune in tune_all.items():
        tune = {k: tune[k] for k in sorted(shared)}
        if not tune:
            continue
        p = tune_p(tune)
        for meth in ("member", "mean", "prob", "pm"):
            s = per_sequence(tune, meth, p if meth == "prob" else None)
            tune_scores[(v, meth)] = (float(np.nanmean(s[:, :, 2])), p if meth == "prob" else None)
    if not tune_scores:
        raise SystemExit("no tuning data yet")
    print(f"{a.model} on {a.dataset}: tuning year {a.tune_year}, {len(shared)} sequences shared by {list(tune_all)}")
    for (v, meth), (csi, _) in sorted(tune_scores.items(), key=lambda x: -x[1][0]):
        print(f"  tune  {v:5s} {meth:7s} mean CSI {csi:.4f}")
    best = max(tune_scores, key=lambda k: tune_scores[k][0])
    report["selected"] = {"variant": best[0], "method": best[1], "p": tune_scores[best][1],
                          "tune_mean_csi": tune_scores[best][0]}
    if best[1] in ("pm", "mean"):
        tune_best = {k: tune_all[best[0]][k] for k in sorted(shared)}
        tx = extra_scores(tune_best, None, best[1], tune_scores[best][1])
        heavy = lambda key: float(np.mean([np.nanmean(tx[f"csi_{t}"][key]) for t in (2.5, 7.6)]))
        use_pmi = heavy("model_pmi") > heavy("model")
        report["intensity_matching"] = {"tune_heavy_csi_pm": heavy("model"), "tune_heavy_csi_pmi": heavy("model_pmi"),
                                        "adopted": use_pmi}
        print(f"  heavy-rain CSI on {a.tune_year} (mean of 2.5 & 7.6 mm/h): pm {heavy('model'):.4f}, "
              f"pmi {heavy('model_pmi'):.4f} -> {'adopt' if use_pmi else 'keep'} intensity matching")
    report["tune_ranking"] = [{"variant": v, "method": m, "mean_csi": c} for (v, m), (c, _) in
                              sorted(tune_scores.items(), key=lambda x: -x[1][0])]

    test = {v: load(a.model, a.dataset, v, a.test_year) for v in a.variants}
    if test.get(best[0]) == {}:
        raise SystemExit(f"Selected configuration {best[0]}/{best[1]} has no {a.test_year} forecasts; "
                         f"generate them with tools/gen_validation.py --variants {best[0]}")
    common = set.intersection(*[set(t) for t in test.values() if t]) if any(test.values()) else set()
    if not common:
        print("no test-year data yet"); report["test"] = None
    else:
        test = {v: {k: t[k] for k in sorted(common)} for v, t in test.items() if t}
        pers = per_sequence(next(iter(test.values())), "persistence")
        res = {"n_sequences": len(common), "persistence": np.nanmean(pers, 0).tolist()}
        rows = [("persistence", pers)]
        for v in test:
            for meth in ("member", "prob", "pm", "mean"):
                p = tune_scores.get((v, meth), (None, None))[1]
                s = per_sequence(test[v], meth, p)
                lo, hi = bootstrap_diff(s[:, :, 2], pers[:, :, 2])
                res[f"{v}/{meth}"] = {"mean": np.nanmean(s, 0).tolist(), "csi_minus_persistence_ci95": [lo.tolist(), hi.tolist()]}
                rows.append((f"{v}/{meth}", s))
        st_tune, st_test = load_steps(a.model, a.dataset, a.tune_year), load_steps(a.model, a.dataset, a.test_year)
        if st_test and set(common) <= set(st_test):
            grid = [x / 50 for x in range(1, 50)]
            p_steps = [max((csi_only([it["prob"][k] >= q for it in st_tune.values()],
                                      [it["truth"][k] >= THR for it in st_tune.values()]), q) for q in grid)[1]
                       for k in range(6)] if st_tune else [0.5] * 6
            st_test = {k: st_test[k] for k in sorted(common)}
            for meth, pp in (("mean", None), ("prob", p_steps)):
                s_ = steps_seq(st_test, meth, pp)
                lo, hi = bootstrap_diff(s_[:, :, 2], pers[:, :, 2])
                res[f"pysteps/{meth}"] = {"mean": np.nanmean(s_, 0).tolist(), "csi_minus_persistence_ci95": [lo.tolist(), hi.tolist()],
                                          "p": pp}
                rows.append((f"pysteps/{meth}", s_))
            sel = f"{best[0]}/{best[1]}"
            sel_s = per_sequence(test[best[0]], best[1], tune_scores[best][1])
            st_best = steps_seq(st_test, "prob", p_steps)
            lo, hi = bootstrap_diff(sel_s[:, :, 2], st_best[:, :, 2])
            res["selected_minus_pysteps_prob_ci95"] = [lo.tolist(), hi.tolist()]
        res["extra"] = extra_scores(test[best[0]], st_test if (st_test and set(common) <= set(st_test)) else None,
                                    best[1], tune_scores[best][1])
        report["test"] = res
        print(f"\nTEST year {a.test_year}: {len(common)} sequences (threshold {THR} mm/h, per-sequence mean)")
        print(f"{'config':18s} " + " ".join(f"CSI+{30 * (k + 1):<3d}" for k in range(6)) + "  POD+90  FAR+90  FSS44+90")
        for name, s in rows:
            mmn = np.nanmean(s, 0)
            star = " <- selected on tuning year" if name == f"{best[0]}/{best[1]}" else ""
            print(f"{name:18s} " + " ".join(f"{x:7.3f}" for x in mmn[:, 2]) + f"  {mmn[2, 0]:6.3f}  {mmn[2, 1]:6.3f}  {mmn[2, 5]:7.3f}{star}")
    ex = (report.get("test") or {}).get("extra")
    if ex:
        print("\nExtra scores on the test year (per lead +30 ... +180):")
        for key in ("csi_2.5", "csi_7.6", "brier_skill_vs_persistence", "mae_mm_h", "rmse_mm_h"):
            for name, vals in ex[key].items():
                print(f"  {key:28s} {name:14s} " + " ".join(f"{v:7.3f}" for v in vals))
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(report, open(os.path.join(a.out, f"val_study_{a.model}_{a.dataset}.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
