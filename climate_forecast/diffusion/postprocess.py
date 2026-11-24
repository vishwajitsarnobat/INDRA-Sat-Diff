"""Turn an ensemble of diffusion samples into one deterministic forecast field.

All fields are in the model's normalised space (log1p(mm/h) / log1p(clip)). Methods:
  member  first sample (what a single draw gives)
  mean    ensemble mean
  prob    ensemble mean where at least a fraction p of members exceed the threshold, else held below it
  pm      probability matching: cells ranked by ensemble mean; the number of cells above the threshold is set
          to the number observed at T0 (the last input frame); other cells are held below the threshold
  pmi     as pm, and the rain rates of those cells are the observed T0 rates, sorted (intensity matching)
Validated on 2022 and tested on 2023 (tools/val_study.py): `pm` with KA x1 (South) and `prob` (Northeast)
beat persistence at every lead time; a single member does not.
"""
import math

import numpy as np

METHODS = ("member", "mean", "prob", "pm", "pmi")


def match_intensities(mean_mm, last_mm, thr_mm):
    """Probability matching with intensities (Ebert 2001), per lead time: cells are ranked by the ensemble
    mean; the n wettest get the n largest observed rain rates at T0 (n = observed cells >= threshold), the
    rest stay below the threshold. The model decides where it rains; the last observation decides how much
    area and how intense. mean_mm: (T, H, W); last_mm: (H, W); returns mm/h."""
    obs = np.sort(np.asarray(last_mm, dtype=np.float64).ravel())[::-1]
    n = int((obs >= thr_mm).sum())
    out = np.minimum(mean_mm, np.nextafter(thr_mm, 0.0)).astype(np.float64)
    if n == 0:
        return out
    for k in range(mean_mm.shape[0]):
        flat = mean_mm[k].ravel()
        idx = np.argpartition(-flat, n - 1)[:n]
        idx = idx[np.argsort(-flat[idx], kind="stable")]
        o = out[k].ravel()
        o[idx] = obs[:n]
        out[k] = o.reshape(mean_mm[k].shape)
    return out


def norm_threshold(thr_mm, clip=100.0):
    return math.log1p(thr_mm) / math.log1p(clip)


def _f32_at_least(x):
    """Smallest float32 >= x, so values lifted to a threshold still pass a float64 `>= threshold` test."""
    v = np.float32(x)
    return np.nextafter(v, np.float32(np.inf)) if float(v) < x else v


def combine(ensemble, last_obs, method="pm", thr_mm=1.0, p=0.5, clip=100.0):
    """ensemble: (M, T, H, W) normalised; last_obs: (H, W) normalised; returns (T, H, W)."""
    ens = np.asarray(ensemble, dtype=np.float32)
    if method == "member" or ens.shape[0] == 1:
        return ens[0]
    k = math.log1p(clip)
    # average in mm/h (as in tools/val_study.py, where the settings were validated), back to normalised space
    mean = (np.log1p(np.expm1(ens * k).mean(0)) / k).astype(np.float32)
    if method == "mean":
        return mean
    thr = norm_threshold(thr_mm, clip)
    thr_up = _f32_at_least(thr)
    below = np.nextafter(np.float32(thr), np.float32(0))
    if method == "pmi":
        obs_mm = np.expm1(np.asarray(last_obs, dtype=np.float64) * k)
        mm = match_intensities(np.expm1(mean.astype(np.float64) * k), obs_mm, thr_mm)
        out = (np.log1p(np.clip(mm, 0, clip)) / k).astype(np.float32)
        return np.where(mm >= thr_mm, np.maximum(out, thr_up), np.minimum(out, below))
    if method == "prob":
        ps = np.broadcast_to(np.asarray(p, dtype=np.float32).reshape(-1, 1, 1), mean.shape)
        keep = (ens >= thr).mean(0) >= ps
    elif method == "pm":
        n = int((np.asarray(last_obs) >= thr).sum())
        keep = np.zeros_like(mean, dtype=bool)
        if n > 0:
            for k in range(mean.shape[0]):
                cut = np.partition(mean[k].ravel(), -n)[-n]
                keep[k] = (mean[k] >= cut) & (mean[k] > 0)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    out = np.where(keep, np.maximum(mean, thr_up), np.minimum(mean, below))
    return out.astype(np.float32)
