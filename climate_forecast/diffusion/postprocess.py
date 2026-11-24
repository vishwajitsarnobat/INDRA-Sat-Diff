"""Turn an ensemble of diffusion samples into one deterministic forecast field.

All fields are in the model's normalised space (log1p(mm/h) / log1p(clip)). Methods:
  member  first sample (what a single draw gives)
  mean    ensemble mean
  prob    ensemble mean where at least a fraction p of members exceed the threshold, else held below it
  pm      probability matching: cells ranked by ensemble mean; the number of cells above the threshold is set
          to the number observed at T0 (the last input frame); other cells are held below the threshold
Validated on 2022 and tested on 2023 (tools/val_study.py): `pm` with KA x1 (South) and `prob` (Northeast)
beat persistence at every lead time; a single member does not.
"""
import math

import numpy as np

METHODS = ("member", "mean", "prob", "pm")


def norm_threshold(thr_mm, clip=100.0):
    return math.log1p(thr_mm) / math.log1p(clip)


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
    below = np.nextafter(np.float32(thr), np.float32(0))
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
    out = np.where(keep, np.maximum(mean, thr), np.minimum(mean, below))
    return out.astype(np.float32)
