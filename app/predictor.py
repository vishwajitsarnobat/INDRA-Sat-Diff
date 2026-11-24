"""Keeps one forecasting model warm on the GPU and runs predictions for the app."""
import gc
import threading
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from climate_forecast.diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from climate_forecast.diffusion.postprocess import combine
from climate_forecast.pipelines.forecast import resample_grid
from climate_forecast.utils.config_loader import load_default_config
from climate_forecast.training.indra_sat_diff_module import IndraSatDiffModule

from . import registry

THRESHOLDS_MM = [1.0, 2.5, 7.6, 16.0]
INFERENCE = load_default_config("train.yaml")["inference"]

_lock = threading.RLock()
_state = {"id": None, "model": None, "cfg": None}


def unload():
    with _lock:
        _state.update(id=None, model=None, cfg=None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _load(model_id):
    if _state["id"] == model_id:
        return _state["model"], _state["cfg"]
    unload()
    m = registry.model_by_id(model_id)
    cfg = OmegaConf.load(registry.os.path.join(registry.ROOT, m["config"]))
    model = IndraSatDiffModule(cfg)
    model.torch_nn_module.load_state_dict(torch.load(registry.os.path.join(registry.ROOT, m["weights"]),
                                                     map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    _state.update(id=model_id, model=model, cfg=cfg)
    return model, cfg


def categorical(pred_mm, true_mm, thr):
    p, o = pred_mm >= thr, true_mm >= thr
    hits = float(np.sum(p & o)); fa = float(np.sum(p & ~o)); miss = float(np.sum(~p & o))
    nan = None
    return {
        "pod": hits / (hits + miss) if hits + miss else nan,
        "far": fa / (hits + fa) if hits + fa else nan,
        "csi": hits / (hits + miss + fa) if hits + miss + fa else nan,
        "bias": (hits + fa) / (hits + miss) if hits + miss else nan,
    }


def predict(model_id, dataset_id, t0, use_alignment=True, members=None, sampler=None, steps=None,
            precision="fp16", progress=None, seed=None, ka_multiplier=None, postprocess=None, prob_threshold=None):
    """Forecast from the frame at unix time t0 (last context frame). Returns dict of numpy arrays + metrics.

    Unset options come from the `inference` defaults (climate_forecast/configs/train.yaml). `forecast` is the
    post-processed ensemble (see climate_forecast/diffusion/postprocess.py); `ensemble` holds all samples.
    """
    d = INFERENCE
    members = int(members if members is not None else d["ensemble_size"])
    sampler = sampler or d["sampler"]
    steps = int(steps if steps is not None else d["steps"])
    ka_multiplier = float(ka_multiplier if ka_multiplier is not None else d["ka_multiplier"])
    postprocess = postprocess or d["postprocess"]
    prob_threshold = prob_threshold if prob_threshold is not None else d["prob_threshold"]
    ds = registry.datasets()[dataset_id]
    with _lock:
        model, cfg = _load(model_id)
        in_len, out_len = cfg.layout.in_len, cfg.layout.out_len
        i0 = ds.index_of(t0)
        if not ds.contiguous(i0 - in_len + 1, in_len):
            raise ValueError(f"Need {in_len} consecutive frames ending at the chosen time.")
        has_truth = ds.contiguous(i0 - in_len + 1, in_len + out_len)
        seq = registry.sequence(ds, i0, in_len, out_len if has_truth else 0)
        context, truth = seq[:in_len], (seq[in_len:] if has_truth else None)

        device = next(model.parameters()).device
        x = torch.from_numpy(context)[None, ..., None].to(device)           # N T H W C
        data_hw = tuple(context.shape[1:3])
        model_hw = (cfg.layout.img_height, cfg.layout.img_width)
        x = resample_grid(x, *model_hw)
        perm = ["NTHWC".find(c) for c in cfg.layout.layout]
        y = x.permute(*perm).expand(members, *x.permute(*perm).shape[1:]).contiguous()
        if seed is not None:
            torch.manual_seed(int(seed))
        amp = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision) if device.type == "cuda" else None
        t_start = time.time()
        with torch.no_grad(), torch.autocast(device.type, dtype=amp or torch.float32, enabled=amp is not None):
            target = AverageIntensityAlignment.model_objective(y) * float(ka_multiplier)
            out = model.sample(cond={"y": y}, batch_size=members, use_alignment=use_alignment,
                               alignment_kwargs={"avg_x_gt": target} if use_alignment else None,
                               sampler=sampler, ddim_steps=steps,
                               callback=(lambda i, n: progress(i / n)) if progress else None)
        seconds = time.time() - t_start
        out = out.float().permute(*[cfg.layout.layout.find(c) for c in "NTHWC"])
        ens = resample_grid(out, *data_hw).clamp(0, 1)[..., 0].cpu().numpy()   # M T H W

    clip = float(ds.meta.get("clip_value", 100.0))
    forecast = combine(ens, context[-1], postprocess, float(d["verify_threshold_mm"]), prob_threshold, clip)
    result = {"context": context, "forecast": forecast, "ensemble": ens, "truth": truth,
              "settings": {"members": members, "sampler": sampler, "steps": steps, "ka": use_alignment,
                           "ka_multiplier": ka_multiplier, "postprocess": postprocess if members > 1 else "member"},
              "times_context": [registry.iso(t) for t in ds.times[i0 - in_len + 1:i0 + 1]],
              "lead_minutes": [(k + 1) * ds.interval // 60 for k in range(out_len)],
              "seconds": round(seconds, 2), "members": members}
    if members > 1:
        ens_mm = registry.to_mm(ens, clip)
        result["exceedance"] = {str(t): (ens_mm >= t).mean(axis=0) for t in THRESHOLDS_MM}
    if truth is not None:
        f_mm, o_mm = registry.to_mm(forecast, clip), registry.to_mm(truth, clip)
        p_mm = np.repeat(registry.to_mm(context[-1:], clip), out_len, axis=0)   # persistence
        result["metrics"] = {
            str(thr): {"model": [categorical(f_mm[k], o_mm[k], thr) for k in range(out_len)],
                       "persistence": [categorical(p_mm[k], o_mm[k], thr) for k in range(out_len)]}
            for thr in THRESHOLDS_MM}
    return result
