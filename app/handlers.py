"""Job implementations for the app (registered into jobs.HANDLERS)."""
import csv
import os
import re
from datetime import datetime, timezone

import numpy as np
import yaml

from . import configs, jobs, registry


# ----------------------------------------------------------------------------- predict
@jobs.handler("predict")
def predict_job(job):
    from . import predictor          # heavy (torch); only loaded when a job needs the model
    p = job.params
    res = predictor.predict(
        model_id=p["model"], dataset_id=p["dataset"], t0=registry.parse_time(p["t0"]),
        use_alignment=bool(p.get("alignment", True)), members=p.get("members"),
        sampler=p.get("sampler"), steps=p.get("steps"), ka_multiplier=p.get("ka_multiplier"),
        postprocess=p.get("postprocess"), precision=p.get("precision", "fp16"), seed=p.get("seed"),
        progress=lambda v: (setattr(job, "progress", v), job.emit("progress", value=v)))
    arrays = {"context": res["context"], "forecast": res["forecast"]}
    if res["members"] > 1:
        arrays["ensemble"] = res["ensemble"]
        arrays["spread"] = res["ensemble"].std(axis=0)
        for thr, prob in res["exceedance"].items():
            arrays[f"exceed_{thr}"] = prob
    if res["truth"] is not None:
        arrays["truth"] = res["truth"]
    np.savez_compressed(os.path.join(job.dir, "result.npz"), **arrays)
    return {k: res[k] for k in ("times_context", "lead_minutes", "seconds", "members", "settings")} | {
        "arrays": sorted(arrays), "metrics": res.get("metrics"), "has_truth": res["truth"] is not None}


# ----------------------------------------------------------------------------- evaluate
@jobs.handler("evaluate")
def evaluate_job(job):
    """Model vs persistence over a period: contingency counts per lead time and threshold."""
    from . import predictor
    p = job.params
    ds = registry.datasets()[p["dataset"]]
    m = registry.model_by_id(p["model"])
    in_len, out_len = m["in_len"], m["out_len"]
    start = registry.parse_time(p["start"]) if p.get("start") else ds.times[0]
    end = registry.parse_time(p["end"]) if p.get("end") else ds.times[-1]
    stride = max(1, int(p.get("stride", 6)))
    variants = {"with_ka": True, "no_ka": False} if p.get("compare_ka", True) else \
        {"with_ka" if p.get("alignment", True) else "no_ka": bool(p.get("alignment", True))}
    starts = [i for i in range(in_len - 1, len(ds.times) - out_len, stride)
              if start <= ds.times[i] <= end and ds.contiguous(i - in_len + 1, in_len + out_len)]
    if not starts:
        raise ValueError("No complete input+truth windows in that period.")
    thrs = predictor.THRESHOLDS_MM
    counts = {name: np.zeros((len(thrs), out_len, 3)) for name in [*variants, "persistence"]}  # hits, fa, miss
    clip = float(ds.meta.get("clip_value", 100.0))
    total = len(starts) * len(variants)
    done = 0
    for i in starts:
        for name, align in variants.items():
            if job.cancel_flag:
                raise InterruptedError
            r = predictor.predict(p["model"], ds.id, ds.times[i], use_alignment=align, members=p.get("members"),
                                  steps=p.get("steps"), ka_multiplier=p.get("ka_multiplier"),
                                  postprocess=p.get("postprocess"), precision=p.get("precision", "fp16"), seed=0)
            f_mm, o_mm = registry.to_mm(r["forecast"], clip), registry.to_mm(r["truth"], clip)
            _accumulate(counts[name], f_mm, o_mm, thrs)
            if name == list(variants)[0]:
                pers = np.repeat(registry.to_mm(r["context"][-1:], clip), out_len, axis=0)
                _accumulate(counts["persistence"], pers, o_mm, thrs)
            done += 1
            job.progress = done / total
            job.emit("progress", value=job.progress, detail=f"{done}/{total} forecasts")
    table = []
    for name, c in counts.items():
        for ti, thr in enumerate(thrs):
            for k in range(out_len):
                h, fa, mi = c[ti, k]
                table.append({"variant": name, "threshold_mm": thr, "lead_min": (k + 1) * ds.interval // 60,
                              "pod": _div(h, h + mi), "far": _div(fa, h + fa), "csi": _div(h, h + mi + fa),
                              "bias": _div(h + fa, h + mi), "hits": int(h), "false_alarms": int(fa),
                              "misses": int(mi)})
    out_dir = os.path.join(registry.ART, "results", f"eval_{m['id']}_{ds.id}_{job.id}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "scores.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    yaml.safe_dump({"model": m["id"], "dataset": ds.id, "start": registry.iso(ds.times[starts[0]]),
                    "end": registry.iso(ds.times[starts[-1]]), "forecasts": len(starts), "stride": stride,
                    "steps": int(p.get("steps", 50)), "sampler": "ddim",
                    "note": "KA target is computed from the context frames (causal)."},
                   open(os.path.join(out_dir, "info.yaml"), "w"))
    return {"table": table, "forecasts": len(starts), "dir": registry.rel(out_dir)}


def _accumulate(c, f_mm, o_mm, thrs):
    for ti, thr in enumerate(thrs):
        for k in range(f_mm.shape[0]):
            pr, ob = f_mm[k] >= thr, o_mm[k] >= thr
            c[ti, k] += (np.sum(pr & ob), np.sum(pr & ~ob), np.sum(~pr & ob))


def _div(a, b):
    return float(a / b) if b else None


# ----------------------------------------------------------------------------- fit-check
@jobs.handler("fitcheck")
def fitcheck_job(job):
    p = job.params
    cfg = {"pipeline": {"output_dir": os.path.join(job.dir, "out")}}
    if p.get("grid"):
        cfg["layout"] = {"img_height": int(p["grid"][0]), "img_width": int(p["grid"][1])}
    path = os.path.join(job.dir, "config.yaml")
    yaml.safe_dump(cfg, open(path, "w"))
    levels = [str(x) for x in p.get("levels", [16, 8, 4, 2, 1])]
    seen = {"n": 0}

    def prog(line):
        if "Trainer.fit" in line:
            seen["n"] += 1
            return min(0.99, seen["n"] / (3 * len(levels)))
    tail = jobs.run_cli(job, ["fit-check", "--config", path, "--levels", *levels], prog)
    rows = []
    for line in tail:
        m = re.match(r"\s*(\d+)\s+(\d+)x(\d+)\s+\S+\s*\|(.*)\|\s*(fits.*|does not fit)\s*$", line)
        if m:
            cells = [c.strip() for c in m.group(4).split("|")]
            rows.append({"level": int(m.group(1)), "grid": [int(m.group(2)), int(m.group(3))],
                         "vae": cells[0], "alignment": cells[1], "diffusion": cells[2], "verdict": m.group(5)})
    rec = next((int(re.search(r"compute_level: (\d+)", l).group(1)) for l in tail if "Recommended: compute_level" in l), None)
    return {"rows": rows, "recommended": rec}


# ----------------------------------------------------------------------------- train
@jobs.handler("train")
def train_job(job):
    p = job.params
    if p.get("regrid"):                       # re-cut the raw files to another box / resolution first
        rg_cfg, rg_name = configs.regrid_config(p)
        base = os.path.join(registry.DATA_DIR, rg_name)
        os.makedirs(base, exist_ok=True)
        rg_path = os.path.join(base, "download.yaml")
        yaml.safe_dump(rg_cfg, open(rg_path, "w"), sort_keys=False)
        job.emit("log", line=f"Preprocessing {rg_name} from raw files...")
        jobs.run_cli(job, ["preprocess", "--config", rg_path])
        yaml.safe_dump(configs.regrid_dataset_meta(p, job.id), open(os.path.join(base, "dataset.yaml"), "w"),
                       sort_keys=False)
        registry.datasets(refresh=True)
        p = {**p, "dataset": rg_name, "regrid": None}
    cfg, name, out = configs.train_config(p)
    ds = registry.datasets()[p["dataset"]]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "train_request.yaml")
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)
    stage = {"i": 0, "epochs": int(p.get("max_epochs", 20))}
    order = {"VAE": 0, "ALIGNMENT": 1, "INDRA_SAT_DIFF": 2}

    def prog(line):
        m = re.search(r"\[Stage: (\w+)\]", line)
        if m and m.group(1) in order:
            stage["i"] = order[m.group(1)]
            return stage["i"] / 3
        m = re.search(r"Epoch (\d+)", line)
        if m:
            return (stage["i"] + min(1.0, int(m.group(1)) / stage["epochs"])) / 3
    jobs.run_cli(job, ["train", "--config", path], prog)
    yaml.safe_dump({"name": name, "dataset": ds.id, "compute_level": cfg["compute_level"],
                    "trained": datetime.now(timezone.utc).isoformat(timespec="minutes"),
                    "trained_on": "app train job " + job.id, "data_span": [registry.iso(ds.times[0]),
                                                                            registry.iso(ds.times[-1])]},
                   open(os.path.join(out, "model_card.yaml"), "w"), sort_keys=False)
    return {"model": name}


# ----------------------------------------------------------------------------- download + preprocess
@jobs.handler("download")
def download_job(job):
    p = job.params
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", p["name"])
    base = os.path.join(registry.DATA_DIR, name)
    raw, processed = os.path.join(base, "raw"), os.path.join(base, "processed")
    cfg = {
        "download": {"product": p.get("product", "IMERG_FINAL_HH_V07"),
                     "periods": [{"start": p["start"], "end": p["end"]}], "output_dir": raw,
                     "workers": int(p.get("workers", 6))},
        "preprocess": {"input_dir": raw, "output_dir": processed, "lat_range": p["lat_range"],
                       "lon_range": p["lon_range"], "raw_data_coordinate_order": ["lon", "lat"],
                       "downscale_factor": int(p.get("downscale_factor", 1)), "clip_value": 100.0,
                       "num_processes": 4},
        "data": {"path": processed, "channels": ["precipitation"], "latitude_variable_name": "lat",
                 "longitude_variable_name": "lon", "time_variable_name": "time", "time_interval_minutes": 30},
    }
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, "download.yaml")
    yaml.safe_dump(cfg, open(path, "w"), sort_keys=False)

    def prog(line):
        m = re.search(r"(\d+)%\|", line)
        return int(m.group(1)) / 100 * 0.8 if m else None
    jobs.run_cli(job, ["download", "--config", path], prog)
    job.emit("log", line="Preprocessing...")
    jobs.run_cli(job, ["preprocess", "--config", path])
    yaml.safe_dump({"name": p.get("title", name), "region": name, "lat_range": p["lat_range"],
                    "lon_range": p["lon_range"], "channel": "precipitation", "time_variable": "time",
                    "clip_value": 100.0, "interval_minutes": 30, "time_epoch": "unix", "array_order": "lon_lat",
                    "source": f"{cfg['download']['product']} via app job {job.id}"},
                   open(os.path.join(base, "dataset.yaml"), "w"), sort_keys=False)
    registry.datasets(refresh=True)
    return {"dataset": name}
