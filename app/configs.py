"""Builds the YAML configs for app-launched training (no heavy imports, so the UI can preview them)."""
import os
import re

import yaml

from . import registry

LEVELS = (1, 2, 4, 8, 16)
NAME = re.compile(r"[^A-Za-z0-9_.-]")


def _years(v):
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",") if x.strip()]
    return [int(x) if str(x).isdigit() else str(x) for x in (v or [])]


def regrid_shape(p):
    """(n_lat, n_lon) cells of a re-cut region at 0.1 deg / downscale_factor."""
    rg = p["regrid"]
    f = max(1, int(rg.get("downscale_factor", 1)))
    return [int(round((rg["lat_range"][1] - rg["lat_range"][0]) * 10)) // f,
            int(round((rg["lon_range"][1] - rg["lon_range"][0]) * 10)) // f]


def regrid_name(p):
    rg = p["regrid"]
    box = "_".join(f"{v:g}" for v in (*rg["lat_range"], *rg["lon_range"])).replace(".", "p").replace("-", "m")
    return NAME.sub("_", f"{p['dataset']}_{box}_x{int(rg.get('downscale_factor', 1))}")


def regrid_config(p):
    """Preprocess config that re-cuts the raw files of a downloaded dataset. Returns (config, new dataset id)."""
    src = registry.datasets()[p["dataset"]]
    base_src = os.path.dirname(src.path)
    raw = os.path.join(base_src, "raw")
    if not os.path.isdir(raw):
        raise ValueError("This dataset has no raw files; region and resolution are fixed. Use Download instead.")
    rg, name = p["regrid"], regrid_name(p)
    old = {}
    if os.path.exists(os.path.join(base_src, "download.yaml")):
        old = yaml.safe_load(open(os.path.join(base_src, "download.yaml"))) or {}
    processed = os.path.join(registry.DATA_DIR, name, "processed")
    pre = {**old.get("preprocess", {}), "input_dir": raw, "output_dir": processed,
           "lat_range": [float(x) for x in rg["lat_range"]], "lon_range": [float(x) for x in rg["lon_range"]],
           "downscale_factor": max(1, int(rg.get("downscale_factor", 1)))}
    pre.setdefault("raw_data_coordinate_order", ["lon", "lat"])
    pre.setdefault("clip_value", 100.0)
    data = {**old.get("data", {"channels": ["precipitation"], "latitude_variable_name": "lat",
                               "longitude_variable_name": "lon", "time_variable_name": "time",
                               "time_interval_minutes": 30}), "path": processed}
    return {"preprocess": pre, "data": data}, name


def regrid_dataset_meta(p, job_id):
    src = registry.datasets()[p["dataset"]]
    rg = p["regrid"]
    return {**src.meta, "name": f"{src.meta.get('name', src.id)} (re-gridded x{int(rg.get('downscale_factor', 1))})",
            "lat_range": [float(x) for x in rg["lat_range"]], "lon_range": [float(x) for x in rg["lon_range"]],
            "grid": regrid_shape(p), "source": f"re-gridded from {src.id} by app job {job_id}"}


def train_config(p):
    """Training request for `climate-forecast train`. Returns (config dict, model name, output dir)."""
    ds = registry.datasets()[p["dataset"]]
    meta = ds.meta
    level = int(p.get("compute_level", 16))
    if level not in LEVELS:
        raise ValueError(f"compute_level must be one of {LEVELS}")
    name = NAME.sub("_", p.get("name") or f"{ds.id}_L{level}")
    out = os.path.join(registry.MODEL_DIR, name)
    cfg = {
        "compute_level": level,
        "pipeline": {"output_dir": registry.rel(out),
                     "model_checkpoint_path": registry.rel(os.path.join(out, "indra_sat_diff", "checkpoints",
                                                                        "indra_sat_diff_final.pt"))},
        "preprocess": {"lat_range": meta.get("lat_range"), "lon_range": meta.get("lon_range")},
        "data": {"path": registry.rel(ds.path), "channels": [meta.get("channel", "Grid/precipitation")],
                 "time_variable_name": meta.get("time_variable", "Grid/time"),
                 "time_interval_minutes": ds.interval // 60},
        "layout": {"img_height": int(meta["grid"][0]), "img_width": int(meta["grid"][1])},
        "optim": {"max_epochs": int(p.get("max_epochs", 20))},
    }
    if p.get("split") == "year":
        cfg["data"]["split"] = {"mode": "year", "train_years": _years(p.get("train_years")),
                                "val_years": _years(p.get("val_years")),
                                "test_years": _years(p.get("test_years")) or _years(p.get("val_years"))}
        if not cfg["data"]["split"]["train_years"] or not cfg["data"]["split"]["val_years"]:
            raise ValueError("Year split needs train years and validation years.")
    else:
        cfg["data"]["split"] = {"mode": "random"}
        for k in ("val_ratio", "test_ratio"):
            if p.get(k) not in (None, ""):
                cfg["data"][k] = float(p[k])
    if p.get("lr") not in (None, ""):
        cfg["optim"]["lr"] = float(p["lr"])
    if p.get("total_batch_size") not in (None, ""):
        cfg["optim"]["total_batch_size"] = int(p["total_batch_size"])
    return cfg, name, out


def preview(p):
    """YAML text of what a train request would write (and of the re-grid step, when requested)."""
    parts = []
    if p.get("regrid"):
        rg_cfg, rg_name = regrid_config(p)
        parts.append(f"# step 1: artifacts/data/{rg_name}/download.yaml  (climate-forecast preprocess)\n"
                     + yaml.safe_dump(rg_cfg, sort_keys=False))
        ds = registry.datasets()[p["dataset"]]
        shape = regrid_shape(p)
        # the re-gridded dataset does not exist yet: describe it for the training config
        cfg, name, out = train_config({**p, "regrid": None})
        cfg["preprocess"] = {"lat_range": rg_cfg["preprocess"]["lat_range"],
                             "lon_range": rg_cfg["preprocess"]["lon_range"]}
        cfg["data"]["path"] = registry.rel(rg_cfg["data"]["path"])
        cfg["data"]["channels"] = rg_cfg["data"].get("channels", cfg["data"]["channels"])
        cfg["layout"] = {"img_height": shape[0], "img_width": shape[1]}
        parts.append(f"# step 2: {registry.rel(out)}/train_request.yaml  (climate-forecast train)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
    else:
        cfg, name, out = train_config(p)
        parts.append(f"# {registry.rel(out)}/train_request.yaml  (climate-forecast train)\n"
                     + yaml.safe_dump(cfg, sort_keys=False))
    return "\n".join(parts)
