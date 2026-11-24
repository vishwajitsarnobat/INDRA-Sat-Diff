import os
import sys
import torch
import torch.nn.functional as F
import h5py
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime, timezone

from ..training.indra_sat_diff_module import IndraSatDiffModule
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..datasets.visualization import visualize_sequence
from ..diffusion.postprocess import combine
from ..utils.config_loader import load_default_config


def inference_settings(cfg):
    """Inference settings: package defaults < the model's resolved config < forecast overrides."""
    out = dict(load_default_config("train.yaml").get("inference", {}))
    out.update(OmegaConf.to_container(cfg.get("inference", {}) or {}, resolve=True))
    f = cfg.get("forecast", {}) or {}
    for key in ("sampler", "steps", "ensemble_size", "postprocess", "prob_threshold", "ka_multiplier"):
        if f.get(key) is not None:
            out[key] = f.get(key)
    return out


def _find_sequence(input_last_file: str, data_dir: str, in_len: int, out_len: int = 0) -> list:
    """Locate the `in_len` context files ending at `input_last_file` (T=0), plus `out_len` files after it."""
    last_abs = os.path.abspath(input_last_file)
    data_dir_abs = os.path.abspath(data_dir)

    if not os.path.exists(last_abs):
        raise FileNotFoundError(f"Input file not found: {last_abs}")

    all_files = sorted([
        os.path.abspath(os.path.join(data_dir_abs, f))
        for f in os.listdir(data_dir_abs) if f.lower().endswith(('.h5', '.hdf5', '.nc', '.nc4'))
    ])

    try:
        last_index = all_files.index(last_abs)
    except ValueError:
        raise ValueError(f"File not found in scanned data: {last_abs}")

    start_index = last_index - in_len + 1
    if start_index < 0:
        raise ValueError(f"Not enough files for input context. Need {in_len} ending at the input file, have {last_index + 1}.")

    end_index = last_index + 1 + out_len
    if end_index > len(all_files):
        raise ValueError(f"Not enough files for ground truth. Need {out_len} after the input file, have {len(all_files) - last_index - 1}.")

    return all_files[start_index:end_index]


def resample_grid(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resample an NTHWC sequence to (height, width): bilinear when enlarging, area-average when shrinking."""
    n, t, h, w, c = x.shape
    if (h, w) == (height, width):
        return x
    frames = x.permute(0, 1, 4, 2, 3).reshape(n * t, c, h, w)
    enlarging = height * width > h * w
    frames = F.interpolate(frames, size=(height, width), mode="bilinear" if enlarging else "area",
                           **({"align_corners": False} if enlarging else {}))
    return frames.reshape(n, t, c, height, width).permute(0, 1, 3, 4, 2).contiguous()


def run(config: dict):
    """Generate a forecast using a trained INDRA-Sat-Diff model."""
    initial_cfg = OmegaConf.create(config)
    training_output_dir = initial_cfg.pipeline.output_dir
    resolved_config_path = os.path.join(training_output_dir, "indra_sat_diff", "resolved_config.yaml")

    if not os.path.exists(resolved_config_path):
        print(f"FATAL: Resolved config not found at {resolved_config_path}", file=sys.stderr)
        sys.exit(1)

    cfg = OmegaConf.load(resolved_config_path)
    if "forecast" in initial_cfg:
        cfg = OmegaConf.merge(cfg, {"forecast": initial_cfg.forecast})
    if initial_cfg.pipeline.get("model_checkpoint_path"):
        cfg.pipeline.model_checkpoint_path = initial_cfg.pipeline.model_checkpoint_path

    model_ckpt_path = cfg.pipeline.get('model_checkpoint_path')
    if not model_ckpt_path or not os.path.exists(model_ckpt_path):
        print(f"FATAL: Checkpoint not found: {model_ckpt_path}", file=sys.stderr)
        sys.exit(1)

    f_cfg = cfg.forecast
    print("--- Starting Forecast ---")
    print(f"  > Config: {resolved_config_path}")
    print(f"  > Model: {model_ckpt_path}")

    model = IndraSatDiffModule(cfg)
    model.torch_nn_module.load_state_dict(torch.load(model_ckpt_path, map_location="cpu"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  > Model on {device}.")

    in_len, out_len = cfg.layout.in_len, cfg.layout.out_len
    with_ground_truth = f_cfg.get("with_ground_truth", False)
    ground_truth_len = out_len if with_ground_truth else 0

    all_files = _find_sequence(
        input_last_file=f_cfg.input_file,
        data_dir=cfg.data.path,
        in_len=in_len,
        out_len=ground_truth_len,
    )
    print(f"  > Found {len(all_files)} files.")

    all_frames = []
    for file_path in all_files:
        with h5py.File(file_path, 'r') as hf:
            frame_channels = [np.squeeze(hf[channel][:]) for channel in cfg.data.channels]
            all_frames.append(np.stack(frame_channels, axis=-1))
    all_frames_np = np.array(all_frames, dtype=np.float32)

    input_seq_np = all_frames_np[:in_len]
    ground_truth_np = all_frames_np[in_len:] if with_ground_truth else None
    last_input_path = all_files[in_len - 1]

    input_tensor = torch.from_numpy(input_seq_np).unsqueeze(0).to(device)

    # A model trained on one grid can forecast on another (e.g. coarser data, other region size):
    # inputs are resampled to the model grid and the forecast is resampled back to the data grid.
    data_hw = tuple(input_seq_np.shape[1:3])
    model_hw = (cfg.layout.img_height, cfg.layout.img_width)
    if data_hw != model_hw:
        print(f"  > Data grid {data_hw[0]}x{data_hw[1]} differs from model grid {model_hw[0]}x{model_hw[1]}; resampling.")
        input_tensor = resample_grid(input_tensor, *model_hw)
    permute_dims = ["NTHWC".find(dim) for dim in cfg.layout.layout]
    input_tensor = input_tensor.permute(*permute_dims)

    # Read timestamp for visualization
    try:
        with h5py.File(last_input_path, 'r') as hf:
            ts = hf[cfg.data.time_variable_name][()]
        start_time = datetime.fromtimestamp(float(np.asarray(ts).reshape(-1)[0]), tz=timezone.utc).isoformat()
        OmegaConf.update(cfg, "visualization.start_time", start_time, merge=True)
        print(f"  > Forecast start (UTC): {start_time}")
    except Exception as e:
        print(f"  > Warning: Could not read timestamp: {e}")

    # Run diffusion sampling
    print("  > Generating forecast...")
    use_alignment = f_cfg.get("use_alignment", cfg.model.get("use_alignment", True))
    print(f"  > Knowledge alignment: {'on' if use_alignment else 'off'}")
    precision = str(f_cfg.get("precision", "fp32")).lower()
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    if amp_dtype is not None and device != "cuda":
        print(f"  > {precision} needs a GPU; using fp32.")
        amp_dtype = None
    print(f"  > Inference precision: {precision if amp_dtype else 'fp32'}")
    with torch.no_grad(), torch.autocast(device_type=device, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        inf = inference_settings(cfg)
        members, sampler, steps = int(inf["ensemble_size"]), str(inf["sampler"]), int(inf["steps"])
        print(f"  > Sampler: {sampler}" + (f", {steps} steps" if sampler == "ddim" else "") + f", {members} member(s)"
              f", post-processing: {inf['postprocess'] if members > 1 else 'none'}, KA x{inf['ka_multiplier']}")
        cond = {"y": input_tensor.expand(members, *input_tensor.shape[1:]).contiguous()}
        alignment_target = AverageIntensityAlignment.model_objective(cond['y']) * float(inf["ka_multiplier"])
        ensemble = model.sample(
            cond=cond, batch_size=members, use_alignment=use_alignment,
            alignment_kwargs={"avg_x_gt": alignment_target} if use_alignment else None,
            sampler=sampler, ddim_steps=steps,
        ).contiguous()
        cond = {"y": cond["y"][:1]}

    output_dir = f_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")

    gif_path = os.path.join(output_dir, f"forecast_{ts_str}.gif")
    png_path = os.path.join(output_dir, f"forecast_{ts_str}.png")
    npy_path = os.path.join(output_dir, f"forecast_data_{ts_str}.npz")

    # Convert to NTHWC on the data grid for saving/visualization
    context = cond['y'].float()
    perm_vis = [cfg.layout.layout.find(c) for c in "NTHWC"]
    if cfg.layout.layout != "NTHWC":
        context = context.permute(*perm_vis)
    context_vis = resample_grid(context, *data_hw).squeeze(0).cpu().numpy()
    ens = ensemble.float()
    if cfg.layout.layout != "NTHWC":
        ens = ens.permute(*perm_vis)
    ens_vis = resample_grid(ens, *data_hw).clamp(0, 1).cpu().numpy()            # M T H W C
    forecast_vis = combine(ens_vis[..., 0], context_vis[-1, ..., 0], inf["postprocess"],
                           float(inf["verify_threshold_mm"]), inf["prob_threshold"],
                           float(cfg.visualization.get("denorm_clip_value", 100.0)))[..., None]

    # Save raw data
    save_dict = {"input_context": context_vis, "forecast": forecast_vis}
    if ens_vis.shape[0] > 1:
        save_dict["ensemble"] = ens_vis
    if ground_truth_np is not None:
        save_dict["ground_truth"] = ground_truth_np
    np.savez_compressed(npy_path, **save_dict)
    print(f"  > Raw data: {npy_path}")

    vis_config = OmegaConf.to_container(cfg, resolve=True)
    start_label = vis_config.get('visualization', {}).get('start_time', 'N/A')

    if ground_truth_np is not None:
        visualize_sequence(
            save_path=png_path,
            sequences=[context_vis, forecast_vis, ground_truth_np],
            labels=["Input Context", "Forecast", "Ground Truth"],
            config=vis_config,
            title=f"Forecast vs Ground Truth \u2014 {start_label}",
        )
    else:
        visualize_sequence(
            save_path=png_path,
            sequences=[context_vis, forecast_vis],
            labels=["Input Context", "Forecast"],
            config=vis_config,
            title=f"Forecast \u2014 {start_label}",
        )

    visualize_sequence(
        save_path=gif_path,
        sequences=forecast_vis,
        labels="Forecast",
        config=vis_config,
    )

    print(f"  > PNG: {png_path}")
    print(f"  > GIF: {gif_path}")
    print(f"\n--- \u2705 Forecast Complete ---")
