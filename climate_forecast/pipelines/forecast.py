# climate_forecast/pipelines/forecast.py

import os
import sys
import torch
import h5py
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

from ..training.indra_sat_diff_module import IndraSatDiffModule
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..datasets.visualization import visualize_sequence

def _find_input_sequence(end_file_path: str, in_len: int, data_dir: str) -> list:
    """Finds the sequence of `in_len` files ending at `end_file_path`."""
    if not os.path.exists(end_file_path):
        raise FileNotFoundError(f"Input file not found: {end_file_path}")
        
    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.h5', '.hdf5'))])
    
    try:
        end_index = all_files.index(os.path.abspath(end_file_path))
    except ValueError:
        raise ValueError(f"End file {end_file_path} not found in the processed data directory {data_dir}.")
    
    start_index = end_index - in_len + 1
    if start_index < 0:
        raise ValueError(f"Not enough preceding files to form a context of length {in_len} for the input file {end_file_path}.")
    
    sequence_files = all_files[start_index : end_index + 1]
    return sequence_files

def run(config: dict):
    # Load the user's config to find the main output directory
    initial_cfg = OmegaConf.create(config)
    training_output_dir = initial_cfg.pipeline.output_dir
    resolved_config_path = os.path.join(training_output_dir, "indra_sat_diff", "resolved_config.yaml")

    if not os.path.exists(resolved_config_path):
        print(
            f"FATAL: The resolved training configuration was not found.\n"
            f"       Looked for: {resolved_config_path}\n"
            "Please ensure that the 'pipeline.output_dir' in your config points to a valid, completed training directory.",
            file=sys.stderr
        )
        sys.exit(1)
        
    # Load the exact configuration the model was trained with. This is the source of truth.
    cfg = OmegaConf.load(resolved_config_path)
    # Merge runtime args like `input_file` and `output_dir` from the command line.
    cfg.merge_with(initial_cfg)
    
    model_ckpt_path = cfg.pipeline.get('model_checkpoint_path')
    if not model_ckpt_path or not os.path.exists(model_ckpt_path):
        print(f"FATAL: Model checkpoint not found at path: {model_ckpt_path}", file=sys.stderr)
        sys.exit(1)

    f_cfg = cfg.forecast

    print("--- Starting Forecast ---")
    print(f"  > Using resolved config from: {resolved_config_path}")
    print(f"  > Loading final INDRA-Sat-Diff model from: {model_ckpt_path}")

    # Initialize the model with the fully resolved and correct configuration.
    model = IndraSatDiffModule(cfg)
    state_dict = torch.load(model_ckpt_path, map_location="cpu")
    model.torch_nn_module.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  > Model loaded and moved to {device}.")

    in_len = cfg.layout.in_len
    print(f"  > Searching for {in_len} input files ending at: {f_cfg.input_file}")
    sequence_files = _find_input_sequence(f_cfg.input_file, in_len, cfg.data.path)
    print(f"  > Found input sequence of {len(sequence_files)} files.")

    input_frames = []
    for file_path in sequence_files:
        with h5py.File(file_path, 'r') as hf:
            frame_channels = [np.squeeze(hf[channel][:]) for channel in cfg.data.channels]
            input_frames.append(np.stack(frame_channels, axis=-1))

    input_seq_np = np.array(input_frames, dtype=np.float32)
    input_seq_tensor_batch = torch.from_numpy(input_seq_np).unsqueeze(0).to(device)

    source_layout = "NTHWC"
    target_layout = cfg.layout.layout
    permute_dims = [source_layout.find(dim) for dim in target_layout]
    input_seq_tensor = input_seq_tensor_batch.permute(*permute_dims)

    try:
        with h5py.File(sequence_files[-1], 'r') as hf:
            last_timestamp_unix = hf[cfg.data.time_variable_name][()]
        forecast_start_time = datetime.fromtimestamp(int(last_timestamp_unix))
        print(f"  > Forecast start time (from last input file): {forecast_start_time.isoformat()}")
        OmegaConf.update(cfg, "visualization.start_time", forecast_start_time.isoformat(), merge=True)
    except Exception as e:
        print(f"  > Warning: Could not read timestamp. GIF will have relative time. Error: {e}")

    print("  > Generating forecast...")
    with torch.no_grad():
        cond = {"y": input_seq_tensor}
        alignment_target = AverageIntensityAlignment.model_objective(cond['y'])
        alignment_kwargs = {"avg_x_gt": alignment_target}
        forecast_seq_norm = model.sample(cond=cond, batch_size=1, use_alignment=True, alignment_kwargs=alignment_kwargs).contiguous()

    output_dir = f_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"forecast_{timestamp_str}.gif")
    png_path = os.path.join(output_dir, f"forecast_{timestamp_str}.png")
    npy_path = os.path.join(output_dir, f"forecast_data_{timestamp_str}.npz")

    current_layout = cfg.layout.layout
    if current_layout != "NTHWC":
        permute_to_vis = [current_layout.find(c) for c in "NTHWC"]
        forecast_for_vis = forecast_seq_norm.permute(*permute_to_vis).squeeze(0).cpu().numpy()
        context_for_vis = cond['y'].permute(*permute_to_vis).squeeze(0).cpu().numpy()
    else:
        forecast_for_vis = forecast_seq_norm.squeeze(0).cpu().numpy()
        context_for_vis = cond['y'].squeeze(0).cpu().numpy()

    print(f"  > Saving raw forecast data to: {npy_path}")
    np.savez_compressed(npy_path, input_context=context_for_vis, forecast=forecast_for_vis)

    vis_config = OmegaConf.to_container(cfg, resolve=True)

    print(f"  > Saving static forecast visualization to: {png_path}")
    visualize_sequence(
        save_path=png_path,
        sequences=[context_for_vis, forecast_for_vis],
        labels=["Input Context", "INDRA-Sat-Diff Forecast"],
        config=vis_config,
        title=f"Forecast starting at {vis_config.get('visualization', {}).get('start_time', 'N/A')}"
    )

    print(f"  > Saving animated forecast GIF to: {gif_path}")
    visualize_sequence(
        save_path=gif_path,
        sequences=forecast_for_vis,
        labels="INDRA-Sat-Diff Forecast",
        config=vis_config
    )

    print(f"\n--- ✅ Forecast Complete ---")