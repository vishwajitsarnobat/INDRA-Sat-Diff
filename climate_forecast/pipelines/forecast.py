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

def _find_sequence(
    input_context_start_file: str,
    data_dir: str,
    in_len: int,
    out_len: int = 0
) -> list:
    """
    Finds a sequence of files starting from the specified file.

    The function locates the `in_len` files that constitute the input context,
    STARTING with the specified `input_context_start_file`. If `out_len` is
    greater than zero, it then finds the next `out_len` consecutive files to
    serve as the ground truth.

    Args:
        input_context_start_file: The path to the FIRST HDF5 file of the input sequence.
        data_dir: The directory containing all the processed data files.
        in_len: The length of the input context sequence.
        out_len: The length of the ground truth sequence to find after the context.

    Returns:
        A list of absolute paths to the files in the full sequence.
    """
    context_start_abs_path = os.path.abspath(input_context_start_file)
    data_dir_abs_path = os.path.abspath(data_dir)

    if not os.path.exists(context_start_abs_path):
        raise FileNotFoundError(f"Input file not found at resolved path: {context_start_abs_path}")

    all_files_abs = sorted([
        os.path.abspath(os.path.join(data_dir_abs_path, f))
        for f in os.listdir(data_dir_abs_path)
        if f.lower().endswith(('.h5', '.hdf5'))
    ])

    try:
        start_index = all_files_abs.index(context_start_abs_path)
    except ValueError:
        raise ValueError(
            f"Input context start file '{context_start_abs_path}' was not found in the list of files "
            f"scanned from the data directory '{data_dir_abs_path}'."
        )

    total_sequence_len = in_len + out_len
    end_index = start_index + total_sequence_len

    if end_index > len(all_files_abs):
        files_available = len(all_files_abs) - start_index
        if files_available < in_len:
             raise ValueError(
                f"Not enough subsequent files to form the input context. "
                f"Required: {in_len}, Available (including start file): {files_available} in '{data_dir_abs_path}'."
            )
        else:
            raise ValueError(
                f"Not enough subsequent files to form the ground truth sequence. "
                f"Required: {out_len}, Available after input context: {files_available - in_len} in '{data_dir_abs_path}'."
            )

    return all_files_abs[start_index : end_index]

def run(config: dict):
    initial_cfg = OmegaConf.create(config)
    training_output_dir = initial_cfg.pipeline.output_dir
    resolved_config_path = os.path.join(
        training_output_dir, "indra_sat_diff", "resolved_config.yaml"
    )

    if not os.path.exists(resolved_config_path):
        print(
            f"FATAL: The resolved training configuration was not found.\n"
            f"       Looked for: {resolved_config_path}\n"
            "Please ensure that 'pipeline.output_dir' in your config points to a valid, completed training directory.",
            file=sys.stderr,
        )
        sys.exit(1)
        
    cfg = OmegaConf.load(resolved_config_path)
    
    if "forecast" in initial_cfg:
        cfg = OmegaConf.merge(cfg, {"forecast": initial_cfg.forecast})

    if initial_cfg.pipeline.get("model_checkpoint_path"):
        cfg.pipeline.model_checkpoint_path = initial_cfg.pipeline.model_checkpoint_path
    
    model_ckpt_path = cfg.pipeline.get('model_checkpoint_path')
    if not model_ckpt_path or not os.path.exists(model_ckpt_path):
        print(f"FATAL: Model checkpoint not found at path: {model_ckpt_path}", file=sys.stderr)
        sys.exit(1)

    f_cfg = cfg.forecast

    print("--- Starting Forecast ---")
    print(f"  > Using resolved config from: {resolved_config_path}")
    print(f"  > Loading final INDRA-Sat-Diff model from: {model_ckpt_path}")

    model = IndraSatDiffModule(cfg)
    state_dict = torch.load(model_ckpt_path, map_location="cpu")
    model.torch_nn_module.load_state_dict(state_dict)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  > Model loaded and moved to {device}.")

    in_len = cfg.layout.in_len
    out_len = cfg.layout.out_len
    with_ground_truth = f_cfg.get("with_ground_truth", False)
    
    ground_truth_len_to_load = out_len if with_ground_truth else 0

    if with_ground_truth:
        total_frames = in_len + out_len
        print(f"  > Ground truth requested. Will load {total_frames} total frames ({in_len} input + {out_len} truth).")

    print(f"  > Searching for sequence with input context starting at: {f_cfg.input_file}")

    all_files = _find_sequence(
        input_context_start_file=f_cfg.input_file,
        data_dir=cfg.data.path,
        in_len=in_len,
        out_len=ground_truth_len_to_load
    )
    print(f"  > Found full sequence of {len(all_files)} files.")

    all_frames = []
    for file_path in all_files:
        with h5py.File(file_path, 'r') as hf:
            frame_channels = [np.squeeze(hf[channel][:]) for channel in cfg.data.channels]
            all_frames.append(np.stack(frame_channels, axis=-1))

    all_frames_np = np.array(all_frames, dtype=np.float32)

    input_seq_np = all_frames_np[:in_len]
    ground_truth_np = all_frames_np[in_len:] if with_ground_truth else None
    
    last_input_file_path = all_files[in_len - 1]

    input_seq_tensor_batch = torch.from_numpy(input_seq_np).unsqueeze(0).to(device)

    source_layout = "NTHWC"
    target_layout = cfg.layout.layout
    permute_dims = [source_layout.find(dim) for dim in target_layout]
    input_seq_tensor = input_seq_tensor_batch.permute(*permute_dims)

    try:
        with h5py.File(last_input_file_path, 'r') as hf:
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
    if with_ground_truth:
        np.savez_compressed(
            npy_path, 
            input_context=context_for_vis, 
            forecast=forecast_for_vis,
            ground_truth=ground_truth_np
        )
    else:
        np.savez_compressed(npy_path, input_context=context_for_vis, forecast=forecast_for_vis)

    vis_config = OmegaConf.to_container(cfg, resolve=True)

    # The static PNG handles the full comparison when ground truth is available.
    if with_ground_truth and ground_truth_np is not None:
        print(f"  > Saving static forecast visualization with ground truth to: {png_path}")
        visualize_sequence(
            save_path=png_path,
            sequences=[context_for_vis, forecast_for_vis, ground_truth_np],
            labels=["Input Context", "INDRA-Sat-Diff Forecast", "Ground Truth"],
            config=vis_config,
            title=f"Forecast vs. Ground Truth starting at {vis_config.get('visualization', {}).get('start_time', 'N/A')}"
        )
    else:
        print(f"  > Saving static forecast visualization to: {png_path}")
        visualize_sequence(
            save_path=png_path,
            sequences=[context_for_vis, forecast_for_vis],
            labels=["Input Context", "INDRA-Sat-Diff Forecast"],
            config=vis_config,
            title=f"Forecast starting at {vis_config.get('visualization', {}).get('start_time', 'N/A')}"
        )

    # The animated GIF always shows only the forecast prediction.
    print(f"  > Saving animated forecast GIF to: {gif_path}")
    visualize_sequence(
        save_path=gif_path,
        sequences=forecast_for_vis,
        labels="INDRA-Sat-Diff Forecast",
        config=vis_config
    )

    print(f"\n--- ✅ Forecast Complete ---")