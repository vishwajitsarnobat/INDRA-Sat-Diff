# climate_forecast/pipelines/forecast.py

import os
import torch
import h5py
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

# Import necessary modules from the framework
from ..training.prediff_module import PreDiffModule
from ..diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from ..datasets.visualization import visualize_sequence

def _find_input_sequence(start_file_path: str, in_len: int, data_dir: str) -> list:
    """Finds the contiguous sequence of `in_len` files relative to the start file."""
    if not os.path.exists(start_file_path):
        raise FileNotFoundError(f"Input file not found: {start_file_path}")

    all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.h5', '.hdf5'))])

    try:
        # Ensure we are working with absolute paths for comparison
        start_index = all_files.index(os.path.abspath(start_file_path))
    except ValueError:
        raise ValueError(f"Start file {start_file_path} not found in the processed data directory {data_dir}.")

    if start_index + in_len > len(all_files):
        raise ValueError(f"Cannot find {in_len} contiguous files starting from {start_file_path}. Not enough files in directory.")

    sequence_files = all_files[start_index : start_index + in_len]
    return sequence_files

def run(config: dict):
    """
    Main entry point for running a forecast.
    """
    cfg = OmegaConf.create(config)
    f_cfg = cfg.forecast
    
    print("--- Starting Forecast ---")

    # --- 1. Load the Trained PreDiff Model ---
    print(f"  > Loading final PreDiff model from: {f_cfg.prediff_checkpoint_path}")
    
    # The config passed to the PreDiffModule MUST be the one used for training.
    # The 'config' argument to this function already contains the merged training config.
    model = PreDiffModule(cfg)
    
    # Load the extracted .pt weights, not the full lightning checkpoint
    state_dict = torch.load(f_cfg.prediff_checkpoint_path, map_location="cpu")
    model.torch_nn_module.load_state_dict(state_dict)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  > Model loaded and moved to {device}.")

    # --- 2. Load and Prepare Input Data ---
    in_len = cfg.layout.in_len
    print(f"  > Searching for {in_len} input files starting from: {f_cfg.input_file}")
    sequence_files = _find_input_sequence(f_cfg.input_file, in_len, cfg.data.path)

    input_frames = []
    for file_path in sequence_files:
        with h5py.File(file_path, 'r') as hf:
            frame_channels = [np.squeeze(hf[channel][:]) for channel in cfg.data.channels]
            input_frames.append(np.stack(frame_channels, axis=-1))

    # Create the tensor in the layout specified by the training config
    input_seq_np = np.array(input_frames, dtype=np.float32)
    input_seq_tensor = torch.from_numpy(input_seq_np).unsqueeze(0).to(device) # Add batch dim 'N'
    
    source_layout = "THWC" # The layout of our numpy array
    target_layout = cfg.layout.layout.replace('N', '')
    permute_dims = [source_layout.find(dim) for dim in target_layout]
    # Add batch dim permutation at the start
    final_permute = [0] + [d + 1 for d in permute_dims]
    input_seq_tensor = input_seq_tensor.permute(*final_permute)


    # --- Automatic Timestamp Extraction for Visualization ---
    try:
        with h5py.File(sequence_files[-1], 'r') as hf:
            last_timestamp_unix = hf[cfg.data.time_variable_name][()]
        forecast_start_time = datetime.fromtimestamp(int(last_timestamp_unix))
        print(f"  > Forecast start time (from last input file): {forecast_start_time.isoformat()}")
        # Inject this into the config for the visualization function
        OmegaConf.update(cfg, "visualization.start_time", forecast_start_time.isoformat(), merge=True)
    except Exception as e:
        print(f"  > Warning: Could not read timestamp. GIF will have relative time. Error: {e}")


    # --- 3. Run Inference ---
    print("  > Generating forecast...")
    with torch.no_grad():
        _, cond, _ = model.get_input(input_seq_tensor, return_verbose=True)

        # Use the context itself for the alignment guide target
        alignment_target = AverageIntensityAlignment.model_objective(cond['y'])
        alignment_kwargs = {"avg_x_gt": alignment_target}

        forecast_seq_norm = model.sample(
            cond=cond, batch_size=1, use_alignment=True, alignment_kwargs=alignment_kwargs
        ).contiguous()

    # --- 4. Save Outputs ---
    output_dir = f_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"forecast_{timestamp_str}.gif")
    npy_path = os.path.join(output_dir, f"forecast_data_{timestamp_str}.npz")

    # Ensure forecast tensor is in a standard layout (NTHWC) for saving/visualization
    current_layout = cfg.layout.layout
    if current_layout != "NTHWC":
        permute_to_vis = [current_layout.find(c) for c in "NTHWC"]
        forecast_for_vis = forecast_seq_norm.permute(*permute_to_vis).squeeze(0).cpu().numpy()
        context_for_vis = cond['y'].permute(*permute_to_vis).squeeze(0).cpu().numpy()
    else:
        forecast_for_vis = forecast_seq_norm.squeeze(0).cpu().numpy()
        context_for_vis = cond['y'].squeeze(0).cpu().numpy()


    # Save raw data to a compressed .npz file
    print(f"  > Saving raw forecast data to: {npy_path}")
    np.savez_compressed(npy_path, input_context=context_for_vis, forecast=forecast_for_vis)

    # Save animated GIF visualization
    print(f"  > Saving forecast visualization to: {gif_path}")
    visualize_sequence(
        save_path=gif_path,
        sequences=forecast_for_vis,
        labels="Forecast",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

    print(f"\n--- ✅ Forecast Complete ---")