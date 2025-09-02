import os
import torch
import h5py
import numpy as np
from omegaconf import OmegaConf
from datetime import datetime

from climate_forecast.training.prediff_module import PreDiffModule
from climate_forecast.diffusion.knowledge_alignment.alignment_guides import AverageIntensityAlignment
from climate_forecast.datasets.visualization import visualize_sequence

def _find_input_sequence(start_file_path: str, in_len: int) -> list:
    """Finds the contiguous sequence of `in_len` files starting from `start_file_path`."""
    if not os.path.exists(start_file_path):
        raise FileNotFoundError(f"Input file not found: {start_file_path}")

    directory = os.path.dirname(start_file_path)
    all_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.h5', '.hdf5'))])

    try:
        start_index = all_files.index(start_file_path)
    except ValueError:
        raise ValueError(f"Start file {start_file_path} not found in its directory.")

    if start_index + in_len > len(all_files):
        raise ValueError(f"Cannot find {in_len} contiguous files starting from {start_file_path}. Not enough files in directory.")

    sequence_files = all_files[start_index : start_index + in_len]
    return sequence_files

def run(config: dict):
    """
    The main entry point for running a forecast.

    Loads the model, finds a sequence of input files, generates a forecast,
    and saves the output as raw data and an animated GIF.
    """
    cfg = OmegaConf.create(config)
    f_cfg = cfg.forecast

    print("--- Starting Forecast ---")

    # --- 1. Load the Trained PreDiff Model ---
    print(f"  > Loading final PreDiff model from: {f_cfg.prediff_checkpoint_path}")
    train_output_dir = os.path.dirname(os.path.dirname(f_cfg.prediff_checkpoint_path))
    train_config_path = os.path.join(train_output_dir, "config.yaml")
    if not os.path.exists(train_config_path):
        raise FileNotFoundError(f"The original training config ('config.yaml') was not found in {train_output_dir}")

    train_cfg = OmegaConf.load(train_config_path)
    model = PreDiffModule(OmegaConf.to_container(train_cfg, resolve=True))
    state_dict = torch.load(f_cfg.prediff_checkpoint_path, map_location="cpu")
    model.torch_nn_module.load_state_dict(state_dict)
    model.eval()

    # --- 2. Load and Prepare Input Data ---
    in_len = train_cfg.layout.in_len
    print(f"  > Searching for {in_len} input files starting from: {f_cfg.input_file}")
    sequence_files = _find_input_sequence(f_cfg.input_file, in_len)

    input_frames = []
    for file_path in sequence_files:
        with h5py.File(file_path, 'r') as hf:
            frame_channels = [hf[channel][:] for channel in train_cfg.data.channels]
            input_frames.append(np.stack(frame_channels, axis=-1))

    input_seq = np.array(input_frames) # (T, H, W, C)
    input_seq_tensor = torch.from_numpy(input_seq).float().unsqueeze(0) # (1, T, H, W, C)

    # --- Automatic Timestamp Extraction ---
    last_input_file = sequence_files[-1]
    with h5py.File(last_input_file, 'r') as hf:
        last_timestamp_unix = hf[train_cfg.data.time_variable_name][()]

    forecast_start_time = datetime.fromtimestamp(last_timestamp_unix)
    print(f"  > Automatically determined forecast start time: {forecast_start_time.isoformat()}")
    OmegaConf.update(cfg, "visualization.start_time", forecast_start_time.isoformat(), merge=True)

    # --- 3. Run Inference ---
    print("  > Generating forecast...")
    with torch.no_grad():
        _, cond = model.get_input(input_seq_tensor)

        # Use the context itself for the alignment guide target
        alignment_target = AverageIntensityAlignment.calculate_ground_truth_target(cond['y'])
        alignment_kwargs = {"avg_x_gt": alignment_target}

        forecast_seq_norm = model.sample(
            cond=cond, batch_size=1, use_alignment=True, alignment_kwargs=alignment_kwargs
        ).contiguous()

    # --- 4. Save Outputs ---
    output_dir = f_cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    gif_path = os.path.join(output_dir, "forecast.gif")
    npy_path = os.path.join(output_dir, "forecast_data.npz")

    # Save raw data to a compressed .npz file
    print(f"  > Saving raw forecast data to: {npy_path}")
    context_to_save = cond['y'].squeeze(0).cpu().numpy()
    forecast_to_save = forecast_seq_norm.squeeze(0).cpu().numpy()
    save_dict = {"input_context": context_to_save, "forecast": forecast_to_save}
    np.savez_compressed(npy_path, **save_dict)

    # Save animated GIF visualization
    print(f"  > Saving forecast visualization to: {gif_path}")
    vis_config = OmegaConf.merge(train_cfg, cfg)
    visualize_sequence(
        save_path=gif_path,
        sequences=forecast_seq_norm.squeeze(0).cpu().numpy(),
        labels="Forecast",
        config=OmegaConf.to_container(vis_config, resolve=True)
    )

    print("\n--- ✅ Forecast Complete ---")