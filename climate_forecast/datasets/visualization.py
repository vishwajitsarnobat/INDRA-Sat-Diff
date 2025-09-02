# climate_forecast/datasets/visualization.py

import os
import warnings
import traceback
from typing import Optional, Sequence, Union, Dict, List
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation

# --- Optional Dependency: Cartopy for GIS plotting ---
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False
    warnings.warn(
        "Cartopy is not installed. GIS plotting for GIFs will not be available. "
        "Install with 'pip install cartopy'"
    )

# --- DEFAULTS ---
# Used if the user provides no visualization config.
DEFAULT_BOUNDARIES = [0.0, 0.1, 2.5, 7.6, 16.0, 50.0, 100.0]
DEFAULT_CMAP_DATA = [
    (1.0, 1.0, 1.0), (0.7, 0.85, 0.95), (0.2, 0.6, 0.85),
    (0.5, 0.85, 0.5), (1.0, 0.8, 0.2), (0.9, 0.0, 0.0),
]
DEFAULT_CLIP_VALUE = 100.0

# --- Helper Functions ---

def _denormalize_log(data: np.ndarray, clip_value: float) -> np.ndarray:
    """Internal helper to denormalize data that is log-normalized."""
    if clip_value <= 0:
        return data.astype(np.float32)
    data = data.astype(np.float32)
    scale_factor = np.log1p(clip_value)
    if scale_factor == 0:
        return np.zeros_like(data)
    denormalized = np.expm1(data * scale_factor)
    denormalized = np.nan_to_num(denormalized, nan=0.0, posinf=clip_value, neginf=0.0)
    return np.clip(denormalized, 0.0, clip_value)

def _setup_colormap(config: Dict):
    """Creates colormap and norm from config, falling back to defaults."""
    vis_cfg = config.get('visualization', {})
    boundaries = vis_cfg.get('boundaries', DEFAULT_BOUNDARIES)
    cmap_data = vis_cfg.get('cmap_data', DEFAULT_CMAP_DATA)

    plot_cmap = ListedColormap(cmap_data)
    plot_cmap.set_over(cmap_data[-1])
    plot_cmap.set_under(cmap_data[0])
    norm = BoundaryNorm(boundaries, plot_cmap.N, clip=False)
    return plot_cmap, norm, boundaries

def _create_static_image(
    save_path: str,
    sequences_denorm: List[np.ndarray],
    labels: List[str],
    cmap, norm, boundaries,
    config: Dict
):
    """
    Generates a single PNG file with a grid of all sequences,
    correctly handling sequences of different lengths.
    """
    vis_cfg = config.get('visualization', {})
    data_cfg = config.get('data', {})

    fs = vis_cfg.get('fontsize', 12)
    dpi = vis_cfg.get('dpi', 150)
    interval_minutes = data_cfg.get('time_interval_minutes', 30)

    # --- Handle variable sequence lengths ---
    # Determine grid dimensions based on the longest sequence
    num_rows = len(sequences_denorm)
    sequence_lengths = [seq.shape[0] for seq in sequences_denorm]
    num_cols = max(sequence_lengths) if sequence_lengths else 0

    if num_rows == 0 or num_cols == 0:
        warnings.warn("No sequences provided to visualize.")
        return

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols,
        figsize=(num_cols * 2.5, num_rows * 2.6), # Slightly more height for labels
        squeeze=False,
        subplot_kw={'xticks': [], 'yticks': []}
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for i in range(num_rows):
        # Set the row label (e.g., "Context", "Target") on the first column
        axes[i, 0].set_ylabel(labels[i], fontsize=fs, fontweight='bold', labelpad=20)

        current_seq_len = sequence_lengths[i]

        for j in range(num_cols):
            ax = axes[i, j]
            # Only plot if the current frame index 'j' is valid for this sequence
            if j < current_seq_len:
                ax.imshow(sequences_denorm[i][j], cmap=cmap, norm=norm, interpolation='nearest')

                # Add time labels to prediction rows
                if labels[i] in ["Target", "Ground Truth", "Prediction", "Forecast"]:
                    time_min = int(interval_minutes * (j + 1))
                    ax.set_xlabel(f"+{time_min} min", fontsize=fs - 2)
            else:
                # If there's no frame, turn the axis off completely
                ax.axis('off')

    plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)


def _create_animated_gif(save_path: str, sequence_denorm: np.ndarray, label: str, cmap, norm, boundaries, config: Dict):
    """Generates a single, geo-referenced GIF for a forecast sequence."""
    if not CARTOPY_AVAILABLE:
        warnings.warn("Cannot create geo-referenced GIF because Cartopy is not installed.")
        return

    # Extract Config
    vis_cfg = config.get('visualization', {})
    data_cfg = config.get('data', {})
    preprocess_cfg = config.get('preprocess', {})

    lon_range = preprocess_cfg.get('lon_range')
    lat_range = preprocess_cfg.get('lat_range')
    if not lon_range or not lat_range:
        raise ValueError("Config must contain 'preprocess.lon_range' and 'preprocess.lat_range' for a geo-referenced GIF.")

    start_time_str = vis_cfg.get('start_time')
    start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now()
    interval_minutes = data_cfg.get('time_interval_minutes', 30)
    dpi = vis_cfg.get('dpi', 120)
    colorbar_label = vis_cfg.get('colorbar_label', "Precipitation Rate (mm/hr)")

    num_frames = sequence_denorm.shape[0]

    # Create Figure with Map
    fig = plt.figure(figsize=(8, 7))
    proj = ccrs.PlateCarree()
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    ax.set_extent(extent, crs=proj)

    # Animation Update Function
    def update(frame_index):
        ax.clear() # Clear previous frame's data
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False

        frame_data = sequence_denorm[frame_index]
        im = ax.imshow(frame_data, extent=extent, origin='upper', cmap=cmap, norm=norm, transform=proj)

        current_time = start_time + timedelta(minutes=interval_minutes * (frame_index + 1))
        time_str = current_time.strftime('%Y-%m-%d %H:%M UTC')
        ax.set_title(f"{label}\n{time_str}", fontsize=12, fontweight='bold')
        return [im]

    # Create and Save Animation
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=boundaries, extend='max')
    cbar.set_label(colorbar_label, fontsize=10)
    fig.subplots_adjust(bottom=0.15)

    ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
    ani.save(save_path, writer='pillow', dpi=dpi)
    plt.close(fig)

# --- PUBLIC API ---

def visualize_sequence(
    save_path: str,
    sequences: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Union[str, Sequence[str]],
    config: Dict,
):
    """
    Primary visualization function to generate static grids or animated GIFs.

    - '.png', '.jpg' -> Creates a static grid of all sequences.
    - '.gif' -> Creates an animated, geo-referenced plot of the *last* sequence provided.

    Args:
        save_path (str): The path to save the output file (e.g., "vis.png").
        sequences (Union[np.ndarray, Sequence[np.ndarray]]):
            A single data sequence (T, H, W, C) or a list of sequences.
            Data is expected to be NORMALIZED.
        labels (Union[str, Sequence[str]]): A label or list of labels for the sequences.
        config (Dict): Configuration dictionary.
    """
    try:
        # Standardize Inputs
        seq_list = [sequences] if isinstance(sequences, np.ndarray) else list(sequences)
        label_list = [labels] if isinstance(labels, str) else list(labels)
        if len(seq_list) != len(label_list):
            raise ValueError("Number of sequences must match number of labels.")

        # Setup Colormap and Denormalize Data
        cmap, norm, boundaries = _setup_colormap(config)
        clip_value = config.get('preprocess', {}).get('clip_value', DEFAULT_CLIP_VALUE)
        sequences_denorm = [_denormalize_log(seq.squeeze(-1), clip_value) for seq in seq_list]

        # Dispatch to Appropriate Plotting Function
        output_format = os.path.splitext(save_path)[1].lower()

        if output_format == '.gif':
            # For GIFs, animate the LAST sequence (assumed to be the forecast)
            print(f"Creating animated GIF for sequence: '{label_list[-1]}'")
            _create_animated_gif(save_path, sequences_denorm[-1], label_list[-1], cmap, norm, boundaries, config)
        else:
            print(f"Creating static image with sequences: {label_list}")
            _create_static_image(save_path, sequences_denorm, label_list, cmap, norm, boundaries, config)

        print(f"Visualization saved to: {save_path}")

    except Exception as e:
        warnings.warn(f"Visualization failed: {e}")
        print(traceback.format_exc())
    finally:
        plt.close('all') # Ensure figures are closed