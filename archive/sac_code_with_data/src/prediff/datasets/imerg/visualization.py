import os
import math
import warnings
import traceback
from typing import Optional, Sequence, Union, Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm as matplotlib_cm

# --- CONSTANTS ---

# Boundaries for precipitation rate in mm/hr, defining color intervals
IMERG_BOUNDARIES = [0.0, 0.1, 2.5, 7.6, 16.0, 50.0, 100.0]

# Colors corresponding to the intervals defined by IMERG_BOUNDARIES
IMERG_CMAP_DATA = [
    (1.0, 1.0, 1.0),    # White: [0.0, 0.1) - No significant rain
    (0.7, 0.85, 0.95),  # Light Blue: [0.1, 2.5) - Light rain
    (0.2, 0.6, 0.85),   # Medium Blue: [2.5, 7.6) - Moderate rain
    (0.5, 0.85, 0.5),   # Light Green: [7.6, 16.0) - Heavy rain
    (1.0, 0.8, 0.2),    # Yellow/Orange: [16.0, 50.0) - Very heavy rain
    (0.9, 0.0, 0.0),    # Red: [50.0, 100.0) - Extreme rain
]

# Default clipping value for denormalization
DEFAULT_CLIP_VALUE = 100.0

# --- NAMING IMPROVEMENT (User Request) ---
# Mapping for more descriptive and attractive plot labels
LABEL_MAPPING = {
    "Context": "Input Context",
    "Target": "Ground Truth",
    "Prediff with KA": "Prediction (Adv. Loss)",
    "Prediff": "Prediction",
}

def _denormalize_log(data: np.ndarray, clip_value: float = DEFAULT_CLIP_VALUE) -> np.ndarray:
    """Internal helper to denormalize data that is log-normalized."""
    if clip_value <= 0:
        warnings.warn(f"Clip value ({clip_value}) must be positive. Returning original data.")
        return data.astype(np.float32)
    data = data.astype(np.float32)
    scale_factor = np.log1p(clip_value)
    if scale_factor == 0:
        scale_factor = 1e-6
    try:
        denormalized = np.expm1(data * scale_factor)  # e^(data * scale) - 1
    except OverflowError:
        warnings.warn("Overflow encountered during exponential denormalization. Clipping result.")
        denormalized = np.full_like(data, clip_value)

    # Replace non-finite values and ensure data is within [0, clip_value]
    denormalized = np.nan_to_num(denormalized, nan=0.0, posinf=clip_value, neginf=0.0)
    return np.clip(denormalized, 0.0, clip_value)

def vis_imerg_seq(
    save_path: str,
    seq: Union[np.ndarray, Sequence[np.ndarray]],
    label: Union[str, Sequence[str]] = "Sequence",
    denormalize: bool = True,
    denorm_clip_value: float = DEFAULT_CLIP_VALUE,
    colorbar_label: str = "Precipitation Rate (mm/hr)",
    interval_real_time: float = 30.0,
    plot_stride: int = 1,
    label_rotation: int = 0,
    label_pad: int = 20,
    label_ha: str = 'right',
    label_avg_int: bool = False,
    fs: int = 12,
    max_cols: int = 7,
    fig_title: Optional[str] = None,
    show_colorbar: bool = True,
    dpi: int = 150,
):
    """
    Visualizes one or more sequences of 2D precipitation fields with a colorbar,
    producing high-quality, readable, and attractive plots.

    Args:
        save_path (str): Path to save the output visualization.
        seq (Union[np.ndarray, Sequence[np.ndarray]]): A single sequence (T, H, W)
            or a list/tuple of sequences. Assumed to be normalized if denormalize=True.
        label (Union[str, Sequence[str]], optional): Label(s) for the sequence(s).
            Special labels like "Context", "Target" are automatically beautified. Defaults to "Sequence".
        denormalize (bool, optional): Whether to apply log denormalization. Defaults to True.
        denorm_clip_value (float, optional): Maximum value for clipping. Defaults to DEFAULT_CLIP_VALUE.
        colorbar_label (str, optional): Label for the colorbar. Defaults to "Precipitation Rate (mm/hr)".
        interval_real_time (float, optional): Time interval between frames in minutes. Defaults to 30.0.
        plot_stride (int, optional): Plot every Nth frame. Defaults to 1.
        label_rotation (int, optional): Rotation angle for row labels. Defaults to 0.
        label_pad (int, optional): Padding for row labels. Defaults to 20.
        label_ha (str, optional): Horizontal alignment for row labels. Defaults to 'right'.
        label_avg_int (bool, optional): Append average intensity to row labels. Defaults to False.
        fs (int, optional): Base font size. Defaults to 12.
        max_cols (int, optional): Maximum number of plot columns per row. Defaults to 7.
        fig_title (Optional[str], optional): Overall figure title. A title like "Test Epc 8 Idx 0"
            will be automatically formatted to "Test Epoch 8, Index 0". Defaults to None.
        show_colorbar (bool, optional): Whether to display the colorbar. Defaults to True.
        dpi (int, optional): Dots per inch for the saved figure. Defaults to 150.
    """
    fig = None
    try:
        # --- 1. SETUP & PREPARATION ---

        # Reformat figure title for better readability (User Request)
        if fig_title:
            try:
                parts = fig_title.strip().split()
                if len(parts) == 5 and parts[0].lower() == 'test' and parts[1].lower() == 'epc' and parts[3].lower() == 'idx':
                    fig_title = f"Test Epoch {parts[2]}, Index {parts[4]}"
            except Exception:
                warnings.warn(f"Could not parse figure title '{fig_title}'. Using original.")

        # Setup colormap and boundaries
        if len(IMERG_BOUNDARIES) != len(IMERG_CMAP_DATA) + 1:
            raise ValueError("Boundaries length must be one more than cmap_data length.")
        
        plot_cmap = ListedColormap(IMERG_CMAP_DATA)
        plot_cmap.set_over(IMERG_CMAP_DATA[-1])  # Color for values >= max boundary
        plot_cmap.set_under(IMERG_CMAP_DATA[0]) # Color for values < min boundary
        cmap_norm = BoundaryNorm(IMERG_BOUNDARIES, plot_cmap.N, clip=False)

        # --- 2. DATA PROCESSING ---

        # Standardize input sequences and labels
        raw_seq_list = [seq] if isinstance(seq, np.ndarray) else list(seq)
        if isinstance(label, str):
            label_list_in = [label] * len(raw_seq_list)
        elif isinstance(label, Sequence) and len(label) == len(raw_seq_list):
            label_list_in = list(label)
        else:
            warnings.warn("Label mismatch or incorrect type. Using default labels.")
            label_list_in = [f"Seq {i+1}" for i in range(len(raw_seq_list))]

        # Process each sequence: denormalize, apply proper names, and calculate stats
        processed_seq_list, final_label_list = [], []
        for s_raw, lbl in zip(raw_seq_list, label_list_in):
            s_proc = s_raw.astype(np.float32)
            if s_proc.ndim == 4 and s_proc.shape[-1] == 1:
                s_proc = s_proc.squeeze(-1) # Handle (T, H, W, 1)
            if s_proc.ndim != 3:
                raise ValueError(f"Input sequence must be 3-dimensional (T, H, W), but got {s_proc.ndim} dims.")

            s_denorm = _denormalize_log(s_proc, clip_value=denorm_clip_value) if denormalize else np.clip(s_proc, 0.0, denorm_clip_value)
            
            # Apply pretty name mapping (User Request)
            current_label = LABEL_MAPPING.get(lbl, lbl)
            
            if label_avg_int:
                mean_val = np.nanmean(s_denorm)
                current_label = f"{current_label}\nAvg: {mean_val:.2f}"
            
            processed_seq_list.append(s_denorm)
            final_label_list.append(current_label)

        # Subsample sequences based on plot_stride
        plot_seq_list = [s[::plot_stride] for s in processed_seq_list]
        seq_len_list_strided = [len(s) for s in plot_seq_list]

        if not any(seq_len_list_strided):
            warnings.warn("No frames to plot after applying stride.")
            return

        # --- 3. FIGURE LAYOUT ---

        # Wrap long sequences into multiple plot rows
        effective_ncols = min(max(seq_len_list_strided), max_cols)
        wrapped_rows_info = []
        for seq_arr, lbl, seq_len in zip(plot_seq_list, final_label_list, seq_len_list_strided):
            num_plot_rows = math.ceil(seq_len / effective_ncols) if effective_ncols > 0 else 1
            for j_row in range(num_plot_rows):
                start, end = j_row * effective_ncols, min(seq_len, (j_row + 1) * effective_ncols)
                wrapped_rows_info.append({
                    "data": seq_arr[start:end],
                    "label": lbl if j_row == 0 else "",
                    "len": end - start,
                })

        if not wrapped_rows_info:
            warnings.warn("No data rows to plot after wrapping.")
            return

        # Create figure and axes
        nrows = len(wrapped_rows_info)
        width_per_col, height_per_row = 3, 3
        fig_width = width_per_col * effective_ncols
        fig_height = height_per_row * nrows + (0.5 if any(s_len > 0 for s_len in seq_len_list_strided) else 0)
        if show_colorbar:
            fig_width += 1.5

        fig, axes = plt.subplots(nrows=nrows, ncols=effective_ncols, figsize=(fig_width, fig_height), squeeze=False)
        
        # Adjust spacing for a clean look
        left_margin = 0.22 if label_rotation == 0 else 0.15
        right_margin = 0.88 if show_colorbar else 0.98
        # MODIFICATION: Reduced hspace and wspace to pack plots tighter, making them larger.
        plt.subplots_adjust(hspace=0.05, wspace=0.05, top=0.92, bottom=0.15, left=left_margin, right=right_margin)
        if fig_title:
            fig.suptitle(fig_title, fontsize=fs + 2, fontweight='bold', y=0.98)

        # --- 4. PLOTTING ---

        # Plot each frame
        for i, row_info in enumerate(wrapped_rows_info):
            if row_info["label"]:
                axes[i, 0].set_ylabel(
                    row_info["label"], rotation=label_rotation, labelpad=label_pad,
                    fontsize=fs, fontweight='bold', ha=label_ha, va='center'
                )
            for j in range(effective_ncols):
                ax = axes[i, j]
                if j >= row_info["len"]:
                    ax.axis('off')
                    continue
                
                frame_data = np.nan_to_num(row_info["data"][j])
                ax.imshow(frame_data, cmap=plot_cmap, norm=cmap_norm, interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])

        # Add time labels to the bottom-most axis of each column
        max_frames_in_row = max(info["len"] for info in wrapped_rows_info)
        for j in range(max_frames_in_row):
            original_frame_index = j * plot_stride
            time_minutes = int(interval_real_time * (original_frame_index + 1))
            axes[nrows - 1, j].set_title(f"+{time_minutes} min", y=-0.2, fontsize=fs - 1)

        # Add colorbar
        if show_colorbar:
            cax = fig.add_axes([
                axes[0, -1].get_position().x1 + 0.01,
                axes[-1, -1].get_position().y0,
                0.015,
                axes[0, 0].get_position().y1 - axes[-1, -1].get_position().y0
            ])
            sm = matplotlib_cm.ScalarMappable(norm=cmap_norm, cmap=plot_cmap)
            cbar = fig.colorbar(sm, cax=cax, orientation='vertical', ticks=IMERG_BOUNDARIES, extend='max', spacing='proportional')
            cbar.set_label(colorbar_label, fontsize=fs, labelpad=10)
            cbar.ax.tick_params(labelsize=fs - 1)
            tick_labels = [f"{b:.1f}" if 0 < b < 10 and b % 1 != 0 else f"{int(b)}" for b in IMERG_BOUNDARIES]
            cbar.ax.set_yticklabels(tick_labels)

        # --- 5. SAVE & CLEANUP ---
        
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Saving visualization to: {save_path}")

    except Exception as e:
        warnings.warn(f"Visualization failed: {e}")
        print(traceback.format_exc())
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)