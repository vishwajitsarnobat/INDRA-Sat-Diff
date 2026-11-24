# climate_forecast/datasets/visualization.py

import os
import warnings
import traceback
from typing import Optional, Sequence, Union, Dict, List
from datetime import datetime, timedelta, timezone

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.animation as animation
import matplotlib.cm as cm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.gridspec import GridSpec
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False

DEFAULT_BOUNDARIES = [0.0, 0.1, 2.5, 7.6, 16.0, 50.0, 100.0]
DEFAULT_CLIP_VALUE = 100.0

def _denormalize_log(data: np.ndarray, clip_value: float) -> np.ndarray:
    """Undo log-normalization and clip to precipitation range."""
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
    """Build precipitation colormap as requested."""
    vis_cfg = config.get('visualization', {})
    boundaries = vis_cfg.get('boundaries', DEFAULT_BOUNDARIES)
    cmap_data = vis_cfg.get('cmap_data')
    if cmap_data:
        cmap = ListedColormap(np.array(cmap_data))
    else:
        cmap = matplotlib.colormaps["turbo"].resampled(len(boundaries) - 1)
    norm = BoundaryNorm(boundaries, cmap.N, clip=False)
    return cmap, norm, boundaries

def _create_vae_reconstruction_image(
    save_path,
    gt_seq,
    recon_seq,
    cmap,
    norm,
    boundaries,
    config,
    title,
):
    """Ground Truth vs Reconstruction plots with research-paper quality layout."""
    dpi = 200
    cbar_label = config.get("visualization", {}).get("colorbar_label", "Precipitation (mm/hr)")

    num_cols = min(gt_seq.shape[0], recon_seq.shape[0], 8)
    if num_cols < 1:
        warnings.warn("Not enough images to create VAE plot.")
        return

    # Responsive sizing: cap total width to prevent overly wide figures
    subplot_w = 3.5
    subplot_h = 3.5
    fig_width = min(num_cols * subplot_w, 22)
    fig_height = 2 * subplot_h + 1.5  # two rows + room for title + colorbar
    fig, axes = plt.subplots(
        2,
        num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        subplot_kw={"xticks": [], "yticks": []},
    )

    # Responsive font sizes: larger for few columns, smaller for many
    fs = max(13, min(18, 20 - max(0, num_cols - 3) * 1.5))
    fig.suptitle(title, fontsize=fs + 2, fontweight="bold")

    for j in range(num_cols):
        axes[0, j].imshow(gt_seq[j], cmap=cmap, norm=norm, interpolation="nearest")
        axes[1, j].imshow(recon_seq[j], cmap=cmap, norm=norm, interpolation="nearest")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=fs, fontweight="bold")
    axes[1, 0].set_ylabel("Reconstruction", fontsize=fs, fontweight="bold")

    # Generous margins to prevent text overlap
    plt.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.16,
                        hspace=0.20, wspace=0.15)
    fig.align_ylabels(axes[:, 0])

    # Dynamically position colorbar below the subplot area
    bboxes = [ax.get_position() for ax in axes.flat if ax.get_visible()]
    cbar_left = min(b.x0 for b in bboxes)
    cbar_right = max(b.x1 for b in bboxes)
    cbar_bottom = min(b.y0 for b in bboxes) - 0.06
    cbar_width = cbar_right - cbar_left
    cbar_height = 0.025
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                        ticks=boundaries, extend="max")
    cbar.ax.tick_params(labelsize=fs - 4)
    cbar.set_label(cbar_label, fontsize=fs - 2, fontweight="bold")

    plt.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _create_static_image(
    save_path: str,
    sequences_denorm: List[np.ndarray],
    labels: List[str],
    cmap,
    norm,
    boundaries,
    config: Dict,
    title: str = "Model Output",
):
    """General static image plotting for forecasts with publication-quality layout."""
    is_vae_recon = (len(sequences_denorm) == 2 and
                    "Reconstruction" in labels and
                    "Ground Truth" in labels)
    if is_vae_recon:
        gt_index = labels.index("Ground Truth")
        recon_index = labels.index("Reconstruction")
        _create_vae_reconstruction_image(
            save_path, sequences_denorm[gt_index], sequences_denorm[recon_index], cmap, norm, boundaries, config, title
        )
        return

    vis_cfg = config.get("visualization", {})
    data_cfg = config.get("data", {})
    dpi = 200
    interval_minutes = data_cfg.get("time_interval_minutes", 30)
    cbar_label = vis_cfg.get("colorbar_label", "Precipitation (mm/hr)")

    num_rows = len(sequences_denorm)
    num_cols = max(seq.shape[0] for seq in sequences_denorm) if sequences_denorm else 0
    if num_rows == 0 or num_cols == 0:
        return

    # Responsive sizing with caps to prevent sprawling figures
    subplot_w = 3.0
    subplot_h = 3.0
    fig_width = min(num_cols * subplot_w, 22)
    fig_height = min(num_rows * subplot_h + 1.5, 14)
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        subplot_kw={"xticks": [], "yticks": []},
    )

    # Responsive font sizes: smaller for many columns, larger for few
    fs = max(12, min(18, 20 - max(0, num_cols - 4) * 1.0))
    fig.suptitle(title, fontsize=fs + 2, fontweight="bold")

    for i, seq in enumerate(sequences_denorm):
        axes[i, 0].set_ylabel(labels[i], fontsize=fs, fontweight="bold")

        for j in range(num_cols):
            ax = axes[i, j]
            if j < seq.shape[0]:
                ax.imshow(seq[j], cmap=cmap, norm=norm, interpolation="nearest")
                time_min = (-int(interval_minutes * (seq.shape[0] - 1 - j))
                            if "Input Context" in labels[i]
                            else int(interval_minutes * (j + 1)))
                title_text = "T=0" if time_min == 0 else f"T{time_min:+} min"
                ax.set_title(title_text, fontsize=max(10, fs - 5), fontweight="normal")
            else:
                ax.axis("off")

    # Generous margins, critical for preventing text overlap
    fig.subplots_adjust(left=0.10, right=0.95, top=0.90, bottom=0.14,
                        hspace=0.25, wspace=0.20)
    fig.align_ylabels(axes[:, 0])

    # Dynamically position colorbar below the subplot bounding box
    bboxes = [ax.get_position() for ax in axes.flat if ax.get_visible()]
    cbar_left = min(b.x0 for b in bboxes)
    cbar_right = max(b.x1 for b in bboxes)
    cbar_bottom = min(b.y0 for b in bboxes) - 0.07
    cbar_width = cbar_right - cbar_left
    cbar_height = 0.020
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal",
                        ticks=boundaries, extend="max")
    cbar.ax.tick_params(labelsize=max(9, fs - 6))
    cbar.set_label(cbar_label, fontsize=fs - 2, fontweight="bold")

    plt.savefig(save_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.close(fig)

def _create_animated_gif(
    save_path: str,
    sequence_denorm: np.ndarray,
    label: str,
    cmap,
    norm,
    boundaries,
    config: Dict,
):
    """Geo-referenced animated GIFs using Cartopy."""
    if not CARTOPY_AVAILABLE:
        warnings.warn("Cartopy not installed. Cannot create geo-referenced GIF.")
        return

    vis_cfg, data_cfg, preprocess_cfg = (
        config.get("visualization", {}),
        config.get("data", {}),
        config.get("preprocess", {}),
    )
    lon_range = preprocess_cfg.get("lon_range")
    lat_range = preprocess_cfg.get("lat_range")
    if not lon_range or not lat_range:
        warnings.warn("Geo-referenced GIF requires 'preprocess.lon_range' and 'lat_range' in config.")
        return

    start_time_str = vis_cfg.get("start_time")
    start_time = datetime.fromisoformat(start_time_str) if start_time_str else datetime.now(timezone.utc)
    interval_minutes = data_cfg.get("time_interval_minutes", 30)
    dpi = vis_cfg.get("dpi", 150)
    fs = 16
    cbar_label = vis_cfg.get("colorbar_label", "Precipitation (mm/hr)")

    fig = plt.figure(figsize=(8, 8.5))
    proj = ccrs.PlateCarree()
    gs = GridSpec(2, 1, height_ratios=[20, 1], hspace=0.15, figure=fig)
    ax = fig.add_subplot(gs[0, 0], projection=proj)
    cbar_ax = fig.add_subplot(gs[1, 0])

    def update(frame_index):
        ax.clear()
        extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
        ax.set_extent(extent, crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6, zorder=2)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
        gl.top_labels, gl.right_labels = False, False

        im = ax.imshow(sequence_denorm[frame_index], extent=extent, origin="upper",
                       cmap=cmap, norm=norm, alpha=0.8, zorder=1)
        current_time = start_time + timedelta(minutes=interval_minutes * (frame_index + 1))
        ax.set_title(f"{label}\n{current_time.strftime('%Y-%m-%d %H:%M UTC')}",
                     fontsize=fs, fontweight="bold")
        return [im]

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal", ticks=boundaries, extend="max")
    cbar.set_label(cbar_label, fontsize=fs, fontweight="bold")
    cbar.ax.tick_params(labelsize=fs - 2)

    ani = animation.FuncAnimation(fig, update, frames=sequence_denorm.shape[0],
                                  blit=True, interval=2000)  # 2s per frame
    ani.save(save_path, writer="pillow", dpi=dpi)
    plt.close(fig)

def visualize_sequence(
    save_path: str,
    sequences: Union[np.ndarray, Sequence[np.ndarray]],
    labels: Union[str, Sequence[str]],
    config: Dict,
    title: Optional[str] = None,
):
    """Main visualization entry point."""
    try:
        seq_list = [sequences] if isinstance(sequences, np.ndarray) else list(sequences)
        label_list = [labels] if isinstance(labels, str) else list(labels)
        if len(seq_list) != len(label_list):
            raise ValueError("Number of sequences must match number of labels.")

        cmap, norm, boundaries = _setup_colormap(config)
        clip_value = config.get("visualization", {}).get("denorm_clip_value", DEFAULT_CLIP_VALUE)
        sequences_denorm = [_denormalize_log(seq.squeeze(-1), clip_value) for seq in seq_list]

        default_title = f"Visualization for {os.path.basename(save_path)}"
        plot_title = title if title is not None else default_title

        output_format = os.path.splitext(save_path)[1].lower()
        if output_format == ".gif":
            if len(sequences_denorm) > 1:
                warnings.warn(f"GIF creation only supports one sequence. Visualizing the first: '{label_list[0]}'.")
            _create_animated_gif(save_path, sequences_denorm[0], label_list[0], cmap, norm, boundaries, config)
        else:
            _create_static_image(save_path, sequences_denorm, label_list, cmap, norm, boundaries, config, title=plot_title)

    except Exception as e:
        warnings.warn(f"Visualization failed: {e}")
        traceback.print_exc()
    finally:
        plt.close("all")