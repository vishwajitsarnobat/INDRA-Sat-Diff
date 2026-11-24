import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime, timedelta
import warnings
import argparse
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import multiprocessing

# --- PLOTTING CONFIGURATION (for Publication Quality) ---
plt.rcParams.update({
    'font.size': 14, 'font.family': 'sans-serif', 'axes.titlesize': 18,
    'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12, 'font.weight': 'bold', 'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})
warnings.filterwarnings("ignore", message="The figure layout has changed to tight")
warnings.filterwarnings("ignore", message="`torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.")


# --- USER CONFIGURATION ---
# TODO: Update this path to your shapefile location.
SHAPEFILE_PATH = "Shapefile/states_IND.shp" 
BATCH_SIZE = 4
NUM_WORKERS = min(4, multiprocessing.cpu_count())


from ..prediff.imerg.train_imerg_prediff import PreDiffIMERGPLModule

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warnings.warn("Basemap not found. Spatial plots will be disabled. Install with 'conda install -c conda-forge basemap'")
    Basemap = None

# --- CONSTANTS (from reference visualization code) ---
DATA_EPOCH = datetime(1980, 1, 1)
IMERG_BOUNDARIES = [0.0, 0.1, 2.5, 7.6, 16.0, 50.0, 100.0]
IMERG_CMAP_DATA = [
    (1.0, 1.0, 1.0),    # White: [0.0, 0.1)
    (0.7, 0.85, 0.95),  # Light Blue: [0.1, 2.5)
    (0.2, 0.6, 0.85),   # Medium Blue: [2.5, 7.6)
    (0.5, 0.85, 0.5),   # Light Green: [7.6, 16.0)
    (1.0, 0.8, 0.2),    # Yellow/Orange: [16.0, 50.0)
    (0.9, 0.0, 0.0),    # Red: [50.0, 100.0)
]
CMAP_BINS = mcolors.ListedColormap(IMERG_CMAP_DATA)
CMAP_BINS.set_over(IMERG_CMAP_DATA[-1])
CMAP_BINS.set_under(IMERG_CMAP_DATA[0])
NORM_BINS = mcolors.BoundaryNorm(IMERG_BOUNDARIES, CMAP_BINS.N, clip=False)


def get_region_config(region_name):
    base_region = region_name.split('_')[0]
    is_test_set = '_test' in region_name
    if base_region == 'south':
        config = {'name': 'Southern India', 'pretty_name': 'Southern India', 'x1': 71.0, 'x2': 83.8, 'y1': 5.0, 'y2': 17.8}
    elif base_region == 'ne':
        config = {'name': 'Northeastern India', 'pretty_name': 'Northeastern India', 'x1': 86.1, 'x2': 98.9, 'y1': 19.1, 'y2': 31.9}
    else: raise ValueError(f"Unknown base region '{base_region}'.")
    if is_test_set: config['pretty_name'] += ' (Test Set)'
    return config

def extract_timestamp(fpath: str) -> int:
    try:
        with h5py.File(fpath, 'r') as f: return int(f['time'][()])
    except Exception: return None

def seconds_from_epoch_to_datetime(seconds: int) -> datetime:
    return DATA_EPOCH + timedelta(seconds=seconds)

def load_sequence(filepaths: list) -> np.ndarray:
    frames = [np.squeeze(h5py.File(fpath, 'r')['precipitationCal'][:], axis=0) for fpath in filepaths]
    return np.stack(frames)

def inverse_preprocess_data(norm_data: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    scale_factor = np.log1p(clip_value)
    with warnings.catch_warnings(): warnings.simplefilter("ignore", RuntimeWarning)
    original_data = np.expm1(norm_data * scale_factor)
    original_data[original_data < 0] = 0.0
    return original_data

class PreDiffPredictor:
    def __init__(self, prediff_model: PreDiffIMERGPLModule, device):
        self.model = prediff_model.to(device); self.model.eval(); self.device = device

    def predict(self, full_seq_tensor: torch.Tensor):
        full_seq_tensor = full_seq_tensor.to(self.device)
        batch_size, in_len = full_seq_tensor.size(0), self.model.hparams.layout.in_len
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            _, cond = self.model.get_input(full_seq_tensor)
            pred_unaligned = self.model.sample(cond=cond, batch_size=batch_size, use_alignment=False)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float32):
            full_seq_tensor_fp32 = full_seq_tensor.float()
            _, cond_fp32 = self.model.get_input(full_seq_tensor_fp32)
            context_seq_fp32 = full_seq_tensor_fp32[:, :in_len, ...]
            alignment_kwargs = self._get_alignment_kwargs_for_testing(context_seq_fp32)
            pred_aligned = self.model.sample(cond=cond_fp32, batch_size=batch_size, use_alignment=True, alignment_kwargs=alignment_kwargs)
        return pred_unaligned.cpu(), pred_aligned.cpu()

    @staticmethod
    def _get_alignment_kwargs_for_testing(context_seq: torch.Tensor) -> dict:
        multiplier, batch_size = 2.0, context_seq.shape[0]
        avg_intensity = torch.mean(context_seq.view(batch_size, -1), dim=1, keepdim=True)
        return {"avg_x_gt": avg_intensity * multiplier}


class VisualizationDataset(Dataset):
    def __init__(self, data_dir: str, forecast_dir: str, seq_len: int, interval_minutes: int):
        self.seq_len = seq_len; self.data_dir = data_dir
        all_files_sorted = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.HDF5')])
        self.valid_starts = self._find_valid_nonoverlapping_sequences(all_files_sorted, forecast_dir, interval_minutes)
        self.all_files = all_files_sorted
        if not self.valid_starts: print("\nINFO: Found 0 new sequences to process for generation.")
        else: print(f"\nINFO: Found {len(self.valid_starts)} new non-overlapping sequences for forecast generation.")

    def _find_valid_nonoverlapping_sequences(self, all_files_sorted: list, forecast_dir: str, interval_minutes: int) -> list:
        if len(all_files_sorted) < self.seq_len: return []
        expected_delta_sec, tolerance_sec = interval_minutes * 60, 5 * 60; valid_starts = []
        timestamp_cache = {f: extract_timestamp(f) for f in tqdm(all_files_sorted, desc="Caching Timestamps", leave=False)}
        i = 0
        pbar = tqdm(total=len(all_files_sorted), desc="Finding Sequences")
        while i <= len(all_files_sorted) - self.seq_len:
            pbar.update(1)
            first_file_ts = timestamp_cache.get(all_files_sorted[i])
            if first_file_ts is None: i += 1; continue
            datetime_str = seconds_from_epoch_to_datetime(first_file_ts).strftime('%Y%m%d%H%M')
            if os.path.exists(os.path.join(forecast_dir, f"{datetime_str}_unaligned.npy")):
                i += self.seq_len; pbar.update(self.seq_len - 1); continue
            candidate_files = all_files_sorted[i : i + self.seq_len]
            timestamps = [timestamp_cache.get(f) for f in candidate_files]
            if any(ts is None for ts in timestamps): i += 1; continue
            is_contiguous = all(abs(timestamps[j+1] - timestamps[j] - expected_delta_sec) <= tolerance_sec for j in range(self.seq_len - 1))
            if is_contiguous:
                valid_starts.append(i); i += self.seq_len; pbar.update(self.seq_len - 1)
            else: i += 1
        pbar.close()
        return valid_starts

    def __len__(self): return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx = self.valid_starts[idx]
        seq_files = self.all_files[start_idx : start_idx + self.seq_len]
        try:
            seq_data = load_sequence(seq_files)
            seq_data[~np.isfinite(seq_data)] = 0.0
            return torch.from_numpy(seq_data).float().unsqueeze(-1), extract_timestamp(seq_files[0])
        except Exception as e:
            print(f"ERROR loading sequence at index {start_idx}: {e}"); return None, None

def _draw_map_features(basemap, fontsize=10):
    if Basemap is None: return
    parallels = np.arange(0., 90., 5.); basemap.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=fontsize, color='gray', linewidth=0.5, zorder=10)
    meridians = np.arange(0., 180., 5.); basemap.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=fontsize, color='gray', linewidth=0.5, zorder=10)
    try:
        if not os.path.exists(SHAPEFILE_PATH): raise FileNotFoundError(f"Shapefile not found: {SHAPEFILE_PATH}")
        shapefile_base = os.path.splitext(SHAPEFILE_PATH)[0]
        basemap.readshapefile(shapefile_base, 'boundary', drawbounds=True, color='dimgray', linewidth=1.5, zorder=11)
    except Exception as e: warnings.warn(f"Could not read shapefile: {e}")
    basemap.drawcoastlines(linewidth=1.2, color='black', zorder=12)
    basemap.drawcountries(linewidth=1.2, color='black', zorder=12)

# --- REWRITTEN PLOTTING FUNCTION ---
def plot_forecast_panel(out_path, input_seq, target_seq, forecast_seq, start_dt, model_name, metadata):
    """
    Creates a high-quality, 3-row plot with a fixed layout as requested:
    - Row 1: Input Context (up to 7 frames)
    - Row 2: Ground Truth (first 6 forecast frames)
    - Row 3: Prediction (first 6 forecast frames)
    """
    if Basemap is None:
        print(f"Skipping static plot for {model_name} because Basemap is not installed."); return

    MAX_COLS = 7  # Max frames per row, inspired by reference code
    FRAMES_TO_SHOW_LEAD_TIME = 6  # Show first 6 frames for target/prediction

    # Prepare data for each row, truncating where necessary
    plot_data_rows = [
        {'label': 'Input Context', 'data': input_seq},
        {'label': 'Ground Truth',  'data': target_seq[:FRAMES_TO_SHOW_LEAD_TIME]},
        {'label': model_name,      'data': forecast_seq[:FRAMES_TO_SHOW_LEAD_TIME]},
    ]
    
    fig, axes = plt.subplots(3, MAX_COLS, figsize=(MAX_COLS * 3, 10.5), facecolor='white',
                             gridspec_kw={'hspace': 0.15, 'wspace': 0.05})

    ny, nx = input_seq[0].shape
    lons, lats = np.linspace(metadata['x1'], metadata['x2'], nx), np.linspace(metadata['y1'], metadata['y2'], ny)
    im = None

    for row_idx, row_info in enumerate(plot_data_rows):
        # Set the Y-label for the row
        axes[row_idx, 0].set_ylabel(row_info['label'], labelpad=40)
        
        for col_idx in range(MAX_COLS):
            ax = axes[row_idx, col_idx]
            
            # Check if there is data to plot in this cell
            if col_idx < len(row_info['data']):
                m = Basemap(projection='cyl', resolution='i', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'],
                            llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], ax=ax)
                _draw_map_features(m)
                
                # Use imshow with 'nearest' for sharp, non-blurry pixels and correct colors
                im = m.imshow(row_info['data'][col_idx], cmap=CMAP_BINS, norm=NORM_BINS, interpolation='nearest', alpha=0.9)

                # Add time labels only on the top row
                if row_idx == 0:
                    time_label = f"Input\nT-{ (len(input_seq) - col_idx -1) * 30 } min"
                    ax.set_title(time_label, pad=15)
                # Add forecast lead time labels to the second row (for both target and prediction)
                elif row_idx == 1:
                     time_label = f"Forecast\nT+{ (col_idx + 1) * 30 } min"
                     ax.set_title(time_label, pad=15)
            else:
                # If no data, turn off the axis to leave it blank
                ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=IMERG_BOUNDARIES)
    cbar.set_label("Precipitation Rate (mm/hr)", labelpad=10)
    cbar.ax.set_xticklabels([f"{b:.1f}" if 0 < b < 10 and b % 1 != 0 else f"{int(b)}" for b in IMERG_BOUNDARIES])

    fig.suptitle(f"Forecast vs. Ground Truth for {metadata['pretty_name']}\nStart: {start_dt.strftime('%Y-%m-%d %H:%M UTC')}", y=1.0)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def create_comparison_gif(out_path, target_seq, forecast_seq, start_dt, model_name, metadata):
    if Basemap is None:
        print(f"Skipping GIF for {model_name} because Basemap is not installed."); return

    frames = []
    for i in range(target_seq.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8), facecolor='white', gridspec_kw={'wspace': 0.1})
        fig.suptitle(f"{metadata['pretty_name']}: Ground Truth vs. {model_name} Forecast")

        plot_info = [("Ground Truth", target_seq), (f"PreDiff {model_name}", forecast_seq)]
        for j, (title, data) in enumerate(plot_info):
            ax = axes[j]
            m = Basemap(projection='cyl', resolution='i', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'],
                        llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], ax=ax)
            
            # Use imshow with 'nearest' for sharp pixels and the correct color map
            m.imshow(data[i], cmap=CMAP_BINS, norm=NORM_BINS, interpolation='nearest', alpha=0.9)
            _draw_map_features(m, fontsize=12)
            ax.set_title(title, pad=10)

        current_dt = start_dt + timedelta(minutes=i * 30)
        time_text = f"{current_dt.strftime('%Y-%m-%d %H:%M UTC')} (T+{(i + 1) * 30}min)"
        fig.text(0.5, 0.05, time_text, ha='center', fontsize=plt.rcParams['axes.labelsize'])
        
        plt.subplots_adjust(bottom=0.1, top=0.88)
        fig.canvas.draw()
        img = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_argb())
        frames.append(img)
        plt.close(fig)

    if frames:
        frames[0].save(out_path, save_all=True, append_images=frames[1:], duration=750, loop=0)

def main(args):
    metadata = get_region_config(args.region)
    base_dir = f'/home/sac/data_67/Nowcasting/data/required_valid_{args.region}'
    forecast_dir = os.path.abspath(f'forecasts_prediff_case_study_{args.region}')
    plot_dir = os.path.abspath(f'plots_prediff_case_study_{args.region}')
    os.makedirs(forecast_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)

    print(f"\n--- Running Analysis for Region: {metadata['pretty_name']} ---")
    print(f"Data Source: {base_dir}\nForecast Cache: {forecast_dir}\nOutput Plots: {plot_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Loading model...")
    if args.ckpt.endswith(".ckpt"):
        pl_model = PreDiffIMERGPLModule.load_from_checkpoint(
            checkpoint_path=args.ckpt, map_location='cpu', oc_file=args.cfg, total_num_steps=1)
    elif args.ckpt.endswith(".pt"):
        pl_model = PreDiffIMERGPLModule(total_num_steps=1, oc_file=args.cfg)
        pl_model.torch_nn_module.load_state_dict(torch.load(args.ckpt, map_location='cpu'), strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint file extension: {args.ckpt}. Must be '.ckpt' or '.pt'.")

    predictor = PreDiffPredictor(pl_model, device)
    params = {'in_len': pl_model.hparams.layout.in_len, 'out_len': pl_model.hparams.layout.out_len, 'interval_minutes': 30, 'clip_value': 100.0}
    loader_kwargs = {"batch_size": BATCH_SIZE, "num_workers": NUM_WORKERS, "shuffle": False, "pin_memory": True}

    print("\n--- PHASE 1: GENERATING/CACHING FORECASTS ---")
    dataset = VisualizationDataset(base_dir, forecast_dir, params['in_len'] + params['out_len'], params['interval_minutes'])
    
    if len(dataset) > 0:
        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
            if not batch: return None, None
            return torch.stack([b[0] for b in batch]), [b[1] for b in batch]
        
        dataloader = DataLoader(dataset, **loader_kwargs, collate_fn=collate_fn)
        for seq_tensors, start_timestamps in tqdm(dataloader, desc="Generating Forecasts"):
            if seq_tensors is None: continue
            preds_unaligned, preds_aligned = predictor.predict(seq_tensors)
            for i, start_ts in enumerate(start_timestamps):
                datetime_str = seconds_from_epoch_to_datetime(start_ts).strftime('%Y%m%d%H%M')
                np.save(os.path.join(forecast_dir, f"{datetime_str}_unaligned.npy"), preds_unaligned[i].squeeze(-1).numpy())
                np.save(os.path.join(forecast_dir, f"{datetime_str}_aligned.npy"), preds_aligned[i].squeeze(-1).numpy())
    else:
        print("No new forecasts to generate. Proceeding to visualization.")

    print("\n--- PHASE 2: CREATING PLOTS AND GIFS ---")
    all_forecast_files = sorted([f for f in os.listdir(forecast_dir) if f.endswith('_unaligned.npy')])
    if not all_forecast_files: print("No forecast files found to visualize."); return
        
    all_data_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.HDF5')])
    timestamp_to_filepath_map = {extract_timestamp(f): f for f in all_data_files if extract_timestamp(f) is not None}

    for fname in tqdm(all_forecast_files, desc="Creating Visualizations"):
        datetime_str = fname.replace("_unaligned.npy", "")
        plot_unaligned_path = os.path.join(plot_dir, f"{datetime_str}_plot_unaligned.png")
        plot_aligned_path = os.path.join(plot_dir, f"{datetime_str}_plot_aligned.png")
        gif_unaligned_path = os.path.join(plot_dir, f"{datetime_str}_gif_unaligned.gif")
        gif_aligned_path = os.path.join(plot_dir, f"{datetime_str}_gif_aligned.gif")
        if all(map(os.path.exists, [plot_unaligned_path, plot_aligned_path, gif_unaligned_path, gif_aligned_path])): continue
        
        start_dt = datetime.strptime(datetime_str, '%Y%m%d%H%M')
        seq_files, is_sequence_complete = [], True
        for i in range(params['in_len'] + params['out_len']):
            current_dt = start_dt + timedelta(minutes=i * 30)
            current_ts = int((current_dt - DATA_EPOCH).total_seconds())
            gt_file_path = min(timestamp_to_filepath_map.keys(), key=lambda ts: abs(ts - current_ts), default=None)
            if gt_file_path is not None and abs(gt_file_path - current_ts) < 180:
                seq_files.append(timestamp_to_filepath_map[gt_file_path])
            else:
                tqdm.write(f"WARNING: Could not find ground truth file near {current_dt}."); is_sequence_complete = False; break
        if not is_sequence_complete: continue

        full_gt_physical = inverse_preprocess_data(load_sequence(seq_files), params['clip_value'])
        input_gt, target_gt = full_gt_physical[:params['in_len']], full_gt_physical[params['in_len']:]
        forecast_unaligned = inverse_preprocess_data(np.load(os.path.join(forecast_dir, fname)), params['clip_value'])
        forecast_aligned = inverse_preprocess_data(np.load(os.path.join(forecast_dir, fname.replace("unaligned", "aligned"))), params['clip_value'])
        
        if not os.path.exists(plot_unaligned_path):
            plot_forecast_panel(plot_unaligned_path, input_gt, target_gt, forecast_unaligned, start_dt, "PreDiff Unaligned", metadata)
        if not os.path.exists(plot_aligned_path):
            plot_forecast_panel(plot_aligned_path, input_gt, target_gt, forecast_aligned, start_dt, "PreDiff Aligned", metadata)
        
        gif_start_dt = start_dt + timedelta(minutes=params['in_len'] * 30)
        if not os.path.exists(gif_unaligned_path):
            create_comparison_gif(gif_unaligned_path, target_gt, forecast_unaligned, gif_start_dt, "Unaligned", metadata)
        if not os.path.exists(gif_aligned_path):
            create_comparison_gif(gif_aligned_path, target_gt, forecast_aligned, gif_start_dt, "Aligned", metadata)

    print(f"\nProcessing complete. All visualizations saved in '{plot_dir}'.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate and visualize PreDiff model forecasts.")
    parser.add_argument('--region', type=str, choices=['south', 'ne', 'south_test', 'ne_test'], required=True)
    parser.add_argument('--cfg', type=str, required=True, help="Path to the model's YAML configuration file.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model's .pt or .ckpt checkpoint file.")
    main(parser.parse_args())