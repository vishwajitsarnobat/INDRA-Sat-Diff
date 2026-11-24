import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings
import argparse
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import scipy.ndimage  # Added for FSS calculation

# Removed: from pysteps import verification

from ..prediff.imerg.train_imerg_prediff import PreDiffIMERGPLModule

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warnings.warn("Basemap not found. Spatial plots will be disabled. Install with 'conda install -c conda-forge basemap'")
    Basemap = None

METRICS_TO_PLOT = ['POD', 'FAR', 'CSI']
DATA_EPOCH = datetime(1980, 1, 1)

# ----------------------------------------------------------------------------------
# NEW SELF-CONTAINED FUNCTIONS (REPLACING PYSTEPS)
# ----------------------------------------------------------------------------------

def calculate_det_cat_fct(pred: np.ndarray, obs: np.ndarray, thr: float, scores: list = ['POD', 'FAR', 'CSI']) -> dict:
    """
    Calculates deterministic categorical scores.
    This is a replacement for pysteps.verification.det_cat_fct.
    
    Args:
        pred (np.ndarray): Prediction array.
        obs (np.ndarray): Observation array.
        thr (float): The threshold for event/no-event.
        scores (list): A list of scores to compute.
    
    Returns:
        dict: A dictionary containing the calculated scores.
    """
    pred_bin = pred >= thr
    obs_bin = obs >= thr
    
    hits = np.sum(np.logical_and(pred_bin, obs_bin))
    misses = np.sum(np.logical_and(np.logical_not(pred_bin), obs_bin))
    false_alarms = np.sum(np.logical_and(pred_bin, np.logical_not(obs_bin)))
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-6
    
    out = {}
    if 'POD' in scores:
        out['POD'] = hits / (hits + misses + epsilon)
    if 'FAR' in scores:
        out['FAR'] = false_alarms / (hits + false_alarms + epsilon)
    if 'CSI' in scores:
        out['CSI'] = hits / (hits + misses + false_alarms + epsilon)
        
    return out

def calculate_fss(pred: np.ndarray, obs: np.ndarray, thr: float, scale: int) -> float:
    """
    Calculates the Fractions Skill Score (FSS).
    This is a replacement for the pysteps FSS verifier.
    
    Args:
        pred (np.ndarray): Prediction array.
        obs (np.ndarray): Observation array.
        thr (float): The threshold for event/no-event.
        scale (int): The spatial scale (window size) in pixels.
        
    Returns:
        float: The FSS score.
    """
    pred_bin = (pred >= thr).astype(float)
    obs_bin = (obs >= thr).astype(float)

    # Calculate fractional coverage using a uniform filter (fast convolution)
    pred_frac = scipy.ndimage.uniform_filter(pred_bin, size=scale)
    obs_frac = scipy.ndimage.uniform_filter(obs_bin, size=scale)
    
    # Calculate MSE and reference MSE
    mse = np.mean((pred_frac - obs_frac) ** 2)
    mse_ref = np.mean(pred_frac ** 2) + np.mean(obs_frac ** 2)
    
    # Add epsilon to avoid division by zero if both fields are all zeros
    epsilon = 1e-6
    if mse_ref < epsilon:
        return 1.0  # Perfect score if both fields are empty
    
    return 1.0 - (mse / mse_ref)

# ----------------------------------------------------------------------------------
# END OF NEW FUNCTIONS
# ----------------------------------------------------------------------------------

def setup_matplotlib_for_publication():
    plt.style.use('seaborn-v0_8-paper')
    params = {'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'figure.titlesize': 18, 'lines.linewidth': 2.0, 'lines.markersize': 6, 'axes.grid': True, 'grid.linestyle': ':', 'grid.linewidth': 0.7, 'savefig.dpi': 300, 'savefig.bbox': 'tight'}
    plt.rcParams.update(params)

MODEL_STYLES = {'PreDiff (Unaligned)': {'color': '#1f77b4', 'marker': 'o'}, 'PreDiff (Aligned)':   {'color': '#ff7f0e', 'marker': 'o'}, 'Persistence':         {'color': '#2ca02c', 'marker': 's'}}

def get_region_config(region_name):
    base_region = region_name.split('_')[0]
    is_test_set = '_test' in region_name
    if base_region == 'south': config = {'name': 'Southern India', 'pretty_name': 'Southern India', 'x1': 71.0, 'x2': 83.8, 'y1': 5.0, 'y2': 17.8, 'xpixelsize': 0.1}
    elif base_region == 'ne': config = {'name': 'Northeastern India', 'pretty_name': 'Northeastern India', 'x1': 86.1, 'x2': 98.9, 'y1': 19.1, 'y2': 31.9, 'xpixelsize': 0.1}
    else: raise ValueError(f"Unknown base region '{base_region}'.")
    if is_test_set: config['pretty_name'] += ' (Test Set)'
    return config

def extract_timestamp(fpath: str) -> int:
    try:
        with h5py.File(fpath, 'r') as f: return int(f['time'][()])
    except Exception: return None

def seconds_from_epoch_to_yyyymmddhhmm(seconds: int) -> str:
    return (DATA_EPOCH + timedelta(seconds=seconds)).strftime('%Y%m%d%H%M')

def load_sequence(filepaths: list) -> np.ndarray:
    frames = [np.squeeze(h5py.File(fpath, 'r')['precipitationCal'][:], axis=0) for fpath in filepaths]
    return np.stack(frames)

def inverse_preprocess_data(norm_data: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    scale_factor = np.log1p(clip_value)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        original_data = np.expm1(norm_data * scale_factor)
    original_data[original_data < 0] = 0.0
    return original_data

def get_contingency_arrays(pred, obs, thr):
    predb, obsb = pred > thr, obs > thr
    hits = np.logical_and(predb, obsb); misses = np.logical_and(np.logical_not(predb), obsb); false_alarms = np.logical_and(predb, np.logical_not(obsb))
    return {'hits': hits, 'misses': misses, 'false_alarms': false_alarms}

class PreDiffPredictor:
    def __init__(self, prediff_model: PreDiffIMERGPLModule):
        self.model = prediff_model
        self.in_slice = self.model.in_slice
    def predict(self, full_seq_tensor: torch.Tensor):
        with torch.no_grad(), torch.cuda.amp.autocast():
            _, cond = self.model.get_input(full_seq_tensor)
            pred_unaligned = self.model.sample(cond=cond, batch_size=full_seq_tensor.size(0), use_alignment=False)
        with torch.enable_grad(), torch.cuda.amp.autocast(enabled=False):
            full_seq_tensor_fp32 = full_seq_tensor.float()
            _, cond_fp32 = self.model.get_input(full_seq_tensor_fp32)
            context_seq_fp32 = full_seq_tensor_fp32[self.in_slice]
            alignment_kwargs = self._get_alignment_kwargs_for_testing(context_seq_fp32)
            pred_aligned = self.model.sample(cond=cond_fp32, batch_size=full_seq_tensor.size(0), use_alignment=True, alignment_kwargs=alignment_kwargs)
        return pred_unaligned, pred_aligned
    @staticmethod
    def _get_alignment_kwargs_for_testing(context_seq: torch.Tensor) -> dict:
        multiplier = 2.0; batch_size = context_seq.shape[0]
        avg_intensity = torch.mean(context_seq.view(batch_size, -1), dim=1, keepdim=True)
        return {"avg_x_gt": avg_intensity * multiplier}

class IMERGPredictionDataset(Dataset):
    def __init__(self, data_dir: str, forecast_dir: str, in_len: int, out_len: int, interval_minutes: int):
        self.seq_len = in_len + out_len; self.in_len = in_len
        all_files_sorted = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.HDF5')])
        self.valid_starts = self._find_sequences(all_files_sorted, forecast_dir, interval_minutes)
        self.all_files = all_files_sorted
        if not self.valid_starts: print("\nWARNING: Found 0 new sequences to process.")
        else: print(f"\nFound {len(self.valid_starts)} new sequences to process for generation.")
    def _find_sequences(self, all_files_sorted: list, forecast_dir: str, interval_minutes: int) -> list:
        if len(all_files_sorted) < self.seq_len: return []
        expected_delta_sec, tolerance_sec = interval_minutes * 60, 5 * 60
        valid_starts = []
        i_iterator = iter(range(len(all_files_sorted) - self.seq_len + 1))
        for i in i_iterator:
            candidate_files = all_files_sorted[i : i + self.seq_len]
            try:
                timestamps = [extract_timestamp(f) for f in candidate_files]
                if any(ts is None for ts in timestamps): continue
                is_contiguous = all(abs(timestamps[j+1] - timestamps[j] - expected_delta_sec) <= tolerance_sec for j in range(self.seq_len - 1))
                if is_contiguous:
                    datetime_str = seconds_from_epoch_to_yyyymmddhhmm(timestamps[self.in_len])
                    if not os.path.exists(os.path.join(forecast_dir, f"{datetime_str}_unaligned.npy")):
                        valid_starts.append((i, datetime_str))
                    for _ in range(self.seq_len - 1): next(i_iterator, None)
            except Exception: continue
        return valid_starts
    def __len__(self): return len(self.valid_starts)
    def __getitem__(self, idx):
        start_idx, datetime_str = self.valid_starts[idx]
        seq_files = self.all_files[start_idx : start_idx + self.seq_len]
        try:
            seq_data = load_sequence(seq_files)
            seq_data[~np.isfinite(seq_data)] = 0.0
            seq_tensor = torch.from_numpy(seq_data).float().unsqueeze(-1)
            return seq_tensor, datetime_str
        except Exception as e:
            print(f"ERROR loading sequence at index {start_idx}: {e}"); return None, None

def _draw_map_features(basemap, shapefile_path):
    if Basemap is None: return
    parallels = np.arange(0., 90., 5.); basemap.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    meridians = np.arange(0., 180., 5.); basemap.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    shapefile_base = os.path.splitext(shapefile_path)[0]
    basemap.readshapefile(shapefile_base, 'boundary', drawbounds=True, color='dimgray', linewidth=0.75, zorder=11)
    basemap.drawcoastlines(linewidth=0.75, color='black', zorder=12)
    basemap.drawcountries(linewidth=0.75, color='black', zorder=12)

def plot_spatial_maps(plots_dir, total_hits, total_misses, total_fas, threshold, metadata, shapefile_path, model_name_tag):
    if Basemap is None: print(f"Skipping spatial plots for {model_name_tag} because Basemap is not installed."); return
    print(f"Generating spatial score maps for: PreDiff {model_name_tag}")
    epsilon = 1e-6
    spatial_pod = total_hits / (total_hits + total_misses + epsilon)
    spatial_far = total_fas / (total_hits + total_fas + epsilon)
    spatial_csi = total_hits / (total_hits + total_misses + total_fas + epsilon)
    SPATIAL_CMAP = 'magma'
    spatial_metrics = {'POD': {'data': spatial_pod, 'cmap': SPATIAL_CMAP}, 'FAR': {'data': spatial_far, 'cmap': SPATIAL_CMAP}, 'CSI': {'data': spatial_csi, 'cmap': SPATIAL_CMAP}}
    ny, nx = spatial_pod.shape
    lons, lats = np.linspace(metadata['x1'], metadata['x2'], nx), np.linspace(metadata['y1'], metadata['y2'], ny)
    for name, metric in spatial_metrics.items():
        fig, ax = plt.subplots(figsize=(9, 9))
        m = Basemap(projection='cyl', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'], llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], resolution='i', ax=ax)
        im = m.pcolormesh(lons, lats, metric['data'], latlon=True, cmap=metric['cmap'], vmin=0, vmax=1, zorder=5, alpha=0.75)
        _draw_map_features(m, shapefile_path)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7); cbar.set_label(f'{name} Score')
        ax.set_title(f'Spatially Aggregated {name} (PreDiff {model_name_tag})\nOver {metadata["pretty_name"]} (Threshold: {threshold} mm/hr)')
        plt.savefig(os.path.join(plots_dir, f"plot_spatial_{name}_prediff_{model_name_tag.lower()}.png"))
        plt.close(fig)

def plot_fss(plots_dir, lead_times, mean_scores, fss_scales, p_size_km, thr, region_name):
    print("Generating Plot: Multi-panel FSS")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True); axes = axes.flatten()
    fig.suptitle(f'Fractions Skill Score (FSS) for {region_name}\n(Threshold: {thr} mm/hr)')
    for i, scale_pix in enumerate(fss_scales):
        ax, scale_km = axes[i], scale_pix * p_size_km
        for model_name, style in MODEL_STYLES.items():
            key_name = 'persistence' if 'Persistence' in model_name else ('aligned' if 'Aligned' in model_name else 'unaligned')
            ax.plot(lead_times, mean_scores[f'{key_name}_fss'][i, :], label=model_name, **style)
        ax.set_title(f'Scale: {scale_pix} pixels (~{scale_km:.0f} km)'); ax.set_xticks(lead_times); ax.set_ylim(0, 1.05)
    fig.supxlabel('Lead Time (minutes)'); fig.supylabel('FSS (Higher is Better)')
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, "plot_fss_multipanel.png"))
    plt.close(fig)

def plot_categorical_scores(plots_dir, lead_times, mean_scores, thr, region_name):
    print("Generating Plot: Categorical Scores (POD, FAR, CSI)")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    fig.suptitle(f'Categorical Skill Scores for {region_name}\n(Threshold: {thr} mm/hr)', y=1.02)
    titles = {'POD': 'POD', 'FAR': 'FAR', 'CSI': 'CSI'}
    y_labels = {'POD': 'POD (Higher is Better)', 'FAR': 'FAR (Lower is Better)', 'CSI': 'CSI (Higher is Better)'}
    for i, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        for model_name, style in MODEL_STYLES.items():
            key_name = 'persistence' if 'Persistence' in model_name else ('aligned' if 'Aligned' in model_name else 'unaligned')
            ax.plot(lead_times, mean_scores[f'{key_name}_cat'][metric], label=model_name, **style)
        ax.set_title(titles[metric]); ax.set_xlabel('Lead Time (minutes)'); ax.set_xticks(lead_times); ax.legend()
        if i == 0: ax.set_ylabel(y_labels[metric])
        else: ax.set_title(y_labels[metric])
    axes[0].set_ylim(-0.05, 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, "plot_categorical_scores.png"))
    plt.close(fig)

def generate_and_save_forecasts(model, device, data_dir, forecast_dir, params, loader_kwargs):
    print("\n--- PHASE 1: GENERATING FORECASTS ---")
    dataset = IMERGPredictionDataset(data_dir, forecast_dir, params['in_len'], params['out_len'], params['interval_minutes'])
    if not dataset.valid_starts:
        print("No new sequences to process. All forecasts may already exist.")
        return
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
        if not batch: return None, None
        seq_tensors, paths = zip(*batch)
        return torch.stack(seq_tensors), paths
    dataloader = DataLoader(dataset, **loader_kwargs, collate_fn=collate_fn)
    model.model.to(device); model.model.eval()
    for batch in tqdm(dataloader, desc="Generating Forecasts"):
        seq_tensors, datetime_strs = batch
        if seq_tensors is None: continue
        seq_tensors = seq_tensors.to(device)
        preds_unaligned, preds_aligned = model.predict(seq_tensors)
        preds_unaligned, preds_aligned = preds_unaligned.cpu(), preds_aligned.cpu()
        for i in range(len(datetime_strs)):
            base_path = os.path.join(forecast_dir, datetime_strs[i])
            np.save(base_path + "_unaligned.npy", preds_unaligned[i].squeeze(-1).numpy())
            np.save(base_path + "_aligned.npy", preds_aligned[i].squeeze(-1).numpy())

def calculate_metrics_and_plot(base_dir, forecast_dir, plot_dir, shapefile_path, params, metadata):
    print("\n--- PHASE 2: VERIFICATION AND PLOTTING ---")
    all_forecast_files = [f for f in os.listdir(forecast_dir) if f.endswith('_unaligned.npy')]
    if not all_forecast_files:
        print("No forecast files found. Aborting verification.")
        return

    all_scores = { 'unaligned': [], 'aligned': [], 'persistence': [] }
    total_contingency = {'unaligned': {'hits': 0, 'misses': 0, 'fas': 0}, 'aligned': {'hits': 0, 'misses': 0, 'fas': 0}}
    print(f"Verifying {len(all_forecast_files)} forecast sets...")
    
    all_data_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.HDF5')])
    datetime_str_to_idx = { seconds_from_epoch_to_yyyymmddhhmm(ts): i for i, f in enumerate(all_data_files) if (ts := extract_timestamp(f)) is not None }
    
    # Removed: fss_verifier = verification.get_method("FSS")
    
    for fname in tqdm(all_forecast_files, desc="Verifying forecasts"):
        datetime_str = fname.replace("_unaligned.npy", "")
        unaligned_path = os.path.join(forecast_dir, fname)
        aligned_path = os.path.join(forecast_dir, f"{datetime_str}_aligned.npy")
        if not os.path.exists(aligned_path): continue
        if datetime_str not in datetime_str_to_idx: continue
        gt_start_idx = datetime_str_to_idx[datetime_str]
        input_start_idx = gt_start_idx - params['in_len']
        gt_end_idx = gt_start_idx + params['out_len']
        if input_start_idx < 0 or gt_end_idx > len(all_data_files): continue
        obs_physical = inverse_preprocess_data(load_sequence(all_data_files[gt_start_idx:gt_end_idx]), params['clip_value'])
        last_input_physical = inverse_preprocess_data(load_sequence([all_data_files[input_start_idx + params['in_len'] - 1]]), params['clip_value'])
        models = {'unaligned': inverse_preprocess_data(np.load(unaligned_path), params['clip_value']), 'aligned': inverse_preprocess_data(np.load(aligned_path), params['clip_value']), 'persistence': np.stack([last_input_physical[0]] * params['out_len'])}
        for name, pred_inv in models.items():
            cat_scores = {m: np.zeros(params['out_len']) for m in METRICS_TO_PLOT}
            fss_scores = np.zeros((len(params['fss_scales_pixels']), params['out_len']))
            for k in range(params['out_len']):
                pred_k, obs_k = np.squeeze(pred_inv[k]), np.squeeze(obs_physical[k])
                
                # --- REPLACEMENT CALLS ---
                cat_dict = calculate_det_cat_fct(pred_k, obs_k, thr=params['verification_threshold'], scores=METRICS_TO_PLOT)
                for m in METRICS_TO_PLOT: cat_scores[m][k] = cat_dict.get(m, np.nan)
                
                for s, scale in enumerate(params['fss_scales_pixels']):
                    fss_scores[s, k] = calculate_fss(pred_k, obs_k, thr=params['verification_threshold'], scale=scale)
                # --- END REPLACEMENT ---

                if name != 'persistence':
                    contingency = get_contingency_arrays(pred_k, obs_k, thr=params['verification_threshold'])
                    total_contingency[name]['hits'] += contingency['hits']
                    total_contingency[name]['misses'] += contingency['misses']
                    total_contingency[name]['fas'] += contingency['false_alarms']
            all_scores[name].append({'cat': cat_scores, 'fss': fss_scores})

    if not all_scores['unaligned']: print("No valid scores were calculated."); return

    mean_scores = { f'{name}_cat': {m: np.nanmean([s['cat'][m] for s in all_scores[name]], axis=0) for m in METRICS_TO_PLOT} for name in all_scores }
    for name in all_scores: mean_scores[f'{name}_fss'] = np.nanmean([s['fss'] for s in all_scores[name]], axis=0)
    
    setup_matplotlib_for_publication()
    pixel_size_km = metadata['xpixelsize'] * 111.0
    lead_times = np.arange(1, params['out_len'] + 1) * params['interval_minutes']
    
    plot_fss(plot_dir, lead_times, mean_scores, params['fss_scales_pixels'], pixel_size_km, params['verification_threshold'], metadata['pretty_name'])
    plot_categorical_scores(plot_dir, lead_times, mean_scores, params['verification_threshold'], metadata['pretty_name'])
    plot_spatial_maps(plot_dir, total_contingency['unaligned']['hits'], total_contingency['unaligned']['misses'], total_contingency['unaligned']['fas'], params['verification_threshold'], metadata, shapefile_path, "Unaligned")
    plot_spatial_maps(plot_dir, total_contingency['aligned']['hits'], total_contingency['aligned']['misses'], total_contingency['aligned']['fas'], params['verification_threshold'], metadata, shapefile_path, "Aligned")

def main(args):
    metadata = get_region_config(args.region)
    base_dir = f'/home/sac/data_67/Nowcasting/data/required_{args.region}'
    forecast_dir = os.path.abspath(f'forecasts_prediff_{args.region}')
    plot_dir = os.path.abspath(f'plots_prediff_{args.region}')
    os.makedirs(forecast_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    print(f"--- Running analysis for region: {metadata['pretty_name']} ---")

    print("Loading model from config...")
    pl_model = PreDiffIMERGPLModule(total_num_steps=1, oc_file=args.cfg)
    
    print(f"Loading weights from: {args.ckpt}")
    if args.ckpt.endswith(".ckpt"):
        checkpoint = torch.load(args.ckpt, map_location='cpu', weights_only=False)
        state_dict = checkpoint['state_dict']
        model_weights = {k.replace("torch_nn_module.", ""): v for k, v in state_dict.items() if k.startswith("torch_nn_module.")}
        pl_model.torch_nn_module.load_state_dict(model_weights, strict=True)
        print("Successfully loaded weights from .ckpt file.")
    elif args.ckpt.endswith(".pt"):
        state_dict = torch.load(args.ckpt, map_location='cpu')
        pl_model.torch_nn_module.load_state_dict(state_dict, strict=True)
        print("Successfully loaded weights from .pt file.")
    else:
        raise ValueError("Checkpoint file must have a .ckpt or .pt extension")

    predictor = PreDiffPredictor(pl_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {"batch_size": 8, "num_workers": args.num_workers, "shuffle": False, "pin_memory": True}
    
    params = {'in_len': pl_model.hparams.layout.in_len, 'out_len': pl_model.hparams.layout.out_len, 'interval_minutes': 30, 'clip_value': 100.0, 'verification_threshold': 1.0, 'fss_scales_pixels': [1, 2, 4, 8]}

    generate_and_save_forecasts(predictor, device, base_dir, forecast_dir, params, loader_kwargs)
    calculate_metrics_and_plot(base_dir, forecast_dir, plot_dir, args.shapefile, params, metadata)
    
    print("\nProcessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PreDiff model verification for a specific region.")
    parser.add_argument('--region', type=str, choices=['south', 'ne', 'south_test', 'ne_test'], required=True, help="The geographic region to process.")
    parser.add_argument('--shapefile', type=str, required=True, help="Path to the shapefile for map boundaries (e.g., of India).")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the model's YAML configuration file.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model's checkpoint file (.ckpt or .pt).")
    parser.add_argument('--num-workers', type=int, default=4, help="Number of worker processes for data loading.")
    main(parser.parse_args())