import os
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pysteps import verification
from datetime import datetime, timedelta
import warnings
import argparse

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# This import assumes your script is in a location where it can "see" the `src` directory
from ..prediff.imerg.train_imerg_prediff import PreDiffIMERGPLModule

# --- Geospatial plotting libraries ---
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    warnings.warn("Basemap not found. Spatial plots will be disabled. Install with 'conda install -c conda-forge basemap'")
    Basemap = None


# --- CONFIGURATION and SETUP ---
METRICS_TO_PLOT = ['POD', 'FAR', 'CSI']

def setup_matplotlib_for_publication():
    plt.style.use('seaborn-v0_8-paper')
    params = {'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 12, 'figure.titlesize': 18, 'lines.linewidth': 2.5, 'lines.markersize': 7, 'axes.grid': True, 'grid.linestyle': ':', 'grid.linewidth': 0.7, 'savefig.dpi': 300, 'savefig.bbox': 'tight'}
    plt.rcParams.update(params)

MODEL_STYLES = {'PreDiff (Unaligned)': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'}, 'PreDiff (Aligned)':   {'color': '#ff7f0e', 'marker': '^', 'linestyle': '-'}, 'Persistence':         {'color': '#555555', 'marker': 's', 'linestyle': '--'}}

# --- DATA & HELPER FUNCTIONS ---

def get_region_config(region_name):
    base_region = region_name.split('_')[0]
    is_test_set = '_test' in region_name
    if base_region == 'south':
        config = {'name': 'Southern India', 'pretty_name': 'Southern India', 'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}, 'x1': 71.0, 'x2': 83.8, 'y1': 5.0, 'y2': 17.8, 'xpixelsize': 0.1, 'ypixelsize': 0.1, 'cartesian': False, 'zerovalue': 0.0}
    elif base_region == 'ne':
        config = {'name': 'Northeastern India', 'pretty_name': 'Northeastern India', 'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'}, 'x1': 86.1, 'x2': 98.9, 'y1': 19.1, 'y2': 31.9, 'xpixelsize': 0.1, 'ypixelsize': 0.1, 'cartesian': False, 'zerovalue': 0.0}
    else:
        raise ValueError(f"Unknown base region '{base_region}'.")
    if is_test_set:
        config['pretty_name'] += ' (Test Set)'
    return config

def extract_timestamp(fpath: str) -> int:
    try:
        with h5py.File(fpath, 'r') as f: return int(f['time'][()])
    except Exception: return None

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
    hits = np.logical_and(predb, obsb)
    misses = np.logical_and(np.logical_not(predb), obsb)
    false_alarms = np.logical_and(predb, np.logical_not(obsb))
    return {'hits': hits, 'misses': misses, 'false_alarms': false_alarms}


# --- MODEL INFERENCE CLASSES ---

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
            pred_aligned = self.model.sample(
                cond=cond_fp32, batch_size=full_seq_tensor.size(0),
                use_alignment=True, alignment_kwargs=alignment_kwargs
            )
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
        self.valid_starts = self._find_valid_sequences(all_files_sorted, forecast_dir, interval_minutes)
        self.all_files = all_files_sorted
        if not self.valid_starts: print("\nWARNING: Found 0 new sequences to process.")
        else: print(f"\nFound {len(self.valid_starts)} new sequences to process.")

    def _find_valid_sequences(self, all_files_sorted: list, forecast_dir: str, interval_minutes: int) -> list:
        if len(all_files_sorted) < self.seq_len: return []
        expected_delta_sec, tolerance_sec = interval_minutes * 60, 5 * 60
        valid_starts, ref_date = [], datetime(1970, 1, 1)
        for i in range(len(all_files_sorted) - self.seq_len + 1):
            candidate_files = all_files_sorted[i : i + self.seq_len]
            try:
                timestamps = [extract_timestamp(f) for f in candidate_files]
                if any(ts is None for ts in timestamps): continue
                is_contiguous = all(abs(timestamps[j+1] - timestamps[j] - expected_delta_sec) <= tolerance_sec for j in range(self.seq_len - 1))
                if not is_contiguous: continue
                forecast_start_ts = timestamps[self.in_len]
                datetime_str = (ref_date + timedelta(seconds=forecast_start_ts)).strftime('%Y%m%d%H%M')
                out_path_base = os.path.join(forecast_dir, datetime_str)
                if not os.path.exists(out_path_base + "_unaligned.npy"): valid_starts.append((i, out_path_base))
            except Exception: continue
        return valid_starts

    def __len__(self): return len(self.valid_starts)

    def __getitem__(self, idx):
        start_idx, out_path_base = self.valid_starts[idx]
        seq_files = self.all_files[start_idx : start_idx + self.seq_len]
        try:
            seq_data = load_sequence(seq_files)
            seq_data[~np.isfinite(seq_data)] = 0.0
            seq_tensor = torch.from_numpy(seq_data).float().unsqueeze(-1)
            return seq_tensor, out_path_base
        except Exception as e:
            print(f"ERROR loading sequence at index {start_idx}: {e}"); return None, None


# --- PLOTTING FUNCTIONS ---

def _draw_map_features(basemap, shapefile_path):
    if Basemap is None: return
    parallels = np.arange(0., 90., 5.); basemap.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    meridians = np.arange(0., 180., 5.); basemap.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    shapefile_base = os.path.splitext(shapefile_path)[0]
    basemap.readshapefile(shapefile_base, 'boundary', drawbounds=True, color='dimgray', linewidth=0.75, zorder=11)
    basemap.drawcoastlines(linewidth=0.75, color='black', zorder=12)
    basemap.drawcountries(linewidth=0.75, color='black', zorder=12)

def plot_spatial_maps(plots_dir, total_hits, total_misses, total_fas, threshold, metadata, shapefile_path, model_name_tag):
    if Basemap is None: 
        print(f"Skipping spatial plots for {model_name_tag} because Basemap is not installed.")
        return
    
    print(f"Generating spatial score maps for: {model_name_tag}")
    epsilon = 1e-6
    spatial_pod = total_hits / (total_hits + total_misses + epsilon)
    spatial_far = total_fas / (total_hits + total_fas + epsilon)
    spatial_csi = total_hits / (total_hits + total_misses + total_fas + epsilon)
    
    spatial_metrics = {'POD': {'data': spatial_pod, 'cmap': 'viridis'}, 'FAR': {'data': spatial_far, 'cmap': 'plasma'}, 'CSI': {'data': spatial_csi, 'cmap': 'magma'}}
    ny, nx = spatial_pod.shape
    lons, lats = np.linspace(metadata['x1'], metadata['x2'], nx), np.linspace(metadata['y1'], metadata['y2'], ny)

    for name, metric in spatial_metrics.items():
        fig, ax = plt.subplots(figsize=(9, 9))
        m = Basemap(projection='cyl', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'], llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], resolution='i', ax=ax)
        im = m.pcolormesh(lons, lats, metric['data'], latlon=True, cmap=metric['cmap'], vmin=0, vmax=1, zorder=5, alpha=0.7)
        _draw_map_features(m, shapefile_path)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7); cbar.set_label(f'{name} Score')
        ax.set_title(f'Spatially Aggregated {name} ({model_name_tag})\nOver {metadata["pretty_name"]} (Threshold: {threshold} mm/hr)')
        plt.savefig(os.path.join(plots_dir, f"plot_spatial_{name}_{model_name_tag.lower()}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_non_spatial_metrics(plots_dir, lead_times, mean_scores, fss_scales, p_size_km, thr, region_name):
    print("Generating non-spatial skill score plots...")
    # FSS Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True); axes = axes.flatten()
    fig.suptitle(f'Fractions Skill Score (FSS) for {region_name}\n(Threshold: {thr} mm/hr)', weight='bold')
    for i, scale_pix in enumerate(fss_scales):
        ax, scale_km = axes[i], scale_pix * p_size_km
        for model_name, style in MODEL_STYLES.items():
            key_name = 'persistence' if 'Persistence' in model_name else ('aligned' if 'Aligned' in model_name else 'unaligned')
            ax.plot(lead_times, mean_scores[f'{key_name}_fss'][i, :], label=model_name, **style)
        ax.set_title(f'Scale: {scale_pix} pixels (~{scale_km:.0f} km)'); ax.set_xticks(lead_times); ax.set_ylim(0, 1)
    fig.text(0.5, 0.03, 'Lead Time (minutes)', ha='center', weight='bold'); fig.text(0.04, 0.5, 'FSS (Higher is Better)', ha='center', rotation='vertical', weight='bold')
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, hspace=0.3, wspace=0.2)
    plt.savefig(os.path.join(plots_dir, "plot_fss_comparison_pub.png")); plt.close(fig)
    
    # Categorical Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'Categorical Skill Scores for {region_name}\n(Threshold: {thr} mm/hr)', weight='bold')
    titles = {'POD': 'POD (Higher is Better)', 'FAR': 'FAR (Lower is Better)', 'CSI': 'CSI (Higher is Better)'}
    for i, metric in enumerate(METRICS_TO_PLOT):
        ax = axes[i]
        for model_name, style in MODEL_STYLES.items():
            key_name = 'persistence' if 'Persistence' in model_name else ('aligned' if 'Aligned' in model_name else 'unaligned')
            ax.plot(lead_times, mean_scores[f'{key_name}_cat'][metric], label=model_name, **style)
        ax.set_title(titles[metric], weight='bold'); ax.set_xlabel('Lead Time (minutes)'); ax.set_xticks(lead_times)
    axes[0].set_ylabel('Skill Score'); axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, frameon=False)
    plt.subplots_adjust(left=0.08, bottom=0.2, right=0.98, top=0.88, wspace=0.2)
    plt.savefig(os.path.join(plots_dir, "plot_categorical_scores_pub.png")); plt.close(fig)


# --- MAIN WORKFLOW FUNCTIONS ---

# --- THIS IS THE RESTORED FUNCTION ---
def generate_and_save_forecasts(model, device, data_dir, forecast_dir, params, loader_kwargs):
    """Orchestrates the forecast generation process using a manual loop."""
    print("\n--- PHASE 1: GENERATING AND SAVING FORECASTS (Manual Loop) ---")
    dataset = IMERGPredictionDataset(data_dir, forecast_dir, params['in_len'], params['out_len'], params['interval_minutes'])
    if not dataset.valid_starts:
        print("No new sequences found to process. Skipping forecast generation.")
        return

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
        if not batch: return None, None
        seq_tensors, paths = zip(*batch)
        return torch.stack(seq_tensors), paths

    dataloader = DataLoader(dataset, **loader_kwargs, collate_fn=collate_fn)
    
    model.model.to(device)
    model.model.eval()

    for batch in tqdm(dataloader, desc="Predicting Batches"):
        seq_tensors, out_paths_base = batch
        if seq_tensors is None: continue
        
        seq_tensors = seq_tensors.to(device)
        
        preds_unaligned, preds_aligned = model.predict(seq_tensors)
        
        preds_unaligned, preds_aligned = preds_unaligned.cpu(), preds_aligned.cpu()
        for i in range(len(out_paths_base)):
            base_path = out_paths_base[i]
            np.save(base_path + "_unaligned.npy", preds_unaligned[i].squeeze(-1).numpy())
            np.save(base_path + "_aligned.npy", preds_aligned[i].squeeze(-1).numpy())


def calculate_metrics_and_plot(data_dir, forecast_dir, plot_dir, shapefile_path, params, metadata):
    """Calculates verification scores and generates all plots."""
    print("\n--- PHASE 2: CALCULATING METRICS AND PLOTTING ---")
    all_unaligned_files = sorted([os.path.join(forecast_dir, f) for f in os.listdir(forecast_dir) if f.endswith('_unaligned.npy')])
    if not all_unaligned_files:
        print("No forecast files found. Aborting verification.")
        return

    all_scores = { 'unaligned': [], 'aligned': [], 'persistence': [] }
    total_contingency = {'unaligned': {'hits': 0, 'misses': 0, 'fas': 0}, 'aligned': {'hits': 0, 'misses': 0, 'fas': 0}}
    print(f"Verifying {len(all_unaligned_files)} forecast sets...")
    
    all_data_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.HDF5')])
    ref_date = datetime(1970, 1, 1)
    datetime_str_to_idx = { (ref_date + timedelta(seconds=ts)).strftime('%Y%m%d%H%M'): i for i, f in enumerate(all_data_files) if (ts := extract_timestamp(f)) is not None }
    
    fss_verifier = verification.get_method("FSS")
    for unaligned_path in tqdm(all_unaligned_files, desc="Verifying forecasts"):
        base_path = unaligned_path.replace("_unaligned.npy", "");
        if not os.path.exists(base_path + "_aligned.npy"): continue
        forecast_dt_str = os.path.basename(base_path)
        if forecast_dt_str not in datetime_str_to_idx: continue
        
        gt_start_idx = datetime_str_to_idx[forecast_dt_str]
        input_start_idx = gt_start_idx - params['in_len']
        gt_end_idx = gt_start_idx + params['out_len']
        if input_start_idx < 0 or gt_end_idx > len(all_data_files): continue
        
        obs_physical = inverse_preprocess_data(load_sequence(all_data_files[gt_start_idx:gt_end_idx]), params['clip_value'])
        last_input_physical = inverse_preprocess_data(load_sequence([all_data_files[input_start_idx + params['in_len'] - 1]]), params['clip_value'])
        
        models = {
            'unaligned': inverse_preprocess_data(np.load(unaligned_path), params['clip_value']),
            'aligned': inverse_preprocess_data(np.load(base_path + "_aligned.npy"), params['clip_value']),
            'persistence': np.stack([last_input_physical[0]] * params['out_len'])
        }

        for name, pred_inv in models.items():
            cat_scores = {m: np.zeros(params['out_len']) for m in METRICS_TO_PLOT}
            fss_scores = np.zeros((len(params['fss_scales_pixels']), params['out_len']))
            for k in range(params['out_len']):
                pred_k, obs_k = np.squeeze(pred_inv[k]), np.squeeze(obs_physical[k])
                cat_dict = verification.det_cat_fct(pred_k, obs_k, thr=params['verification_threshold'])
                for m in METRICS_TO_PLOT: cat_scores[m][k] = cat_dict.get(m, np.nan)
                for s, scale in enumerate(params['fss_scales_pixels']): fss_scores[s, k] = fss_verifier(pred_k, obs_k, thr=params['verification_threshold'], scale=scale)
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
    plot_non_spatial_metrics(plot_dir, np.arange(1, params['out_len'] + 1) * params['interval_minutes'], mean_scores, params['fss_scales_pixels'], pixel_size_km, params['verification_threshold'], metadata['pretty_name'])
    plot_spatial_maps(plot_dir, total_contingency['unaligned']['hits'], total_contingency['unaligned']['misses'], total_contingency['unaligned']['fas'], params['verification_threshold'], metadata, shapefile_path, "Unaligned")
    plot_spatial_maps(plot_dir, total_contingency['aligned']['hits'], total_contingency['aligned']['misses'], total_contingency['aligned']['fas'], params['verification_threshold'], metadata, shapefile_path, "Aligned")

    print("\nProcessing complete.")


def main(args):
    metadata = get_region_config(args.region)
    base_dir = f'/home/sac/data_67/Nowcasting/data/required_valid_{args.region}'
    forecast_dir = os.path.abspath(f'forecasts_prediff_{args.region}')
    plot_dir = os.path.abspath(f'plots_prediff_{args.region}')
    os.makedirs(forecast_dir, exist_ok=True); os.makedirs(plot_dir, exist_ok=True)
    print(f"--- Running analysis for region: {metadata['pretty_name']} ---")

    pl_model = PreDiffIMERGPLModule(total_num_steps=1, oc_file=args.cfg)
    state_dict = torch.load(args.ckpt, map_location='cpu')
    pl_model.torch_nn_module.load_state_dict(state_dict, strict=True)
    predictor = PreDiffPredictor(pl_model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_kwargs = {"batch_size": 16, "num_workers": 8, "shuffle": False, "pin_memory": True}
    params = {'in_len': pl_model.hparams.layout.in_len, 'out_len': pl_model.hparams.layout.out_len, 'interval_minutes': 30, 'clip_value': 100.0, 'verification_threshold': 1.0, 'fss_scales_pixels': [1, 2, 4, 8]}

    generate_and_save_forecasts(predictor, device, base_dir, forecast_dir, params, loader_kwargs)
    calculate_metrics_and_plot(base_dir, forecast_dir, plot_dir, args.shapefile, params, metadata)
    
    print("\nProcessing complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run PreDiff model verification for a specific region.")
    parser.add_argument('--region', type=str, choices=['south', 'ne', 'south_test', 'ne_test'], required=True, help="The geographic region to process.")
    parser.add_argument('--shapefile', type=str, required=True, help="Path to the shapefile for map boundaries (e.g., of India).")
    parser.add_argument('--cfg', type=str, required=True, help="Path to the model's YAML configuration file.")
    parser.add_argument('--ckpt', type=str, required=True, help="Path to the model's .pt checkpoint file.")
    main(parser.parse_args())
# To run:
# CUDA_VISIBLE_DEVICES=1 uv run -m scripts.test.prediff_test --region south --shapefile Shapefile/states_IND --cfg /home/sac/data_67/Nowcasting/vishwajit/PreDiff_ISRO/scripts/prediff/imerg/prediff_imerg_ne.yaml --ckpt models_ne/imerg_earthformerunet.pt