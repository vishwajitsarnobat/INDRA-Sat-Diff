import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import pandas as pd
from pysteps import motion, nowcasts, verification

# --- Geospatial plotting libraries ---
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print("This script requires Basemap for map plotting.")
    print("It is highly recommended to install it with conda:")
    print("conda install -c conda-forge basemap")
    exit()


# --- PLOTTING CONFIGURATION (for Publication Quality) ---
plt.rcParams.update({
    'font.size': 14, 'font.family': 'sans-serif', 'axes.titlesize': 18,
    'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


# --- DATA & MODEL FUNCTIONS (Unchanged) ---

def get_region_config(region_name):
    base_region = region_name.split('_')[0]
    is_test_set = '_test' in region_name
    if base_region == 'south':
        config = {
            'name': 'Southern India', 'pretty_name': 'Southern India',
            'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'},
            'x1': 71.0, 'x2': 83.8, 'y1': 5.0, 'y2': 17.8,
            'xpixelsize': 0.1, 'ypixelsize': 0.1,
            'cartesian': False, 'zerovalue': 0.0
        }
    elif base_region == 'ne':
        config = {
            'name': 'Northeastern India', 'pretty_name': 'Northeastern India',
            'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'},
            'x1': 86.1, 'x2': 98.9, 'y1': 19.1, 'y2': 31.9,
            'xpixelsize': 0.1, 'ypixelsize': 0.1,
            'cartesian': False, 'zerovalue': 0.0
        }
    else:
        raise ValueError(f"Unknown base region '{base_region}'.")
    if is_test_set:
        config['pretty_name'] += ' (Test Set)'
    return config

def extract_timestamp(fpath):
    with h5py.File(fpath, 'r') as f: return f['time'][()]

def is_sequence_valid(filepaths, interval_minutes=30, tolerance_seconds=300):
    expected_delta = interval_minutes * 60
    for i in range(len(filepaths) - 1):
        try:
            time_diff = int(extract_timestamp(filepaths[i+1])) - int(extract_timestamp(filepaths[i]))
            if abs(time_diff - expected_delta) > tolerance_seconds: return False
        except Exception: return False
    return True

def load_sequence(filepaths):
    frames = [h5py.File(fpath, 'r')['precipitationCal'][0] for fpath in filepaths]
    return np.stack(frames)

def transform_to_physical(norm_data, clip_value=100):
    scale_factor = np.log(clip_value + 1)
    original_data = np.exp(norm_data * scale_factor) - 1
    original_data[original_data < 0] = 0.0
    return original_data

def compute_motion(train_data_physical):
    oflow_method = motion.get_method('LK')
    return oflow_method(train_data_physical)

def generate_ensemble_forecast(train_data_physical, motion_field, n_timesteps, n_ens_members, metadata, interval_minutes, precip_thr):
    nowcast_method = nowcasts.get_method('steps')
    pixel_size_km = metadata['xpixelsize'] * 111.0
    precip_input = np.copy(train_data_physical)
    precip_input[~np.isfinite(precip_input)] = metadata['zerovalue']
    forecast_physical = nowcast_method(precip_input, motion_field, n_timesteps, n_ens_members,
                                       kmperpixel=pixel_size_km, timestep=interval_minutes,
                                       precip_thr=precip_thr, seed=42, vel_pert_method=None)
    return forecast_physical

def get_contingency_arrays(pred, obs, thr):
    predb, obsb = pred > thr, obs > thr
    hits = np.logical_and(predb, obsb)
    misses = np.logical_and(np.logical_not(predb), obsb)
    false_alarms = np.logical_and(predb, np.logical_not(obsb))
    return {'hits': hits, 'misses': misses, 'false_alarms': false_alarms}


# --- PUBLICATION-QUALITY PLOTTING FUNCTIONS (Using Basemap with Alpha Blending) ---

def _draw_map_features(basemap, shapefile_path):
    """Helper function to draw map features with a clear visual hierarchy."""
    # Draw faint graticules in the background
    parallels = np.arange(0., 90., 5.)
    basemap.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    meridians = np.arange(0., 180., 5.)
    basemap.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    
    # Draw shapefile boundaries with a soft color
    shapefile_base = os.path.splitext(shapefile_path)[0]
    basemap.readshapefile(shapefile_base, 'boundary', drawbounds=True, color='dimgray', linewidth=0.75, zorder=11)
    
    # Draw major coastlines and borders on top
    basemap.drawcoastlines(linewidth=0.75, color='black', zorder=12)
    basemap.drawcountries(linewidth=0.75, color='black', zorder=12)

def plot_spatial_maps(plots_dir, total_hits, total_misses, total_fas, threshold, metadata, shapefile_path):
    print("Generating Plot 3: Spatial Score Maps on Geography (Basemap)")
    epsilon = 1e-6
    spatial_pod = total_hits / (total_hits + total_misses + epsilon)
    spatial_far = total_fas / (total_hits + total_fas + epsilon)
    spatial_csi = total_hits / (total_hits + total_misses + total_fas + epsilon)
    
    spatial_metrics = {
        'POD': {'data': spatial_pod, 'cmap': 'viridis', 'label': 'Probability of Detection'},
        'FAR': {'data': spatial_far, 'cmap': 'plasma', 'label': 'False Alarm Ratio'},
        'CSI': {'data': spatial_csi, 'cmap': 'magma', 'label': 'Critical Success Index'}
    }
    
    ny, nx = spatial_pod.shape
    lons = np.linspace(metadata['x1'], metadata['x2'], nx)
    lats = np.linspace(metadata['y1'], metadata['y2'], ny)

    for name, metric in spatial_metrics.items():
        fig, ax = plt.subplots(figsize=(9, 9))
        
        m = Basemap(
            projection='cyl', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'],
            llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], resolution='i', ax=ax
        )
        
        # Plot the data with alpha blending to make it semi-transparent
        im = m.pcolormesh(lons, lats, metric['data'], latlon=True, cmap=metric['cmap'], 
                          vmin=0, vmax=1, zorder=5, alpha=0.7)
        
        # Draw map features on top
        _draw_map_features(m, shapefile_path)
        
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7)
        cbar.set_label(f'{name} Score')
        ax.set_title(f'Spatially Aggregated {metric["label"]}\nOver {metadata["pretty_name"]} (Threshold: {threshold} mm/hr)')
        
        plt.savefig(os.path.join(plots_dir, f"plot3_spatial_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_contingency_map(plots_dir, forecast_sample, obs_sample, threshold, fss_verifier, metadata, shapefile_path):
    print("Generating Plot 4: Sample Contingency Map on Geography (Basemap)")
    contingency_arrays = get_contingency_arrays(forecast_sample, obs_sample, thr=threshold)
    cat_map = np.zeros(forecast_sample.shape)
    cat_map[contingency_arrays['misses']] = 1
    cat_map[contingency_arrays['false_alarms']] = 2
    cat_map[contingency_arrays['hits']] = 3
    
    cat_map_masked = np.ma.masked_where(cat_map == 0, cat_map)
    fss_val = fss_verifier(forecast_sample, obs_sample, thr=threshold, scale=4)
    cmap = mcolors.ListedColormap(['#e41a1c', '#377eb8', '#4daf4a']) # Red, Blue, Green
    
    ny, nx = forecast_sample.shape
    lons = np.linspace(metadata['x1'], metadata['x2'], nx)
    lats = np.linspace(metadata['y1'], metadata['y2'], ny)
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    m = Basemap(
        projection='cyl', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'],
        llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], resolution='i', ax=ax
    )
    
    # Plot the categorical data with alpha blending
    m.pcolormesh(lons, lats, cat_map_masked, latlon=True, cmap=cmap, vmin=1, vmax=3, zorder=5, alpha=0.7)
    
    _draw_map_features(m, shapefile_path)
    
    patches = [plt.Rectangle((0,0),1,1, color=c) for c in ['#4daf4a', '#e41a1c', '#377eb8']]
    labels = ["Hit", "Miss", "False Alarm"]
    ax.legend(patches, labels, loc='upper left', title='Categories', frameon=True, facecolor='white', framealpha=0.7)
    ax.set_title(f'Sample Contingency Map (STEPS vs. Obs)\nOver {metadata["pretty_name"]} (FSS: {fss_val:.3f})')

    plt.savefig(os.path.join(plots_dir, "plot4_contingency_map.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

# --- NON-SPATIAL PLOTTING and METRICS (Unchanged) ---
def plot_fss(plots_dir, lead_times, mean_model_fss, mean_pers_fss, fss_scales, p_size_km, thr, region_name):
    print("Generating Plot 1: FSS (Multi-panel)")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {'model': '#1f77b4', 'persistence': '#ff7f0e'}
    for i, scale_pix in enumerate(fss_scales):
        ax = axes[i]
        scale_km = scale_pix * p_size_km
        ax.plot(lead_times, mean_model_fss[i, :], marker='o', color=colors['model'], label='STEPS')
        ax.plot(lead_times, mean_pers_fss[i, :], marker='s', linestyle='--', color=colors['persistence'], label='Persistence')
        ax.set_title(f'Scale: {scale_pix} pixels (~{scale_km:.0f} km)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7); ax.set_ylim(0, 1.05); ax.set_xticks(lead_times); ax.legend()
    fig.suptitle(f'Fractions Skill Score (FSS) for {region_name}\n(Threshold: {thr} mm/hr)', y=0.98)
    fig.supxlabel('Lead Time (minutes)'); fig.supylabel('FSS (Higher is Better)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, "plot1_fss_multipanel.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_categorical_scores(plots_dir, lead_times, mean_model_cat, mean_pers_cat, thr, region_name):
    print("Generating Plot 2: CSI, FAR, POD")
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21, 6), sharey=True)
    metrics, y_labels = ['POD', 'FAR', 'CSI'], ['POD', 'FAR', 'CSI']
    colors = {'model': '#2ca02c', 'persistence': '#d62728'}
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.plot(lead_times, mean_model_cat[metric], marker='o', color=colors['model'], label='STEPS')
        ax.plot(lead_times, mean_pers_cat[metric], marker='s', linestyle='--', color=colors['persistence'], label='Persistence')
        ax.set_title(metric); ax.set_xlabel('Lead Time (minutes)'); ax.set_ylabel(f'{y_labels[i]} (Higher is Better)' if metric != 'FAR' else f'{y_labels[i]} (Lower is Better)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7); ax.set_xticks(lead_times); ax.legend()
    axes[0].set_ylim(-0.05, 1.05)
    fig.suptitle(f'Categorical Scores for {region_name}\n(Threshold: {thr} mm/hr)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot2_categorical_scores.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_and_save_metrics(plots_dir, lead_times, model_scores, persistence_scores, fss_scales, p_size_km, region_name):
    data = []
    for metric in ['POD', 'FAR', 'CSI']:
        for i, lead_time in enumerate(lead_times):
            data.append(['Categorical', metric, f'{lead_time} min', model_scores[metric][i], persistence_scores[metric][i]])
    for s_idx, scale_pix in enumerate(fss_scales):
        metric_name = f'FSS ({(scale_pix * p_size_km):.0f} km)'
        for i, lead_time in enumerate(lead_times):
            data.append(['FSS', metric_name, f'{lead_time} min', model_scores['FSS'][s_idx, i], persistence_scores['FSS'][s_idx, i]])
    df = pd.DataFrame(data, columns=['Score Type', 'Metric', 'Lead Time', 'STEPS', 'Persistence']).round(4)
    print(f"\n\n--- FINAL AVERAGED SCORES ({region_name}) ---")
    print(df.to_string())
    output_path = os.path.join(plots_dir, f"final_scores_table_{region_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt")
    with open(output_path, 'w') as f:
        f.write(f"--- FINAL AVERAGED SCORES ({region_name}) ---\n\n" + df.to_string())
    print(f"\nFinal scores table saved to '{output_path}'")


def main(region, shapefile_path):
    # --- This function remains identical ---
    metadata = get_region_config(region)
    base_dir = f'/home/sac/data_67/Nowcasting/data/required_valid_{region}'
    plots_dir = f'plots_pysteps_valid_{region}'
    os.makedirs(plots_dir, exist_ok=True)
    print(f"--- Running analysis for region: {metadata['pretty_name']} ---")
    print(f"--- Data source: {base_dir} ---")
    print(f"--- Plots will be saved to: {plots_dir} ---")

    seq_len, n_train, interval_minutes = 13, 7, 30
    n_obs = seq_len - n_train
    n_ens_members = 50
    verification_threshold = 1.0
    fss_scales_pixels = [1, 2, 4, 8]

    all_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.HDF5')])
    n_seq = len(all_files) // seq_len
    if n_seq == 0:
        print("Not enough files to form a complete sequence."); return

    fss_verifier = verification.get_method("FSS")
    all_scores_model_fss, all_scores_pers_fss = [], []
    all_scores_model_cat, all_scores_pers_cat = [], []
    try:
        with h5py.File(all_files[0], 'r') as f:
            spatial_shape = f['precipitationCal'][0].shape
    except (IOError, IndexError):
        print("Error reading the first file. Exiting."); return

    total_hits_model, total_misses_model, total_fas_model = np.zeros(spatial_shape), np.zeros(spatial_shape), np.zeros(spatial_shape)
    last_forecast_sample, last_obs_sample = None, None

    for i in range(n_seq):
        print(f"\nProcessing sequence {i+1}/{n_seq}")
        seq_files = all_files[i*seq_len:(i+1)*seq_len]
        if not is_sequence_valid(seq_files, interval_minutes): continue
        
        seq_data_norm = load_sequence(seq_files)
        seq_data_norm[~np.isfinite(seq_data_norm)] = 0.0
        
        train_norm, obs_norm = seq_data_norm[:n_train], seq_data_norm[n_train:]
        train_physical = transform_to_physical(train_norm)
        motion_field = compute_motion(train_physical)
        
        forecast_physical_ensemble = generate_ensemble_forecast(
            train_physical, motion_field, n_obs, n_ens_members, metadata, 
            interval_minutes, verification_threshold
        )
        
        forecast_mean_physical = np.mean(forecast_physical_ensemble, axis=0)
        obs_physical = transform_to_physical(obs_norm)
        persistence_physical = np.stack([train_physical[-1, :, :]] * n_obs)

        if i == n_seq - 1:
            last_forecast_sample, last_obs_sample = forecast_mean_physical, obs_physical

        seq_model_fss, seq_pers_fss = np.zeros((len(fss_scales_pixels), n_obs)), np.zeros((len(fss_scales_pixels), n_obs))
        seq_model_cat = {'CSI': np.zeros(n_obs), 'POD': np.zeros(n_obs), 'FAR': np.zeros(n_obs)}
        seq_pers_cat = {'CSI': np.zeros(n_obs), 'POD': np.zeros(n_obs), 'FAR': np.zeros(n_obs)}

        for k in range(n_obs):
            model_cat = verification.det_cat_fct(forecast_mean_physical[k], obs_physical[k], thr=verification_threshold)
            pers_cat = verification.det_cat_fct(persistence_physical[k], obs_physical[k], thr=verification_threshold)
            for score in ['CSI', 'POD', 'FAR']:
                seq_model_cat[score][k] = model_cat.get(score, np.nan)
                seq_pers_cat[score][k] = pers_cat.get(score, np.nan)
            for s, scale in enumerate(fss_scales_pixels):
                seq_model_fss[s, k] = fss_verifier(forecast_mean_physical[k], obs_physical[k], thr=verification_threshold, scale=scale)
                seq_pers_fss[s, k] = fss_verifier(persistence_physical[k], obs_physical[k], thr=verification_threshold, scale=scale)
            
            contingency_arrays = get_contingency_arrays(forecast_mean_physical[k], obs_physical[k], thr=verification_threshold)
            total_hits_model += contingency_arrays['hits']; total_misses_model += contingency_arrays['misses']; total_fas_model += contingency_arrays['false_alarms']
            
        all_scores_model_fss.append(seq_model_fss); all_scores_pers_fss.append(seq_pers_fss)
        all_scores_model_cat.append(seq_model_cat); all_scores_pers_cat.append(seq_pers_cat)

    if not all_scores_model_fss:
        print("No valid sequences were processed."); return

    print("\nAveraging scores and generating publication-quality plots...")
    mean_model_fss = np.nanmean(np.stack(all_scores_model_fss), axis=0)
    mean_pers_fss = np.nanmean(np.stack(all_scores_pers_fss), axis=0)
    mean_model_cat = {s: np.nanmean([d[s] for d in all_scores_model_cat], axis=0) for s in ['CSI', 'POD', 'FAR']}
    mean_pers_cat = {s: np.nanmean([d[s] for d in all_scores_pers_cat], axis=0) for s in ['CSI', 'POD', 'FAR']}
    lead_times = np.arange(1, n_obs + 1) * interval_minutes
    pixel_size_km = metadata['xpixelsize'] * 111.0
    
    plot_fss(plots_dir, lead_times, mean_model_fss, mean_pers_fss, fss_scales_pixels, pixel_size_km, verification_threshold, metadata['pretty_name'])
    plot_categorical_scores(plots_dir, lead_times, mean_model_cat, mean_pers_cat, verification_threshold, metadata['pretty_name'])
    plot_spatial_maps(plots_dir, total_hits_model, total_misses_model, total_fas_model, verification_threshold, metadata, shapefile_path)
    if last_forecast_sample is not None and last_obs_sample is not None:
        plot_contingency_map(plots_dir, last_forecast_sample[-1], last_obs_sample[-1], verification_threshold, fss_verifier, metadata, shapefile_path)

    print(f"\nProcessing complete. All plots saved in '{plots_dir}'.")
    
    model_scores = {**mean_model_cat, 'FSS': mean_model_fss}
    persistence_scores = {**mean_pers_cat, 'FSS': mean_pers_fss}
    print_and_save_metrics(plots_dir, lead_times, model_scores, persistence_scores, fss_scales_pixels, pixel_size_km, metadata['pretty_name'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run pysteps verification for a specific region.")
    parser.add_argument('--region', type=str, choices=['south', 'ne', 'south_test', 'ne_test'], required=True, help="The geographic region to process.")
    parser.add_argument('--shapefile', type=str, required=True, help="Path to the shapefile for map boundaries (e.g., of India).")
    args = parser.parse_args()
    
    main(region=args.region, shapefile_path=args.shapefile)