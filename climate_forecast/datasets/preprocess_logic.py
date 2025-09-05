import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone

def preprocess_data(data: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    """
    Preprocesses a single data array by clipping, applying a logarithmic
    transformation, and scaling.
    """
    clipped = np.clip(data, 0, clip_value)
    logged = np.log1p(clipped)
    scale_factor = np.log1p(clip_value)
    if scale_factor == 0:
        return np.zeros_like(data)
    norm_data = logged / scale_factor
    return norm_data

def process_single_file(
    input_file: str,
    output_file: str,
    config: dict
) -> None:
    """
    Processes a single raw HDF5 file based on the provided configuration.
    This function is generic and relies on the config to understand the
    structure of the raw data files.
    """
    p_cfg = config['preprocess']
    d_cfg = config['data']

    lat_range = p_cfg['lat_range']
    lon_range = p_cfg['lon_range']
    clip_value = p_cfg.get('clip_value', 100.0)
    downscale_factor = p_cfg.get('downscale_factor', 1)

    lat_var = d_cfg['latitude_variable_name']
    lon_var = d_cfg['longitude_variable_name']
    time_var = d_cfg['time_variable_name']
    data_channels = d_cfg['channels']

    # --- FIX: Read time epoch settings from config ---
    time_epoch_start_str = d_cfg.get('time_epoch_start', '1970-01-01T00:00:00Z')
    # Convert ISO format string from config to a timezone-aware datetime object
    try:
        # Handle 'Z' for UTC properly
        if time_epoch_start_str.endswith('Z'):
            time_epoch_start_str = time_epoch_start_str[:-1] + '+00:00'
        epoch_start_dt = datetime.fromisoformat(time_epoch_start_str)
    except ValueError:
        raise ValueError(f"Invalid `time_epoch_start` format: '{time_epoch_start_str}'. Please use ISO 8601 format, e.g., 'YYYY-MM-DDTHH:MM:SSZ'.")

    coord_order = p_cfg['raw_data_coordinate_order']
    if set(coord_order) != {'lat', 'lon'} or len(coord_order) != 2:
        raise ValueError(f"Invalid `raw_data_coordinate_order`: {coord_order}.")

    with h5py.File(input_file, 'r') as fin:
        lat = fin[lat_var][:]
        lon = fin[lon_var][:]
        
        # --- FIX: Read time offset and convert to standard Unix timestamp ---
        time_offset_seconds = fin[time_var][()]
        actual_datetime = epoch_start_dt + timedelta(seconds=float(time_offset_seconds))
        unix_timestamp = actual_datetime.timestamp()

        lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]

        if lat_indices.size == 0 or lon_indices.size == 0:
            raise ValueError(f"No valid lat/lon indices in {input_file} for the specified range.")

        lat_slice = slice(lat_indices[0], lat_indices[-1] + 1)
        lon_slice = slice(lon_indices[0], lon_indices[-1] + 1)

        slice_tuple = [Ellipsis]
        if coord_order == ['lat', 'lon']:
            slice_tuple.extend([lat_slice, lon_slice])
        else:
            slice_tuple.extend([lon_slice, lat_slice])

        processed_channels = {}
        for channel in data_channels:
            raw_data = fin[channel][:]
            subset_data = raw_data[tuple(slice_tuple)]

            original_shape = subset_data.shape
            squeezed_shape = [dim for dim in original_shape if dim != 1]

            if len(squeezed_shape) != 2:
                if subset_data.ndim > 2:
                    subset_data = subset_data[-1, ...]
                else:
                    raise ValueError(f"Channel '{channel}' in {input_file} has an unsupported shape "
                                     f"after slicing: {original_shape}. Expected a 2D array or "
                                     f"one that can be squeezed to 2D.")

            subset_data = np.squeeze(subset_data)

            if subset_data.ndim != 2:
                raise ValueError(f"Failed to produce a 2D array for channel '{channel}' in {input_file}. "
                                 f"Shape after processing: {subset_data.shape}")

            if downscale_factor > 1:
                tensor_input = torch.from_numpy(subset_data.copy()).unsqueeze(0).unsqueeze(0)
                downscaled = F.avg_pool2d(tensor_input, kernel_size=downscale_factor)
                subset_data = downscaled.squeeze(0).squeeze(0).numpy()

            processed_channels[channel] = preprocess_data(subset_data, clip_value)

    with h5py.File(output_file, 'w') as fout:
        for channel, data in processed_channels.items():
            fout.create_dataset(channel, data=data, compression="gzip")
        # --- FIX: Save the standardized Unix timestamp ---
        fout.create_dataset(time_var, data=unix_timestamp)