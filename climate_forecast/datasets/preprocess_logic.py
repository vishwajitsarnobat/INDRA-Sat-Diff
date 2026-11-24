import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timedelta, timezone


def preprocess_data(data: np.ndarray, clip_value: float = 100.0) -> np.ndarray:
    """Apply clipping, log-transform, and scale a precipitation data array."""
    clipped = np.clip(data, 0, clip_value)
    logged = np.log1p(clipped)
    scale_factor = np.log1p(clip_value)
    if scale_factor == 0:
        return np.zeros_like(data)
    return logged / scale_factor


_UNIT_SECONDS = {"second": 1.0, "seconds": 1.0, "minute": 60.0, "minutes": 60.0,
                 "hour": 3600.0, "hours": 3600.0, "day": 86400.0, "days": 86400.0}


def _epoch_from_units(units):
    """Parse CF time units like 'seconds since 1980-01-06 00:00:00 UTC' -> (aware datetime, seconds per unit)."""
    if units is None:
        return None, None
    if isinstance(units, (bytes, np.bytes_)):
        units = units.decode()
    units = str(units).strip()
    if " since " not in units:
        return None, None
    unit, ref = units.split(" since ", 1)
    ref = ref.replace("UTC", "").replace("Z", "").strip().replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(ref)
    except ValueError:
        return None, None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt, _UNIT_SECONDS.get(unit.strip().lower())


def process_single_file(input_file: str, output_file: str, config: dict) -> None:
    """Process a single raw HDF5 file: spatially slice, downscale, and normalize."""
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

    # Parse epoch start (default Unix epoch) for converting file time offsets to timestamps
    time_epoch_start_str = d_cfg.get('time_epoch_start', '1970-01-01T00:00:00Z')
    if time_epoch_start_str.endswith('Z'):
        time_epoch_start_str = time_epoch_start_str[:-1] + '+00:00'
    try:
        epoch_start_dt = datetime.fromisoformat(time_epoch_start_str)
    except ValueError:
        raise ValueError(f"Invalid `time_epoch_start` format: '{time_epoch_start_str}'.")

    coord_order = p_cfg['raw_data_coordinate_order']
    if set(coord_order) != {'lat', 'lon'} or len(coord_order) != 2:
        raise ValueError(f"Invalid `raw_data_coordinate_order`: {coord_order}.")

    with h5py.File(input_file, 'r') as fin:
        lat = fin[lat_var][:]
        lon = fin[lon_var][:]

        # Convert file time offset to a Unix timestamp. The epoch comes from the variable's CF `units`
        # attribute when present (IMERG: "seconds since 1980-01-06 00:00:00 UTC"), else from config.
        time_ds = fin[time_var]
        time_offset_seconds = float(np.asarray(time_ds[()]).reshape(-1)[0])
        file_epoch, unit_seconds = _epoch_from_units(time_ds.attrs.get('units'))
        epoch = file_epoch or epoch_start_dt
        actual_datetime = epoch + timedelta(seconds=time_offset_seconds * (unit_seconds or 1.0))
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

            # Handle degenerate leading dimensions
            squeezed_dims = [d for d in subset_data.shape if d != 1]
            if len(squeezed_dims) != 2:
                if subset_data.ndim > 2:
                    subset_data = subset_data[-1, ...]
                else:
                    raise ValueError(f"Channel '{channel}' in {input_file} has unsupported shape {subset_data.shape}.")
            subset_data = np.squeeze(subset_data)
            if subset_data.ndim != 2:
                raise ValueError(f"Failed to produce 2D array for channel '{channel}' in {input_file}.")

            if downscale_factor > 1:
                tensor_input = torch.from_numpy(subset_data.copy()).unsqueeze(0).unsqueeze(0)
                downscaled = F.avg_pool2d(tensor_input, kernel_size=downscale_factor)
                subset_data = downscaled.squeeze(0).squeeze(0).numpy()

            processed_channels[channel] = preprocess_data(subset_data, clip_value)

    with h5py.File(output_file, 'w') as fout:
        for channel, data in processed_channels.items():
            fout.create_dataset(channel, data=data, compression="gzip")
        fout.create_dataset(time_var, data=unix_timestamp)