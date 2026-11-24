import os
import h5py
import numpy as np
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import torch
import torch.nn.functional as F

def save_lat_lon(input_file, output_dir, lat_range=(5, 17.8), lon_range=(71, 83.8)):
    """
    Extracts latitude and longitude arrays from the input HDF5 file,
    applies filtering based on updated ranges, and saves them to a new HDF5 file.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        with h5py.File(input_file, 'r') as fin:
            lat = fin['Grid/lat'][:]
            lon = fin['Grid/lon'][:]
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Create boolean masks for valid lat and lon values using new ranges.
    lat_mask = (lat >= lat_range[0]) & (lat <= lat_range[1])
    lon_mask = (lon >= lon_range[0]) & (lon <= lon_range[1])

    # Extract the valid lat and lon values.
    valid_lat = lat[lat_mask]
    valid_lon = lon[lon_mask]

    print("Lat size: ", len(valid_lat))
    print("Lon size: ", len(valid_lon))

    output_file = os.path.join(output_dir, 'lat_lon.HDF5')
    try:
        with h5py.File(output_file, 'w') as fout:
            ds_lat = fout.create_dataset('lat', data=valid_lat)
            ds_lon = fout.create_dataset('lon', data=valid_lon)

            ds_lat.attrs['units'] = 'degrees_north'
            ds_lon.attrs['units'] = 'degrees_east'
            fout.attrs['description'] = 'Latitude and longitude data extracted and filtered for Indian region from original IMERG data.'

        print(f"Lat/lon data saved to {output_file}")

    except Exception as e:
        print(f"Error writing output file: {e}")
        return

def preprocess_data(data, clip_value=100):
    """
    Preprocess data by clipping, applying a logarithmic transformation,
    and then scaling by a fixed constant instead of min-max normalization.
    
    Parameters:
      data (np.ndarray): Input data.
      clip_value (float): Maximum value to clip the data (e.g., 100).
    
    Returns:
      np.ndarray: Transformed data scaled to [0, 1].
    """
    # Clip the data to the range [0, clip_value]
    clipped = np.clip(data, 0, clip_value)
    
    # Apply logarithmic transformation (using log1p to avoid log(0))
    logged = np.log(clipped + 1)
    
    # Use a fixed scaling factor (log(clip_value + 1)) to normalize
    scale_factor = np.log(clip_value + 1)
    norm_data = logged / scale_factor
    
    return norm_data

def process_file(input_file, output_file, lat_range=(5, 17.8), lon_range=(71, 83.8), clip_value=100, downscale=False):
    """
    Processes a single HDF5 file by:
      - Reading the precipitation and time datasets (with their attributes) from the 'Grid' group.
      - Subsetting the data based on the provided latitude and longitude ranges.
      - Preprocessing the precipitation data:
          (a) clipping to [0, clip_value],
          (b) applying log normalization,
          (c) applying min-max normalization.
      - Writing the transformed precipitation data and the full time dataset (with attributes) to the output file.
    
    Parameters:
      input_file (str): Path to the input HDF5 file.
      output_file (str): Path to the output HDF5 file.
      lat_range (tuple): Latitude bounds (min, max).
      lon_range (tuple): Longitude bounds (min, max).
      clip_value (float): The maximum value for clipping before transformation.
    """
    with h5py.File(input_file, 'r') as fin:
        # Read coordinate arrays.
        lat = fin['Grid/lat'][:]
        lon = fin['Grid/lon'][:]
        
        # Read time dataset.
        time_dset = fin['Grid/time']
        time_data = time_dset[:]
        
        # Read precipitation dataset.
        precip_dset = fin['Grid/precipitationCal']

        # Filter and retain important attributes for time and precipitation.
        time_keys = ['DimensionNames', 'Units', 'units', 'standard_name', 'LongName', 'bounds', 'axis', 'calendar']
        precip_keys = ['DimensionNames', 'Units', 'units', 'coordinates', '_FillValue', 'CodeMissingValue', 'LongName']
        time_attrs = {k: time_dset.attrs[k] for k in time_keys if k in time_dset.attrs}
        precip_attrs = {k: precip_dset.attrs[k] for k in precip_keys if k in precip_dset.attrs}

        # Find indices that satisfy the lat/lon conditions, since data is contiguous.
        lat_indices = np.where((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
        lon_indices = np.where((lon >= lon_range[0]) & (lon <= lon_range[1]))[0]
        
        if lat_indices.size == 0 or lon_indices.size == 0:
            raise ValueError(f"No valid lat/lon indices found in {input_file} for the specified range.")
        
        lat_start, lat_end = lat_indices[0], lat_indices[-1] + 1
        lon_start, lon_end = lon_indices[0], lon_indices[-1] + 1
        
        precip_data_full = precip_dset[:]
        precip_subset = precip_data_full[-1, lon_start:lon_end, lat_start:lat_end]

        
        if downscale:
            tensor_input = torch.tensor(precip_subset).unsqueeze(0).unsqueeze(0)
            downscaled = F.avg_pool2d(tensor_input, kernel_size=4)
            precip_downscaled = downscaled.squeeze(1).squeeze(1).numpy()
            # Apply normalization:
            precip_transformed = preprocess_data(precip_downscaled, clip_value)

        else:
            tensor_input = torch.tensor(precip_subset).unsqueeze(0).unsqueeze(0)
            precip_squeezed = tensor_input.squeeze(1).squeeze(1).numpy()
            # Apply normalization:
            precip_transformed = preprocess_data(precip_squeezed, clip_value)
    
    # Write the new HDF5 file with the transformed precipitation and the time datasets.
    with h5py.File(output_file, 'w') as fout:
        ds_precip = fout.create_dataset("precipitationCal", data=precip_transformed, compression="gzip")
        # Retain important precipitation attributes.
        for key, value in precip_attrs.items():
            ds_precip.attrs[key] = value
        
        ds_time = fout.create_dataset("time", data=time_data[-1])
        # Retain important time attributes.
        for key, value in time_attrs.items():
            ds_time.attrs[key] = value

    

def process_file_wrapper(args):
    """
    Wrapper function that:
      - Creates the corresponding output subdirectory (preserving the input directory structure).
      - Extracts the last time value from the 'Grid/time' dataset to name the output file.
      - Calls process_file() with the specified clip_value.
    
    Expected args tuple:
      (input_file, input_dir, output_dir, lat_range, lon_range, clip_value)
    """
    input_file, input_dir, output_dir, lat_range, lon_range, clip_value, downscale = args

    # Compute relative subdirectory and create it.
    rel_path = os.path.relpath(os.path.dirname(input_file), input_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Open file to extract the final timestamp.
    try:
        with h5py.File(input_file, 'r') as fin:
            time_array = fin['Grid/time'][:]
            last_time_val = time_array[-1]
    except Exception as e:
        print(f"Error reading time from {input_file}: {e}")
        return

    # Convert the last time value to a datetime string.
    ref_date = datetime(1970, 1, 1, 0, 0, 0)
    final_datetime = ref_date + timedelta(seconds=float(last_time_val))
    datetime_str = final_datetime.strftime('%Y%m%d%H%M')
    
    # Use the datetime string as the output file name.
    output_file_name = f"{datetime_str}.HDF5"
    output_file_path = os.path.join(output_subdir, output_file_name)
    
    try:
        process_file(input_file, output_file_path, lat_range=lat_range, lon_range=lon_range, clip_value=clip_value, downscale=False)
        print(f"Processed:  {input_file} -> {output_file_path}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main(input_dir, output_dir, lat_range=(5, 17.8), lon_range=(71, 83.8), processes=8, clip_value=100, downscale=False):
    """
    Recursively collects HDF5 files from the input directory and processes each file in parallel.
    
    Parameters:
      input_dir (str): Directory containing input HDF5 files.
      output_dir (str): Directory to save processed HDF5 files.
      lat_range (tuple): Latitude bounds (min, max).
      lon_range (tuple): Longitude bounds (min, max).
      processes (int): Number of processes for parallel processing.
      clip_value (float): Value for clipping before transformation.
    """
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.h5', '.hdf5')):
                input_file = os.path.join(root, file)
                tasks.append((input_file, input_dir, output_dir, lat_range, lon_range, clip_value, downscale))
    
    if processes is None:
        processes = cpu_count()
    print(f"Found {len(tasks)} files. Using {processes} processes.")
    
    with Pool(processes=processes) as pool:
        pool.map(process_file_wrapper, tasks)

if __name__ == "__main__":

    # For India: lat = (-6.2, 45) and lon = (53.8, 105)
    input_dir = '/sacarch6/Abhishek_Chhari/IMERG/IMERG'
    output_dir = 'data/processed_south'
    processes = 50

    main(input_dir, output_dir, lat_range=(5, 17.8), lon_range=(71, 83.8), processes=processes, clip_value=100, downscale=False)
