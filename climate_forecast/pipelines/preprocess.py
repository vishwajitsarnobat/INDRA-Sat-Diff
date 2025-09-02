import os
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

# Import the core logic from our generalized module
from climate_forecast.datasets.preprocess_logic import process_single_file

def _process_file_wrapper(args_tuple, config):
    """
    A wrapper to unpack arguments and call the core processing function.
    This is necessary for use with multiprocessing.Pool.
    """
    input_file, input_dir, output_dir = args_tuple
    
    # Preserve the directory structure from input to output
    rel_path = os.path.relpath(os.path.dirname(input_file), input_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Create a unique, timestamp-based filename for the processed file
    # This is more robust than relying on the input filename.
    # Note: We read the timestamp from the file itself.
    # We will simply use the original filename for the new one to avoid re-reading.
    base_name = os.path.basename(input_file)
    output_file_path = os.path.join(output_subdir, base_name)
    
    try:
        process_single_file(input_file, output_file_path, config)
        print(f"Processed: {input_file} -> {output_file_path}")
    except Exception as e:
        print(f"ERROR processing {input_file}: {e}")

def run(config: dict):
    """
    The main entry point for the preprocessing pipeline.
    
    Recursively finds all HDF5 files in the input directory and processes them in parallel.
    """
    p_cfg = config['preprocess']
    input_dir = p_cfg['input_dir']
    output_dir = p_cfg['output_dir']
    
    print("--- Starting Preprocessing ---")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all files to be processed
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.h5', '.hdf5')):
                input_file_path = os.path.join(root, file)
                tasks.append((input_file_path, input_dir, output_dir))
    
    if not tasks:
        print("No HDF5 files found in the input directory. Exiting.")
        return
        
    num_processes = p_cfg.get('num_processes', cpu_count())
    print(f"Found {len(tasks)} files to process using {num_processes} parallel workers.")
    
    # Create a partial function to pass the static 'config' object to the wrapper
    worker_func = partial(_process_file_wrapper, config=config)
    
    with Pool(processes=num_processes) as pool:
        pool.map(worker_func, tasks)
        
    print("--- Preprocessing Complete ---")