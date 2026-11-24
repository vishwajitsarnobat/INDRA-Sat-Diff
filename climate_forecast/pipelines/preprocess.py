import os
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from ..datasets.preprocess_logic import process_single_file


def _process_file_wrapper(args_tuple, config):
    input_file, input_dir, output_dir = args_tuple
    rel_path = os.path.relpath(os.path.dirname(input_file), input_dir)
    output_subdir = os.path.join(output_dir, rel_path)
    os.makedirs(output_subdir, exist_ok=True)

    base_name = os.path.basename(input_file)
    output_file_path = os.path.join(output_subdir, base_name)

    try:
        process_single_file(input_file, output_file_path, config)
    except Exception as e:
        print(f"ERROR processing {input_file}: {e}")


def run(config: dict):
    """Preprocess all HDF5 files in the input directory using parallel workers."""
    p_cfg = config['preprocess']
    input_dir = p_cfg['input_dir']
    output_dir = p_cfg['output_dir']

    print("--- Starting Preprocessing ---")
    print(f"Input: {input_dir}  Output: {output_dir}")

    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.h5', '.hdf5', '.nc', '.nc4')):
                tasks.append((os.path.join(root, file), input_dir, output_dir))

    if not tasks:
        print("No HDF5 files found. Exiting.")
        return

    num_processes = p_cfg.get('num_processes', cpu_count())
    print(f"Found {len(tasks)} files, {num_processes} workers.")

    worker_func = partial(_process_file_wrapper, config=config)

    with Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap_unordered(worker_func, tasks),
                  total=len(tasks), desc="Preprocessing", unit="file", ncols=80))

    print("--- Preprocessing Complete ---")