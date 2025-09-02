# climate_forecast/datasets/dataloader.py

import os
import warnings
import numpy as np
import pandas as pd
import h5py
from typing import List, Dict

class GriddedDataLoader:
    """
    A generic data loader for gridded, time-series climate data stored in HDF5 files.
    It identifies and loads valid, contiguous time sequences based on timestamps.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.data_cfg = config['data']
        self.layout_cfg = config['layout']

        self.data_dir = self.data_cfg['path']
        self.input_seq_len = self.layout_cfg['in_len']
        self.output_seq_len = self.layout_cfg['out_len']
        self.seq_len = self.input_seq_len + self.output_seq_len

        self.time_interval_seconds = self.data_cfg['time_interval_minutes'] * 60
        self.tolerance_seconds = self.data_cfg.get('time_tolerance_seconds', 120)

        self.time_variable_name = self.data_cfg['time_variable_name']

        self.shuffle = self.data_cfg.get('shuffle', False)
        self.csv_file = self.data_cfg.get('csv_file', None)

        if self.csv_file:
            self._load_sequences_from_csv()
        else:
            self.file_info = self._scan_files()
            if len(self.file_info) < self.seq_len:
                raise ValueError("Not enough files to form a sequence.")
            self.valid_sequences = self._build_valid_sequences()

        if not self.valid_sequences:
            raise ValueError("No valid sequences found in the specified data directory.")
        if self.shuffle:
            np.random.shuffle(self.valid_sequences)

        # --- NEW: DATA INTEGRITY WARNING ---
        if not self.csv_file:
            num_files_found = len(self.file_info)
            num_valid_sequences = len(self.valid_sequences)
            # A rough estimate of the maximum possible sequences
            max_possible_sequences = max(0, num_files_found - self.seq_len + 1)

            if max_possible_sequences > 0 and num_valid_sequences < max_possible_sequences * 0.95:
                warnings.warn(
                    f"Data Integrity Check: Found {num_files_found} preprocessed files, "
                    f"but was only able to create {num_valid_sequences} contiguous sequences out of a possible {max_possible_sequences}. "
                    f"A large number of sequences might be missing due to gaps (missing files) in your dataset. "
                    f"Please verify your preprocessed data is temporally continuous."
                )


    def _load_sequences_from_csv(self):
        df = pd.read_csv(self.csv_file, header=None)
        if df.shape[1] != self.seq_len:
            raise ValueError(f"CSV must have {self.seq_len} columns, got {df.shape[1]}.")
        valid = []
        for row in df.itertuples(index=False):
            paths = [os.path.join(self.data_dir, fn) for fn in row]
            try:
                times = [h5py.File(f, 'r')[self.time_variable_name][()] for f in paths]
            except Exception:
                continue
            if all(abs(times[i] - times[i-1] - self.time_interval_seconds) <= self.tolerance_seconds
                   for i in range(1, self.seq_len)):
                valid.append(paths)
        self.valid_sequences = valid

    def _scan_files(self):
        files = [os.path.join(self.data_dir, f)
                 for f in os.listdir(self.data_dir)
                 if f.lower().endswith(('.h5', '.hdf5'))]
        info = []
        for f in files:
            try:
                with h5py.File(f, 'r') as hf:
                    t = hf[self.time_variable_name][()]
                info.append((t, f))
            except Exception:
                pass
        info.sort(key=lambda x: x[0])
        return info

    def _build_valid_sequences(self) -> List[List[str]]:
        times = [t for t, _ in self.file_info]
        files = [f for _, f in self.file_info]
        seqs = []
        for i in range(len(times) - self.seq_len + 1):
            window = times[i:i+self.seq_len]
            if all(abs(window[j] - window[j-1] - self.time_interval_seconds) <= self.tolerance_seconds
                   for j in range(1, len(window))):
                seqs.append(files[i:i+self.seq_len])
        return seqs

    def __len__(self):
        return len(self.valid_sequences)