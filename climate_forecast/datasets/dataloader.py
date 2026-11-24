import os
import warnings
import numpy as np
import pandas as pd
import h5py
from typing import List, Dict


def _to_seconds(t) -> float:
    return float(np.asarray(t).reshape(-1)[0])


class GriddedDataLoader:
    """Discover and validate contiguous temporal sequences of gridded HDF5 climate data."""

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
        # (first, last) Unix timestamp of each sequence, parallel to valid_sequences
        self.sequence_times = []

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
            order = np.random.permutation(len(self.valid_sequences))
            self.valid_sequences = [self.valid_sequences[i] for i in order]
            self.sequence_times = [self.sequence_times[i] for i in order]

        # Warn if many files are present but few contiguous sequences can be formed
        if not self.csv_file:
            max_possible = max(0, len(self.file_info) - self.seq_len + 1)
            if max_possible > 0 and len(self.valid_sequences) < max_possible * 0.95:
                warnings.warn(
                    f"Found {len(self.file_info)} files but only {len(self.valid_sequences)} / {max_possible} "
                    f"contiguous sequences could be formed. Verify temporal continuity of input data."
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
            if all(abs(times[i] - times[i - 1] - self.time_interval_seconds) <= self.tolerance_seconds
                   for i in range(1, self.seq_len)):
                valid.append(paths)
                self.sequence_times.append((_to_seconds(times[0]), _to_seconds(times[-1])))
        self.valid_sequences = valid

    def _scan_files(self):
        files = [os.path.join(self.data_dir, f)
                 for f in os.listdir(self.data_dir)
                 if f.lower().endswith(('.h5', '.hdf5', '.nc', '.nc4'))]
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
            window = times[i:i + self.seq_len]
            if all(abs(window[j] - window[j - 1] - self.time_interval_seconds) <= self.tolerance_seconds
                   for j in range(1, len(window))):
                seqs.append(files[i:i + self.seq_len])
                self.sequence_times.append((_to_seconds(window[0]), _to_seconds(window[-1])))
        return seqs

    def __len__(self):
        return len(self.valid_sequences)