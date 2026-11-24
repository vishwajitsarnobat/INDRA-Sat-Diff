import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.nn.functional import avg_pool2d
from typing import List
from einops import rearrange


class IMERGDataLoader:
    """
    Parameters
    ----------
    data_dir : str
        Directory where the HDF5 files are stored.
    input_seq_len : int
        Number of consecutive frames to use as inputs.
    output_seq_len : int
        Number of consecutive frames to use as outputs.
    batch_size : int
        Number of sequences per batch.
    time_interval : float
        Expected time interval between consecutive files in seconds.
    tolerance : float
        Allowed tolerance (in seconds) for the time difference.
    shuffle : bool
        Whether to shuffle the valid sequences.
    csv_file : str, optional
        Path to CSV with pre-defined sequences (each row has seq_len filenames).
    """

    def __init__(
        self,
        data_dir,
        input_seq_len=6,
        output_seq_len=6,
        batch_size=4,
        time_interval=1800,
        tolerance=120,
        shuffle=False,
        csv_file: str = None,
    ):
        self.data_dir = data_dir
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.seq_len = input_seq_len + output_seq_len
        self.batch_size = batch_size
        self.time_interval = time_interval
        self.tolerance = tolerance
        self.shuffle = shuffle
        self.csv_file = csv_file

        # if CSV provided, load sequences directly
        if self.csv_file:
            df = pd.read_csv(self.csv_file, header=None)
            if df.shape[1] != self.seq_len:
                raise ValueError(f"CSV must have {self.seq_len} columns, got {df.shape[1]}.")
            valid = []
            for row in df.itertuples(index=False):
                paths = [os.path.join(self.data_dir, fn) for fn in row]
                try:
                    times = []
                    for f in paths:
                        with h5py.File(f, 'r') as hf:
                            times.append(hf['time'][()])
                except Exception:
                    continue
                if all(abs(times[i] - times[i-1] - self.time_interval) <= self.tolerance
                       for i in range(1, self.seq_len)):
                    valid.append(paths)
            self.valid_sequences = valid
        else:
            # scan directory and build sequences
            self.file_info = self._scan_files()
            if len(self.file_info) < self.seq_len:
                raise ValueError("Not enough files to form a sequence.")
            self.valid_sequences = self._build_valid_sequences()

        if not self.valid_sequences:
            raise ValueError("No valid sequences found.")
        if self.shuffle:
            np.random.shuffle(self.valid_sequences)
        self.curr_idx = 0

    def _scan_files(self):
        files = [os.path.join(self.data_dir, f)
                 for f in os.listdir(self.data_dir)
                 if f.lower().endswith(('.h5', '.hdf5'))]
        info = []
        for f in files:
            try:
                with h5py.File(f, 'r') as hf:
                    t = hf['time'][()]
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
            if all(abs(window[j] - window[j-1] - self.time_interval) <= self.tolerance
                   for j in range(1, len(window))):
                seqs.append(files[i:i+self.seq_len])
        return seqs

    def __iter__(self):
        self.curr_idx = 0
        return self

    def __next__(self):
        if self.curr_idx >= len(self.valid_sequences):
            raise StopIteration
        batch, cnt = [], 0
        while cnt < self.batch_size and self.curr_idx < len(self.valid_sequences):
            batch.append(self.valid_sequences[self.curr_idx])
            self.curr_idx += 1
            cnt += 1
        if not batch:
            raise StopIteration
        inputs, outputs = [], []
        for seq in batch:
            data = [h5py.File(f, 'r')['precipitationCal'][:] for f in seq]
            arr = np.array(data)
            inputs.append(arr[:self.input_seq_len])
            outputs.append(arr[self.input_seq_len:])
        return {
            "input": torch.from_numpy(np.array(inputs)),
            "output": torch.from_numpy(np.array(outputs))
        }

    def __len__(self):
        return len(self.valid_sequences) // self.batch_size