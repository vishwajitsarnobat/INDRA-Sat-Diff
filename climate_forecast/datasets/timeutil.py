"""Timestamps of data files (UTC seconds), from the embedded time variable or the file name."""
import os
import re
from datetime import datetime, timezone

import h5py
import numpy as np

_PATTERNS = (re.compile(r"(\d{8})-S(\d{4})"),   # IMERG: 3B-HHR....20230601-S003000-...
             re.compile(r"(\d{8})(\d{4})"))     # SAC files: 202306010030.HDF5


def time_from_name(path):
    """UTC seconds from a file name holding YYYYMMDD-SHHMM (IMERG) or YYYYMMDDHHMM; None if absent."""
    name = os.path.basename(path)
    for pat in _PATTERNS:
        m = pat.search(name)
        if m:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp()
    return None


def file_time(path, time_variable, source="file"):
    """UTC seconds of one file. source="filename" trusts the file name (needed for the SAC-processed data,
    whose embedded times are five days late); otherwise the embedded (already converted) time is used."""
    if source == "filename":
        t = time_from_name(path)
        if t is None:
            raise ValueError(f"No timestamp in file name: {path}")
        return t
    with h5py.File(path, "r") as hf:
        return float(np.asarray(hf[time_variable][()]).reshape(-1)[0])
