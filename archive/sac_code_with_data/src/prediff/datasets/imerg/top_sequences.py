import os
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
from functools import partial
import random

# --- Config ---
data_dir = "data/processed_south"
key = 'precipitationCal'
sequence_length = 13
top_percentage = 10  # Percentage (e.g., 20 for 20%)
random_percentage = 1 # Percentage (e.g., 1 for 1%)
output_csv = 'src/prediff/datasets/imerg/input_sequences_south.csv'
num_workers = 100  # Number of parallel processes

# --- Step 1: Load and sort filenames ---
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.HDF5')])

def extract_timestamp(f):
    return datetime.strptime(f[:12], '%Y%m%d%H%M')

all_files.sort(key=extract_timestamp)

# --- Step 2: Build sequences (assumes file contiguity) ---
sequences = []
for i in range(len(all_files) - sequence_length + 1):
    seq = all_files[i:i + sequence_length]
    sequences.append(seq)

# --- Step 3: Compute precipitation sum per sequence (parallelized) ---
def compute_sequence_sum(seq, data_dir, key):
    total = 0
    for fname in seq:
        try:
            with h5py.File(os.path.join(data_dir, fname), 'r') as f:
                hem = f[key][()]
                total += np.sum(hem)
        except (KeyError, OSError):
            return -np.inf  # Mark invalid sequences
    return total

compute_sum_partial = partial(compute_sequence_sum, data_dir=data_dir, key=key)

print("Computing precipitation totals using multiprocessing...")
with Pool(num_workers) as pool:
    results = list(tqdm(
        pool.imap(compute_sum_partial, sequences, chunksize=100),
        total=len(sequences),
        desc="Processing sequences"
    ))

sequence_scores = []
for seq, total_sum in zip(sequences, results):
    if total_sum != -np.inf:
        sequence_scores.append((seq, total_sum))

# --- Step 4: Select top N% and random M% ---
sequence_scores.sort(key=lambda x: x[1], reverse=True)

num_total_valid = len(sequence_scores)
num_top = int((top_percentage / 100.0) * num_total_valid)
top_sequences = [s[0] for s in sequence_scores[:num_top]]

remaining_sequences = [s[0] for s in sequence_scores[num_top:]]
num_random = min(int((random_percentage / 100.0) * num_total_valid), len(remaining_sequences))

random.seed(42) # Use python's random for random.sample
random_sequences = random.sample(remaining_sequences, num_random)

# --- Step 5: Save to CSV ---
selected_sequences = top_sequences + random_sequences
df = pd.DataFrame(selected_sequences)
df.to_csv(output_csv, index=False, header=False)

print(f"Saved {len(selected_sequences)} sequences to {output_csv}")