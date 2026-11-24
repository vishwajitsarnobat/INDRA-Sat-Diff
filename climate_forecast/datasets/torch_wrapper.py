import os
from datetime import datetime, timezone
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lightning import LightningDataModule, seed_everything
import h5py
from omegaconf import OmegaConf

from .dataloader import GriddedDataLoader
from .augmentation import TransformsFixRotation


class ClimateDataset(Dataset):
    """PyTorch Dataset loading multi-channel gridded climate sequences from HDF5 files."""

    def __init__(self, config: dict, valid_sequences: list):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.valid_sequences = valid_sequences
        self.channels = self.config.data.channels
        self.aug_mode = self.config.data.get('aug_mode', '0')
        self.output_type = np.float32

        if self.aug_mode == "0":
            self.aug = nn.Identity()
        elif self.aug_mode == "1":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=180),
            )
        elif self.aug_mode == "2":
            self.aug = nn.Sequential(
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                TransformsFixRotation(angles=[0, 90, 180, 270]),
            )
        else:
            raise NotImplementedError(f"Augmentation mode {self.aug_mode} not implemented.")

    def __len__(self):
        return len(self.valid_sequences)

    def __getitem__(self, index):
        seq_files = self.valid_sequences[index]
        seq_data_frames = []

        for file_path in seq_files:
            with h5py.File(file_path, 'r') as hf:
                frame_channels = []
                for channel_name in self.channels:
                    data = np.squeeze(hf[channel_name][:])
                    if data.ndim != 2:
                        raise ValueError(f"Channel '{channel_name}' in {file_path} is not 2D after squeeze.")
                    frame_channels.append(data)
                multi_channel_frame = np.stack(frame_channels, axis=-1)
                seq_data_frames.append(multi_channel_frame)

        seq_data = np.array(seq_data_frames, dtype=self.output_type)
        seq_tensor = torch.from_numpy(seq_data)
        # Convert to (T, C, H, W) for transforms, then back to target layout
        seq_tensor = seq_tensor.permute(0, 3, 1, 2)

        if self.aug_mode != "0":
            t, c, h, w = seq_tensor.shape
            seq_tensor = self.aug(seq_tensor.reshape(t * c, 1, h, w))
            seq_tensor = seq_tensor.reshape(t, c, h, w)

        target_layout = self.config.layout.layout.replace('N', '')
        permute_dims = ["TCHW".find(dim) for dim in target_layout]
        return seq_tensor.permute(*permute_dims).contiguous()


def _parse_years(entries) -> set:
    """Expand a list of years and inclusive "start-end" ranges into a set of ints."""
    years = set()
    for e in entries or []:
        if isinstance(e, str) and '-' in e:
            start, end = (int(x) for x in e.split('-'))
            years.update(range(start, end + 1))
        else:
            years.add(int(e))
    return years


def _split_by_year(sequences: list, sequence_times: list, split_cfg) -> tuple:
    """Assign each sequence to train/val/test by the UTC year of its frames; drop year-crossing windows."""
    year_sets = [_parse_years(split_cfg.get(k)) for k in ('train_years', 'val_years', 'test_years')]
    test_is_val = not year_sets[2] or year_sets[2] == year_sets[1]   # no separate test years: test on val
    pairs = [(0, 1), (0, 2)] + ([] if test_is_val else [(1, 2)])
    for i, j in pairs:
        if year_sets[i] & year_sets[j]:
            raise ValueError(f"data.split years overlap: {sorted(year_sets[i] & year_sets[j])}")
    if test_is_val:
        year_sets[2] = set()
    splits = ([], [], [])
    for seq, (t_first, t_last) in zip(sequences, sequence_times):
        y_first = datetime.fromtimestamp(t_first, tz=timezone.utc).year
        y_last = datetime.fromtimestamp(t_last, tz=timezone.utc).year
        if y_first != y_last:
            continue
        for split, years in zip(splits, year_sets):
            if y_first in years:
                split.append(seq)
                break
    if test_is_val:
        splits[2].extend(splits[1])
    return splits


class ClimateDataModule(LightningDataModule):
    """LightningDataModule that discovers, splits, and loads climate sequences."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = OmegaConf.create(config)
        self.config_dict = OmegaConf.to_container(self.config, resolve=True)
        self.val_ratio = self.config.data.get('val_ratio', 0.1)
        self.test_ratio = self.config.data.get('test_ratio', 0.1)
        self.seed = self.config.optim.get('seed', 42)

    def prepare_data(self) -> None:
        if not os.path.exists(self.config.data.path):
            raise FileNotFoundError(f"Processed data directory not found: {self.config.data.path}")

    def setup(self, stage: str = None) -> None:
        seed_everything(self.seed)
        sequence_loader = GriddedDataLoader(self.config_dict)
        all_sequences = sequence_loader.valid_sequences

        split_cfg = self.config.data.get('split', None) or {}
        split_mode = split_cfg.get('mode', 'random')
        if split_mode == 'year':
            train_seqs, val_seqs, test_seqs = _split_by_year(
                all_sequences, sequence_loader.sequence_times, split_cfg)
            print(f"  > Year split: {len(train_seqs)} train / {len(val_seqs)} val / {len(test_seqs)} test sequences")
            if not train_seqs or not val_seqs or not test_seqs:
                raise ValueError("Year split produced an empty train/val/test set; check data.split years.")
        elif split_mode == 'random':
            total_len = len(all_sequences)
            val_len = int(total_len * self.val_ratio)
            test_len = int(total_len * self.test_ratio)
            train_len = total_len - val_len - test_len
            if train_len < 1:
                raise ValueError("Not enough samples for training after splitting dataset.")

            train_seqs, val_seqs, test_seqs = random_split(
                all_sequences, [train_len, val_len, test_len],
                generator=torch.Generator().manual_seed(self.seed),
            )
        else:
            raise ValueError(f"Unknown data.split.mode '{split_mode}'; use 'random' or 'year'.")

        self.train_dataset = ClimateDataset(self.config_dict, train_seqs)
        self.val_dataset = ClimateDataset(self.config_dict, val_seqs)
        self.test_dataset = ClimateDataset(self.config_dict, test_seqs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=self.config.data.get('shuffle', True),
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True,
        )

    @property
    def num_train_samples(self):
        return len(self.train_dataset)

    @property
    def num_val_samples(self):
        return len(self.val_dataset)

    @property
    def num_test_samples(self):
        return len(self.test_dataset)