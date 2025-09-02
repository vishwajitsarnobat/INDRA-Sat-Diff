import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lightning import LightningDataModule, seed_everything
import h5py
import torch.nn.functional as F
from omegaconf import OmegaConf

# These now use relative paths within our package, which is the correct way.
from .dataloader import GriddedDataLoader
from .augmentation import TransformsFixRotation


class ClimateDataset(Dataset):
    """
    A generic PyTorch Dataset for loading sequences of gridded climate data.
    It is configured via a dictionary and handles multiple data channels.
    """
    def __init__(self, config: dict, valid_sequences: list):
        super().__init__()
        # Store config and the list of file paths for this split (train/val/test)
        self.config = OmegaConf.create(config)
        self.valid_sequences = valid_sequences
        
        # Extract necessary settings from config
        self.channels = self.config.data.channels
        self.aug_mode = self.config.data.get('aug_mode', '0')
        self.output_type = np.float32

        # Set up augmentation transforms
        if self.aug_mode == "0":
            self.aug = nn.Identity() # Use Identity for clarity
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
                # --- CRITICAL GENERALIZED LOGIC ---
                # Loop through the channels specified in the config and stack them.
                frame_channels = []
                for channel_name in self.channels:
                    # Read data and ensure it's at least 2D
                    data = np.squeeze(hf[channel_name][:])
                    if data.ndim != 2:
                        raise ValueError(f"Data for channel '{channel_name}' in {file_path} is not 2D after squeeze.")
                    frame_channels.append(data)
                
                # Stack along the last axis to create a (H, W, C) array
                multi_channel_frame = np.stack(frame_channels, axis=-1)
                seq_data_frames.append(multi_channel_frame)

        # Stack frames to create a (T, H, W, C) array
        seq_data = np.array(seq_data_frames, dtype=self.output_type)
        seq_data_tensor = torch.from_numpy(seq_data)

        # Permute to (T, C, H, W) for PyTorch-native transforms
        seq_data_tensor = seq_data_tensor.permute(0, 3, 1, 2)

        # Apply augmentation if enabled
        if self.aug_mode != "0":
            # Apply the same random transform to all frames in the sequence
            # This is done by stacking them into a "super-batch"
            t, c, h, w = seq_data_tensor.shape
            seq_data_tensor = self.aug(seq_data_tensor.reshape(t * c, 1, h, w))
            seq_data_tensor = seq_data_tensor.reshape(t, c, h, w)

        # Permute the final tensor to match the layout specified by the user
        # (e.g., NTHWC). The batch dim 'N' is added by the DataLoader.
        source_layout = "TCHW"
        target_layout = self.config.layout.layout.replace('N', '') # e.g., "THWC"
        
        # Create the permutation mapping
        permute_dims = [source_layout.find(dim) for dim in target_layout]
        seq_data_tensor = seq_data_tensor.permute(*permute_dims)

        return seq_data_tensor.contiguous()


class ClimateDataModule(LightningDataModule):
    """
    A generic LightningDataModule for climate datasets, driven by a config file.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = OmegaConf.create(config)
        # Store a copy of the raw dict for the dataset constructor
        self.config_dict = OmegaConf.to_container(self.config, resolve=True)

        self.val_ratio = self.config.data.get('val_ratio', 0.1)
        self.test_ratio = self.config.data.get('test_ratio', 0.1)
        self.seed = self.config.optim.get('seed', 42)

    def prepare_data(self) -> None:
        """Checks if the processed data directory exists."""
        if not os.path.exists(self.config.data.path):
            raise FileNotFoundError(f"Processed data directory not found: {self.config.data.path}")

    def setup(self, stage: str = None) -> None:
        """
        Finds all valid data sequences and splits them into train, val, and test sets.
        """
        seed_everything(self.seed)
        
        # Instantiate the loader to find all valid, contiguous file sequences
        sequence_loader = GriddedDataLoader(self.config_dict)
        all_sequences = sequence_loader.valid_sequences
        
        # Split the list of valid file paths, not the dataset object
        total_len = len(all_sequences)
        val_len = int(total_len * self.val_ratio)
        test_len = int(total_len * self.test_ratio)
        train_len = total_len - val_len - test_len
        
        if train_len < 1:
            raise ValueError("Not enough samples for training after splitting dataset.")
        
        # Perform the split
        train_seqs, val_seqs, test_seqs = random_split(
            all_sequences,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )
        
        # Create separate Dataset instances for each split, passing the config and the file lists
        self.train_dataset = ClimateDataset(self.config_dict, train_seqs)
        self.val_dataset = ClimateDataset(self.config_dict, val_seqs)
        self.test_dataset = ClimateDataset(self.config_dict, test_seqs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=self.config.data.get('shuffle', True),
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.optim.micro_batch_size,
            shuffle=False,
            num_workers=self.config.data.get('num_workers', 4),
            pin_memory=True
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