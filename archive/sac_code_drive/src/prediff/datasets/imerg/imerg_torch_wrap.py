import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from lightning import LightningDataModule, seed_everything
from einops import rearrange
import h5py
import torch.nn.functional as F  # for interpolation

from src.prediff.datasets.imerg.imerg_dataloader import IMERGDataLoader
from src.prediff.datasets.augmentation import TransformsFixRotation


class IMERGTorchDataset(Dataset):
    """
    A PyTorch Dataset wrapper for IMERG data,
    with optional CSV-driven sequence selection.
    """
    def __init__(
        self,
        input_seq_len: int = 6,
        output_seq_len: int = 6,
        data_dir: str = None,
        shuffle: bool = False,
        output_type=np.float32,
        aug_mode: str = "0",
        ret_contiguous: bool = True,
        downsample: int = 0,
        csv_file: str = None,
    ):
        super().__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.seq_len = input_seq_len + output_seq_len
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.output_type = output_type
        self.ret_contiguous = ret_contiguous
        self.aug_mode = aug_mode
        self.downsample = downsample
        self.csv_file = csv_file

        # initialize underlying loader (batch_size=1) and get valid sequences
        self.imerg_dataloader = IMERGDataLoader(
            data_dir=self.data_dir,
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            batch_size=1,
            shuffle=self.shuffle,
            csv_file=self.csv_file,
        )
        self.valid_sequences = self.imerg_dataloader.valid_sequences
        if not self.valid_sequences:
            raise ValueError("No valid sequences found.")

        # set up augmentation
        if self.aug_mode == "0":
            self.aug = lambda x: x
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

    def _apply_downsample(self, frame_tensor):
        if self.downsample == 0:
            return frame_tensor
        elif self.downsample == 1:
            return F.interpolate(
                frame_tensor.unsqueeze(0), size=(256, 256), mode='bilinear',
                align_corners=False, antialias=True
            ).squeeze(0)
        elif self.downsample == 2:
            return F.interpolate(
                frame_tensor.unsqueeze(0), size=(128, 128), mode='bilinear',
                align_corners=False, antialias=True
            ).squeeze(0)
        else:
            raise ValueError("Invalid downsample value. Use 0, 1, or 2.")

    def __getitem__(self, index):
        seq_files = self.valid_sequences[index]
        seq_data = []
        for file in seq_files:
            with h5py.File(file, 'r') as hf:
                arr = hf['precipitationCal'][:][0]
            seq_data.append(arr)

        seq_data = np.expand_dims(np.array(seq_data).astype(self.output_type), axis=-1)

        # apply downsampling
        if self.downsample:
            frames = []
            for i in range(seq_data.shape[0]):
                frame = torch.from_numpy(seq_data[i]).float().permute(2, 0, 1)
                frame = self._apply_downsample(frame)
                frames.append(frame.permute(1, 2, 0))
            seq_data = torch.stack(frames, dim=0)
        else:
            seq_data = torch.from_numpy(seq_data)

        # apply augmentation
        if self.aug_mode != "0":
            auged = []
            for frame in seq_data:
                f = frame.permute(2, 0, 1)
                f = self.aug(f)
                auged.append(f.permute(1, 2, 0))
            seq_data = torch.stack(auged, dim=0)

        if self.ret_contiguous:
            seq_data = seq_data.contiguous()
        return seq_data

    def __len__(self):
        return len(self.valid_sequences)


class IMERGLightningDataModule(LightningDataModule):
    """
    A LightningDataModule for IMERG data,
    with optional CSV-driven sequence selection.
    """
    def __init__(
        self,
        input_seq_len: int = 7,
        output_seq_len: int = 6,
        data_dir: str = 'data/processed',
        aug_mode: str = "0",
        ret_contiguous: bool = True,
        val_ratio: float = 0.1,
        batch_size: int = 1,
        num_workers: int = 1,
        seed: int = 0,
        downsample: int = 0,
        csv_file: str = None,
    ):
        super().__init__()
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.data_dir = data_dir
        self.aug_mode = aug_mode
        self.ret_contiguous = ret_contiguous
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.downsample = downsample
        self.csv_file = csv_file

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"IMERG data directory {self.data_dir} not found!")

    def setup(self, stage=None) -> None:
        seed_everything(self.seed)
        full_dataset = IMERGTorchDataset(
            input_seq_len=self.input_seq_len,
            output_seq_len=self.output_seq_len,
            data_dir=self.data_dir,
            shuffle=True,
            output_type=np.float32,
            aug_mode=self.aug_mode,
            ret_contiguous=self.ret_contiguous,
            downsample=self.downsample,
            csv_file=self.csv_file,
        )
        total = len(full_dataset)
        val_len = int(total * self.val_ratio)
        test_len = int(total * self.val_ratio)
        train_len = total - val_len - test_len
        if train_len < 1:
            raise ValueError("Not enough samples for training after splitting.")
        self.imerg_train, self.imerg_val, self.imerg_test = random_split(
            full_dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(
            self.imerg_train, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.imerg_val, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.imerg_test, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )

    @property
    def num_train_samples(self):
        return len(self.imerg_train)

    @property
    def num_val_samples(self):
        return len(self.imerg_val)

    @property
    def num_test_samples(self):
        return len(self.imerg_test)