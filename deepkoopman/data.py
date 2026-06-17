from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning as L
from torch.utils.data import DataLoader, Dataset

from .config import DeepKoopmanConfig


def stack_data(data: np.ndarray, num_shifts: int, len_time: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    n = data.shape[1]
    num_traj = data.shape[0] // len_time
    new_len_time = len_time - num_shifts
    tensor = np.zeros((num_shifts + 1, num_traj * new_len_time, n), dtype=np.float64)
    for j in range(num_shifts + 1):
        for t in range(num_traj):
            src_start = t * len_time + j
            src_end = src_start + new_len_time
            dst_start = t * new_len_time
            dst_end = dst_start + new_len_time
            tensor[j, dst_start:dst_end, :] = data[src_start:src_end, :]
    return tensor


def stack_data_windows(
    data: np.ndarray,
    num_shifts: int,
    len_time: int,
    window_indices: np.ndarray,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    n = data.shape[1]
    num_traj = data.shape[0] // len_time
    if data.shape[0] != num_traj * len_time:
        raise ValueError(f"Data length {data.shape[0]} is not divisible by len_time={len_time}")
    if num_shifts >= len_time:
        raise ValueError(f"num_shifts={num_shifts} must be smaller than len_time={len_time}")

    indices = np.asarray(window_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("window_indices must be 1-D")
    if indices.size and (indices.min() < 0 or indices.max() >= num_traj):
        raise IndexError(f"window index out of range for {num_traj} trajectories")

    out_dtype = np.dtype(dtype) if dtype is not None else data.dtype
    new_len_time = len_time - num_shifts
    tensor = np.empty((num_shifts + 1, indices.size * new_len_time, n), dtype=out_dtype)
    for j in range(num_shifts + 1):
        for dst_t, src_t in enumerate(indices):
            src_start = int(src_t) * len_time + j
            src_end = src_start + new_len_time
            dst_start = dst_t * new_len_time
            dst_end = dst_start + new_len_time
            tensor[j, dst_start:dst_end, :] = data[src_start:src_end, :]
    return tensor


class WindowedTrajectoryDataset(Dataset):
    def __init__(self, data: np.ndarray, config: DeepKoopmanConfig):
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[0] % config.data.len_time != 0:
            raise ValueError(f"Data length {data.shape[0]} is not divisible by len_time={config.data.len_time}")
        self.data = data
        self.config = config
        self.max_shift = max([1] + config.data.shifts + config.data.middle_shifts)
        if self.max_shift >= config.data.len_time:
            raise ValueError(f"max shift {self.max_shift} must be smaller than len_time={config.data.len_time}")
        self.dtype = np.float32 if config.runtime.dtype == "float32" else np.float64
        self.num_trajectories = data.shape[0] // config.data.len_time

    def __len__(self) -> int:
        return self.num_trajectories

    def __getitem__(self, index: int) -> torch.Tensor:
        stacked = stack_data_windows(
            self.data,
            self.max_shift,
            self.config.data.len_time,
            np.asarray([index], dtype=np.int64),
            dtype=self.dtype,
        )
        return torch.as_tensor(stacked)


class DeepKoopmanDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        config: DeepKoopmanConfig,
        test_data: np.ndarray | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            WindowedTrajectoryDataset(self.train_data, self.config),
            batch_size=self.config.trainer.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            WindowedTrajectoryDataset(self.val_data, self.config),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_data is None:
            return None
        return DataLoader(
            WindowedTrajectoryDataset(self.test_data, self.config),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
