from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import h5py

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning as L
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import DeepKoopmanConfig
from .io import H5SplitData, TrajectoryData


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


def stack_window(window: np.ndarray, num_shifts: int, dtype: np.dtype | type | None = None) -> np.ndarray:
    if window.ndim == 1:
        window = window[:, None]
    if num_shifts >= window.shape[0]:
        raise ValueError(f"num_shifts={num_shifts} must be smaller than window length={window.shape[0]}")
    out_dtype = np.dtype(dtype) if dtype is not None else window.dtype
    new_len_time = window.shape[0] - num_shifts
    tensor = np.empty((num_shifts + 1, new_len_time, window.shape[1]), dtype=out_dtype)
    for j in range(num_shifts + 1):
        tensor[j] = window[j : j + new_len_time]
    return tensor


def stack_window_starts(
    window: np.ndarray,
    num_shifts: int,
    starts: np.ndarray,
    dtype: np.dtype | type | None = None,
) -> np.ndarray:
    if window.ndim == 1:
        window = window[:, None]
    if num_shifts >= window.shape[0]:
        raise ValueError(f"num_shifts={num_shifts} must be smaller than window length={window.shape[0]}")
    starts = np.asarray(starts, dtype=np.int64)
    max_start = window.shape[0] - num_shifts
    if starts.ndim != 1:
        raise ValueError("starts must be 1-D")
    if starts.size and (starts.min() < 0 or starts.max() >= max_start):
        raise IndexError(f"start index out of range for max_start={max_start}")
    out_dtype = np.dtype(dtype) if dtype is not None else window.dtype
    tensor = np.empty((num_shifts + 1, starts.size, window.shape[1]), dtype=out_dtype)
    for j in range(num_shifts + 1):
        tensor[j] = window[starts + j]
    return tensor


def trajectory_count(data: TrajectoryData, len_time: int) -> int:
    if isinstance(data, H5SplitData):
        return data.shape[0]
    if data.ndim == 1:
        data = data[:, None]
    return data.shape[0] // len_time


def read_trajectories(data: TrajectoryData, indices: np.ndarray, len_time: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if isinstance(data, H5SplitData):
        if indices.size == 0:
            return np.empty((0, data.shape[1], data.shape[2]), dtype=np.dtype(data.dtype))
        order = np.argsort(indices)
        sorted_indices = indices[order]
        with h5py.File(data.path, "r") as f:
            values = np.asarray(f[data.dataset_path][sorted_indices])
        inverse = np.empty_like(order)
        inverse[order] = np.arange(order.size)
        return values[inverse]

    if data.ndim == 1:
        data = data[:, None]
    usable = (data.shape[0] // len_time) * len_time
    trajectories = data[:usable].reshape(usable // len_time, len_time, data.shape[1])
    return trajectories[indices]


def read_conditions(data: TrajectoryData, indices: np.ndarray, len_time: int) -> np.ndarray | None:
    indices = np.asarray(indices, dtype=np.int64)
    if not isinstance(data, H5SplitData) or data.condition_dataset_path is None:
        return None
    if indices.size == 0:
        return np.empty((0, len_time), dtype=np.int64)
    order = np.argsort(indices)
    sorted_indices = indices[order]
    with h5py.File(data.path, "r") as f:
        values = np.asarray(f[data.condition_dataset_path][sorted_indices])
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return values[inverse]


class WindowedTrajectoryDataset(Dataset):
    def __init__(self, data: TrajectoryData, config: DeepKoopmanConfig, *, random_starts: bool = False):
        if isinstance(data, H5SplitData):
            if data.shape[1] != config.data.len_time:
                raise ValueError(f"HDF5 window length {data.shape[1]} does not match len_time={config.data.len_time}")
        elif data.ndim == 1:
            data = data[:, None]
        if not isinstance(data, H5SplitData) and data.shape[0] % config.data.len_time != 0:
            raise ValueError(f"Data length {data.shape[0]} is not divisible by len_time={config.data.len_time}")
        self.data = data
        self.config = config
        self.max_shift = max([1] + config.data.shifts + config.data.middle_shifts)
        if self.max_shift >= config.data.len_time:
            raise ValueError(f"max shift {self.max_shift} must be smaller than len_time={config.data.len_time}")
        self.dtype = np.float32 if config.runtime.dtype == "float32" else np.float64
        self.num_trajectories = trajectory_count(data, config.data.len_time)
        self.random_starts = random_starts
        self.starts_per_sequence = config.data.starts_per_sequence
        self._h5_file: h5py.File | None = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5_file"] = None
        return state

    def __del__(self):
        if getattr(self, "_h5_file", None) is not None:
            self._h5_file.close()

    def _h5_dataset(self):
        if not isinstance(self.data, H5SplitData):
            raise TypeError("HDF5 dataset requested for non-HDF5 data")
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data.path, "r")
        return self._h5_file[self.data.dataset_path]

    def _h5_condition_dataset(self):
        if not isinstance(self.data, H5SplitData) or self.data.condition_dataset_path is None:
            return None
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data.path, "r")
        return self._h5_file[self.data.condition_dataset_path]

    def condition_labels(self) -> np.ndarray | None:
        if not isinstance(self.data, H5SplitData) or self.data.condition_dataset_path is None:
            return None
        with h5py.File(self.data.path, "r") as f:
            cond = np.asarray(f[self.data.condition_dataset_path][:, 0], dtype=np.int64)
        return cond

    def __len__(self) -> int:
        return self.num_trajectories

    def _start_indices(self) -> np.ndarray | None:
        count = self.starts_per_sequence
        if count is None:
            return None
        max_start = self.config.data.len_time - self.max_shift
        if count >= max_start:
            return np.arange(max_start, dtype=np.int64)
        if self.random_starts:
            return np.random.randint(0, max_start, size=count, dtype=np.int64)
        return np.linspace(0, max_start - 1, num=count, dtype=np.int64)

    def __getitem__(self, index: int) -> torch.Tensor:
        starts = self._start_indices()
        if isinstance(self.data, H5SplitData):
            window = np.asarray(self._h5_dataset()[index])
            if starts is None:
                stacked_x = stack_window(window, self.max_shift, dtype=self.dtype)
            else:
                stacked_x = stack_window_starts(window, self.max_shift, starts, dtype=self.dtype)
            x = torch.as_tensor(stacked_x)
            condition_dataset = self._h5_condition_dataset()
            if condition_dataset is None:
                return x
            condition_window = np.asarray(condition_dataset[index], dtype=np.int64)
            if starts is None:
                condition = stack_window(condition_window, self.max_shift, dtype=np.int64)
            else:
                condition = stack_window_starts(condition_window, self.max_shift, starts, dtype=np.int64)
            return x, torch.as_tensor(condition[..., 0], dtype=torch.long)
        if starts is not None:
            if self.data.ndim == 1:
                data = self.data[:, None]
            else:
                data = self.data
            start = index * self.config.data.len_time
            stacked = stack_window_starts(
                data[start : start + self.config.data.len_time],
                self.max_shift,
                starts,
                dtype=self.dtype,
            )
        else:
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
        train_data: TrajectoryData,
        val_data: TrajectoryData,
        config: DeepKoopmanConfig,
        test_data: TrajectoryData | None = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.config = config
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        dataset = WindowedTrajectoryDataset(self.train_data, self.config, random_starts=True)
        sampler = None
        shuffle = True
        labels = dataset.condition_labels()
        if labels is not None and labels.size:
            counts = np.bincount(labels, minlength=max(int(labels.max()) + 1, 1)).astype(np.float64)
            counts[counts == 0] = 1.0
            weights = 1.0 / counts[labels]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(weights, dtype=torch.double),
                num_samples=len(labels),
                replacement=True,
            )
            shuffle = False
        return DataLoader(
            dataset,
            batch_size=self.config.trainer.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            WindowedTrajectoryDataset(self.val_data, self.config, random_starts=False),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader | None:
        if self.test_data is None:
            return None
        return DataLoader(
            WindowedTrajectoryDataset(self.test_data, self.config, random_starts=False),
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
