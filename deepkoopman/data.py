from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset


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


def build_dataset(stacked: np.ndarray) -> TensorDataset:
    # Keep the full stacked trajectory tensor as one training sample.
    x = torch.from_numpy(stacked).to(torch.float64).unsqueeze(0)
    return TensorDataset(x)
