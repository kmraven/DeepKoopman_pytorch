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


def build_dataset(stacked: np.ndarray) -> TensorDataset:
    # Keep the full stacked trajectory tensor as one training sample.
    x = torch.from_numpy(stacked).to(torch.float64).unsqueeze(0)
    return TensorDataset(x)
