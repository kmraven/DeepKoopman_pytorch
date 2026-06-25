from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


@dataclass(frozen=True)
class H5SplitData:
    path: Path
    dataset_path: str
    shape: tuple[int, int, int]
    dtype: str
    condition_dataset_path: str | None = None
    condition_shape: tuple[int, ...] | None = None


TrajectoryData = np.ndarray | H5SplitData


def train_paths_for_dataset(data_dir: str | Path, dataset: str, count: int) -> list[Path]:
    data_dir = Path(data_dir)
    return [data_dir / f"{dataset}_train{i}_x.csv" for i in range(1, count + 1)]


def h5_path_for_dataset(data_dir: str | Path, dataset: str) -> Path:
    return Path(data_dir) / f"{dataset}.h5"


def _h5_split(path: Path, split: str) -> H5SplitData:
    dataset_path = f"/{split}/x"
    condition_dataset_path = f"/{split}/condition"
    with h5py.File(path, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"{dataset_path} not found in {path}")
        dataset = f[dataset_path]
        if dataset.ndim != 3:
            raise ValueError(f"Expected {dataset_path} to be 3-D (windows, time, dim), got {dataset.shape}")
        condition_shape = None
        if condition_dataset_path in f:
            condition = f[condition_dataset_path]
            if tuple(condition.shape[:2]) != tuple(dataset.shape[:2]):
                raise ValueError(
                    f"{condition_dataset_path} shape {condition.shape} does not match x time axes {dataset.shape[:2]}"
                )
            condition_shape = tuple(int(v) for v in condition.shape)
        else:
            condition_dataset_path = None
        return H5SplitData(
            path=path,
            dataset_path=dataset_path,
            shape=tuple(int(v) for v in dataset.shape),
            dtype=str(dataset.dtype),
            condition_dataset_path=condition_dataset_path,
            condition_shape=condition_shape,
        )


def load_train_data(data_dir: str | Path, dataset: str, train_files: int) -> TrajectoryData:
    h5_path = h5_path_for_dataset(data_dir, dataset)
    if h5_path.exists():
        return _h5_split(h5_path, "train")

    paths = train_paths_for_dataset(data_dir, dataset, train_files)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training data files for {dataset}: {missing}")
    return np.concatenate([np.loadtxt(path, delimiter=",", dtype=np.float64) for path in paths], axis=0)


def load_split_data(data_dir: str | Path, dataset: str, train_files: int) -> dict[str, TrajectoryData]:
    data_dir = Path(data_dir)
    h5_path = h5_path_for_dataset(data_dir, dataset)
    if h5_path.exists():
        return {
            "train": _h5_split(h5_path, "train"),
            "val": _h5_split(h5_path, "val"),
            "test": _h5_split(h5_path, "test"),
        }

    val_path = data_dir / f"{dataset}_val_x.csv"
    test_path = data_dir / f"{dataset}_test_x.csv"
    missing = [str(path) for path in [val_path, test_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing validation/test data files for {dataset}: {missing}")
    return {
        "train": load_train_data(data_dir, dataset, train_files),
        "val": np.loadtxt(val_path, delimiter=",", dtype=np.float64),
        "test": np.loadtxt(test_path, delimiter=",", dtype=np.float64),
    }
