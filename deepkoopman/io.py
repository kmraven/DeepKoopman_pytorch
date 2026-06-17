from __future__ import annotations

from pathlib import Path

import numpy as np


def train_paths_for_dataset(data_dir: str | Path, dataset: str, count: int) -> list[Path]:
    data_dir = Path(data_dir)
    return [data_dir / f"{dataset}_train{i}_x.csv" for i in range(1, count + 1)]


def load_train_data(data_dir: str | Path, dataset: str, train_files: int) -> np.ndarray:
    paths = train_paths_for_dataset(data_dir, dataset, train_files)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training data files for {dataset}: {missing}")
    return np.concatenate([np.loadtxt(path, delimiter=",", dtype=np.float64) for path in paths], axis=0)


def load_split_data(data_dir: str | Path, dataset: str, train_files: int) -> dict[str, np.ndarray]:
    data_dir = Path(data_dir)
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
