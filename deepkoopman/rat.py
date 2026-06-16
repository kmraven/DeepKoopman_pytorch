from __future__ import annotations

import csv
import importlib.util
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np
from scipy import io as scipy_io
from scipy import signal


@dataclass(frozen=True)
class RatRecord:
    rat_id: str
    music_type: str
    time_point: str
    date: str
    number: int
    filename: str
    yyyymmdd: str
    path: Path | None = None


@dataclass(frozen=True)
class WindowRecord:
    rat_id: str
    music_type: str
    time_point: str
    split: str
    source_file: str
    window_index: int
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float


def yymmdd_to_yyyymmdd(value: str | int) -> str:
    text = str(value).strip().zfill(6)
    yy = int(text[:2])
    century = 2000 if yy < 70 else 1900
    return f"{century + yy:04d}{text[2:]}"


def load_metadata(path: str | Path = "rat_data/rat_id.csv") -> list[RatRecord]:
    records: list[RatRecord] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            records.append(
                RatRecord(
                    rat_id=row["rat_id"],
                    music_type=row["music_type"],
                    time_point=row["time_point"],
                    date=str(row["date"]).strip(),
                    number=int(row["number"]),
                    filename=row["filename"],
                    yyyymmdd=yymmdd_to_yyyymmdd(row["date"]),
                )
            )
    return records


def load_data_root_template(env_path: str | Path = "rat_data/env.py") -> str:
    env_path = Path(env_path)
    spec = importlib.util.spec_from_file_location("rat_data_env", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return str(module.data_root_path)


def resolve_case_insensitive(root: Path, filename: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    lowered = filename.lower()
    matches = [p for p in root.iterdir() if p.name.lower() == lowered]
    if len(matches) == 1:
        return matches[0]
    direct = root / filename
    if direct.exists():
        return direct
    if not matches:
        raise FileNotFoundError(f"No case-insensitive match for {filename} under {root}")
    raise FileExistsError(f"Multiple case-insensitive matches for {filename} under {root}: {matches}")


def attach_paths(records: list[RatRecord], root_template: str) -> list[RatRecord]:
    out: list[RatRecord] = []
    for record in records:
        root = Path(root_template.format(yyyymmdd=record.yyyymmdd))
        out.append(replace(record, path=resolve_case_insensitive(root, record.filename)))
    return out


def filter_existing_records(records: list[RatRecord], root_template: str) -> list[RatRecord]:
    out: list[RatRecord] = []
    for record in records:
        try:
            out.extend(attach_paths([record], root_template))
        except FileNotFoundError:
            continue
    return out


def load_analog_waves(path: str | Path, variable: str = "analogWaves") -> np.ndarray:
    path = Path(path)
    try:
        with h5py.File(path, "r") as f:
            if variable not in f:
                raise KeyError(f"{variable} not found in {path}; keys={list(f.keys())}")
            arr = np.asarray(f[variable])
    except OSError:
        mat = scipy_io.loadmat(path)
        if variable not in mat:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            raise KeyError(f"{variable} not found in {path}; keys={keys}")
        arr = np.asarray(mat[variable])

    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D analog waves, got {arr.shape} in {path}")
    if arr.shape[0] == 66:
        arr = arr.T
    if arr.shape[1] != 66:
        raise ValueError(f"Expected analog waves shaped (time, 66) or (66, time), got {arr.shape} in {path}")
    return arr.astype(np.float64, copy=False)


def extract_ephys_channels(waves: np.ndarray) -> np.ndarray:
    if waves.ndim != 2 or waves.shape[1] != 66:
        raise ValueError(f"Expected (time, 66), got {waves.shape}")
    return waves[:, 2:66]


def preprocess_ephys(
    data: np.ndarray,
    *,
    raw_fs: float = 1000.0,
    target_fs: float = 250.0,
    line_freq: float = 50.0,
    bandpass: tuple[float, float] = (1.0, 100.0),
) -> np.ndarray:
    x = signal.detrend(data, axis=0, type="constant")
    nyq = raw_fs / 2.0

    if line_freq > 0:
        for freq in np.arange(line_freq, nyq, line_freq):
            b, a = signal.iirnotch(freq / nyq, Q=30.0)
            x = signal.filtfilt(b, a, x, axis=0)

    low, high = bandpass
    if low <= 0 or high >= nyq or low >= high:
        raise ValueError(f"Invalid bandpass {bandpass} for raw_fs={raw_fs}")
    sos = signal.butter(4, [low, high], btype="bandpass", fs=raw_fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=0)

    if target_fs != raw_fs:
        ratio = Fraction(target_fs / raw_fs).limit_denominator(1000)
        target_len = int(round(x.shape[0] * target_fs / raw_fs))
        x = signal.resample_poly(x, ratio.numerator, ratio.denominator, axis=0)
        x = x[:target_len]
    return x.astype(np.float64, copy=False)


def make_windows(
    data: np.ndarray,
    *,
    fs: float = 250.0,
    window_sec: float = 1.0,
    stride_sec: float = 0.5,
    max_windows: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    window_len = int(round(window_sec * fs)) + 1
    stride = int(round(stride_sec * fs))
    if stride <= 0:
        raise ValueError("stride_sec must produce at least one sample")
    starts = list(range(0, data.shape[0] - window_len + 1, stride))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError(f"No windows can be made from data shape {data.shape}")
    windows = np.stack([data[start : start + window_len] for start in starts], axis=0)
    spans = [(start, start + window_len) for start in starts]
    return windows.astype(np.float64, copy=False), spans


def fit_zscore(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = windows.reshape(-1, windows.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_zscore(windows: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (windows - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)


def flatten_windows(windows: np.ndarray) -> np.ndarray:
    return windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2])


def split_for_rat(rat_id: str) -> str:
    number = int(rat_id.split("_")[1])
    if number <= 11:
        return "train"
    if number <= 13:
        return "val"
    return "test"
