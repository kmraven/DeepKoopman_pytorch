from __future__ import annotations

import csv
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
    block_id: str = ""
    section: str = ""
    condition: str = ""
    condition_id: int = 0
    sequence_index: int = 0
    start_window: int = 0
    end_window: int = 0
    global_window_id: int = 0


@dataclass(frozen=True)
class FeatureBlock:
    record: RatRecord
    split: str
    block_id: str
    fs: float
    features: np.ndarray
    artifact_mask: np.ndarray
    valid_mask: np.ndarray
    spans: list[tuple[int, int]]
    bad_channels: tuple[int, ...]


FREQUENCY_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma_40hz", 35.0, 45.0),
    ("high_gamma", 65.0, 150.0),
)

CONDITION_NAMES = ("silence", "normal_music", "gamma_music", "gamma_click")
CONDITION_TO_ID = {name: idx for idx, name in enumerate(CONDITION_NAMES)}


def yymmdd_to_yyyymmdd(value: str | int) -> str:
    text = str(value).strip().zfill(6)
    yy = int(text[:2])
    century = 2000 if yy < 70 else 1900
    return f"{century + yy:04d}{text[2:]}"


def load_metadata(path: str | Path = "data/rat_id.csv") -> list[RatRecord]:
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
    return waves[:, :64]


def section_label(time_point: str) -> str:
    mapping = {
        "before": "pre",
        "during_a": "during1",
        "during_b": "during2",
        "after": "post",
    }
    return mapping.get(time_point, time_point)


def condition_label(music_type: str, time_point: str) -> str:
    if time_point in {"before", "after"}:
        return "silence"
    mapping = {
        "conventional": "normal_music",
        "normal": "normal_music",
        "gamma": "gamma_music",
        "control": "gamma_click",
        "click": "gamma_click",
    }
    return mapping.get(music_type, music_type)


def block_id_for_record(record: RatRecord) -> str:
    stem = Path(record.filename).stem
    return f"{record.rat_id}_{record.music_type}_{record.time_point}_{stem}"


def _mad(values: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray:
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis, keepdims=keepdims)
    return np.maximum(mad / 0.6744897501960817, 1e-12)


def robust_zscore(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    scale = float(_mad(values))
    return (values - median) / scale


def detect_bad_channels(
    data: np.ndarray,
    *,
    flat_std: float = 1e-10,
    high_variance_factor: float = 10.0,
    saturation_threshold: float | None = None,
    saturation_fraction: float = 0.01,
    abnormal_amplitude_z: float = 12.0,
) -> np.ndarray:
    if data.ndim != 2 or data.shape[1] != 64:
        raise ValueError(f"Expected ephys data shaped (time, 64), got {data.shape}")
    std = np.nanstd(data, axis=0)
    good_std = std[std > flat_std]
    median_std = float(np.nanmedian(good_std)) if good_std.size else 1.0
    bad = std <= flat_std
    bad |= std > median_std * high_variance_factor
    peak = np.nanpercentile(np.abs(data), 99.9, axis=0)
    bad |= np.abs(robust_zscore(peak)) > abnormal_amplitude_z
    if saturation_threshold is not None:
        saturated = np.mean(np.abs(data) >= saturation_threshold, axis=0)
        bad |= saturated >= saturation_fraction
    return bad


def common_average_reference(data: np.ndarray, bad_channels: np.ndarray) -> np.ndarray:
    good = ~bad_channels
    if not np.any(good):
        raise ValueError("Cannot re-reference when all 64 channels are marked bad")
    referenced = data.copy()
    referenced[:, good] = referenced[:, good] - referenced[:, good].mean(axis=1, keepdims=True)
    referenced[:, bad_channels] = np.nan
    return referenced


def interpolate_bad_channels(data: np.ndarray, bad_channels: np.ndarray, grid_shape: tuple[int, int] = (8, 8)) -> np.ndarray:
    if data.shape[1] != grid_shape[0] * grid_shape[1]:
        raise ValueError(f"Data has {data.shape[1]} channels, incompatible with grid_shape={grid_shape}")
    if not np.any(bad_channels):
        return data
    out = data.copy()
    rows, cols = grid_shape
    good_average = np.nanmean(out[:, ~bad_channels], axis=1)
    bad_grid = bad_channels.reshape(rows, cols)
    for channel in np.flatnonzero(bad_channels):
        r, c = divmod(int(channel), cols)
        neighbors: list[int] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < rows and 0 <= cc < cols and not bad_grid[rr, cc]:
                    neighbors.append(rr * cols + cc)
        if neighbors:
            out[:, channel] = np.nanmean(out[:, neighbors], axis=1)
        else:
            out[:, channel] = good_average
    return out


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


def preprocess_ephys_block(
    data: np.ndarray,
    *,
    raw_fs: float = 1000.0,
    target_fs: float = 1000.0,
    line_freq: float = 50.0,
    bandpass: tuple[float, float] = (1.0, 200.0),
    max_bad_channels: int = 10,
    flat_std: float = 1e-10,
    high_variance_factor: float = 10.0,
    saturation_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    bad = detect_bad_channels(
        data,
        flat_std=flat_std,
        high_variance_factor=high_variance_factor,
        saturation_threshold=saturation_threshold,
    )
    if int(bad.sum()) >= max_bad_channels:
        raise ValueError(f"Too many bad channels: {int(bad.sum())} / 64")
    rereferenced = common_average_reference(data, bad)
    interpolated = interpolate_bad_channels(rereferenced, bad)
    filtered = preprocess_ephys(
        interpolated,
        raw_fs=raw_fs,
        target_fs=target_fs,
        line_freq=line_freq,
        bandpass=bandpass,
    )
    return filtered, bad


def bandpower_features(
    data: np.ndarray,
    *,
    fs: float,
    window_sec: float = 2.0,
    stride_sec: float = 0.5,
    bands: tuple[tuple[str, float, float], ...] = FREQUENCY_BANDS,
    eps: float = 1e-12,
    max_windows: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    if data.ndim != 2 or data.shape[1] != 64:
        raise ValueError(f"Expected filtered ephys data shaped (time, 64), got {data.shape}")
    nyq = fs / 2.0
    invalid = [(name, low, high) for name, low, high in bands if high >= nyq]
    if invalid:
        raise ValueError(f"Frequency bands exceed Nyquist={nyq:g} Hz: {invalid}")
    window_len = int(round(window_sec * fs))
    stride = int(round(stride_sec * fs))
    if window_len <= 0 or stride <= 0:
        raise ValueError("window_sec and stride_sec must produce positive sample counts")
    starts = list(range(0, data.shape[0] - window_len + 1, stride))
    if max_windows is not None:
        starts = starts[:max_windows]
    if not starts:
        raise ValueError(f"No feature windows can be made from data shape {data.shape}")

    features = np.empty((len(starts), 8, 8, len(bands)), dtype=np.float32)
    spans: list[tuple[int, int]] = []
    nperseg = min(window_len, max(8, int(round(fs))))
    for idx, start in enumerate(starts):
        end = start + window_len
        freqs, power = signal.welch(data[start:end], fs=fs, axis=0, nperseg=nperseg)
        band_values = []
        for _, low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            if not np.any(mask):
                raise ValueError(f"No Welch frequency bins for band {(low, high)} at fs={fs}")
            band_values.append(np.log(power[mask].mean(axis=0) + eps))
        channel_band = np.stack(band_values, axis=1)
        features[idx] = channel_band.reshape(8, 8, len(bands))
        spans.append((start, end))
    return features, spans


def artifact_window_mask(
    data: np.ndarray,
    features: np.ndarray,
    spans: list[tuple[int, int]],
    *,
    amplitude_z: float = 8.0,
    derivative_z: float = 8.0,
    power_z: float = 8.0,
    simultaneous_channel_fraction: float = 0.25,
) -> np.ndarray:
    if len(spans) < 10:
        return np.zeros(len(spans), dtype=bool)
    amplitude = np.empty((len(spans), 64), dtype=np.float64)
    derivative = np.empty((len(spans), 64), dtype=np.float64)
    for idx, (start, end) in enumerate(spans):
        segment = data[start:end]
        amplitude[idx] = np.nanmax(np.abs(segment), axis=0)
        derivative[idx] = np.nanmax(np.abs(np.diff(segment, axis=0)), axis=0)

    amp_z = np.abs((amplitude - np.nanmedian(amplitude, axis=0)) / _mad(amplitude, axis=0, keepdims=True))
    diff_z = np.abs((derivative - np.nanmedian(derivative, axis=0)) / _mad(derivative, axis=0, keepdims=True))
    flat_features = features.reshape(features.shape[0], -1)
    feat_z = np.abs((flat_features - np.nanmedian(flat_features, axis=0)) / _mad(flat_features, axis=0, keepdims=True))

    channel_fraction = np.mean((amp_z > amplitude_z) | (diff_z > derivative_z), axis=1)
    return (
        np.any(amp_z > amplitude_z, axis=1)
        | np.any(diff_z > derivative_z, axis=1)
        | np.any(feat_z > power_z, axis=1)
        | (channel_fraction >= simultaneous_channel_fraction)
    )


def fit_feature_zscore(blocks: list[FeatureBlock]) -> tuple[np.ndarray, np.ndarray]:
    values = [block.features[block.valid_mask] for block in blocks if block.split == "train" and np.any(block.valid_mask)]
    if not values:
        raise ValueError("No valid training feature windows were found for z-score fitting")
    train = np.concatenate(values, axis=0)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_feature_zscore(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((features - mean.reshape(1, *mean.shape)) / std.reshape(1, *std.shape)).astype(np.float32)


def make_feature_sequences(
    block: FeatureBlock,
    *,
    sequence_length: int = 64,
    sequence_stride: int = 1,
    max_sequences: int | None = None,
    pad_short: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[WindowRecord]]:
    if sequence_length <= 0 or sequence_stride <= 0:
        raise ValueError("sequence_length and sequence_stride must be positive")
    n = block.features.shape[0]
    condition = condition_label(block.record.music_type, block.record.time_point)
    condition_id = CONDITION_TO_ID[condition]
    section = section_label(block.record.time_point)
    x_parts: list[np.ndarray] = []
    c_parts: list[np.ndarray] = []
    rows: list[WindowRecord] = []

    starts = list(range(0, n - sequence_length + 1, sequence_stride))
    for start in starts:
        end = start + sequence_length
        if not bool(np.all(block.valid_mask[start:end])):
            continue
        x_parts.append(block.features[start:end].reshape(sequence_length, -1))
        c_parts.append(np.full(sequence_length, condition_id, dtype=np.int64))
        start_sample, _ = block.spans[start]
        _, end_sample = block.spans[end - 1]
        rows.append(
            WindowRecord(
                rat_id=block.record.rat_id,
                music_type=block.record.music_type,
                time_point=block.record.time_point,
                split=block.split,
                source_file=str(block.record.path),
                window_index=len(rows),
                start_sample=int(start_sample),
                end_sample=int(end_sample),
                start_sec=float(start_sample / block.fs),
                end_sec=float(end_sample / block.fs),
                block_id=block.block_id,
                section=section,
                condition=condition,
                condition_id=condition_id,
                sequence_index=len(rows),
                start_window=start,
                end_window=end,
            )
        )
        if max_sequences is not None and len(rows) >= max_sequences:
            break

    if not rows and pad_short and n > 0 and np.any(block.valid_mask):
        valid_indices = np.flatnonzero(block.valid_mask)
        first = int(valid_indices[0])
        contiguous = [first]
        for idx in valid_indices[1:]:
            if int(idx) == contiguous[-1] + 1:
                contiguous.append(int(idx))
            elif len(contiguous) >= sequence_length:
                break
            else:
                contiguous = [int(idx)]
        selected = np.asarray(contiguous[:sequence_length], dtype=np.int64)
        if selected.size == 0:
            selected = valid_indices[:1]
        if selected.size < sequence_length:
            selected = np.pad(selected, (0, sequence_length - selected.size), mode="edge")
        x_parts.append(block.features[selected].reshape(sequence_length, -1))
        c_parts.append(np.full(sequence_length, condition_id, dtype=np.int64))
        start_sample, _ = block.spans[int(selected[0])]
        _, end_sample = block.spans[int(selected[-1])]
        rows.append(
            WindowRecord(
                rat_id=block.record.rat_id,
                music_type=block.record.music_type,
                time_point=block.record.time_point,
                split=block.split,
                source_file=str(block.record.path),
                window_index=0,
                start_sample=int(start_sample),
                end_sample=int(end_sample),
                start_sec=float(start_sample / block.fs),
                end_sec=float(end_sample / block.fs),
                block_id=block.block_id,
                section=section,
                condition=condition,
                condition_id=condition_id,
                sequence_index=0,
                start_window=int(selected[0]),
                end_window=int(selected[-1]) + 1,
            )
        )

    if not x_parts:
        return (
            np.empty((0, sequence_length, block.features.shape[1] * block.features.shape[2] * block.features.shape[3]), dtype=np.float32),
            np.empty((0, sequence_length), dtype=np.int64),
            [],
        )
    return np.stack(x_parts).astype(np.float32), np.stack(c_parts), rows


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


def split_for_rat_fold(
    rat_id: str,
    *,
    fold: int = 0,
    num_rats: int = 15,
    test_rats: int = 3,
    val_rats: int = 2,
) -> str:
    rat_number = int(rat_id.split("_")[1])
    rat_index = rat_number - 1
    order = list(range(num_rats))
    start = (fold % max(num_rats // test_rats, 1)) * test_rats
    rotated = order[start:] + order[:start]
    test = set(rotated[:test_rats])
    val = set(rotated[test_rats : test_rats + val_rats])
    if rat_index in test:
        return "test"
    if rat_index in val:
        return "val"
    return "train"
