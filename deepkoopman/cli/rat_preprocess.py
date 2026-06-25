from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml
import h5py
from tqdm.auto import tqdm

from deepkoopman.config import DataConfig, DeepKoopmanConfig, LossConfig, ModelConfig, OptimizerConfig, RuntimeConfig, TrainerConfig
from deepkoopman.rat import (
    CONDITION_NAMES,
    FREQUENCY_BANDS,
    FeatureBlock,
    WindowRecord,
    apply_zscore,
    attach_paths,
    bandpower_features,
    artifact_window_mask,
    apply_feature_zscore,
    block_id_for_record,
    condition_label,
    extract_ephys_channels,
    fit_feature_zscore,
    filter_existing_records,
    fit_zscore,
    load_analog_waves,
    load_metadata,
    make_windows,
    make_feature_sequences,
    preprocess_ephys,
    preprocess_ephys_block,
    section_label,
    split_for_rat,
    split_for_rat_fold,
)


@dataclass
class RatMetadataConfig:
    path: str = "data/rat_id.csv"


@dataclass
class RatSourceConfig:
    data_root_template: str = "rat_data_tmp/{yyyymmdd}/data_mat"


@dataclass
class RatInputConfig:
    metadata: RatMetadataConfig | dict = field(default_factory=RatMetadataConfig)
    source: RatSourceConfig | dict = field(default_factory=RatSourceConfig)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            self.metadata = RatMetadataConfig(**self.metadata)
        if isinstance(self.source, dict):
            self.source = RatSourceConfig(**self.source)


@dataclass
class RatPreprocessingConfig:
    raw_fs: float = 1000.0
    target_fs: float = 1000.0
    line_freq: float = 50.0
    bandpass_low: float = 1.0
    bandpass_high: float = 200.0
    window_sec: float = 2.0
    stride_sec: float = 0.5
    max_windows_per_record: int | None = None
    mode: str = "bandpower_sequence"
    sequence_length: int = 64
    sequence_stride: int = 1
    max_sequences_per_record: int | None = None
    fold: int = 0
    num_rats: int = 15
    test_rats_per_fold: int = 3
    val_rats_per_fold: int = 2
    max_bad_channels: int = 10
    flat_std: float = 1e-10
    high_variance_factor: float = 10.0
    saturation_threshold: float | None = None
    artifact_amplitude_z: float = 8.0
    artifact_derivative_z: float = 8.0
    artifact_power_z: float = 8.0
    artifact_channel_fraction: float = 0.25
    processed_root: str = "processed"
    pad_short_sequences: bool = False


@dataclass
class RatPreprocessConfig:
    input: RatInputConfig | dict = field(default_factory=RatInputConfig)
    preprocessing: RatPreprocessingConfig | dict = field(default_factory=RatPreprocessingConfig)
    deepkoopman: DeepKoopmanConfig | dict = field(
        default_factory=lambda: DeepKoopmanConfig(
            data=DataConfig(
                name="RatAuditoryCortex",
                len_time=64,
                delta_t=0.5,
                shifts=list(range(1, 21)),
                middle_shifts=list(range(1, 21)),
                input_shape=[8, 8, 6],
                condition_names=list(CONDITION_NAMES),
            ),
            model=ModelConfig(
                widths=[64, 256, 128, 3, 3, 128, 256, 64],
                omega_hidden_widths=[64, 64],
                num_real=1,
                num_complex_pairs=1,
                activation="gelu",
                architecture="rat_conditional_conv",
                condition_dim=len(CONDITION_NAMES),
            ),
            loss=LossConfig(
                reconstruction_weight=1.0,
                prediction_weight=1.0,
                middle_shift_weight=0.5,
                covariance_weight=1e-3,
                linf_weight=0.0,
                l2_weight=0.0,
            ),
            optimizer=OptimizerConfig(name="adamw", lr=1e-3, weight_decay=1e-5),
            trainer=TrainerConfig(batch_size=128, max_epochs=230, autoencoder_warmup_epochs=30, gradient_clip_val=1.0),
            runtime=RuntimeConfig(seed=42, dtype="float32"),
        )
    )

    def __post_init__(self) -> None:
        if isinstance(self.input, dict):
            self.input = RatInputConfig(**self.input)
        if isinstance(self.preprocessing, dict):
            self.preprocessing = RatPreprocessingConfig(**self.preprocessing)
        if isinstance(self.deepkoopman, dict):
            self.deepkoopman = DeepKoopmanConfig(**self.deepkoopman)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RatPreprocessConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        raw = {key: value for key, value in raw.items() if key in {"input", "preprocessing", "deepkoopman"}}
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_records(config: RatPreprocessConfig, *, quick: bool):
    records = load_metadata(config.input.metadata.path)
    root_template = config.input.source.data_root_template
    if quick:
        return filter_existing_records(records, root_template)
    return attach_paths(records, root_template)


def _split_map(records, quick: bool, preprocessing: RatPreprocessingConfig | None = None) -> dict:
    if not quick:
        if preprocessing is None:
            return {record: split_for_rat(record.rat_id) for record in records}
        return {
            record: split_for_rat_fold(
                record.rat_id,
                fold=preprocessing.fold,
                num_rats=preprocessing.num_rats,
                test_rats=preprocessing.test_rats_per_fold,
                val_rats=preprocessing.val_rats_per_fold,
            )
            for record in records
        }
    mapping = {}
    for idx, record in enumerate(records):
        mapping[record] = "train" if idx == 0 else "val"
    return mapping


def _copy_window_records(records: list[WindowRecord], source_split: str, target_split: str) -> list[WindowRecord]:
    return [
        WindowRecord(
            rat_id=row.rat_id,
            music_type=row.music_type,
            time_point=row.time_point,
            split=target_split,
            source_file=row.source_file,
            window_index=row.window_index,
            start_sample=row.start_sample,
            end_sample=row.end_sample,
            start_sec=row.start_sec,
            end_sec=row.end_sec,
            block_id=row.block_id,
            section=row.section,
            condition=row.condition,
            condition_id=row.condition_id,
            sequence_index=row.sequence_index,
            start_window=row.start_window,
            end_window=row.end_window,
            global_window_id=row.global_window_id,
        )
        for row in records
        if row.split == source_split
    ]


def _prepare_windows(
    config: RatPreprocessConfig,
    records,
    *,
    quick: bool,
    show_progress: bool,
) -> tuple[dict[str, np.ndarray], list[WindowRecord], dict[str, object]]:
    split_map = _split_map(records, quick)
    raw_by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    metadata: list[WindowRecord] = []
    preprocessing = config.preprocessing

    iterator = tqdm(records, desc="preprocess", unit="file", disable=not show_progress)
    for record in iterator:
        waves = load_analog_waves(record.path)
        ephys = extract_ephys_channels(waves)
        processed = preprocess_ephys(
            ephys,
            raw_fs=preprocessing.raw_fs,
            target_fs=preprocessing.target_fs,
            line_freq=preprocessing.line_freq,
            bandpass=(preprocessing.bandpass_low, preprocessing.bandpass_high),
        )
        windows, spans = make_windows(
            processed,
            fs=preprocessing.target_fs,
            window_sec=preprocessing.window_sec,
            stride_sec=preprocessing.stride_sec,
            max_windows=preprocessing.max_windows_per_record,
        )
        split = split_map[record]
        raw_by_split[split].append(windows)
        for index, (start, end) in enumerate(spans):
            metadata.append(
                WindowRecord(
                    rat_id=record.rat_id,
                    music_type=record.music_type,
                    time_point=record.time_point,
                    split=split,
                    source_file=str(record.path),
                    window_index=index,
                    start_sample=start,
                    end_sample=end,
                    start_sec=start / preprocessing.target_fs,
                    end_sec=end / preprocessing.target_fs,
                )
            )

    if not raw_by_split["train"]:
        raise ValueError("No training windows were found.")
    if not raw_by_split["val"]:
        raw_by_split["val"] = raw_by_split["train"][:1]
        metadata.extend(_copy_window_records(metadata, "train", "val"))
    if not raw_by_split["test"]:
        raw_by_split["test"] = raw_by_split["val"][:1]
        metadata.extend(_copy_window_records(metadata, "val", "test"))

    train_windows = np.concatenate(raw_by_split["train"], axis=0)
    mean, std = fit_zscore(train_windows)

    windows_by_split = {}
    counts = {}
    for split, parts in raw_by_split.items():
        windows = np.concatenate(parts, axis=0)
        windows = apply_zscore(windows, mean, std)
        windows_by_split[split] = windows
        counts[split] = int(windows.shape[0])

    stats = {
        "quick_mode": bool(quick),
        "zscore_mean": mean.tolist(),
        "zscore_std": std.tolist(),
        "window_counts": counts,
    }
    return windows_by_split, metadata, stats


def _resolve_processed_root(config: RatPreprocessConfig, data_dir: Path, output_data_dir: str | Path | None) -> Path:
    root = Path(config.preprocessing.processed_root)
    if root.is_absolute():
        return root
    if output_data_dir is not None:
        return data_dir / root
    return root


def _write_processed_block(processed_root: Path, block: FeatureBlock, standardized_features: np.ndarray) -> dict[str, str]:
    block_dir = processed_root / block.record.rat_id / block.block_id
    block_dir.mkdir(parents=True, exist_ok=True)
    feature_path = block_dir / "features.npy"
    artifact_path = block_dir / "artifact_mask.npy"
    labels_path = block_dir / "labels.csv"
    np.save(feature_path, standardized_features.astype(np.float32))
    np.save(artifact_path, block.artifact_mask.astype(bool))
    rows = []
    section = section_label(block.record.time_point)
    condition = condition_label(block.record.music_type, block.record.time_point)
    for idx, (start, end) in enumerate(block.spans):
        rows.append(
            {
                "rat_id": block.record.rat_id,
                "block_id": block.block_id,
                "music_type": block.record.music_type,
                "time_point": block.record.time_point,
                "section": section,
                "condition": condition,
                "window_index": idx,
                "start_sample": start,
                "end_sample": end,
                "start_sec": start / block.fs,
                "end_sec": end / block.fs,
                "artifact": bool(block.artifact_mask[idx]),
            }
        )
    _write_rows(labels_path, rows)
    return {"features": str(feature_path), "artifact_mask": str(artifact_path), "labels": str(labels_path)}


def _prepare_feature_blocks(
    config: RatPreprocessConfig,
    records,
    *,
    quick: bool,
    show_progress: bool,
) -> tuple[list[FeatureBlock], dict[str, object]]:
    preprocessing = config.preprocessing
    split_map = _split_map(records, quick, preprocessing)
    blocks: list[FeatureBlock] = []
    skipped: list[dict[str, str]] = []
    iterator = tqdm(records, desc="rat features", unit="file", disable=not show_progress)
    for record in iterator:
        try:
            waves = load_analog_waves(record.path)
            ephys = extract_ephys_channels(waves)
            filtered, bad_channels = preprocess_ephys_block(
                ephys,
                raw_fs=preprocessing.raw_fs,
                target_fs=preprocessing.target_fs,
                line_freq=preprocessing.line_freq,
                bandpass=(preprocessing.bandpass_low, preprocessing.bandpass_high),
                max_bad_channels=preprocessing.max_bad_channels,
                flat_std=preprocessing.flat_std,
                high_variance_factor=preprocessing.high_variance_factor,
                saturation_threshold=preprocessing.saturation_threshold,
            )
            features, spans = bandpower_features(
                filtered,
                fs=preprocessing.target_fs,
                window_sec=preprocessing.window_sec,
                stride_sec=preprocessing.stride_sec,
                max_windows=preprocessing.max_windows_per_record,
            )
            artifact_mask = artifact_window_mask(
                filtered,
                features,
                spans,
                amplitude_z=preprocessing.artifact_amplitude_z,
                derivative_z=preprocessing.artifact_derivative_z,
                power_z=preprocessing.artifact_power_z,
                simultaneous_channel_fraction=preprocessing.artifact_channel_fraction,
            )
            blocks.append(
                FeatureBlock(
                    record=record,
                    split=split_map[record],
                    block_id=block_id_for_record(record),
                    fs=preprocessing.target_fs,
                    features=features,
                    artifact_mask=artifact_mask,
                    valid_mask=~artifact_mask,
                    spans=spans,
                    bad_channels=tuple(int(idx) for idx in np.flatnonzero(bad_channels)),
                )
            )
        except Exception as exc:
            skipped.append({"source_file": str(record.path), "reason": str(exc)})

    if not blocks:
        raise ValueError("No rat feature blocks were produced.")
    stats = {
        "quick_mode": bool(quick),
        "num_blocks": len(blocks),
        "skipped_records": skipped,
        "bad_channel_counts": {
            block.block_id: len(block.bad_channels)
            for block in blocks
        },
        "artifact_window_counts": {
            block.block_id: int(block.artifact_mask.sum())
            for block in blocks
        },
        "bands": [{"name": name, "low": low, "high": high} for name, low, high in FREQUENCY_BANDS],
    }
    return blocks, stats


def _feature_sequences_by_split(
    blocks: list[FeatureBlock],
    config: RatPreprocessConfig,
    *,
    processed_root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[WindowRecord], dict[str, object]]:
    preprocessing = config.preprocessing
    mean, std = fit_feature_zscore(blocks)
    x_parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    c_parts: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    metadata: list[WindowRecord] = []
    processed_files: dict[str, dict[str, str]] = {}
    for block in blocks:
        standardized = apply_feature_zscore(block.features, mean, std)
        std_block = FeatureBlock(
            record=block.record,
            split=block.split,
            block_id=block.block_id,
            fs=block.fs,
            features=standardized,
            artifact_mask=block.artifact_mask,
            valid_mask=block.valid_mask,
            spans=block.spans,
            bad_channels=block.bad_channels,
        )
        processed_files[block.block_id] = _write_processed_block(processed_root, block, standardized)
        x, condition, rows = make_feature_sequences(
            std_block,
            sequence_length=preprocessing.sequence_length,
            sequence_stride=preprocessing.sequence_stride,
            max_sequences=preprocessing.max_sequences_per_record,
            pad_short=preprocessing.pad_short_sequences,
        )
        if x.size:
            x_parts[block.split].append(x)
            c_parts[block.split].append(condition)
            metadata.extend(rows)

    for source, target in [("train", "val"), ("val", "test")]:
        if not x_parts[target] and x_parts[source]:
            x_parts[target] = [x_parts[source][0][:1]]
            c_parts[target] = [c_parts[source][0][:1]]
            metadata.extend(_copy_window_records(metadata, source, target)[:1])

    if not x_parts["train"]:
        raise ValueError("No training sequences were produced.")

    sequences_by_split: dict[str, np.ndarray] = {}
    conditions_by_split: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for split in ["train", "val", "test"]:
        if not x_parts[split]:
            raise ValueError(f"No {split} sequences were produced.")
        sequences_by_split[split] = np.concatenate(x_parts[split], axis=0).astype(np.float32)
        conditions_by_split[split] = np.concatenate(c_parts[split], axis=0).astype(np.int64)
        counts[split] = int(sequences_by_split[split].shape[0])

    stats = {
        "zscore_mean": mean.tolist(),
        "zscore_std": std.tolist(),
        "sequence_counts": counts,
        "processed_root": str(processed_root),
        "processed_files": processed_files,
    }
    return sequences_by_split, conditions_by_split, metadata, stats


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_window_metadata(path: Path, rows: list[WindowRecord]) -> None:
    _write_rows(
        path,
        [
            {
                "split": row.split,
                "rat_id": row.rat_id,
                "music_type": row.music_type,
                "time_point": row.time_point,
                "source_file": row.source_file,
                "window_index": row.window_index,
                "start_sample": row.start_sample,
                "end_sample": row.end_sample,
                "start_sec": row.start_sec,
                "end_sec": row.end_sec,
                "block_id": row.block_id,
                "section": row.section,
                "condition": row.condition,
                "condition_id": row.condition_id,
                "sequence_index": row.sequence_index,
                "start_window": row.start_window,
                "end_window": row.end_window,
                "global_window_id": idx if row.global_window_id == 0 else row.global_window_id,
            }
            for idx, row in enumerate(rows)
        ],
    )


def _write_training_hdf5(
    output_dir: Path,
    dataset: str,
    windows_by_split: dict[str, np.ndarray],
    metadata: list[WindowRecord],
    conditions_by_split: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h5_path = output_dir / f"{dataset}.h5"
    metadata_path = output_dir / f"{dataset}_window_metadata.csv"
    with h5py.File(h5_path, "w") as f:
        for split in ["train", "val", "test"]:
            windows = np.asarray(windows_by_split[split], dtype=np.float32)
            chunk_windows = min(max(1, windows.shape[0]), 64)
            group = f.create_group(split)
            group.create_dataset(
                "x",
                data=windows,
                chunks=(chunk_windows, windows.shape[1], windows.shape[2]),
                compression="lzf",
                shuffle=True,
            )
            if conditions_by_split is not None:
                condition = np.asarray(conditions_by_split[split], dtype=np.int64)
                group.create_dataset(
                    "condition",
                    data=condition,
                    chunks=(chunk_windows, condition.shape[1]),
                    compression="lzf",
                    shuffle=True,
                )
        if conditions_by_split is not None:
            f.attrs["condition_names"] = json.dumps(list(CONDITION_NAMES))
            f.attrs["frequency_bands"] = json.dumps(
                [{"name": name, "low": low, "high": high} for name, low, high in FREQUENCY_BANDS]
            )
    _write_window_metadata(metadata_path, metadata)
    return {"hdf5": str(h5_path), "metadata": str(metadata_path)}


def _apply_runtime_overrides(
    config: RatPreprocessConfig,
    *,
    data_root_template: str | None = None,
) -> RatPreprocessConfig:
    if data_root_template is not None:
        config.input.source.data_root_template = data_root_template
    return config


def _apply_quick_overrides(config: RatPreprocessConfig) -> RatPreprocessConfig:
    if config.preprocessing.max_windows_per_record is None:
        config.preprocessing.max_windows_per_record = 8
    else:
        config.preprocessing.max_windows_per_record = min(config.preprocessing.max_windows_per_record, 8)
    config.preprocessing.pad_short_sequences = True
    if config.preprocessing.max_sequences_per_record is None:
        config.preprocessing.max_sequences_per_record = 2
    else:
        config.preprocessing.max_sequences_per_record = min(config.preprocessing.max_sequences_per_record, 2)
    return config


def run_preprocess(
    config: RatPreprocessConfig,
    *,
    output_data_dir: str | Path | None = None,
    quick: bool = False,
    quick_records: int | None = None,
    no_progress: bool = False,
    rebuild: bool = False,
) -> dict[str, object]:
    if quick:
        config = _apply_quick_overrides(config)

    data_dir = Path(output_data_dir) if output_data_dir is not None else Path(config.deepkoopman.data.root)
    dataset = config.deepkoopman.data.name
    h5_path = data_dir / f"{dataset}.h5"
    metadata_path = data_dir / f"{dataset}_window_metadata.csv"
    summary_path = data_dir / f"{dataset}_preprocess_summary.json"
    if h5_path.exists() and metadata_path.exists() and not rebuild:
        summary = {
            "dataset": dataset,
            "data_dir": str(data_dir),
            "quick_mode": bool(quick),
            "reused_existing": True,
            "window_counts": {},
            "data": {"hdf5": str(h5_path), "metadata": str(metadata_path)},
            "config": config.to_dict(),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summary["summary"] = str(summary_path)
        return summary

    records = _load_records(config, quick=quick)
    if quick and quick_records is not None:
        records = records[:quick_records]

    mode = config.preprocessing.mode
    quick_legacy_waveform = False
    if quick and mode == "bandpower_sequence" and config.preprocessing.max_windows_per_record is not None:
        if config.preprocessing.max_windows_per_record <= 2:
            mode = "waveform_window"
            quick_legacy_waveform = True

    if mode == "waveform_window":
        if quick_legacy_waveform:
            config.preprocessing.target_fs = 250.0
            config.preprocessing.bandpass_high = min(config.preprocessing.bandpass_high, 100.0)
            config.preprocessing.window_sec = 1.0
        windows_by_split, metadata, stats = _prepare_windows(config, records, quick=quick, show_progress=not no_progress)
        data_paths = _write_training_hdf5(data_dir, dataset, windows_by_split, metadata)
    elif mode == "bandpower_sequence":
        processed_root = _resolve_processed_root(config, data_dir, output_data_dir)
        blocks, block_stats = _prepare_feature_blocks(config, records, quick=quick, show_progress=not no_progress)
        windows_by_split, conditions_by_split, metadata, sequence_stats = _feature_sequences_by_split(
            blocks,
            config,
            processed_root=processed_root,
        )
        data_paths = _write_training_hdf5(data_dir, dataset, windows_by_split, metadata, conditions_by_split)
        stats = {**block_stats, **sequence_stats}
    else:
        raise ValueError(f"Unsupported rat preprocessing mode: {config.preprocessing.mode}")

    summary = {
        "dataset": dataset,
        "data_dir": str(data_dir),
        "num_records": len(records),
        "quick_mode": bool(quick),
        "reused_existing": False,
        "window_counts": stats.get("window_counts", {}),
        "preprocessing_stats": stats,
        "data": data_paths,
        "config": config.to_dict(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rat_analysis/default.yaml")
    parser.add_argument("--data-root-template", default=None)
    parser.add_argument("--output-data-dir", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=2)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = RatPreprocessConfig.from_yaml(args.config)
    config = _apply_runtime_overrides(
        config,
        data_root_template=args.data_root_template,
    )

    print(
        json.dumps(
            run_preprocess(
                config,
                output_data_dir=args.output_data_dir,
                quick=args.quick,
                quick_records=args.quick_records,
                no_progress=args.no_progress,
                rebuild=args.rebuild,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
