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
    WindowRecord,
    apply_zscore,
    attach_paths,
    extract_ephys_channels,
    filter_existing_records,
    fit_zscore,
    load_analog_waves,
    load_metadata,
    make_windows,
    preprocess_ephys,
    split_for_rat,
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
    target_fs: float = 250.0
    line_freq: float = 50.0
    bandpass_low: float = 1.0
    bandpass_high: float = 100.0
    window_sec: float = 1.0
    stride_sec: float = 0.5
    max_windows_per_record: int | None = None


@dataclass
class RatPreprocessConfig:
    input: RatInputConfig | dict = field(default_factory=RatInputConfig)
    preprocessing: RatPreprocessingConfig | dict = field(default_factory=RatPreprocessingConfig)
    deepkoopman: DeepKoopmanConfig | dict = field(
        default_factory=lambda: DeepKoopmanConfig(
            data=DataConfig(
                name="RatAuditoryCortex",
                len_time=251,
                delta_t=1.0 / 250.0,
                shifts=list(range(1, 11)),
                middle_shifts=list(range(1, 11)),
            ),
            model=ModelConfig(
                widths=[64, 256, 128, 3, 3, 128, 256, 64],
                omega_hidden_widths=[64, 64],
                num_real=1,
                num_complex_pairs=1,
            ),
            loss=LossConfig(
                reconstruction_weight=0.1,
                middle_shift_weight=1.0,
                linf_weight=1e-8,
                l2_weight=1e-12,
            ),
            optimizer=OptimizerConfig(lr=1e-3),
            trainer=TrainerConfig(batch_size=256, max_epochs=5),
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


def _split_map(records, quick: bool) -> dict:
    if not quick:
        return {record: split_for_rat(record.rat_id) for record in records}
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
            }
            for row in rows
        ],
    )


def _write_training_hdf5(
    output_dir: Path,
    dataset: str,
    windows_by_split: dict[str, np.ndarray],
    metadata: list[WindowRecord],
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
        config.preprocessing.max_windows_per_record = 4
    else:
        config.preprocessing.max_windows_per_record = min(config.preprocessing.max_windows_per_record, 4)
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

    windows_by_split, metadata, stats = _prepare_windows(config, records, quick=quick, show_progress=not no_progress)
    data_paths = _write_training_hdf5(data_dir, dataset, windows_by_split, metadata)

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
