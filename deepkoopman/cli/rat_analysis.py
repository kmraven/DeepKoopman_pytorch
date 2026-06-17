from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from deepkoopman.config import DataConfig, DeepKoopmanConfig, LossConfig, ModelConfig, OptimizerConfig, RuntimeConfig, TrainerConfig
from deepkoopman.data import DeepKoopmanDataModule, WindowedTrajectoryDataset
from deepkoopman.lightning import DeepKoopmanLightningModule, build_trainer
from deepkoopman.losses import compute_losses
from deepkoopman.rat import (
    WindowRecord,
    apply_zscore,
    attach_paths,
    extract_ephys_channels,
    filter_existing_records,
    fit_zscore,
    flatten_windows,
    load_analog_waves,
    load_data_root_template,
    load_metadata,
    make_windows,
    preprocess_ephys,
    split_for_rat,
)
from deepkoopman.visualization import load_history, plot_losses, save_history_csv


@dataclass
class RatInputConfig:
    metadata: str = "rat_data/rat_id.csv"
    env: str = "rat_data/env.py"
    data_root: str | None = None


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
class RatCacheConfig:
    preprocessed_dir: str | None = None
    preprocessed_cache_dir: str = "results/rat_preprocessed_cache"
    save_preprocessed: bool = True


@dataclass
class RatAnalysisConfig:
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
    output_dir: str = "results/rat_analysis"
    latent_samples: int = 200
    cache: RatCacheConfig | dict = field(default_factory=RatCacheConfig)

    def __post_init__(self) -> None:
        if isinstance(self.input, dict):
            self.input = RatInputConfig(**self.input)
        if isinstance(self.preprocessing, dict):
            self.preprocessing = RatPreprocessingConfig(**self.preprocessing)
        if isinstance(self.deepkoopman, dict):
            self.deepkoopman = DeepKoopmanConfig(**self.deepkoopman)
        if isinstance(self.cache, dict):
            self.cache = RatCacheConfig(**self.cache)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RatAnalysisConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(**raw)

    def to_dict(self) -> dict:
        return asdict(self)


def _load_records(config: RatAnalysisConfig, *, quick: bool):
    records = load_metadata(config.input.metadata)
    root_template = config.input.data_root or load_data_root_template(config.input.env)
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


def _window_records_for_split(records: list[WindowRecord], source_split: str, target_split: str) -> list[WindowRecord]:
    out = []
    for row in records:
        if row.split == source_split:
            out.append(
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
            )
    return out


def _prepare_windows(
    config: RatAnalysisConfig,
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
        metadata.extend(_window_records_for_split(metadata, "train", "val"))
    if not raw_by_split["test"]:
        raw_by_split["test"] = raw_by_split["val"][:1]
        metadata.extend(_window_records_for_split(metadata, "val", "test"))

    train_windows = np.concatenate(raw_by_split["train"], axis=0)
    mean, std = fit_zscore(train_windows)

    arrays = {}
    counts = {}
    for split, parts in raw_by_split.items():
        windows = np.concatenate(parts, axis=0)
        windows = apply_zscore(windows, mean, std)
        arrays[split] = flatten_windows(windows)
        counts[split] = int(windows.shape[0])
    stats = {
        "quick_mode": bool(quick),
        "zscore_mean": mean.tolist(),
        "zscore_std": std.tolist(),
        "window_counts": counts,
    }
    return arrays, metadata, stats


def _evaluate(module: DeepKoopmanLightningModule, data: np.ndarray, *, show_progress: bool = False, desc: str = "evaluate") -> dict[str, float]:
    loader = torch.utils.data.DataLoader(
        WindowedTrajectoryDataset(data, module.config),
        batch_size=module.config.trainer.batch_size,
        shuffle=False,
    )
    totals: dict[str, float] = {}
    total_examples = 0
    iterator = tqdm(loader, desc=desc, unit="batch", disable=not show_progress)
    module.eval()
    with torch.no_grad():
        for batch in iterator:
            batch = module._prepare_batch(batch.to(module.device))
            losses = compute_losses(module.model, batch, module.config)
            weight = int(batch.shape[1])
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * weight
            total_examples += weight
    return {name: value / total_examples for name, value in totals.items()}


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
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


def _read_window_metadata(path: Path) -> list[WindowRecord]:
    rows: list[WindowRecord] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                WindowRecord(
                    rat_id=row["rat_id"],
                    music_type=row["music_type"],
                    time_point=row["time_point"],
                    split=row["split"],
                    source_file=row["source_file"],
                    window_index=int(row["window_index"]),
                    start_sample=int(row["start_sample"]),
                    end_sample=int(row["end_sample"]),
                    start_sec=float(row["start_sec"]),
                    end_sec=float(row["end_sec"]),
                )
            )
    return rows


def _preprocess_cache_payload(config: RatAnalysisConfig, records, *, quick: bool) -> dict[str, object]:
    split_map = _split_map(records, quick)
    preprocessing = config.preprocessing
    model_config = config.deepkoopman
    payload_records = []
    for record in records:
        stat = Path(record.path).stat()
        payload_records.append(
            {
                "rat_id": record.rat_id,
                "music_type": record.music_type,
                "time_point": record.time_point,
                "date": record.date,
                "filename": record.filename,
                "yyyymmdd": record.yyyymmdd,
                "split": split_map[record],
                "path": str(Path(record.path).resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return {
        "version": 1,
        "quick": bool(quick),
        "raw_fs": preprocessing.raw_fs,
        "target_fs": preprocessing.target_fs,
        "line_freq": preprocessing.line_freq,
        "bandpass_low": preprocessing.bandpass_low,
        "bandpass_high": preprocessing.bandpass_high,
        "window_sec": preprocessing.window_sec,
        "stride_sec": preprocessing.stride_sec,
        "len_time": model_config.data.len_time,
        "max_windows_per_record": preprocessing.max_windows_per_record,
        "records": payload_records,
    }


def _preprocess_cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _preprocess_cache_dir(config: RatAnalysisConfig, cache_key: str) -> Path:
    if config.cache.preprocessed_dir:
        return Path(config.cache.preprocessed_dir)
    return Path(config.cache.preprocessed_cache_dir) / cache_key


def _save_preprocessed_windows(
    out_dir: Path,
    windows_by_split: dict[str, np.ndarray],
    metadata: list[WindowRecord],
    stats: dict[str, object],
    cache_payload: dict[str, object],
    cache_key: str,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split, windows in windows_by_split.items():
        path = out_dir / f"{split}_windows.npy"
        np.save(path, windows)
        paths[f"{split}_windows"] = str(path)
    metadata_path = out_dir / "window_metadata.csv"
    _write_window_metadata(metadata_path, metadata)
    paths["window_metadata"] = str(metadata_path)
    stats_path = out_dir / "preprocessing_stats.json"
    stats_with_key = {**stats, "cache_key": cache_key}
    stats_path.write_text(json.dumps(stats_with_key, indent=2), encoding="utf-8")
    paths["preprocessing_stats"] = str(stats_path)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"cache_key": cache_key, "payload": cache_payload, "paths": paths}, indent=2),
        encoding="utf-8",
    )
    paths["manifest"] = str(manifest_path)
    return paths


def _load_preprocessed_windows(cache_dir: Path) -> tuple[dict[str, np.ndarray], list[WindowRecord], dict[str, object], dict[str, str]]:
    required = {
        "train_windows": cache_dir / "train_windows.npy",
        "val_windows": cache_dir / "val_windows.npy",
        "test_windows": cache_dir / "test_windows.npy",
        "window_metadata": cache_dir / "window_metadata.csv",
        "preprocessing_stats": cache_dir / "preprocessing_stats.json",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Incomplete preprocessed cache under {cache_dir}: missing {missing}")
    windows_by_split = {
        split: np.load(required[f"{split}_windows"], mmap_mode="r")
        for split in ["train", "val", "test"]
    }
    arrays = {
        split: flatten_windows(windows)
        for split, windows in windows_by_split.items()
    }
    metadata = _read_window_metadata(required["window_metadata"])
    stats = json.loads(required["preprocessing_stats"].read_text(encoding="utf-8"))
    paths = {name: str(path) for name, path in required.items()}
    manifest = cache_dir / "manifest.json"
    if manifest.exists():
        paths["manifest"] = str(manifest)
    return arrays, metadata, stats, paths


def _sample_latents(
    module: DeepKoopmanLightningModule,
    arrays: dict[str, np.ndarray],
    metadata: list[WindowRecord],
    out_dir: Path,
    sample_windows: int,
) -> list[dict]:
    rows = []
    meta_by_split: dict[str, list[WindowRecord]] = {"train": [], "val": [], "test": []}
    for row in metadata:
        meta_by_split[row.split].append(row)

    for split, data in arrays.items():
        n_windows = data.shape[0] // module.config.data.len_time
        take = min(sample_windows, n_windows, len(meta_by_split[split]))
        if take <= 0:
            continue
        x0 = np.array(data[: take * module.config.data.len_time : module.config.data.len_time], copy=True)
        dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
        x = torch.as_tensor(x0, dtype=dtype, device=module.device)
        module.model.eval()
        with torch.no_grad():
            latent = module.model.encode(x)
            omegas = module.model._omega_net_apply(latent)
        latent_np = latent.detach().cpu().numpy()
        omega_np = np.concatenate([om.detach().cpu().numpy() for om in omegas], axis=1)

        for idx in range(take):
            info = meta_by_split[split][idx]
            rows.append(
                {
                    "split": split,
                    "rat_id": info.rat_id,
                    "music_type": info.music_type,
                    "time_point": info.time_point,
                    "source_file": info.source_file,
                    "window_index": info.window_index,
                    "z0": latent_np[idx, 0],
                    "z1": latent_np[idx, 1],
                    "z2": latent_np[idx, 2],
                    "frequency_hz": omega_np[idx, 0] / (2 * np.pi),
                    "growth": omega_np[idx, 1],
                    "real_rate": omega_np[idx, 2],
                }
            )
    _write_rows(out_dir / "latent_samples.csv", rows)
    return rows


def _summarize_latents(rows: list[dict], out_dir: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["split"], row["music_type"], row["time_point"]), []).append(row)

    summaries = []
    for (split, music_type, time_point), values in sorted(grouped.items()):
        summaries.append(
            {
                "split": split,
                "music_type": music_type,
                "time_point": time_point,
                "n": len(values),
                "frequency_hz_mean": float(np.mean([v["frequency_hz"] for v in values])),
                "growth_mean": float(np.mean([v["growth"] for v in values])),
                "real_rate_mean": float(np.mean([v["real_rate"] for v in values])),
                "latent_radius_mean": float(np.mean([(v["z0"] ** 2 + v["z1"] ** 2) ** 0.5 for v in values])),
                "z2_mean": float(np.mean([v["z2"] for v in values])),
            }
        )
    _write_rows(out_dir / "latent_summary_by_condition.csv", summaries)

    by_music = {}
    for row in summaries:
        by_music.setdefault((row["split"], row["music_type"]), {})[row["time_point"]] = row
    comparisons = []
    for (split, music_type), points in sorted(by_music.items()):
        before = points.get("before")
        if not before:
            continue
        for time_point in ["during_a", "during_b", "after"]:
            current = points.get(time_point)
            if current is None:
                continue
            comparisons.append(
                {
                    "split": split,
                    "music_type": music_type,
                    "comparison": f"{time_point}/before",
                    "frequency_hz_delta": current["frequency_hz_mean"] - before["frequency_hz_mean"],
                    "growth_delta": current["growth_mean"] - before["growth_mean"],
                    "latent_radius_delta": current["latent_radius_mean"] - before["latent_radius_mean"],
                    "z2_delta": current["z2_mean"] - before["z2_mean"],
                }
            )
    _write_rows(out_dir / "condition_comparisons.csv", comparisons)


def _plot_latents(rows: list[dict], out_path: Path) -> None:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"gamma": "tab:red", "control": "tab:blue", "conventional": "tab:green"}
    for music_type, color in colors.items():
        values = [r for r in rows if r["music_type"] == music_type]
        if not values:
            continue
        ax.scatter(
            [r["z0"] for r in values],
            [r["z1"] for r in values],
            [r["z2"] for r in values],
            s=8,
            alpha=0.65,
            label=music_type,
            color=color,
        )
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_zlabel("z2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _apply_runtime_overrides(
    config: RatAnalysisConfig,
    *,
    output_dir: str | None = None,
    device: str | None = None,
    preprocessed_dir: str | None = None,
    preprocessed_cache_dir: str | None = None,
) -> RatAnalysisConfig:
    if output_dir is not None:
        config.output_dir = output_dir
    if device is not None:
        config.deepkoopman.runtime.device = device
    if preprocessed_dir is not None:
        config.cache.preprocessed_dir = preprocessed_dir
    if preprocessed_cache_dir is not None:
        config.cache.preprocessed_cache_dir = preprocessed_cache_dir
    return config


def _apply_quick_overrides(config: RatAnalysisConfig) -> RatAnalysisConfig:
    config.deepkoopman.trainer.max_epochs = min(config.deepkoopman.trainer.max_epochs, 1)
    config.deepkoopman.data.shifts = config.deepkoopman.data.shifts[:2]
    config.deepkoopman.data.middle_shifts = config.deepkoopman.data.middle_shifts[:2]
    if config.preprocessing.max_windows_per_record is None:
        config.preprocessing.max_windows_per_record = 4
    else:
        config.preprocessing.max_windows_per_record = min(config.preprocessing.max_windows_per_record, 4)
    config.latent_samples = min(config.latent_samples, 16)
    return config


def run(config: RatAnalysisConfig, *, quick: bool = False, quick_records: int | None = None, rebuild_preprocessed: bool = False, no_save_preprocessed: bool = False, no_progress: bool = False) -> dict:
    if quick:
        config = _apply_quick_overrides(config)
    run_dir = Path(config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    table_dir = run_dir / "tables"
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(config, quick=quick)
    if quick and quick_records is not None:
        records = records[:quick_records]

    cache_payload = _preprocess_cache_payload(config, records, quick=quick)
    cache_key = _preprocess_cache_key(cache_payload)
    preprocessed_dir = _preprocess_cache_dir(config, cache_key)
    preprocessed_paths = {}

    save_preprocessed = config.cache.save_preprocessed and not no_save_preprocessed
    if save_preprocessed and preprocessed_dir.exists() and not rebuild_preprocessed:
        try:
            arrays, metadata, stats, preprocessed_paths = _load_preprocessed_windows(preprocessed_dir)
            stats["cache_hit"] = True
            stats["cache_key"] = cache_key
            stats["cache_dir"] = str(preprocessed_dir)
            stats["preprocessed_paths"] = preprocessed_paths
        except FileNotFoundError:
            arrays, metadata, stats = _prepare_windows(config, records, quick=quick, show_progress=not no_progress)
            stats["cache_hit"] = False
            stats["cache_key"] = cache_key
            stats["cache_dir"] = str(preprocessed_dir)
    else:
        arrays, metadata, stats = _prepare_windows(config, records, quick=quick, show_progress=not no_progress)
        stats["cache_hit"] = False
        stats["cache_key"] = cache_key
        stats["cache_dir"] = str(preprocessed_dir)

    if save_preprocessed and not preprocessed_paths:
        windows_by_split = {
            split: data.reshape(-1, config.deepkoopman.data.len_time, 64)
            for split, data in arrays.items()
        }
        preprocessed_paths = _save_preprocessed_windows(
            preprocessed_dir,
            windows_by_split,
            metadata,
            stats,
            cache_payload,
            cache_key,
        )
        stats["preprocessed_paths"] = preprocessed_paths

    cfg = config.deepkoopman
    cfg.trainer.enable_progress_bar = not no_progress
    cfg.logging.save_dir = str(run_dir / "logs")
    module = DeepKoopmanLightningModule(cfg)
    datamodule = DeepKoopmanDataModule(arrays["train"], arrays["val"], cfg, test_data=arrays["test"])
    trainer = build_trainer(cfg, default_root_dir=run_dir, checkpoint_dir=run_dir / "checkpoints", run_name="rat_analysis")
    trainer.fit(module, datamodule=datamodule)
    checkpoint = Path(trainer.checkpoint_callback.best_model_path)
    module = DeepKoopmanLightningModule.load_checkpoint(checkpoint)

    metrics = {
        split: _evaluate(module, data, show_progress=not no_progress, desc=f"eval {split}")
        for split, data in arrays.items()
    }
    (table_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (table_dir / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    history_file = next(run_dir.glob("logs/**/metrics.csv"), None)
    history = load_history(history_file) if history_file else []
    save_history_csv(history, table_dir / "history.csv")
    if history:
        plot_losses(history, fig_dir / "losses.png")

    latent_rows = _sample_latents(module, arrays, metadata, table_dir, sample_windows=config.latent_samples)
    _summarize_latents(latent_rows, table_dir)
    _plot_latents(latent_rows, fig_dir / "latent_3d.png")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "num_records": len(records),
        "quick_mode": bool(quick),
        "quick_mode_note": "quick mode is for pipeline smoke testing only, not scientific interpretation" if quick else "",
        "preprocessed_cache_hit": bool(stats.get("cache_hit", False)),
        "preprocessed_cache_key": cache_key,
        "preprocessed_cache_dir": str(preprocessed_dir),
        "dtype": cfg.runtime.dtype,
        "batch_size": cfg.trainer.batch_size,
        "window_counts": stats.get("window_counts", {}),
        "config": config.to_dict(),
        "metrics": metrics,
        "artifacts": {
            "history": str(table_dir / "history.csv"),
            "metrics": str(table_dir / "metrics.json"),
            "preprocessing_stats": str(table_dir / "preprocessing_stats.json"),
            "preprocessed": preprocessed_paths,
            "latent_samples": str(table_dir / "latent_samples.csv"),
            "latent_summary": str(table_dir / "latent_summary_by_condition.csv"),
            "condition_comparisons": str(table_dir / "condition_comparisons.csv"),
            "latent_3d": str(fig_dir / "latent_3d.png"),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rat_analysis.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=2)
    parser.add_argument("--preprocessed-dir", default=None)
    parser.add_argument("--preprocessed-cache-dir", default=None)
    parser.add_argument("--rebuild-preprocessed", action="store_true")
    parser.add_argument("--no-save-preprocessed", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = RatAnalysisConfig.from_yaml(args.config)
    if args.data_root is not None:
        config.input.data_root = args.data_root
    config = _apply_runtime_overrides(
        config,
        output_dir=args.output_dir,
        device=args.device,
        preprocessed_dir=args.preprocessed_dir,
        preprocessed_cache_dir=args.preprocessed_cache_dir,
    )

    print(
        json.dumps(
            run(
                config,
                quick=args.quick,
                quick_records=args.quick_records,
                rebuild_preprocessed=args.rebuild_preprocessed,
                no_save_preprocessed=args.no_save_preprocessed,
                no_progress=args.no_progress,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
