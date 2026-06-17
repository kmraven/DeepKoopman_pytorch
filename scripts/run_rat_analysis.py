from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
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

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.model import DeepKoopmanModule
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
from deepkoopman.trainer import DeepKoopmanTrainer
from deepkoopman.visualization import plot_losses, save_history_csv


def _rat_config(args: argparse.Namespace) -> DeepKoopmanConfig:
    return DeepKoopmanConfig(
        data_name="RatAuditoryCortex",
        len_time=args.len_time,
        delta_t=1.0 / args.target_fs,
        widths=[64, 256, 128, 3, 3, 128, 256, 64],
        hidden_widths_omega=[64, 64],
        num_real=1,
        num_complex_pairs=1,
        shifts=list(range(1, args.num_shifts + 1)),
        shifts_middle=list(range(1, args.num_shifts_middle + 1)),
        recon_lam=args.recon_lam,
        mid_shift_lam=args.mid_shift_lam,
        Linf_lam=args.linf_lam,
        l2_lam=args.l2_lam,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )


def _load_records(args: argparse.Namespace):
    records = load_metadata(args.metadata)
    root_template = args.data_root or load_data_root_template(args.env)
    if args.quick:
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


def _prepare_windows(args: argparse.Namespace, records) -> tuple[dict[str, np.ndarray], list[WindowRecord], dict[str, object]]:
    split_map = _split_map(records, args.quick)
    raw_by_split: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
    metadata: list[WindowRecord] = []

    iterator = tqdm(records, desc="preprocess", unit="file", disable=args.no_progress)
    for record in iterator:
        waves = load_analog_waves(record.path)
        ephys = extract_ephys_channels(waves)
        processed = preprocess_ephys(
            ephys,
            raw_fs=args.raw_fs,
            target_fs=args.target_fs,
            line_freq=args.line_freq,
            bandpass=(args.bandpass_low, args.bandpass_high),
        )
        windows, spans = make_windows(
            processed,
            fs=args.target_fs,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            max_windows=args.max_windows_per_record,
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
                    start_sec=start / args.target_fs,
                    end_sec=end / args.target_fs,
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
        "quick_mode": bool(args.quick),
        "zscore_mean": mean.tolist(),
        "zscore_std": std.tolist(),
        "window_counts": counts,
    }
    return arrays, metadata, stats


def _evaluate(trainer: DeepKoopmanTrainer, data: np.ndarray) -> dict[str, float]:
    return trainer.evaluate_batched(data)


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


def _preprocess_cache_payload(args: argparse.Namespace, records) -> dict[str, object]:
    split_map = _split_map(records, args.quick)
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
        "quick": bool(args.quick),
        "raw_fs": args.raw_fs,
        "target_fs": args.target_fs,
        "line_freq": args.line_freq,
        "bandpass_low": args.bandpass_low,
        "bandpass_high": args.bandpass_high,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "len_time": args.len_time,
        "max_windows_per_record": args.max_windows_per_record,
        "records": payload_records,
    }


def _preprocess_cache_key(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _preprocess_cache_dir(args: argparse.Namespace, cache_key: str) -> Path:
    if args.preprocessed_dir:
        return Path(args.preprocessed_dir)
    return Path(args.preprocessed_cache_dir) / cache_key


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
    trainer: DeepKoopmanTrainer,
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
        n_windows = data.shape[0] // trainer.config.len_time
        take = min(sample_windows, n_windows, len(meta_by_split[split]))
        if take <= 0:
            continue
        x0 = np.array(data[: take * trainer.config.len_time : trainer.config.len_time], copy=True)
        x = torch.as_tensor(x0, dtype=trainer.torch_dtype, device=trainer.device)
        trainer.model.eval()
        with torch.no_grad():
            latent = trainer.model.encode(x)
            omegas = trainer.model._omega_net_apply(latent)
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


def run(args: argparse.Namespace) -> dict:
    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    table_dir = run_dir / "tables"
    fig_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    records = _load_records(args)
    if args.quick:
        records = records[: args.quick_records]

    cache_payload = _preprocess_cache_payload(args, records)
    cache_key = _preprocess_cache_key(cache_payload)
    preprocessed_dir = _preprocess_cache_dir(args, cache_key)
    preprocessed_paths = {}

    if not args.no_save_preprocessed and preprocessed_dir.exists() and not args.rebuild_preprocessed:
        try:
            arrays, metadata, stats, preprocessed_paths = _load_preprocessed_windows(preprocessed_dir)
            stats["cache_hit"] = True
            stats["cache_key"] = cache_key
            stats["cache_dir"] = str(preprocessed_dir)
            stats["preprocessed_paths"] = preprocessed_paths
        except FileNotFoundError:
            arrays, metadata, stats = _prepare_windows(args, records)
            stats["cache_hit"] = False
            stats["cache_key"] = cache_key
            stats["cache_dir"] = str(preprocessed_dir)
    else:
        arrays, metadata, stats = _prepare_windows(args, records)
        stats["cache_hit"] = False
        stats["cache_key"] = cache_key
        stats["cache_dir"] = str(preprocessed_dir)

    if not args.no_save_preprocessed and not preprocessed_paths:
        windows_by_split = {
            split: data.reshape(-1, args.len_time, 64)
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

    cfg = _rat_config(args)
    model = DeepKoopmanModule(cfg)
    trainer = DeepKoopmanTrainer(model, cfg)
    history = trainer.fit(arrays["train"], arrays["val"])
    checkpoint = run_dir / "rat_deepkoopman.pt"
    trainer.save(checkpoint)

    metrics = {split: _evaluate(trainer, data) for split, data in arrays.items()}
    (table_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (table_dir / "preprocessing_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    save_history_csv(history, table_dir / "history.csv")
    plot_losses(history, fig_dir / "losses.png")

    latent_rows = _sample_latents(trainer, arrays, metadata, table_dir, sample_windows=args.latent_samples)
    _summarize_latents(latent_rows, table_dir)
    _plot_latents(latent_rows, fig_dir / "latent_3d.png")

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "num_records": len(records),
        "quick_mode": bool(args.quick),
        "quick_mode_note": "quick mode is for pipeline smoke testing only, not scientific interpretation" if args.quick else "",
        "preprocessed_cache_hit": bool(stats.get("cache_hit", False)),
        "preprocessed_cache_key": cache_key,
        "preprocessed_cache_dir": str(preprocessed_dir),
        "dtype": cfg.dtype,
        "batch_size": cfg.batch_size,
        "window_counts": stats.get("window_counts", {}),
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
    parser.add_argument("--metadata", default="rat_data/rat_id.csv")
    parser.add_argument("--env", default="rat_data/env.py")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default="results/rat_analysis")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quick-records", type=int, default=2)
    parser.add_argument("--raw-fs", type=float, default=1000.0)
    parser.add_argument("--target-fs", type=float, default=250.0)
    parser.add_argument("--line-freq", type=float, default=50.0)
    parser.add_argument("--bandpass-low", type=float, default=1.0)
    parser.add_argument("--bandpass-high", type=float, default=100.0)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--stride-sec", type=float, default=0.5)
    parser.add_argument("--len-time", type=int, default=251)
    parser.add_argument("--max-windows-per-record", type=int, default=None)
    parser.add_argument("--num-shifts", type=int, default=10)
    parser.add_argument("--num-shifts-middle", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--recon-lam", type=float, default=0.1)
    parser.add_argument("--mid-shift-lam", type=float, default=1.0)
    parser.add_argument("--linf-lam", type=float, default=1e-8)
    parser.add_argument("--l2-lam", type=float, default=1e-12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--latent-samples", type=int, default=200)
    parser.add_argument("--preprocessed-dir", default=None)
    parser.add_argument("--preprocessed-cache-dir", default="results/rat_preprocessed_cache")
    parser.add_argument("--rebuild-preprocessed", action="store_true")
    parser.add_argument("--no-save-preprocessed", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.num_shifts = min(args.num_shifts, 2)
        args.num_shifts_middle = min(args.num_shifts_middle, 2)
        if args.max_windows_per_record is None:
            args.max_windows_per_record = 4
        args.latent_samples = min(args.latent_samples, 16)

    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
