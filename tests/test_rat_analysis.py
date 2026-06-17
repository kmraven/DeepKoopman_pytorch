from __future__ import annotations

from pathlib import Path

import numpy as np

from deepkoopman.rat import (
    apply_zscore,
    attach_paths,
    extract_ephys_channels,
    fit_zscore,
    load_analog_waves,
    load_metadata,
    make_windows,
    preprocess_ephys,
    yymmdd_to_yyyymmdd,
)
from deepkoopman.cli.rat_analysis import RatAnalysisConfig, run


def test_rat_metadata_date_and_group_counts():
    records = load_metadata("data/rat_id.csv")
    assert len(records) == 180
    assert yymmdd_to_yyyymmdd("251125") == "20251125"
    assert yymmdd_to_yyyymmdd("260105") == "20260105"
    assert {r.music_type for r in records} == {"gamma", "control", "conventional"}
    assert {r.time_point for r in records} == {"before", "during_a", "during_b", "after"}


def test_case_insensitive_path_resolution_for_local_examples():
    records = load_metadata("data/rat_id.csv")
    template = "rat_data_tmp/{yyyymmdd}/data_mat"
    selected = [
        r
        for r in records
        if r.rat_id == "rat_001" and r.music_type == "gamma" and r.time_point in {"before", "during_b"}
    ]
    resolved = attach_paths(selected, template)
    assert resolved[0].path is not None and resolved[0].path.name == "datafile251125_017_raw.mat"
    assert resolved[1].path is not None and resolved[1].path.name == "datafile251125_024_RAW.mat"


def test_rat_mat_loader_extracts_ephys_channels():
    arr = load_analog_waves("rat_data_tmp/20251125/data_mat/datafile251125_017_raw.mat")
    assert arr.shape == (600010, 66)
    ephys = extract_ephys_channels(arr)
    assert ephys.shape == (600010, 64)


def test_preprocess_zscore_and_windowing_shapes():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(2000, 64))
    processed = preprocess_ephys(data, raw_fs=1000.0, target_fs=250.0, line_freq=50.0, bandpass=(1.0, 100.0))
    assert processed.shape[1] == 64
    windows, spans = make_windows(processed, fs=250.0, window_sec=1.0, stride_sec=0.5, max_windows=2)
    assert windows.shape == (2, 251, 64)
    assert len(spans) == 2
    mean, std = fit_zscore(windows[:1])
    normalized = apply_zscore(windows, mean, std)
    assert normalized.shape == windows.shape
    np.testing.assert_allclose(normalized[:1].reshape(-1, 64).mean(axis=0), 0.0, atol=1e-8)


def test_rat_analysis_cli_quick(tmp_path: Path):
    cfg = RatAnalysisConfig.from_yaml("configs/rat_analysis/default.yaml")
    assert cfg.input.metadata.path == "data/rat_id.csv"
    assert cfg.input.source.data_root_template == "rat_data_tmp/{yyyymmdd}/data_mat"
    cfg.output_dir = str(tmp_path)
    cfg.cache.preprocessed_cache_dir = str(tmp_path / "cache")
    cfg.preprocessing.max_windows_per_record = 2
    cfg.deepkoopman.runtime.device = "cpu"
    cfg.deepkoopman.trainer.max_epochs = 1
    cfg.deepkoopman.trainer.batch_size = 256
    cfg.latent_samples = 4
    summary = run(cfg, quick=True, quick_records=2, no_progress=True)
    run_dir = Path(summary["run_dir"])
    assert Path(summary["checkpoint"]).suffix == ".ckpt"
    assert Path(summary["checkpoint"]).exists()
    assert (run_dir / "tables" / "metrics.json").exists()
    assert (run_dir / "tables" / "latent_samples.csv").exists()
    assert (run_dir / "tables" / "latent_summary_by_condition.csv").exists()
    assert (run_dir / "figures" / "latent_3d.png").exists()
    cache_dir = Path(summary["preprocessed_cache_dir"])
    assert summary["preprocessed_cache_hit"] is False
    assert (cache_dir / "train_windows.npy").exists()
    assert (cache_dir / "val_windows.npy").exists()
    assert (cache_dir / "test_windows.npy").exists()
    assert (cache_dir / "window_metadata.csv").exists()
    assert (cache_dir / "manifest.json").exists()
    train_windows = np.load(cache_dir / "train_windows.npy", mmap_mode="r")
    assert train_windows.shape == (2, 251, 64)
    assert summary["artifacts"]["preprocessed"]["train_windows"].endswith("train_windows.npy")
    assert summary["dtype"] == "float32"
    assert summary["batch_size"] == 256

    cfg.deepkoopman.trainer.batch_size = 16
    second_summary = run(cfg, quick=True, quick_records=2, no_progress=True)
    assert second_summary["preprocessed_cache_hit"] is True
    assert second_summary["preprocessed_cache_key"] == summary["preprocessed_cache_key"]
    assert second_summary["preprocessed_cache_dir"] == summary["preprocessed_cache_dir"]
    assert second_summary["batch_size"] == 16
