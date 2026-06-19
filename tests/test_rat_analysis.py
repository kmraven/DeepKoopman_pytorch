from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml
import h5py
from scipy import io as scipy_io

from deepkoopman.cli.rat_preprocess import RatPreprocessConfig, run_preprocess
from deepkoopman.cli.train import run_training
from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.io import H5SplitData, load_split_data
from deepkoopman.postprocess import run_postprocess
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


def _write_rat_mat(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    t = np.arange(2200, dtype=np.float64) / 1000.0
    waves = rng.normal(scale=0.05, size=(t.size, 66))
    for channel in range(64):
        waves[:, channel] += np.sin(2 * np.pi * (3 + channel % 5) * t)
    waves[:, 64] = 100.0
    waves[:, 65] = 200.0
    path.parent.mkdir(parents=True, exist_ok=True)
    scipy_io.savemat(path, {"analogWaves": waves})


def _write_small_rat_source(tmp_path: Path) -> Path:
    data_dir = tmp_path / "rat_data" / "20251125" / "data_mat"
    _write_rat_mat(data_dir / "datafile251125_001_raw.mat", seed=1)
    _write_rat_mat(data_dir / "datafile251125_002_raw.mat", seed=2)
    metadata = tmp_path / "rat_id.csv"
    metadata.write_text(
        "\n".join(
            [
                "rat_id,music_type,time_point,date,number,filename",
                "rat_001,gamma,before,251125,1,datafile251125_001_raw.mat",
                "rat_012,control,during_a,251125,2,datafile251125_002_raw.mat",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metadata


def test_rat_metadata_date_and_group_counts():
    records = load_metadata("data/rat_id.csv")
    assert len(records) == 180
    assert yymmdd_to_yyyymmdd("251125") == "20251125"
    assert yymmdd_to_yyyymmdd("260105") == "20260105"
    assert {r.music_type for r in records} == {"gamma", "control", "conventional"}
    assert {r.time_point for r in records} == {"before", "during_a", "during_b", "after"}


def test_case_insensitive_path_resolution_for_local_examples(tmp_path: Path):
    metadata = _write_small_rat_source(tmp_path)
    records = load_metadata(metadata)
    template = str(tmp_path / "rat_data" / "{yyyymmdd}" / "data_mat")
    resolved = attach_paths(records, template)
    assert resolved[0].path is not None and resolved[0].path.name == "datafile251125_001_raw.mat"
    assert resolved[1].path is not None and resolved[1].path.name == "datafile251125_002_raw.mat"


def test_rat_mat_loader_extracts_first_64_ephys_channels(tmp_path: Path):
    mat_path = tmp_path / "source.mat"
    waves = np.arange(5 * 66, dtype=np.float64).reshape(5, 66)
    scipy_io.savemat(mat_path, {"analogWaves": waves})
    arr = load_analog_waves(mat_path)
    ephys = extract_ephys_channels(arr)
    assert ephys.shape == (5, 64)
    np.testing.assert_array_equal(ephys, waves[:, :64])


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


def test_rat_preprocess_writes_hdf5_training_data(tmp_path: Path):
    metadata = _write_small_rat_source(tmp_path)
    cfg = RatPreprocessConfig.from_yaml("configs/rat_analysis/default.yaml")
    cfg.input.metadata.path = str(metadata)
    cfg.input.source.data_root_template = str(tmp_path / "rat_data" / "{yyyymmdd}" / "data_mat")
    cfg.deepkoopman.data.root = str(tmp_path / "data")
    cfg.preprocessing.max_windows_per_record = 2

    summary = run_preprocess(cfg, quick=True, quick_records=2, no_progress=True)
    data_dir = Path(summary["data_dir"])
    assert (data_dir / "RatAuditoryCortex.h5").exists()
    assert (data_dir / "RatAuditoryCortex_window_metadata.csv").exists()

    with h5py.File(data_dir / "RatAuditoryCortex.h5", "r") as f:
        assert f["/train/x"].shape == (2, 251, 64)
        assert f["/train/x"].dtype == np.dtype("float32")
    splits = load_split_data(data_dir, "RatAuditoryCortex", 1)
    assert isinstance(splits["train"], H5SplitData)
    assert splits["train"].shape == (2, 251, 64)
    assert summary["reused_existing"] is False

    second = run_preprocess(cfg, quick=True, quick_records=2, no_progress=True)
    assert second["reused_existing"] is True


def test_rat_train_config_loads():
    cfg = DeepKoopmanConfig.from_yaml("configs/train/rat.yaml")
    assert cfg.data.name == "RatAuditoryCortex"
    assert cfg.model.widths[0] == 64
    assert cfg.model.widths[-1] == 64


def test_rat_postprocess_writes_condition_and_rat_latent_figures(tmp_path: Path):
    rng = np.random.default_rng(3)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with h5py.File(data_dir / "RatAuditoryCortex.h5", "w") as f:
        for split, n_windows in {"train": 2, "val": 1, "test": 1}.items():
            group = f.create_group(split)
            group.create_dataset("x", data=rng.normal(size=(n_windows, 251, 64)).astype(np.float32))
    (data_dir / "RatAuditoryCortex_window_metadata.csv").write_text(
        "\n".join(
            [
                "split,rat_id,music_type,time_point,source_file,window_index,start_sample,end_sample,start_sec,end_sec",
                "train,rat_001,gamma,before,a.mat,0,0,251,0.0,1.0",
                "train,rat_001,gamma,during_a,a.mat,1,125,376,0.5,1.5",
                "val,rat_012,control,before,b.mat,0,0,251,0.0,1.0",
                "test,rat_014,conventional,after,c.mat,0,0,251,0.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = DeepKoopmanConfig.from_yaml("configs/train/rat.yaml")
    cfg.data.root = str(data_dir)
    cfg.data.shifts = [1]
    cfg.data.middle_shifts = [1]
    cfg.model.widths = [64, 8, 3, 3, 8, 64]
    cfg.model.omega_hidden_widths = [8]
    cfg.trainer.max_epochs = 1
    cfg.trainer.batch_size = 2
    cfg.trainer.enable_progress_bar = False
    cfg.runtime.device = "cpu"
    cfg_path = tmp_path / "rat_train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    run_dir = tmp_path / "run"
    run_training(
        Namespace(
            config=str(cfg_path),
            epochs=None,
            batch_size=None,
            output_dir=str(run_dir),
            wandb=False,
            wandb_project=None,
            wandb_entity=None,
            wandb_mode=None,
            run_name=None,
            no_progress=True,
        )
    )
    summary = run_postprocess(
        run_dir,
        data_dir=data_dir,
        samples_per_split=2,
        latent_grid_size=5,
        state_grid_size=5,
    )
    post_dir = Path(summary["output_dir"])
    assert (post_dir / "tables" / "rat_latent_samples.csv").exists()
    assert (post_dir / "figures" / "rat_latent_by_condition_all.png").exists()
    assert (post_dir / "figures" / "rat_latent_by_condition_test.png").exists()
    assert (post_dir / "figures" / "rat_latent_by_rat_all.png").exists()
    assert (post_dir / "figures" / "rat_latent_by_rat_test.png").exists()
