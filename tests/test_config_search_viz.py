from __future__ import annotations

import importlib
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.cli.train import run_training
from deepkoopman.lightning import build_logger
from deepkoopman.postprocess import run_postprocess
from deepkoopman.search import _load_training_data, _sample_space, run_random_search
from deepkoopman.visualization import plot_losses, save_history_csv


def test_config_from_yaml():
    cfg = DeepKoopmanConfig.from_yaml("configs/train/discrete_spectrum.yaml")
    assert cfg.data.name == "DiscreteSpectrumExample"
    assert cfg.num_evals == 2
    assert cfg.data.shifts == list(range(1, 31))
    assert cfg.data.middle_shifts == list(range(1, 51))
    assert cfg.to_dict()["model"]["omega_hidden_widths"] == [10, 10, 10]


def test_logging_defaults_to_csv_and_wandb_is_opt_in(tmp_path: Path):
    cfg = DeepKoopmanConfig.from_yaml("configs/train/discrete_spectrum.yaml")
    cfg.logging.save_dir = str(tmp_path)
    csv_logger = build_logger(cfg)
    assert csv_logger.__class__.__name__ == "CSVLogger"

    cfg.logging.backend = "wandb"
    cfg.logging.wandb.mode = "offline"
    wandb_logger = build_logger(cfg, run_name="offline-smoke")
    assert wandb_logger.__class__.__name__ == "WandbLogger"


def test_cli_modules_are_packaged_and_dead_shims_are_removed():
    assert importlib.import_module("deepkoopman.cli.train")
    assert importlib.import_module("deepkoopman.cli.search")
    assert importlib.import_module("deepkoopman.cli.postprocess")
    assert importlib.util.find_spec("deepkoopman.cli.reproduce") is None
    assert importlib.util.find_spec("deepkoopman.example") is None
    assert importlib.util.find_spec("deepkoopman.trainer") is None
    assert importlib.util.find_spec("scripts") is None


def test_visualization_outputs(tmp_path: Path):
    history = [
        {
            "epoch": 0.0,
            "train_loss": 1.0,
            "train_loss1": 0.5,
            "train_loss2": 0.2,
            "train_loss3": 0.1,
            "train_loss_linf": 0.01,
            "train_loss_l1": 0.0,
            "train_loss_l2": 0.0,
            "loss": 0.9,
            "loss1": 0.4,
            "loss2": 0.2,
            "loss3": 0.1,
            "loss_linf": 0.01,
            "loss_l1": 0.0,
            "loss_l2": 0.0,
        }
    ]
    csv_path = tmp_path / "history.csv"
    fig_path = tmp_path / "losses.png"
    save_history_csv(history, csv_path)
    plot_losses(history, fig_path)
    assert csv_path.exists()
    assert fig_path.exists()


def test_random_search_smoke(tmp_path: Path):
    cfg = {
        "fixed": {
            "data": {
                "name": "DiscreteSpectrumExample",
                "root": "data",
                "len_time": 51,
                "delta_t": 0.02,
                "shifts": [1, 2],
                "middle_shifts": [1, 2],
            },
            "model": {
                "widths": [2, 20, 20, 2, 2, 20, 20, 2],
                "omega_hidden_widths": [8],
                "num_real": 2,
                "num_complex_pairs": 0,
            },
            "loss": {"reconstruction_weight": 0.1, "linf_weight": 1e-8, "l2_weight": 1e-12},
            "optimizer": {"lr": 1e-3},
            "trainer": {"batch_size": 128, "max_epochs": 1},
            "runtime": {"seed": 42},
        },
        "search_spaces": {
            "optimizer.lr": {"type": "choice", "values": [1e-3]},
        },
        "num_trials": 1,
        "seed": 1,
        "metric": "loss",
        "output_dir": str(tmp_path / "search"),
        "data_dir": "data",
    }

    import yaml

    config_path = tmp_path / "search.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    summary = run_random_search(config_path)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "trials.csv").exists()
    assert (run_dir / "best_config.yaml").exists()
    assert (run_dir / "best_checkpoint.ckpt").exists()
    assert (run_dir / "summary.json").exists()

    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["num_trials"] == 1


def test_paper_search_template_spaces_match_expected_ranges():
    import random
    import yaml

    cfg = yaml.safe_load(Path("configs/search/discrete_spectrum.yaml").read_text(encoding="utf-8"))
    rng = random.Random(7)
    widths = _sample_space(cfg["search_spaces"]["model.widths"], rng)
    depth = (len(widths) - 4) // 2
    assert widths[0] == 2
    assert widths[depth + 1 : depth + 3] == [2, 2]
    assert widths[-1] == 2
    assert len(widths) in {6, 8, 10, 12}

    cfg = yaml.safe_load(Path("configs/search/fluid_box.yaml").read_text(encoding="utf-8"))
    rng = random.Random(9)
    omega_widths = _sample_space(cfg["search_spaces"]["model.omega_hidden_widths"], rng)
    assert len(omega_widths) in {1, 2}
    assert all(10 <= width <= 130 for width in omega_widths)


def test_search_training_loader_uses_multiple_train_files():
    one = _load_training_data(Path("data"), "DiscreteSpectrumExample", 1)
    two = _load_training_data(Path("data"), "DiscreteSpectrumExample", 2)
    assert two.shape[0] == one.shape[0] * 2


def _write_quick_config(tmp_path: Path, dataset: str = "DiscreteSpectrumExample", train_files: int = 1) -> Path:
    cfg = DeepKoopmanConfig.from_yaml("configs/train/discrete_spectrum.yaml")
    cfg.data.name = dataset
    cfg.data.train_files = train_files
    cfg.data.shifts = [1, 2]
    cfg.data.middle_shifts = [1, 2]
    cfg.trainer.max_epochs = 1
    cfg.trainer.batch_size = 16
    cfg.trainer.enable_progress_bar = False
    cfg.runtime.device = "cpu"
    path = tmp_path / f"{dataset}.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    return path


def test_train_cli_outputs_only_training_artifacts(tmp_path: Path):
    config_path = _write_quick_config(tmp_path, train_files=2)
    out_dir = tmp_path / "train_run"
    args = Namespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        output_dir=str(out_dir),
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_mode=None,
        run_name=None,
        no_progress=True,
    )
    summary = run_training(args)
    run_dir = Path(summary["run_dir"])
    assert (run_dir / "best_checkpoint.ckpt").exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "summary.json").exists()
    assert list(run_dir.glob("logs/**/metrics.csv"))
    assert not (run_dir / "tables").exists()
    assert not (run_dir / "figures").exists()
    assert not (run_dir / "paper").exists()
    payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert payload["best_val_loss"] is not None
    assert payload["global_step"] > 0


def test_postprocess_outputs_requested_artifacts_for_2d_dataset(tmp_path: Path):
    config_path = _write_quick_config(tmp_path)
    out_dir = tmp_path / "train_run"
    args = Namespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        output_dir=str(out_dir),
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_mode=None,
        run_name=None,
        no_progress=True,
    )
    run_training(args)
    summary = run_postprocess(out_dir, samples_per_split=2, seed=3)
    post_dir = Path(summary["output_dir"])
    assert (post_dir / "tables" / "test_metrics.json").exists()
    assert (post_dir / "tables" / "test_metrics.csv").exists()
    assert (post_dir / "tables" / "sampled_trajectories.csv").exists()
    assert (post_dir / "figures" / "train_data_trajectories.png").exists()
    assert (post_dir / "figures" / "val_latent_true_vs_pred.png").exists()
    assert list((post_dir / "figures").glob("eigen_component_*.png"))
    assert list((post_dir / "figures").glob("eigenfunction_*_heatmap.png"))
    metrics = json.loads((post_dir / "tables" / "test_metrics.json").read_text(encoding="utf-8"))
    assert "pre_regularization_loss" in metrics


def test_postprocess_skips_heatmaps_for_3d_dataset(tmp_path: Path):
    source = np.loadtxt("data/FluidFlowBox_train1_x.csv", delimiter=",", dtype=np.float64)
    val = np.loadtxt("data/FluidFlowBox_val_x.csv", delimiter=",", dtype=np.float64)
    test = np.loadtxt("data/FluidFlowBox_test_x.csv", delimiter=",", dtype=np.float64)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = 101 * 4
    np.savetxt(data_dir / "FluidFlowBox_train1_x.csv", source[:rows], delimiter=",")
    np.savetxt(data_dir / "FluidFlowBox_val_x.csv", val[:rows], delimiter=",")
    np.savetxt(data_dir / "FluidFlowBox_test_x.csv", test[:rows], delimiter=",")

    cfg = DeepKoopmanConfig.from_yaml("configs/train/fluid_box.yaml")
    cfg.data.root = str(data_dir)
    cfg.data.train_files = 1
    cfg.data.shifts = [1, 2]
    cfg.data.middle_shifts = [1, 2]
    cfg.trainer.max_epochs = 1
    cfg.trainer.batch_size = 4
    cfg.trainer.enable_progress_bar = False
    cfg.runtime.device = "cpu"
    config_path = tmp_path / "fluid_box.yaml"
    config_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    out_dir = tmp_path / "fluid_run"
    args = Namespace(
        config=str(config_path),
        epochs=None,
        batch_size=None,
        output_dir=str(out_dir),
        wandb=False,
        wandb_project=None,
        wandb_entity=None,
        wandb_mode=None,
        run_name=None,
        no_progress=True,
    )
    run_training(args)
    summary = run_postprocess(out_dir, data_dir=data_dir, samples_per_split=2, seed=4)
    post_dir = Path(summary["output_dir"])
    assert (post_dir / "figures" / "test_data_trajectories.png").exists()
    assert not list((post_dir / "figures").glob("eigenfunction_*_heatmap.png"))
