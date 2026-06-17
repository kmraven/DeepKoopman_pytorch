from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.lightning import build_logger
from deepkoopman.search import run_random_search
from deepkoopman.visualization import plot_losses, save_history_csv


def test_config_from_yaml():
    cfg = DeepKoopmanConfig.from_yaml("configs/train/discrete.yaml")
    assert cfg.data.name == "DiscreteSpectrumExample"
    assert cfg.num_evals == 2
    assert cfg.to_dict()["model"]["omega_hidden_widths"] == [10, 10]


def test_logging_defaults_to_csv_and_wandb_is_opt_in(tmp_path: Path):
    cfg = DeepKoopmanConfig.from_yaml("configs/train/discrete.yaml")
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
