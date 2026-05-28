from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.search import run_random_search
from deepkoopman.visualization import plot_losses, save_history_csv


def test_config_from_yaml():
    cfg = DeepKoopmanConfig.from_yaml("configs/discrete_train.yaml")
    assert cfg.data_name == "DiscreteSpectrumExample"
    assert cfg.num_evals == 2


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
            "data_name": "DiscreteSpectrumExample",
            "len_time": 51,
            "delta_t": 0.02,
            "num_real": 2,
            "num_complex_pairs": 0,
            "widths": [2, 20, 20, 2, 2, 20, 20, 2],
            "hidden_widths_omega": [8],
            "shifts": [1, 2],
            "shifts_middle": [1, 2],
            "batch_size": 128,
            "learning_rate": 1e-3,
            "recon_lam": 0.1,
            "Linf_lam": 1e-8,
            "l2_lam": 1e-12,
            "max_epochs": 1,
            "seed": 42,
        },
        "search_spaces": {
            "learning_rate": {"type": "choice", "values": [1e-3]},
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
    assert (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()

    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["num_trials"] == 1
