from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.model import DeepKoopmanModule
from deepkoopman.reproduction import paper_config
from deepkoopman.trainer import DeepKoopmanTrainer
from scripts.reproduce_paper import run_dataset


def test_paper_best_params_translation():
    discrete = paper_config("DiscreteSpectrumExample")
    assert discrete.len_time == 51
    assert discrete.delta_t == 0.02
    assert discrete.data_train_len == 1
    assert discrete.num_real == 2
    assert discrete.num_complex_pairs == 0
    assert discrete.widths == [2, 30, 30, 2, 2, 30, 30, 2]
    assert discrete.hidden_widths_omega == [10, 10, 10]
    assert discrete.l2_lam == 1e-15
    assert discrete.max_time == 4 * 60 * 60

    pendulum = paper_config("Pendulum")
    assert pendulum.data_train_len == 3
    assert pendulum.num_complex_pairs == 1
    assert pendulum.init_distribution == "dl"
    assert pendulum.omega_init_distribution == "dl"
    assert pendulum.batch_size == 128
    assert pendulum.autoencoder_warmup_epochs == 1

    attractor = paper_config("FluidFlowOnAttractor")
    assert attractor.len_time == 121
    assert attractor.delta_t == 0.05
    assert attractor.num_evals == 2
    assert attractor.widths == [3, 105, 2, 2, 105, 3]

    box = paper_config("FluidFlowBox")
    assert box.len_time == 101
    assert box.delta_t == 0.01
    assert box.num_real == 1
    assert box.num_complex_pairs == 1
    assert box.num_evals == 3
    assert box.widths == [3, 130, 3, 3, 130, 3]


def test_legacy_training_loop_with_synthetic_files(tmp_path: Path):
    cfg = DeepKoopmanConfig(
        data_name="Synthetic",
        widths=[2, 4, 2, 2, 4, 2],
        hidden_widths_omega=[3],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.1,
        len_time=5,
        shifts=[1],
        shifts_middle=[1],
        batch_size=2,
        data_train_len=2,
        num_passes_per_file=1,
        num_steps_per_batch=1,
        num_steps_per_file_pass=0,
        eval_interval=1,
        seed=1,
        device="cpu",
    )
    paths = []
    for idx in range(2):
        arr = np.linspace(0, 1, 20, dtype=np.float64).reshape(10, 2) + idx
        path = tmp_path / f"Synthetic_train{idx + 1}_x.csv"
        np.savetxt(path, arr, delimiter=",")
        paths.append(path)
    val = np.linspace(0, 1, 20, dtype=np.float64).reshape(10, 2)

    trainer = DeepKoopmanTrainer(DeepKoopmanModule(cfg), cfg)
    summary = trainer.fit_legacy_files(paths, val, checkpoint_path=tmp_path / "best.pt")

    assert summary["stop_condition"] == "completed requested file passes"
    assert summary["best_val_loss"] < float("inf")
    assert len(trainer.history) == 2
    assert {row["file_num"] for row in trainer.history} == {1.0, 2.0}
    assert (tmp_path / "best.pt").exists()


def test_reproduce_cli_quick_discrete(tmp_path: Path):
    args = Namespace(
        dataset="DiscreteSpectrumExample",
        output_dir=str(tmp_path),
        data_dir="data",
        device="cpu",
        quick=True,
        no_progress=True,
    )
    summary = run_dataset("DiscreteSpectrumExample", args)
    run_dir = Path(summary["run_dir"])

    assert (run_dir / "best_checkpoint.pt").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "tables" / "metrics.json").exists()
    assert (run_dir / "tables" / "latent_coordinates.csv").exists()
    assert (run_dir / "figures" / "losses.png").exists()
    assert summary["quick"] is True
    assert "test" in summary["metrics"]
