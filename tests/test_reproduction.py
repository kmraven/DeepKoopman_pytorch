from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from deepkoopman.cli.reproduce import run_dataset
from deepkoopman.reproduction import paper_config


def test_paper_best_params_translation():
    discrete = paper_config("DiscreteSpectrumExample")
    assert discrete.data.len_time == 51
    assert discrete.data.delta_t == 0.02
    assert discrete.data.train_files == 1
    assert discrete.model.num_real == 2
    assert discrete.model.num_complex_pairs == 0
    assert discrete.model.widths == [2, 30, 30, 2, 2, 30, 30, 2]
    assert discrete.model.omega_hidden_widths == [10, 10, 10]
    assert discrete.loss.l2_weight == 1e-15

    pendulum = paper_config("Pendulum")
    assert pendulum.data.train_files == 3
    assert pendulum.model.num_complex_pairs == 1
    assert pendulum.model.initialization.distribution == "dl"
    assert pendulum.model.initialization.omega_distribution == "dl"
    assert pendulum.trainer.batch_size == 128
    assert pendulum.trainer.autoencoder_warmup_epochs == 1

    attractor = paper_config("FluidFlowOnAttractor")
    assert attractor.data.len_time == 121
    assert attractor.data.delta_t == 0.05
    assert attractor.num_evals == 2
    assert attractor.model.widths == [3, 105, 2, 2, 105, 3]

    box = paper_config("FluidFlowBox")
    assert box.data.len_time == 101
    assert box.data.delta_t == 0.01
    assert box.model.num_real == 1
    assert box.model.num_complex_pairs == 1
    assert box.num_evals == 3
    assert box.model.widths == [3, 130, 3, 3, 130, 3]


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

    assert Path(summary["checkpoint"]).suffix == ".ckpt"
    assert Path(summary["checkpoint"]).exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "tables" / "metrics.json").exists()
    assert (run_dir / "tables" / "latent_coordinates.csv").exists()
    assert (run_dir / "figures" / "losses.png").exists()
    assert summary["quick"] is True
    assert "test" in summary["metrics"]
