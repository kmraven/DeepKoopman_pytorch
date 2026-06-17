import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from deepkoopman.config import DataConfig, DeepKoopmanConfig, ModelConfig, RuntimeConfig, TrainerConfig
from deepkoopman.data import DeepKoopmanDataModule, WindowedTrajectoryDataset, stack_data, stack_data_windows
from deepkoopman.lightning import DeepKoopmanLightningModule
from deepkoopman.losses import compute_losses
from deepkoopman.model import DeepKoopmanModule


def make_config(**overrides) -> DeepKoopmanConfig:
    data = overrides.pop("data", {})
    model = overrides.pop("model", {})
    trainer = overrides.pop("trainer", {})
    runtime = overrides.pop("runtime", {})
    return DeepKoopmanConfig(
        data=DataConfig(len_time=overrides.pop("len_time", 6), delta_t=overrides.pop("delta_t", 0.02), name="Test", **data),
        model=ModelConfig(
            widths=overrides.pop("widths", [2, 8, 8, 2, 2, 8, 8, 2]),
            omega_hidden_widths=overrides.pop("omega_hidden_widths", [4]),
            num_real=overrides.pop("num_real", 2),
            num_complex_pairs=overrides.pop("num_complex_pairs", 0),
            **model,
        ),
        trainer=TrainerConfig(**trainer),
        runtime=RuntimeConfig(**runtime),
        **overrides,
    )


def test_stack_data_shape_and_index():
    len_time = 5
    traj = 2
    n = 2
    arr = np.arange(traj * len_time * n, dtype=np.float64).reshape(traj * len_time, n)
    out = stack_data(arr, num_shifts=2, len_time=len_time)
    assert out.shape == (3, traj * (len_time - 2), n)
    np.testing.assert_array_equal(out[0, 0], arr[0])
    np.testing.assert_array_equal(out[1, 0], arr[1])


def test_stack_data_windows_matches_selected_full_stack():
    len_time = 5
    traj = 3
    n = 2
    arr = np.arange(traj * len_time * n, dtype=np.float64).reshape(traj * len_time, n)
    selected = np.array([2, 0])
    out = stack_data_windows(arr, num_shifts=2, len_time=len_time, window_indices=selected, dtype=np.float32)
    assert out.dtype == np.float32
    assert out.shape == (3, selected.size * (len_time - 2), n)
    np.testing.assert_array_equal(out[0, 0], arr[10])
    np.testing.assert_array_equal(out[1, 3], arr[1])


def test_model_dtype_defaults_to_float64_and_supports_float32():
    default_model = DeepKoopmanModule(make_config(len_time=5))
    assert next(default_model.parameters()).dtype == torch.float64
    float32_model = DeepKoopmanModule(make_config(len_time=5, runtime={"dtype": "float32"}))
    assert next(float32_model.parameters()).dtype == torch.float32


def test_varying_multiply_shape():
    cfg = make_config(len_time=5)
    model = DeepKoopmanModule(cfg)
    y = torch.randn(7, 2, dtype=torch.float64)
    om = [torch.randn(7, 1, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)]
    out = model.varying_multiply(y, om)
    assert out.shape == y.shape


def test_losses_are_finite():
    cfg = make_config(data={"shifts": [1, 2], "middle_shifts": [1, 2, 3]})
    model = DeepKoopmanModule(cfg)
    batch = torch.randn(4, 10, 2, dtype=torch.float64)
    losses = compute_losses(model, batch, cfg)
    assert torch.isfinite(losses["loss"])


def test_windowed_dataset_matches_full_stack():
    cfg = make_config(data={"shifts": [1, 2], "middle_shifts": [1, 2]}, trainer={"batch_size": 2}, runtime={"dtype": "float32"})
    data = np.linspace(0, 1, 4 * 6 * 2, dtype=np.float32).reshape(4 * 6, 2)
    dataset = WindowedTrajectoryDataset(data, cfg)
    assert len(dataset) == 4
    batch = torch.stack([dataset[0], dataset[1]], dim=0)
    prepared = DeepKoopmanLightningModule(cfg)._prepare_batch(batch)
    full = stack_data(data, num_shifts=2, len_time=6).astype(np.float32)
    np.testing.assert_array_equal(prepared.numpy(), full[:, :8, :])


def test_lightning_module_trains_and_loads_checkpoint(tmp_path):
    cfg = make_config(data={"shifts": [1], "middle_shifts": [1]}, trainer={"batch_size": 2, "max_epochs": 1}, runtime={"dtype": "float32", "device": "cpu"})
    cfg.trainer.enable_progress_bar = False
    cfg.logging.save_dir = str(tmp_path / "logs")
    data = np.linspace(0, 1, 4 * 6 * 2, dtype=np.float32).reshape(4 * 6, 2)
    module = DeepKoopmanLightningModule(cfg)
    datamodule = DeepKoopmanDataModule(data, data, cfg)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="val/loss", mode="min")
    trainer = L.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        logger=False,
        enable_progress_bar=False,
        callbacks=[checkpoint],
    )
    trainer.fit(module, datamodule=datamodule)
    assert checkpoint.best_model_path
    loaded = DeepKoopmanLightningModule.load_checkpoint(checkpoint.best_model_path)
    pred = loaded.predict_array(data[:1], steps=2)
    assert pred.shape == (3, 1, 2)
