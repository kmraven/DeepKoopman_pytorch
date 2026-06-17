import numpy as np
import torch

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.data import stack_data, stack_data_windows
from deepkoopman.losses import compute_losses
from deepkoopman.model import DeepKoopmanModule
from deepkoopman.trainer import DeepKoopmanTrainer


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
    base = dict(
        widths=[2, 8, 8, 2, 2, 8, 8, 2],
        hidden_widths_omega=[4],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.02,
        len_time=5,
    )
    default_model = DeepKoopmanModule(DeepKoopmanConfig(**base))
    assert next(default_model.parameters()).dtype == torch.float64
    float32_model = DeepKoopmanModule(DeepKoopmanConfig(**base, dtype="float32"))
    assert next(float32_model.parameters()).dtype == torch.float32


def test_varying_multiply_shape():
    cfg = DeepKoopmanConfig(
        widths=[2, 8, 8, 2, 2, 8, 8, 2],
        hidden_widths_omega=[4],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.02,
        len_time=5,
    )
    model = DeepKoopmanModule(cfg)
    y = torch.randn(7, 2, dtype=torch.float64)
    om = [torch.randn(7, 1, dtype=torch.float64), torch.randn(7, 1, dtype=torch.float64)]
    out = model.varying_multiply(y, om)
    assert out.shape == y.shape


def test_losses_are_finite():
    cfg = DeepKoopmanConfig(
        widths=[2, 8, 8, 2, 2, 8, 8, 2],
        hidden_widths_omega=[4],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.02,
        len_time=6,
        shifts=[1, 2],
        shifts_middle=[1, 2, 3],
    )
    model = DeepKoopmanModule(cfg)
    batch = torch.randn(4, 10, 2, dtype=torch.float64)
    losses = compute_losses(model, batch, cfg)
    assert torch.isfinite(losses["loss"])


def test_batched_evaluation_matches_full_evaluation_without_linf():
    cfg = DeepKoopmanConfig(
        widths=[2, 8, 8, 2, 2, 8, 8, 2],
        hidden_widths_omega=[4],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.02,
        len_time=6,
        shifts=[1, 2],
        shifts_middle=[1, 2],
        batch_size=2,
        dtype="float32",
    )
    data = np.linspace(0, 1, 4 * 6 * 2, dtype=np.float32).reshape(4 * 6, 2)
    trainer = DeepKoopmanTrainer(DeepKoopmanModule(cfg), cfg)
    full = stack_data(data, num_shifts=2, len_time=6).astype(np.float32)
    with torch.no_grad():
        expected = compute_losses(trainer.model, torch.as_tensor(full, device=trainer.device), cfg)
    actual = trainer.evaluate_batched(data, batch_size=2)
    np.testing.assert_allclose(actual["loss"], float(expected["loss"].detach().cpu()), rtol=1e-6)
    np.testing.assert_allclose(actual["loss1"], float(expected["loss1"].detach().cpu()), rtol=1e-6)


def test_trainer_fit_uses_minibatch_path(monkeypatch):
    cfg = DeepKoopmanConfig(
        widths=[2, 8, 8, 2, 2, 8, 8, 2],
        hidden_widths_omega=[4],
        num_real=2,
        num_complex_pairs=0,
        delta_t=0.02,
        len_time=6,
        shifts=[1],
        shifts_middle=[1],
        batch_size=2,
        max_epochs=1,
        dtype="float32",
    )
    data = np.linspace(0, 1, 4 * 6 * 2, dtype=np.float32).reshape(4 * 6, 2)
    trainer = DeepKoopmanTrainer(DeepKoopmanModule(cfg), cfg)

    def fail_full_stack(*args, **kwargs):
        raise AssertionError("fit() should not call full stack_data")

    monkeypatch.setattr("deepkoopman.trainer.stack_data", fail_full_stack)
    history = trainer.fit(data, data)
    assert len(history) == 1
    assert np.isfinite(history[0]["loss"])
