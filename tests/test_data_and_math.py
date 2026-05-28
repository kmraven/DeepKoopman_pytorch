import numpy as np
import torch

from deepkoopman.config import DeepKoopmanConfig
from deepkoopman.data import stack_data
from deepkoopman.losses import compute_losses
from deepkoopman.model import DeepKoopmanModule


def test_stack_data_shape_and_index():
    len_time = 5
    traj = 2
    n = 2
    arr = np.arange(traj * len_time * n, dtype=np.float64).reshape(traj * len_time, n)
    out = stack_data(arr, num_shifts=2, len_time=len_time)
    assert out.shape == (3, traj * (len_time - 2), n)
    np.testing.assert_array_equal(out[0, 0], arr[0])
    np.testing.assert_array_equal(out[1, 0], arr[1])


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
