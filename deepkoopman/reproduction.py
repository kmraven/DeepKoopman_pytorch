from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from .config import DeepKoopmanConfig


PAPER_DATASETS = [
    "DiscreteSpectrumExample",
    "Pendulum",
    "FluidFlowOnAttractor",
    "FluidFlowBox",
]


def _steps_per_file_pass(num_initial_conditions: int, len_time: int, max_shifts: int, batch_size: int, steps_per_batch: int) -> int:
    num_examples = num_initial_conditions * (len_time - max_shifts)
    steps_to_see_all = num_examples / batch_size
    return (int(steps_to_see_all) + 1) * steps_per_batch


def paper_best_params() -> dict[str, dict]:
    discrete_steps = _steps_per_file_pass(5000, 51, 50, 256, 2)
    pendulum_steps = _steps_per_file_pass(5000, 51, 50, 128, 2)
    attractor_steps = _steps_per_file_pass(5000, 121, 120, 256, 2)
    box_steps = _steps_per_file_pass(5000, 101, 100, 128, 2)
    return {
        "DiscreteSpectrumExample": {
            "data_name": "DiscreteSpectrumExample",
            "data_train_len": 1,
            "len_time": 51,
            "delta_t": 0.02,
            "num_real": 2,
            "num_complex_pairs": 0,
            "widths": [2, 30, 30, 2, 2, 30, 30, 2],
            "hidden_widths_omega": [10, 10, 10],
            "num_shifts": 30,
            "num_shifts_middle": 50,
            "recon_lam": 0.1,
            "Linf_lam": 1e-7,
            "L1_lam": 0.0,
            "L2_lam": 1e-15,
            "auto_first": 0,
            "num_passes_per_file": 15 * 6 * 10,
            "num_steps_per_batch": 2,
            "num_steps_per_file_pass": discrete_steps,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "max_time": 4 * 60 * 60,
            "min_5min": 0.5,
            "min_20min": 0.0004,
            "min_40min": 0.00008,
            "min_1hr": 0.00003,
            "min_2hr": 0.00001,
            "min_3hr": 0.000006,
            "min_halfway": 0.000006,
        },
        "Pendulum": {
            "data_name": "Pendulum",
            "data_train_len": 3,
            "len_time": 51,
            "delta_t": 0.02,
            "num_real": 0,
            "num_complex_pairs": 1,
            "widths": [2, 80, 80, 2, 2, 80, 80, 2],
            "hidden_widths_omega": [170],
            "dist_weights": "dl",
            "dist_weights_omega": "dl",
            "num_shifts": 30,
            "num_shifts_middle": 50,
            "recon_lam": 0.001,
            "Linf_lam": 1e-9,
            "L1_lam": 0.0,
            "L2_lam": 1e-14,
            "auto_first": 1,
            "num_passes_per_file": 15 * 6 * 50,
            "num_steps_per_batch": 2,
            "num_steps_per_file_pass": pendulum_steps,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "max_time": 6 * 60 * 60,
            "min_5min": 0.25,
            "min_20min": 0.02,
            "min_40min": 0.002,
            "min_1hr": 0.0002,
            "min_2hr": 0.00002,
            "min_3hr": 0.000004,
            "min_4hr": 0.0000005,
            "min_halfway": 1,
        },
        "FluidFlowOnAttractor": {
            "data_name": "FluidFlowOnAttractor",
            "data_train_len": 3,
            "len_time": 121,
            "delta_t": 0.05,
            "num_real": 0,
            "num_complex_pairs": 1,
            "widths": [3, 105, 2, 2, 105, 3],
            "hidden_widths_omega": [300],
            "num_shifts": 30,
            "num_shifts_middle": 120,
            "recon_lam": 0.1,
            "Linf_lam": 1e-7,
            "L1_lam": 0.0,
            "L2_lam": 1e-13,
            "auto_first": 1,
            "num_passes_per_file": 15 * 6 * 10,
            "num_steps_per_batch": 2,
            "num_steps_per_file_pass": attractor_steps,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "max_time": 6 * 60 * 60,
            "min_5min": 0.45,
            "min_20min": 0.001,
            "min_40min": 0.0005,
            "min_1hr": 0.00025,
            "min_2hr": 0.00005,
            "min_3hr": 0.000005,
            "min_4hr": 0.0000007,
            "min_halfway": 1,
        },
        "FluidFlowBox": {
            "data_name": "FluidFlowBox",
            "data_train_len": 4,
            "len_time": 101,
            "delta_t": 0.01,
            "num_real": 1,
            "num_complex_pairs": 1,
            "widths": [3, 130, 3, 3, 130, 3],
            "hidden_widths_omega": [20, 20],
            "dist_weights": "dl",
            "dist_weights_omega": "dl",
            "num_shifts": 30,
            "num_shifts_middle": 100,
            "recon_lam": 0.1,
            "Linf_lam": 1e-9,
            "L1_lam": 0.0,
            "L2_lam": 1e-13,
            "auto_first": 1,
            "num_passes_per_file": 15 * 6 * 10,
            "num_steps_per_batch": 2,
            "num_steps_per_file_pass": box_steps,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "max_time": 6 * 60 * 60,
            "min_5min": 0.45,
            "min_20min": 0.005,
            "min_40min": 0.0005,
            "min_1hr": 0.00025,
            "min_2hr": 0.00005,
            "min_3hr": 0.000007,
            "min_4hr": 0.000005,
            "min_halfway": 1,
        },
    }


def paper_config(dataset: str, *, quick: bool = False, device: str = "auto") -> DeepKoopmanConfig:
    params = deepcopy(paper_best_params()[dataset])
    if quick:
        params["num_passes_per_file"] = 1
        params["num_steps_per_batch"] = 1
        params["num_steps_per_file_pass"] = 0
        params["batch_size"] = min(params["batch_size"], 16)
        params["max_time"] = 60
        params["shifts"] = [1, 2]
        params["shifts_middle"] = [1, 2]
    cfg = DeepKoopmanConfig.from_legacy_params(params)
    cfg.device = device
    cfg.eval_interval = 1 if quick else 20
    return cfg


def train_paths_for_dataset(data_dir: str | Path, dataset: str, count: int) -> list[Path]:
    data_dir = Path(data_dir)
    return [data_dir / f"{dataset}_train{i}_x.csv" for i in range(1, count + 1)]
