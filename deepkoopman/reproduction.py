from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .config import DeepKoopmanConfig


PAPER_DATASETS = [
    "DiscreteSpectrumExample",
    "Pendulum",
    "FluidFlowOnAttractor",
    "FluidFlowBox",
]

PAPER_CONFIG_FILES = {
    "DiscreteSpectrumExample": "discrete_spectrum.yaml",
    "Pendulum": "pendulum.yaml",
    "FluidFlowOnAttractor": "fluid_attractor.yaml",
    "FluidFlowBox": "fluid_box.yaml",
}

PAPER_TRAIN_CONFIG_DIR = Path("configs/train")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_config_dir(config_dir: str | Path | None = None) -> Path:
    path = Path(config_dir) if config_dir is not None else PAPER_TRAIN_CONFIG_DIR
    if not path.is_absolute():
        path = _repo_root() / path
    return path


def paper_config_path(dataset: str, config_dir: str | Path | None = None) -> Path:
    if dataset not in PAPER_CONFIG_FILES:
        raise KeyError(f"Unknown paper dataset {dataset!r}. Expected one of {PAPER_DATASETS}.")
    return _resolve_config_dir(config_dir) / PAPER_CONFIG_FILES[dataset]


def load_paper_manifest(config_dir: str | Path | None = None) -> dict[str, Any]:
    manifest_path = _resolve_config_dir(config_dir) / "manifest.yaml"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paper_config(dataset: str, *, quick: bool = False, device: str = "auto", config_dir: str | Path | None = None) -> DeepKoopmanConfig:
    cfg = DeepKoopmanConfig.from_yaml(paper_config_path(dataset, config_dir))
    cfg.runtime.device = device
    if quick:
        cfg.trainer.max_epochs = 1
        cfg.trainer.max_steps = -1
        cfg.trainer.batch_size = min(cfg.trainer.batch_size, 16)
        cfg.data.shifts = [1, 2]
        cfg.data.middle_shifts = [1, 2]
        cfg.callbacks.early_stopping.enabled = False
        cfg.trainer.enable_progress_bar = False
    return cfg


def _config_to_legacy_params(cfg: DeepKoopmanConfig) -> dict[str, Any]:
    return {
        "data_name": cfg.data.name,
        "data_train_len": cfg.data.train_files,
        "len_time": cfg.data.len_time,
        "delta_t": cfg.data.delta_t,
        "num_real": cfg.model.num_real,
        "num_complex_pairs": cfg.model.num_complex_pairs,
        "widths": list(cfg.model.widths),
        "hidden_widths_omega": list(cfg.model.omega_hidden_widths),
        "dist_weights": cfg.model.initialization.distribution,
        "dist_weights_omega": cfg.model.initialization.omega_distribution,
        "scale": cfg.model.initialization.scale,
        "scale_omega": cfg.model.initialization.omega_scale,
        "num_shifts": len(cfg.data.shifts),
        "num_shifts_middle": len(cfg.data.middle_shifts),
        "shifts": list(cfg.data.shifts),
        "shifts_middle": list(cfg.data.middle_shifts),
        "recon_lam": cfg.loss.reconstruction_weight,
        "mid_shift_lam": cfg.loss.middle_shift_weight,
        "Linf_lam": cfg.loss.linf_weight,
        "L1_lam": cfg.loss.l1_weight,
        "L2_lam": cfg.loss.l2_weight,
        "learning_rate": cfg.optimizer.lr,
        "batch_size": cfg.trainer.batch_size,
        "max_epochs": cfg.trainer.max_epochs,
        "auto_first": 1 if cfg.trainer.autoencoder_warmup_epochs else 0,
        "dtype": cfg.runtime.dtype,
        "device": cfg.runtime.device,
    }


def paper_best_params(config_dir: str | Path | None = None) -> dict[str, dict[str, Any]]:
    params = {}
    manifest = load_paper_manifest(config_dir)
    for dataset in PAPER_DATASETS:
        cfg = paper_config(dataset, config_dir=config_dir)
        legacy = _config_to_legacy_params(cfg)
        dataset_meta = deepcopy(manifest["datasets"].get(dataset, {}))
        legacy["paper"] = dataset_meta
        params[dataset] = legacy
    return params


def train_paths_for_dataset(data_dir: str | Path, dataset: str, count: int) -> list[Path]:
    data_dir = Path(data_dir)
    return [data_dir / f"{dataset}_train{i}_x.csv" for i in range(1, count + 1)]
