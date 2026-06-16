from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DeepKoopmanConfig:
    widths: list[int]
    hidden_widths_omega: list[int]
    num_real: int
    num_complex_pairs: int
    delta_t: float
    len_time: int
    data_name: str = "DiscreteSpectrumExample"
    shifts: list[int] = field(default_factory=lambda: list(range(1, 31)))
    shifts_middle: list[int] = field(default_factory=lambda: list(range(1, 51)))
    recon_lam: float = 1.0
    mid_shift_lam: float = 1.0
    Linf_lam: float = 0.0
    relative_loss: bool = False
    l1_lam: float = 0.0
    l2_lam: float = 0.0
    learning_rate: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 10
    autoencoder_warmup_epochs: int = 0
    data_train_len: int = 1
    num_passes_per_file: int = 1
    num_steps_per_batch: int = 1
    num_steps_per_file_pass: int | None = None
    max_time: float | None = None
    min_5min: float = 1e-2
    min_20min: float = 1e-3
    min_40min: float = 1e-4
    min_1hr: float = 1e-5
    min_2hr: float = 10 ** (-5.25)
    min_3hr: float = 10 ** (-5.5)
    min_4hr: float = 10 ** (-5.75)
    min_halfway: float = 1e-4
    eval_interval: int = 20
    init_distribution: str = "tn"
    omega_init_distribution: str = "tn"
    init_scale: float = 0.1
    omega_init_scale: float = 0.1
    act_type: str = "relu"
    seed: int = 42
    device: str = "auto"

    @property
    def num_evals(self) -> int:
        return 2 * self.num_complex_pairs + self.num_real

    @classmethod
    def from_legacy_params(cls, params: dict) -> "DeepKoopmanConfig":
        return cls(
            widths=params["widths"],
            hidden_widths_omega=params["hidden_widths_omega"],
            num_real=params["num_real"],
            num_complex_pairs=params["num_complex_pairs"],
            delta_t=params["delta_t"],
            len_time=params["len_time"],
            data_name=params.get("data_name", "DiscreteSpectrumExample"),
            shifts=list(params.get("shifts", range(1, params.get("num_shifts", 0) + 1))),
            shifts_middle=list(
                params.get("shifts_middle", range(1, params.get("num_shifts_middle", 0) + 1))
            ),
            recon_lam=params.get("recon_lam", 1.0),
            mid_shift_lam=params.get("mid_shift_lam", 1.0),
            Linf_lam=params.get("Linf_lam", 0.0),
            relative_loss=bool(params.get("relative_loss", 0)),
            l1_lam=params.get("L1_lam", 0.0),
            l2_lam=params.get("L2_lam", 0.0),
            learning_rate=params.get("learning_rate", 1e-3),
            batch_size=params.get("batch_size", 256),
            max_epochs=params.get("max_epochs", 10),
            autoencoder_warmup_epochs=1 if params.get("auto_first", 0) else 0,
            data_train_len=params.get("data_train_len", 1),
            num_passes_per_file=params.get("num_passes_per_file", 1),
            num_steps_per_batch=params.get("num_steps_per_batch", 1),
            num_steps_per_file_pass=params.get("num_steps_per_file_pass"),
            max_time=params.get("max_time"),
            min_5min=params.get("min_5min", 1e-2),
            min_20min=params.get("min_20min", 1e-3),
            min_40min=params.get("min_40min", 1e-4),
            min_1hr=params.get("min_1hr", 1e-5),
            min_2hr=params.get("min_2hr", 10 ** (-5.25)),
            min_3hr=params.get("min_3hr", 10 ** (-5.5)),
            min_4hr=params.get("min_4hr", 10 ** (-5.75)),
            min_halfway=params.get("min_halfway", 1e-4),
            init_distribution=params.get("dist_weights", "tn"),
            omega_init_distribution=params.get("dist_weights_omega", "tn"),
            init_scale=params.get("scale", 0.1),
            omega_init_scale=params.get("scale_omega", 0.1),
            act_type=params.get("act_type", "relu"),
            seed=params.get("seed", 42),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeepKoopmanConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if "config" in raw:
            raw = raw["config"]
        return cls(**raw)

    def to_dict(self) -> dict:
        return self.__dict__.copy()
