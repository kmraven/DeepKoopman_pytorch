from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class TrainerConfig:
    max_epochs: int | None = None
    max_steps: int = -1
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str | None = None
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True
    val_check_interval: float | int = 1.0


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    monitor: str = "val/loss"
    patience: int = 10
    min_delta: float = 0.0
    mode: str = "min"


@dataclass
class ModelCheckpointConfig:
    monitor: str = "val/loss"
    mode: str = "min"
    save_top_k: int = 1
    filename: str = "best-{epoch:03d}"


@dataclass
class CallbackConfig:
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    model_checkpoint: ModelCheckpointConfig = field(default_factory=ModelCheckpointConfig)


@dataclass
class WandbConfig:
    project: str = "deepkoopman"
    entity: str | None = None
    mode: str = "online"


@dataclass
class LoggingConfig:
    backend: str = "csv"
    save_dir: str = "results"
    name: str | None = None
    wandb: WandbConfig = field(default_factory=WandbConfig)


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
    dtype: str = "float64"
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    callbacks: CallbackConfig = field(default_factory=CallbackConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)
        if isinstance(self.callbacks, dict):
            early = self.callbacks.get("early_stopping", {})
            checkpoint = self.callbacks.get("model_checkpoint", {})
            self.callbacks = CallbackConfig(
                early_stopping=EarlyStoppingConfig(**early),
                model_checkpoint=ModelCheckpointConfig(**checkpoint),
            )
        if isinstance(self.logging, dict):
            wandb = self.logging.get("wandb", {})
            rest = {k: v for k, v in self.logging.items() if k != "wandb"}
            self.logging = LoggingConfig(**rest, wandb=WandbConfig(**wandb))
        if self.dtype not in {"float32", "float64"}:
            raise ValueError(f"dtype must be 'float32' or 'float64', got {self.dtype!r}")
        if self.logging.backend not in {"csv", "wandb"}:
            raise ValueError(f"logging.backend must be 'csv' or 'wandb', got {self.logging.backend!r}")
        if self.callbacks.early_stopping.mode not in {"min", "max"}:
            raise ValueError("callbacks.early_stopping.mode must be 'min' or 'max'")
        if self.callbacks.model_checkpoint.mode not in {"min", "max"}:
            raise ValueError("callbacks.model_checkpoint.mode must be 'min' or 'max'")
        if self.trainer.max_epochs is None:
            self.trainer.max_epochs = int(self.max_epochs)
        else:
            self.trainer.max_epochs = int(self.trainer.max_epochs)
            self.max_epochs = self.trainer.max_epochs
        if self.trainer.precision is None:
            self.trainer.precision = "32-true" if self.dtype == "float32" else "64-true"

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
        raw = cls._normalize_yaml(raw)
        return cls(**raw)

    @classmethod
    def _normalize_yaml(cls, raw: dict) -> dict:
        raw = raw.copy()
        if "model" in raw:
            model = raw.pop("model")
            raw.setdefault("widths", model["widths"])
            raw.setdefault("hidden_widths_omega", model.get("omega_hidden_widths", model.get("hidden_widths_omega")))
            raw.setdefault("num_real", model["num_real"])
            raw.setdefault("num_complex_pairs", model["num_complex_pairs"])
            raw.setdefault("act_type", model.get("activation", model.get("act_type", "relu")))
            raw.setdefault("dtype", model.get("dtype", raw.get("dtype", "float64")))
        if "data" in raw:
            data = raw.pop("data")
            raw.setdefault("data_name", data.get("name", data.get("data_name", "DiscreteSpectrumExample")))
            raw.setdefault("len_time", data["len_time"])
            raw.setdefault("delta_t", data["delta_t"])
            if "shifts" in data:
                raw.setdefault("shifts", data["shifts"])
            if "shifts_middle" in data:
                raw.setdefault("shifts_middle", data["shifts_middle"])
        if "loss" in raw:
            loss = raw.pop("loss")
            raw.setdefault("recon_lam", loss.get("reconstruction_weight", loss.get("recon_lam", 1.0)))
            raw.setdefault("mid_shift_lam", loss.get("middle_shift_weight", loss.get("mid_shift_lam", 1.0)))
            raw.setdefault("Linf_lam", loss.get("linf_weight", loss.get("Linf_lam", 0.0)))
            raw.setdefault("l1_lam", loss.get("l1_weight", loss.get("l1_lam", 0.0)))
            raw.setdefault("l2_lam", loss.get("l2_weight", loss.get("l2_lam", 0.0)))
            raw.setdefault("relative_loss", loss.get("relative", loss.get("relative_loss", False)))
        if "optimizer" in raw:
            optimizer = raw.pop("optimizer")
            raw.setdefault("learning_rate", optimizer.get("lr", optimizer.get("learning_rate", 1e-3)))
        trainer = raw.get("trainer")
        if isinstance(trainer, dict):
            raw.setdefault("max_epochs", trainer.get("max_epochs", raw.get("max_epochs", 10)))
            raw.setdefault("batch_size", trainer.get("batch_size", raw.get("batch_size", 256)))
        return raw

    def to_dict(self) -> dict:
        return asdict(self)
