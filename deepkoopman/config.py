from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    name: str
    len_time: int
    delta_t: float
    root: str = "data"
    shifts: list[int] | dict[str, int] = field(default_factory=lambda: list(range(1, 31)))
    middle_shifts: list[int] | dict[str, int] = field(default_factory=lambda: list(range(1, 51)))
    train_files: int = 1
    input_shape: list[int] | None = None
    condition_names: list[str] = field(
        default_factory=lambda: ["silence", "normal_music", "gamma_music", "gamma_click"]
    )
    starts_per_sequence: int | None = None

    def __post_init__(self) -> None:
        self.shifts = _expand_index_spec(self.shifts, "shifts")
        self.middle_shifts = _expand_index_spec(self.middle_shifts, "middle_shifts")
        if self.starts_per_sequence is not None and self.starts_per_sequence <= 0:
            raise ValueError("data.starts_per_sequence must be positive when set")


@dataclass
class InitializationConfig:
    distribution: str = "tn"
    scale: float = 0.1
    omega_distribution: str = "tn"
    omega_scale: float = 0.1


@dataclass
class ModelConfig:
    widths: list[int]
    omega_hidden_widths: list[int]
    num_real: int
    num_complex_pairs: int
    activation: str = "relu"
    architecture: str = "mlp"
    condition_dim: int = 4
    initialization: InitializationConfig = field(default_factory=InitializationConfig)


@dataclass
class LossConfig:
    reconstruction_weight: float = 1.0
    prediction_weight: float = 1.0
    middle_shift_weight: float = 1.0
    covariance_weight: float = 0.0
    linf_weight: float = 0.0
    l1_weight: float = 0.0
    l2_weight: float = 0.0
    relative: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class TrainerConfig:
    batch_size: int = 256
    max_epochs: int = 10
    max_steps: int = -1
    accelerator: str = "auto"
    devices: str | int = "auto"
    precision: str | None = None
    log_every_n_steps: int = 50
    enable_progress_bar: bool = True
    val_check_interval: float | int = 1.0
    autoencoder_warmup_epochs: int = 0
    gradient_clip_val: float | None = None


@dataclass
class RuntimeConfig:
    seed: int = 42
    dtype: str = "float64"
    device: str = "auto"


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
    save_last: bool = True
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


def _coerce(cls, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__} or dict, got {type(value).__name__}")


def _expand_index_spec(value: Any, name: str) -> list[int]:
    if isinstance(value, list):
        return [int(v) for v in value]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, dict):
        required = {"start", "stop", "interval"}
        missing = required - set(value)
        if missing:
            raise ValueError(f"data.{name} range spec is missing keys: {sorted(missing)}")
        start = int(value["start"])
        stop = int(value["stop"])
        interval = int(value["interval"])
        if interval <= 0:
            raise ValueError(f"data.{name}.interval must be positive")
        return list(range(start, stop, interval))
    raise TypeError(f"data.{name} must be a list or a start/stop/interval dict")


@dataclass
class DeepKoopmanConfig:
    data: DataConfig | dict
    model: ModelConfig | dict
    loss: LossConfig | dict = field(default_factory=LossConfig)
    optimizer: OptimizerConfig | dict = field(default_factory=OptimizerConfig)
    trainer: TrainerConfig | dict = field(default_factory=TrainerConfig)
    callbacks: CallbackConfig | dict = field(default_factory=CallbackConfig)
    logging: LoggingConfig | dict = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig | dict = field(default_factory=RuntimeConfig)

    def __post_init__(self) -> None:
        self.data = _coerce(DataConfig, self.data)
        if isinstance(self.model, dict):
            init = self.model.get("initialization", {})
            self.model = ModelConfig(
                **{k: v for k, v in self.model.items() if k != "initialization"},
                initialization=_coerce(InitializationConfig, init),
            )
        elif isinstance(self.model, ModelConfig) and isinstance(self.model.initialization, dict):
            self.model.initialization = _coerce(InitializationConfig, self.model.initialization)
        elif not isinstance(self.model, ModelConfig):
            raise TypeError(f"Expected ModelConfig or dict, got {type(self.model).__name__}")
        self.loss = _coerce(LossConfig, self.loss)
        self.optimizer = _coerce(OptimizerConfig, self.optimizer)
        self.trainer = _coerce(TrainerConfig, self.trainer)
        self.runtime = _coerce(RuntimeConfig, self.runtime)
        if isinstance(self.callbacks, dict):
            early = self.callbacks.get("early_stopping", {})
            checkpoint = self.callbacks.get("model_checkpoint", {})
            self.callbacks = CallbackConfig(
                early_stopping=_coerce(EarlyStoppingConfig, early),
                model_checkpoint=_coerce(ModelCheckpointConfig, checkpoint),
            )
        elif not isinstance(self.callbacks, CallbackConfig):
            raise TypeError(f"Expected CallbackConfig or dict, got {type(self.callbacks).__name__}")
        if isinstance(self.logging, dict):
            wandb = self.logging.get("wandb", {})
            rest = {k: v for k, v in self.logging.items() if k != "wandb"}
            self.logging = LoggingConfig(**rest, wandb=_coerce(WandbConfig, wandb))
        elif not isinstance(self.logging, LoggingConfig):
            raise TypeError(f"Expected LoggingConfig or dict, got {type(self.logging).__name__}")

        if self.runtime.dtype not in {"float32", "float64"}:
            raise ValueError(f"runtime.dtype must be 'float32' or 'float64', got {self.runtime.dtype!r}")
        if self.logging.backend not in {"csv", "wandb"}:
            raise ValueError(f"logging.backend must be 'csv' or 'wandb', got {self.logging.backend!r}")
        if self.optimizer.name not in {"adam", "adamw"}:
            raise ValueError("optimizer.name must be 'adam' or 'adamw'")
        if self.callbacks.early_stopping.mode not in {"min", "max"}:
            raise ValueError("callbacks.early_stopping.mode must be 'min' or 'max'")
        if self.callbacks.model_checkpoint.mode not in {"min", "max"}:
            raise ValueError("callbacks.model_checkpoint.mode must be 'min' or 'max'")
        if self.trainer.precision is None:
            self.trainer.precision = "32-true" if self.runtime.dtype == "float32" else "64-true"

    @property
    def num_evals(self) -> int:
        return 2 * self.model.num_complex_pairs + self.model.num_real

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DeepKoopmanConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if "config" in raw:
            raw = raw["config"]
        required = {"data", "model"}
        missing = required - set(raw)
        if missing:
            raise ValueError(
                f"{path} uses an unsupported legacy/flat config shape. "
                f"Missing top-level sections: {sorted(missing)}. "
                "Use DeepKoopmanConfig.from_legacy_dict() for legacy dictionaries."
            )
        return cls(**raw)

    @classmethod
    def from_legacy_dict(cls, params: dict) -> "DeepKoopmanConfig":
        return cls(
            data=DataConfig(
                name=params.get("data_name", "DiscreteSpectrumExample"),
                len_time=params["len_time"],
                delta_t=params["delta_t"],
                shifts=list(params.get("shifts", range(1, params.get("num_shifts", 0) + 1))),
                middle_shifts=list(
                    params.get("shifts_middle", range(1, params.get("num_shifts_middle", 0) + 1))
                ),
                train_files=params.get("data_train_len", 1),
            ),
            model=ModelConfig(
                widths=params["widths"],
                omega_hidden_widths=params["hidden_widths_omega"],
                num_real=params["num_real"],
                num_complex_pairs=params["num_complex_pairs"],
                activation=params.get("act_type", "relu"),
                initialization=InitializationConfig(
                    distribution=params.get("dist_weights", "tn"),
                    omega_distribution=params.get("dist_weights_omega", "tn"),
                    scale=params.get("scale", 0.1),
                    omega_scale=params.get("scale_omega", 0.1),
                ),
            ),
            loss=LossConfig(
                reconstruction_weight=params.get("recon_lam", 1.0),
                middle_shift_weight=params.get("mid_shift_lam", 1.0),
                linf_weight=params.get("Linf_lam", 0.0),
                relative=bool(params.get("relative_loss", 0)),
                l1_weight=params.get("L1_lam", params.get("l1_lam", 0.0)),
                l2_weight=params.get("L2_lam", params.get("l2_lam", 0.0)),
            ),
            optimizer=OptimizerConfig(lr=params.get("learning_rate", 1e-3)),
            trainer=TrainerConfig(
                batch_size=params.get("batch_size", 256),
                max_epochs=params.get("max_epochs", 10),
                autoencoder_warmup_epochs=1 if params.get("auto_first", 0) else params.get("autoencoder_warmup_epochs", 0),
            ),
            runtime=RuntimeConfig(
                seed=params.get("seed", 42),
                dtype=params.get("dtype", "float64"),
                device=params.get("device", "auto"),
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)
