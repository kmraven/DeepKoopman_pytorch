from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from .config import DeepKoopmanConfig
from .losses import compute_losses
from .model import DeepKoopmanModule, build_model


LOSS_ALIASES = {
    "loss": "loss",
    "loss1": "reconstruction",
    "loss2": "prediction",
    "loss3": "latent_consistency",
    "loss_cov": "latent_covariance",
    "loss_linf": "linf",
    "loss_l1": "l1",
    "loss_l2": "l2",
}


class DeepKoopmanLightningModule(L.LightningModule):
    def __init__(self, config: DeepKoopmanConfig):
        super().__init__()
        self.config = config
        self.model = build_model(config)
        self.save_hyperparameters({"config": config.to_dict()})

    def forward(self, stacked: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return self.model(stacked, self.config.data.shifts, self.config.data.middle_shifts)

    def _prepare_tensor_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 4:
            batch = batch.permute(1, 0, 2, 3).reshape(batch.shape[1], -1, batch.shape[3])
        if batch.ndim != 3:
            raise ValueError(f"Expected a 3-D or 4-D stacked batch, got shape {tuple(batch.shape)}")
        return batch

    def _prepare_condition_batch(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim == 4 and batch.shape[-1] == 1:
            batch = batch[..., 0]
        if batch.ndim == 3:
            batch = batch.permute(1, 0, 2).reshape(batch.shape[1], -1)
        elif batch.ndim == 2:
            batch = batch.T
        else:
            raise ValueError(f"Expected a 2-D or 3-D condition batch, got shape {tuple(batch.shape)}")
        return batch.long()

    def _prepare_batch(
        self,
        batch: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                stacked = self._prepare_tensor_batch(batch[0])
                conditions = self._prepare_condition_batch(batch[1])
                return stacked, conditions
            batch = batch[0]
        return self._prepare_tensor_batch(batch)

    def _shared_step(self, batch: torch.Tensor, stage: str) -> torch.Tensor:
        prepared = self._prepare_batch(batch)
        if isinstance(prepared, tuple):
            stacked, conditions = prepared
        else:
            stacked, conditions = prepared, None
        losses = compute_losses(self.model, stacked, self.config, conditions)
        metrics = {f"{stage}/{LOSS_ALIASES[name]}": value for name, value in losses.items() if name != "loss"}
        self.log_dict(metrics, on_step=stage == "train", on_epoch=True, prog_bar=False, logger=True, batch_size=stacked.shape[1])
        self.log(f"{stage}/loss", losses["loss"], on_step=stage == "train", on_epoch=True, prog_bar=True, logger=True, batch_size=stacked.shape[1])
        if stage == "train" and self.current_epoch < self.config.trainer.autoencoder_warmup_epochs:
            return losses["loss1"] + losses.get("loss_cov", 0.0) + losses["loss_l1"] + losses["loss_l2"]
        return losses["loss"]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.config.optimizer.name == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        return torch.optim.Adam(
            self.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )

    def predict_array(self, x0, steps: int):
        self.eval()
        device = self.device
        dtype = torch.float32 if self.config.runtime.dtype == "float32" else torch.float64
        x = torch.as_tensor(x0, dtype=dtype, device=device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self.model.predict(x, steps)
        return y.detach().cpu().numpy()

    def reconstruct_array(self, x):
        self.eval()
        device = self.device
        dtype = torch.float32 if self.config.runtime.dtype == "float32" else torch.float64
        t = torch.as_tensor(x, dtype=dtype, device=device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        with torch.no_grad():
            y = self.model.reconstruct(t)
        return y.detach().cpu().numpy()

    @classmethod
    def load_checkpoint(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "DeepKoopmanLightningModule":
        ckpt = torch.load(path, map_location=map_location)
        config_payload = ckpt["hyper_parameters"]["config"]
        config = DeepKoopmanConfig(**config_payload)
        return cls.load_from_checkpoint(path, config=config, map_location=map_location)


def build_logger(config: DeepKoopmanConfig, *, use_wandb: bool = False, run_name: str | None = None):
    if use_wandb or config.logging.backend == "wandb":
        wandb_kwargs: dict[str, Any] = {
            "project": config.logging.wandb.project,
            "name": run_name or config.logging.name,
            "save_dir": config.logging.save_dir,
            "mode": config.logging.wandb.mode,
        }
        if config.logging.wandb.entity:
            wandb_kwargs["entity"] = config.logging.wandb.entity
        return WandbLogger(**wandb_kwargs)
    return CSVLogger(save_dir=config.logging.save_dir, name=run_name or config.logging.name or "deepkoopman")


def build_callbacks(config: DeepKoopmanConfig, checkpoint_dir: str | Path | None = None) -> list:
    checkpoint_cfg = config.callbacks.model_checkpoint
    callbacks: list = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=checkpoint_cfg.monitor,
            mode=checkpoint_cfg.mode,
            save_top_k=checkpoint_cfg.save_top_k,
            save_last=checkpoint_cfg.save_last,
            filename=checkpoint_cfg.filename,
            auto_insert_metric_name=False,
        )
    ]
    early_cfg = config.callbacks.early_stopping
    if early_cfg.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=early_cfg.monitor,
                patience=early_cfg.patience,
                min_delta=early_cfg.min_delta,
                mode=early_cfg.mode,
            )
        )
    return callbacks


def build_trainer(
    config: DeepKoopmanConfig,
    *,
    default_root_dir: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
    use_wandb: bool = False,
    run_name: str | None = None,
    logger=True,
) -> L.Trainer:
    loggers = False if logger is False else build_logger(config, use_wandb=use_wandb, run_name=run_name)
    return L.Trainer(
        accelerator=config.trainer.accelerator if config.runtime.device == "auto" else config.runtime.device,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        max_epochs=config.trainer.max_epochs,
        max_steps=config.trainer.max_steps,
        log_every_n_steps=config.trainer.log_every_n_steps,
        enable_progress_bar=config.trainer.enable_progress_bar,
        val_check_interval=config.trainer.val_check_interval,
        gradient_clip_val=config.trainer.gradient_clip_val,
        logger=loggers,
        callbacks=build_callbacks(config, checkpoint_dir=checkpoint_dir),
        default_root_dir=default_root_dir,
    )
