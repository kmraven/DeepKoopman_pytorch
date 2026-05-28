from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DeepKoopmanConfig
from .data import build_dataset, stack_data
from .losses import compute_losses
from .model import DeepKoopmanModule


class DeepKoopmanTrainer:
    def __init__(self, model: DeepKoopmanModule, config: DeepKoopmanConfig):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.history: list[dict[str, float]] = []

    def _resolve_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _seed_all(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def fit(self, train_data: np.ndarray, val_data: np.ndarray, config: DeepKoopmanConfig | None = None) -> list[dict[str, float]]:
        if config is not None:
            self.config = config
        self._seed_all()

        max_shift = max([1] + self.config.shifts + self.config.shifts_middle)
        train_stacked = stack_data(train_data, max_shift, self.config.len_time)
        val_stacked = stack_data(val_data, max_shift, self.config.len_time)

        train_loader = DataLoader(build_dataset(train_stacked), batch_size=1, shuffle=True)
        val_batch = torch.from_numpy(val_stacked).to(self.device, dtype=torch.float64)

        self.history = []
        for epoch in range(self.config.max_epochs):
            self.model.train()
            train_losses_last = None
            for (batch,) in train_loader:
                batch = batch[0].to(self.device)
                train_losses = compute_losses(self.model, batch, self.config)
                train_losses_last = train_losses
                loss_for_step = train_losses["loss1"] if epoch < self.config.autoencoder_warmup_epochs else train_losses["loss"]
                self.optimizer.zero_grad()
                loss_for_step.backward()
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                val_losses = compute_losses(self.model, val_batch, self.config)

            row = {
                "epoch": float(epoch),
                "train_loss": float(train_losses_last["loss"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss1": float(train_losses_last["loss1"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss2": float(train_losses_last["loss2"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss3": float(train_losses_last["loss3"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss_linf": float(train_losses_last["loss_linf"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss_l1": float(train_losses_last["loss_l1"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "train_loss_l2": float(train_losses_last["loss_l2"].detach().cpu()) if train_losses_last is not None else float("nan"),
                "loss": float(val_losses["loss"].detach().cpu()),
                "loss1": float(val_losses["loss1"].detach().cpu()),
                "loss2": float(val_losses["loss2"].detach().cpu()),
                "loss3": float(val_losses["loss3"].detach().cpu()),
                "loss_linf": float(val_losses["loss_linf"].detach().cpu()),
                "loss_l1": float(val_losses["loss_l1"].detach().cpu()),
                "loss_l2": float(val_losses["loss_l2"].detach().cpu()),
            }
            self.history.append(row)
        return self.history

    def predict(self, x0: np.ndarray, steps: int) -> np.ndarray:
        self.model.eval()
        x = torch.as_tensor(x0, dtype=torch.float64, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self.model.predict(x, steps)
        return y.detach().cpu().numpy()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        t = torch.as_tensor(x, dtype=torch.float64, device=self.device)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        with torch.no_grad():
            y = self.model.reconstruct(t)
        return y.detach().cpu().numpy()

    def save_predictions(self, x: np.ndarray, steps: int, out_dir: str | Path, prefix: str = "pred") -> dict[str, str]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        recon = self.reconstruct(x)
        pred = self.predict(x, steps)
        x_path = out_dir / f"{prefix}_input.csv"
        recon_path = out_dir / f"{prefix}_recon.csv"
        pred_path = out_dir / f"{prefix}_multistep.csv"
        np.savetxt(x_path, np.asarray(x), delimiter=",")
        np.savetxt(recon_path, recon, delimiter=",")
        np.savetxt(pred_path, pred.reshape(pred.shape[0], -1), delimiter=",")
        return {"input": str(x_path), "recon": str(recon_path), "pred": str(pred_path)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config.to_dict()}, path)
        with open(path.with_suffix(".history.json"), "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DeepKoopmanTrainer":
        ckpt = torch.load(path, map_location="cpu")
        config = DeepKoopmanConfig(**ckpt["config"])
        model = DeepKoopmanModule(config)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, config)
