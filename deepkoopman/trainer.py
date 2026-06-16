from __future__ import annotations

import json
import random
import time
from copy import deepcopy
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
        # The implementation uses float64 to match the TensorFlow reference.
        # MPS does not support float64 kernels, so CPU is the portable default on macOS.
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

    def _loss_row(
        self,
        train_losses: dict[str, torch.Tensor],
        val_losses: dict[str, torch.Tensor],
        step: int,
        file_pass: int,
        file_num: int,
        elapsed: float,
    ) -> dict[str, float]:
        return {
            "epoch": float(step),
            "step": float(step),
            "file_pass": float(file_pass),
            "file_num": float(file_num),
            "elapsed_sec": float(elapsed),
            "train_loss": float(train_losses["loss"].detach().cpu()),
            "train_loss1": float(train_losses["loss1"].detach().cpu()),
            "train_loss2": float(train_losses["loss2"].detach().cpu()),
            "train_loss3": float(train_losses["loss3"].detach().cpu()),
            "train_loss_linf": float(train_losses["loss_linf"].detach().cpu()),
            "train_loss_l1": float(train_losses["loss_l1"].detach().cpu()),
            "train_loss_l2": float(train_losses["loss_l2"].detach().cpu()),
            "loss": float(val_losses["loss"].detach().cpu()),
            "loss1": float(val_losses["loss1"].detach().cpu()),
            "loss2": float(val_losses["loss2"].detach().cpu()),
            "loss3": float(val_losses["loss3"].detach().cpu()),
            "loss_linf": float(val_losses["loss_linf"].detach().cpu()),
            "loss_l1": float(val_losses["loss_l1"].detach().cpu()),
            "loss_l2": float(val_losses["loss_l2"].detach().cpu()),
        }

    def _check_progress(
        self,
        elapsed: float,
        best_error: float,
        progress: dict[str, float],
    ) -> tuple[bool, str | None]:
        checks = [
            ("been5min", 5 * 60, self.config.min_5min, "too slowly improving in first 5 min"),
            ("been20min", 20 * 60, self.config.min_20min, "too slowly improving in first 20 min"),
            ("been40min", 40 * 60, self.config.min_40min, "too slowly improving in first 40 min"),
            ("been1hr", 60 * 60, self.config.min_1hr, "too slowly improving in first hour"),
            ("been2hr", 2 * 60 * 60, self.config.min_2hr, "too slowly improving in first two hours"),
            ("been3hr", 3 * 60 * 60, self.config.min_3hr, "too slowly improving in first three hours"),
            ("been4hr", 4 * 60 * 60, self.config.min_4hr, "too slowly improving in first four hours"),
        ]
        for key, threshold, minimum, message in checks:
            if not progress[key] and elapsed > threshold:
                if best_error > minimum:
                    return True, message
                progress[key] = best_error

        if self.config.max_time is not None:
            if not progress["beenHalf"] and elapsed > self.config.max_time / 2:
                if best_error > self.config.min_halfway:
                    return True, "too slowly improving halfway in"
                progress["beenHalf"] = best_error
            if elapsed > self.config.max_time:
                return True, "past max time"
        return False, None

    def fit_legacy_files(
        self,
        train_paths: list[str | Path],
        val_data: np.ndarray,
        checkpoint_path: str | Path | None = None,
    ) -> dict[str, object]:
        self._seed_all()
        if len(train_paths) < self.config.data_train_len:
            raise ValueError(
                f"Expected {self.config.data_train_len} train files for {self.config.data_name}, "
                f"got {len(train_paths)}"
            )

        max_shift = max([1] + self.config.shifts + self.config.shifts_middle)
        val_stacked = stack_data(val_data, max_shift, self.config.len_time)
        val_batch = torch.from_numpy(val_stacked).to(self.device, dtype=torch.float64)
        total_file_passes = self.config.data_train_len * self.config.num_passes_per_file

        self.history = []
        best_error = float("inf")
        best_state = deepcopy(self.model.state_dict())
        stop_condition = "completed requested file passes"
        progress = {
            "been5min": 0.0,
            "been20min": 0.0,
            "been40min": 0.0,
            "been1hr": 0.0,
            "been2hr": 0.0,
            "been3hr": 0.0,
            "been4hr": 0.0,
            "beenHalf": 0.0,
        }
        global_step = 0
        start = time.time()
        checkpoint = Path(checkpoint_path) if checkpoint_path else None

        for file_pass in range(total_file_passes):
            file_num = (file_pass % self.config.data_train_len) + 1
            train_data = np.loadtxt(train_paths[file_num - 1], delimiter=",", dtype=np.float64)
            train_stacked = stack_data(train_data, max_shift, self.config.len_time)
            num_examples = train_stacked.shape[1]
            batch_size = self.config.batch_size if self.config.batch_size > 0 else num_examples
            num_batches = max(1, int(np.floor(num_examples / batch_size)))
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            train_stacked = train_stacked[:, indices, :]
            requested_steps = self.config.num_steps_per_batch * num_batches
            if self.config.num_steps_per_file_pass is not None:
                requested_steps = min(requested_steps, self.config.num_steps_per_file_pass + 1)

            for local_step in range(requested_steps):
                if batch_size < num_examples:
                    offset = (local_step * batch_size) % (num_examples - batch_size)
                else:
                    offset = 0
                batch_np = train_stacked[:, offset : offset + batch_size, :]
                batch = torch.from_numpy(batch_np).to(self.device, dtype=torch.float64)

                self.model.train()
                train_losses = compute_losses(self.model, batch, self.config)
                use_auto = self.config.autoencoder_warmup_epochs > 0 and not progress["been5min"]
                loss_for_step = train_losses["loss1"] + train_losses["loss_l1"] + train_losses["loss_l2"] if use_auto else train_losses["loss"]
                self.optimizer.zero_grad()
                loss_for_step.backward()
                self.optimizer.step()

                if global_step % self.config.eval_interval == 0:
                    self.model.eval()
                    with torch.no_grad():
                        val_losses = compute_losses(self.model, val_batch, self.config)
                    elapsed = time.time() - start
                    row = self._loss_row(train_losses, val_losses, global_step, file_pass, file_num, elapsed)
                    self.history.append(row)
                    val_error = row["loss"]
                    if best_error == float("inf") or val_error < best_error - best_error * 1e-5:
                        best_error = val_error
                        best_state = deepcopy(self.model.state_dict())
                        if checkpoint is not None:
                            self.save(checkpoint)
                    finished, reason = self._check_progress(elapsed, best_error, progress)
                    if finished:
                        stop_condition = reason or "stopped"
                        self.model.load_state_dict(best_state)
                        return {
                            "stop_condition": stop_condition,
                            "best_val_loss": best_error,
                            "steps": global_step + 1,
                            "elapsed_sec": elapsed,
                        }
                global_step += 1

        self.model.load_state_dict(best_state)
        elapsed = time.time() - start
        if checkpoint is not None:
            self.save(checkpoint)
        return {
            "stop_condition": stop_condition,
            "best_val_loss": best_error,
            "steps": global_step,
            "elapsed_sec": elapsed,
        }

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
