from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm.auto import tqdm

from .config import DeepKoopmanConfig
from .data import stack_data, stack_data_windows
from .losses import compute_losses
from .model import DeepKoopmanModule


class DeepKoopmanTrainer:
    def __init__(self, model: DeepKoopmanModule, config: DeepKoopmanConfig):
        self.model = model
        self.config = config
        self.device = self._resolve_device(config.device)
        self.torch_dtype = self._resolve_dtype(config.dtype)
        self.numpy_dtype = np.float32 if self.torch_dtype == torch.float32 else np.float64
        self.model.to(self.device, dtype=self.torch_dtype)
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

    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        if dtype == "float32":
            return torch.float32
        if dtype == "float64":
            return torch.float64
        raise ValueError(f"dtype must be 'float32' or 'float64', got {dtype!r}")

    def _seed_all(self) -> None:
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

    def _max_shift(self) -> int:
        return max([1] + self.config.shifts + self.config.shifts_middle)

    def _num_windows(self, data: np.ndarray) -> int:
        if data.ndim == 1:
            length = data.shape[0]
        else:
            length = data.shape[0]
        if length % self.config.len_time != 0:
            raise ValueError(f"Data length {length} is not divisible by len_time={self.config.len_time}")
        return length // self.config.len_time

    def _stack_window_batch(self, data: np.ndarray, indices: np.ndarray) -> torch.Tensor:
        stacked = stack_data_windows(
            data,
            self._max_shift(),
            self.config.len_time,
            indices,
            dtype=self.numpy_dtype,
        )
        return torch.as_tensor(stacked, dtype=self.torch_dtype, device=self.device)

    def _losses_to_float(self, losses: dict[str, torch.Tensor]) -> dict[str, float]:
        return {name: float(value.detach().cpu()) for name, value in losses.items()}

    def evaluate_batched(
        self,
        data: np.ndarray,
        batch_size: int | None = None,
        show_progress: bool = False,
        desc: str = "validate",
    ) -> dict[str, float]:
        num_windows = self._num_windows(data)
        if num_windows == 0:
            raise ValueError("Cannot evaluate an empty dataset")
        batch_size = batch_size or self.config.batch_size or num_windows
        batch_size = max(1, int(batch_size))
        totals: dict[str, float] = {}
        total_examples = 0
        self.model.eval()
        starts = range(0, num_windows, batch_size)
        iterator = tqdm(starts, desc=desc, unit="batch", leave=False, disable=not show_progress)
        with torch.no_grad():
            for start in iterator:
                indices = np.arange(start, min(start + batch_size, num_windows), dtype=np.int64)
                batch = self._stack_window_batch(data, indices)
                losses = compute_losses(self.model, batch, self.config)
                weight = int(batch.shape[1])
                loss_values = self._losses_to_float(losses)
                for name, value in loss_values.items():
                    totals[name] = totals.get(name, 0.0) + value * weight
                total_examples += weight
        return {name: value / total_examples for name, value in totals.items()}

    def fit(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        config: DeepKoopmanConfig | None = None,
        show_progress: bool = False,
    ) -> list[dict[str, float]]:
        if config is not None:
            self.config = config
            self.torch_dtype = self._resolve_dtype(config.dtype)
            self.numpy_dtype = np.float32 if self.torch_dtype == torch.float32 else np.float64
            self.model.to(self.device, dtype=self.torch_dtype)
        self._seed_all()

        num_train_windows = self._num_windows(train_data)
        if num_train_windows == 0:
            raise ValueError("Cannot train on an empty dataset")
        batch_size = self.config.batch_size if self.config.batch_size > 0 else num_train_windows

        self.history = []
        epoch_iter = tqdm(range(self.config.max_epochs), desc="train", unit="epoch", disable=not show_progress)
        for epoch in epoch_iter:
            self.model.train()
            train_totals: dict[str, float] = {}
            train_examples = 0
            indices_all = np.arange(num_train_windows, dtype=np.int64)
            np.random.shuffle(indices_all)
            starts = range(0, num_train_windows, batch_size)
            batch_iter = tqdm(
                starts,
                desc=f"epoch {epoch + 1}/{self.config.max_epochs}",
                unit="batch",
                leave=False,
                disable=not show_progress,
            )
            for start in batch_iter:
                window_indices = indices_all[start : start + batch_size]
                batch = self._stack_window_batch(train_data, window_indices)
                train_losses = compute_losses(self.model, batch, self.config)
                loss_for_step = train_losses["loss1"] if epoch < self.config.autoencoder_warmup_epochs else train_losses["loss"]
                self.optimizer.zero_grad()
                loss_for_step.backward()
                self.optimizer.step()

                weight = int(batch.shape[1])
                loss_values = self._losses_to_float(train_losses)
                for name, value in loss_values.items():
                    train_totals[name] = train_totals.get(name, 0.0) + value * weight
                train_examples += weight
                if show_progress:
                    batch_iter.set_postfix(loss=loss_values["loss"])

            train_avg = {name: value / train_examples for name, value in train_totals.items()}
            val_losses = self.evaluate_batched(
                val_data,
                batch_size=batch_size,
                show_progress=show_progress,
                desc=f"val {epoch + 1}/{self.config.max_epochs}",
            )

            row = {
                "epoch": float(epoch),
                "train_loss": train_avg["loss"],
                "train_loss1": train_avg["loss1"],
                "train_loss2": train_avg["loss2"],
                "train_loss3": train_avg["loss3"],
                "train_loss_linf": train_avg["loss_linf"],
                "train_loss_l1": train_avg["loss_l1"],
                "train_loss_l2": train_avg["loss_l2"],
                "loss": val_losses["loss"],
                "loss1": val_losses["loss1"],
                "loss2": val_losses["loss2"],
                "loss3": val_losses["loss3"],
                "loss_linf": val_losses["loss_linf"],
                "loss_l1": val_losses["loss_l1"],
                "loss_l2": val_losses["loss_l2"],
            }
            self.history.append(row)
            if show_progress:
                epoch_iter.set_postfix(train_loss=row["train_loss"], val_loss=row["loss"])
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
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> dict[str, object]:
        self._seed_all()
        if len(train_paths) < self.config.data_train_len:
            raise ValueError(
                f"Expected {self.config.data_train_len} train files for {self.config.data_name}, "
                f"got {len(train_paths)}"
            )

        max_shift = max([1] + self.config.shifts + self.config.shifts_middle)
        val_stacked = stack_data(val_data, max_shift, self.config.len_time).astype(self.numpy_dtype, copy=False)
        val_batch = torch.from_numpy(val_stacked).to(self.device, dtype=self.torch_dtype)
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
            train_data = np.loadtxt(train_paths[file_num - 1], delimiter=",", dtype=self.numpy_dtype)
            train_stacked = stack_data(train_data, max_shift, self.config.len_time).astype(self.numpy_dtype, copy=False)
            num_examples = train_stacked.shape[1]
            batch_size = self.config.batch_size if self.config.batch_size > 0 else num_examples
            num_batches = max(1, int(np.floor(num_examples / batch_size)))
            indices = np.arange(num_examples)
            np.random.shuffle(indices)
            train_stacked = train_stacked[:, indices, :]
            requested_steps = self.config.num_steps_per_batch * num_batches
            if self.config.num_steps_per_file_pass is not None:
                requested_steps = min(requested_steps, self.config.num_steps_per_file_pass + 1)
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "file_start",
                        "file_pass": file_pass,
                        "file_num": file_num,
                        "requested_steps": requested_steps,
                        "total_file_passes": total_file_passes,
                    }
                )

            for local_step in range(requested_steps):
                if batch_size < num_examples:
                    offset = (local_step * batch_size) % (num_examples - batch_size)
                else:
                    offset = 0
                batch_np = train_stacked[:, offset : offset + batch_size, :]
                batch = torch.from_numpy(batch_np).to(self.device, dtype=self.torch_dtype)

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
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "eval",
                                "global_step": global_step,
                                "file_pass": file_pass,
                                "file_num": file_num,
                                "elapsed_sec": elapsed,
                                "val_loss": val_error,
                                "best_val_loss": best_error,
                                "train_loss": row["train_loss"],
                            }
                        )
                    finished, reason = self._check_progress(elapsed, best_error, progress)
                    if finished:
                        stop_condition = reason or "stopped"
                        self.model.load_state_dict(best_state)
                        if progress_callback is not None:
                            progress_callback(
                                {
                                    "event": "stop",
                                    "global_step": global_step + 1,
                                    "file_pass": file_pass,
                                    "file_num": file_num,
                                    "elapsed_sec": elapsed,
                                    "stop_condition": stop_condition,
                                    "best_val_loss": best_error,
                                }
                            )
                        return {
                            "stop_condition": stop_condition,
                            "best_val_loss": best_error,
                            "steps": global_step + 1,
                            "elapsed_sec": elapsed,
                        }
                global_step += 1
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "step",
                            "global_step": global_step,
                            "file_pass": file_pass,
                            "file_num": file_num,
                        }
                    )

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
        x = torch.as_tensor(x0, dtype=self.torch_dtype, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            y = self.model.predict(x, steps)
        return y.detach().cpu().numpy()

    def reconstruct(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        t = torch.as_tensor(x, dtype=self.torch_dtype, device=self.device)
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
