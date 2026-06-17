from __future__ import annotations

import json
import os
import csv
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = []
            for row in csv.DictReader(f):
                parsed = {}
                for key, value in row.items():
                    if value == "":
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
                rows.append(parsed)
            return rows
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history_csv(history: list[dict], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        out_csv.write_text("", encoding="utf-8")
        return
    keys = list(history[0].keys())
    lines = [",".join(keys)]
    for row in history:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def plot_losses(history: list[dict], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    history = [row for row in history if "epoch" in row]
    if not history:
        raise ValueError("Cannot plot an empty history")
    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def values(*names: str) -> list[float]:
        for name in names:
            vals = [r.get(name) for r in history]
            if any(v is not None for v in vals):
                return [float(v) if v is not None else np.nan for v in vals]
        return [np.nan for _ in history]

    axes[0, 0].plot(epochs, values("train/loss_epoch", "train/loss", "train_loss"), label="train")
    axes[0, 0].plot(epochs, values("val/loss", "loss"), label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, values("val/reconstruction", "loss1"), label="recon")
    axes[0, 1].plot(epochs, values("val/prediction", "loss2"), label="pred")
    axes[0, 1].plot(epochs, values("val/latent_consistency", "loss3"), label="linearity")
    axes[0, 1].set_title("Loss Components (val)")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, values("val/linf", "loss_linf"), label="Linf")
    axes[1, 0].plot(epochs, values("val/l1", "loss_l1"), label="L1")
    axes[1, 0].plot(epochs, values("val/l2", "loss_l2"), label="L2")
    axes[1, 0].set_title("Regularization (val)")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, np.log10(np.maximum(values("val/loss", "loss"), 1e-16)))
    axes[1, 1].set_title("log10(val loss)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_reconstruction(x_true: np.ndarray, x_recon: np.ndarray, out_path: str | Path, title: str = "Reconstruction") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x_true.flatten(), label="true")
    ax.plot(x_recon.flatten(), label="recon")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_prediction(pred_series: np.ndarray, out_path: str | Path, title: str = "Prediction") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    # pred_series shape: steps+1, batch, features
    for feat in range(pred_series.shape[-1]):
        ax.plot(pred_series[:, 0, feat], label=f"feat{feat}")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
