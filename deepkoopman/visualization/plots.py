from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_history(path: str | Path) -> list[dict]:
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

    epochs = [int(r["epoch"]) for r in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(epochs, [r["train_loss"] for r in history], label="train")
    axes[0, 0].plot(epochs, [r["loss"] for r in history], label="val")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [r["loss1"] for r in history], label="recon")
    axes[0, 1].plot(epochs, [r["loss2"] for r in history], label="pred")
    axes[0, 1].plot(epochs, [r["loss3"] for r in history], label="linearity")
    axes[0, 1].set_title("Loss Components (val)")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [r["loss_linf"] for r in history], label="Linf")
    axes[1, 0].plot(epochs, [r["loss_l1"] for r in history], label="L1")
    axes[1, 0].plot(epochs, [r["loss_l2"] for r in history], label="L2")
    axes[1, 0].set_title("Regularization (val)")
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, np.log10(np.maximum([r["loss"] for r in history], 1e-16)))
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
