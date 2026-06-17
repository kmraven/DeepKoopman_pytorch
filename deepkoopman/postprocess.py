from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import stack_data
from .io import load_split_data
from .lightning import DeepKoopmanLightningModule
from .losses import compute_losses


LOSS_NAMES = {
    "loss": "loss",
    "loss1": "reconstruction",
    "loss2": "prediction",
    "loss3": "latent_consistency",
    "loss_linf": "linf",
    "loss_l1": "l1",
    "loss_l2": "l2",
}


def _as_trajectories(data: np.ndarray, len_time: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    usable = (data.shape[0] // len_time) * len_time
    return data[:usable].reshape(usable // len_time, len_time, data.shape[1])


def evaluate_data(module: DeepKoopmanLightningModule, data: np.ndarray) -> dict[str, float]:
    cfg = module.config
    max_shift = max([1] + cfg.data.shifts + cfg.data.middle_shifts)
    stacked = stack_data(data, max_shift, cfg.data.len_time)
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    batch = torch.from_numpy(stacked).to(module.device, dtype=dtype)
    module.model.eval()
    with torch.no_grad():
        raw = compute_losses(module.model, batch, cfg)
    metrics = {LOSS_NAMES[key]: float(value.detach().cpu()) for key, value in raw.items()}
    metrics["pre_regularization_loss"] = (
        metrics["reconstruction"] + metrics["prediction"] + metrics["latent_consistency"] + metrics["linf"]
    )
    return metrics


def _write_single_row_csv(path: Path, row: dict[str, float | str | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _write_rows(path: Path, rows: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sample_indices(num_trajectories: int, count: int, rng: np.random.Generator) -> np.ndarray:
    if num_trajectories <= count:
        return np.arange(num_trajectories, dtype=np.int64)
    return np.sort(rng.choice(num_trajectories, size=count, replace=False))


def sample_trajectories(
    splits: dict[str, np.ndarray],
    len_time: int,
    samples_per_split: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, int | str]]]:
    rng = np.random.default_rng(seed)
    sampled: dict[str, np.ndarray] = {}
    rows: list[dict[str, int | str]] = []
    for split, data in splits.items():
        trajectories = _as_trajectories(data, len_time)
        indices = _sample_indices(len(trajectories), samples_per_split, rng)
        sampled[split] = trajectories[indices]
        for order, index in enumerate(indices):
            rows.append({"split": split, "sample_order": order, "trajectory_index": int(index)})
    return sampled, rows


def _plot_data_trajectories(sampled: dict[str, np.ndarray], fig_dir: Path, delta_t: float) -> dict[str, str]:
    paths = {}
    for split, trajectories in sampled.items():
        if trajectories.shape[2] > 3:
            continue
        path = fig_dir / f"{split}_data_trajectories.png"
        fig = plt.figure(figsize=(7, 5))
        if trajectories.shape[2] == 3:
            ax = fig.add_subplot(111, projection="3d")
            for traj in trajectories:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1.5)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_zlabel("x3")
        elif trajectories.shape[2] == 2:
            ax = fig.add_subplot(111)
            for traj in trajectories:
                ax.plot(traj[:, 0], traj[:, 1], linewidth=1.5)
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
        else:
            ax = fig.add_subplot(111)
            t = np.arange(trajectories.shape[1]) * delta_t
            for traj in trajectories:
                ax.plot(t, traj[:, 0], linewidth=1.5)
            ax.set_xlabel("time")
            ax.set_ylabel("x1")
        ax.set_title(f"{split} data trajectories")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"{split}_data_trajectories"] = str(path)
    return paths


def _encode_array(module: DeepKoopmanLightningModule, values: np.ndarray) -> np.ndarray:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(values, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        return module.model.encode(x).detach().cpu().numpy()


def _predict_latent_array(module: DeepKoopmanLightningModule, x0: np.ndarray, steps: int) -> np.ndarray:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(x0, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        g0 = module.model.encode(x)
        latents = module.model.predict_latent(g0, steps)
    return torch.stack(latents, dim=0).detach().cpu().numpy()


def _plot_latent_true_vs_pred(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    fig_dir: Path,
) -> dict[str, str]:
    paths = {}
    for split, trajectories in sampled.items():
        path = fig_dir / f"{split}_latent_true_vs_pred.png"
        first = _encode_array(module, trajectories[0])
        latent_dim = first.shape[1]
        fig = plt.figure(figsize=(7, 5))
        if latent_dim >= 3:
            ax = fig.add_subplot(111, projection="3d")
            for traj in trajectories:
                true_latent = _encode_array(module, traj)
                pred_latent = _predict_latent_array(module, traj[:1], traj.shape[0] - 1)[:, 0, :]
                ax.plot(true_latent[:, 0], true_latent[:, 1], true_latent[:, 2], linewidth=1.5)
                ax.plot(pred_latent[:, 0], pred_latent[:, 1], pred_latent[:, 2], linestyle="--", linewidth=1.2)
            ax.set_xlabel("g1")
            ax.set_ylabel("g2")
            ax.set_zlabel("g3")
        elif latent_dim == 2:
            ax = fig.add_subplot(111)
            for traj in trajectories:
                true_latent = _encode_array(module, traj)
                pred_latent = _predict_latent_array(module, traj[:1], traj.shape[0] - 1)[:, 0, :]
                ax.plot(true_latent[:, 0], true_latent[:, 1], linewidth=1.5)
                ax.plot(pred_latent[:, 0], pred_latent[:, 1], linestyle="--", linewidth=1.2)
            ax.set_xlabel("g1")
            ax.set_ylabel("g2")
        else:
            ax = fig.add_subplot(111)
            for traj in trajectories:
                true_latent = _encode_array(module, traj)
                pred_latent = _predict_latent_array(module, traj[:1], traj.shape[0] - 1)[:, 0, :]
                steps = np.arange(traj.shape[0])
                ax.plot(steps, true_latent[:, 0], linewidth=1.5)
                ax.plot(steps, pred_latent[:, 0], linestyle="--", linewidth=1.2)
            ax.set_xlabel("step")
            ax.set_ylabel("g1")
        ax.set_title(f"{split} latent true vs predicted")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"{split}_latent_true_vs_pred"] = str(path)
    return paths


def _omega_components(module: DeepKoopmanLightningModule, latent: np.ndarray) -> dict[str, np.ndarray]:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    g = torch.as_tensor(latent, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        omegas = [omega.detach().cpu().numpy() for omega in module.model._omega_net_apply(g)]
    out: dict[str, np.ndarray] = {}
    for idx in range(module.config.model.num_complex_pairs):
        out[f"complex{idx + 1}_omega"] = omegas[idx][:, 0]
        out[f"complex{idx + 1}_mu"] = omegas[idx][:, 1]
    offset = module.config.model.num_complex_pairs
    for idx in range(module.config.model.num_real):
        out[f"real{idx + 1}_mu"] = omegas[offset + idx][:, 0]
    return out


def _latent_xy(latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = latent[:, 0]
    y = latent[:, 1] if latent.shape[1] > 1 else np.zeros(latent.shape[0])
    return x, y


def _plot_eigen_components(module: DeepKoopmanLightningModule, sampled: dict[str, np.ndarray], fig_dir: Path) -> dict[str, str]:
    all_points = np.concatenate([traj.reshape(-1, traj.shape[-1]) for traj in sampled.values()], axis=0)
    latent = _encode_array(module, all_points)
    x, y = _latent_xy(latent)
    paths = {}
    for name, values in _omega_components(module, latent).items():
        path = fig_dir / f"eigen_component_{name}.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x, y, c=values, s=8, cmap="viridis")
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel("g1")
        ax.set_ylabel("g2")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"eigen_component_{name}"] = str(path)
    return paths


def _plot_eigenfunction_heatmaps(module: DeepKoopmanLightningModule, sampled: dict[str, np.ndarray], fig_dir: Path) -> dict[str, str]:
    all_points = np.concatenate([traj.reshape(-1, traj.shape[-1]) for traj in sampled.values()], axis=0)
    if all_points.shape[1] > 2:
        return {}
    latent = _encode_array(module, all_points)
    x = all_points[:, 0]
    y = all_points[:, 1] if all_points.shape[1] > 1 else np.zeros(all_points.shape[0])
    paths = {}
    for idx in range(latent.shape[1]):
        path = fig_dir / f"eigenfunction_{idx + 1}_heatmap.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x, y, c=latent[:, idx], s=8, cmap="coolwarm")
        fig.colorbar(sc, ax=ax, label=f"g{idx + 1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Eigenfunction g{idx + 1}")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"eigenfunction_{idx + 1}_heatmap"] = str(path)
    return paths


def run_postprocess(
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    dataset: str | None = None,
    output_dir: str | Path | None = None,
    samples_per_split: int = 3,
    seed: int = 42,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    out_dir = Path(output_dir) if output_dir is not None else run_dir / "postprocess"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ckpt = run_dir / "best_checkpoint.ckpt"
    if not ckpt.exists():
        candidates = sorted(run_dir.glob("**/*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found under {run_dir}")
        ckpt = candidates[0]

    module = DeepKoopmanLightningModule.load_checkpoint(ckpt)
    cfg = module.config
    dataset = dataset or cfg.data.name
    splits = load_split_data(data_dir, dataset, cfg.data.train_files)

    test_metrics = evaluate_data(module, splits["test"])
    (table_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    _write_single_row_csv(table_dir / "test_metrics.csv", test_metrics)

    sampled, sample_rows = sample_trajectories(splits, cfg.data.len_time, samples_per_split, seed)
    _write_rows(table_dir / "sampled_trajectories.csv", sample_rows)

    figures = {}
    figures.update(_plot_data_trajectories(sampled, fig_dir, cfg.data.delta_t))
    figures.update(_plot_latent_true_vs_pred(module, sampled, fig_dir))
    figures.update(_plot_eigen_components(module, sampled, fig_dir))
    figures.update(_plot_eigenfunction_heatmaps(module, sampled, fig_dir))

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "dataset": dataset,
        "output_dir": str(out_dir),
        "tables": {
            "test_metrics_json": str(table_dir / "test_metrics.json"),
            "test_metrics_csv": str(table_dir / "test_metrics.csv"),
            "sampled_trajectories": str(table_dir / "sampled_trajectories.csv"),
        },
        "figures": figures,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
