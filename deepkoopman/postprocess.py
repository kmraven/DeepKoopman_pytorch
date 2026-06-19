from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import WindowedTrajectoryDataset, read_trajectories, trajectory_count
from .io import load_split_data
from .io import TrajectoryData
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


def _as_trajectories(data: TrajectoryData, len_time: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        return read_trajectories(data, np.arange(trajectory_count(data, len_time)), len_time)
    if data.ndim == 1:
        data = data[:, None]
    usable = (data.shape[0] // len_time) * len_time
    return data[:usable].reshape(usable // len_time, len_time, data.shape[1])


def evaluate_data(module: DeepKoopmanLightningModule, data: TrajectoryData) -> dict[str, float]:
    cfg = module.config
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    loader = torch.utils.data.DataLoader(
        WindowedTrajectoryDataset(data, cfg),
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
    )
    totals: dict[str, float] = {}
    total_examples = 0
    module.model.eval()
    with torch.no_grad():
        for batch in loader:
            stacked = module._prepare_batch(batch.to(module.device, dtype=dtype))
            raw = compute_losses(module.model, stacked, cfg)
            weight = int(stacked.shape[1])
            for key, value in raw.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * weight
            total_examples += weight
    metrics = {LOSS_NAMES[key]: value / total_examples for key, value in totals.items()}
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
    splits: dict[str, TrajectoryData],
    len_time: int,
    samples_per_split: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, int | str]]]:
    rng = np.random.default_rng(seed)
    sampled: dict[str, np.ndarray] = {}
    rows: list[dict[str, int | str]] = []
    for split, data in splits.items():
        num_trajectories = trajectory_count(data, len_time)
        indices = _sample_indices(num_trajectories, samples_per_split, rng)
        sampled[split] = read_trajectories(data, indices, len_time)
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


def _read_rat_metadata(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_split.setdefault(row["split"], []).append(row)
    return {
        (split, index): row
        for split, rows in rows_by_split.items()
        for index, row in enumerate(rows)
    }


def _rat_latent_rows(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    sample_rows: list[dict[str, int | str]],
    metadata_path: Path,
) -> list[dict[str, int | float | str]]:
    metadata = _read_rat_metadata(metadata_path)
    rows: list[dict[str, int | float | str]] = []
    for sample in sample_rows:
        split = str(sample["split"])
        sample_order = int(sample["sample_order"])
        trajectory_index = int(sample["trajectory_index"])
        info = metadata.get((split, trajectory_index))
        if info is None:
            continue
        latent = _encode_array(module, sampled[split][sample_order, :1])[0]
        row: dict[str, int | float | str] = {
            "split": split,
            "sample_order": sample_order,
            "trajectory_index": trajectory_index,
            "rat_id": info["rat_id"],
            "music_type": info["music_type"],
            "time_point": info["time_point"],
            "condition": f"{info['music_type']}_{info['time_point']}",
        }
        for idx, value in enumerate(latent):
            row[f"z{idx}"] = float(value)
        rows.append(row)
    return rows


def _plot_rat_latent_group(
    rows: list[dict[str, int | float | str]],
    out_path: Path,
    *,
    group_key: str,
    title: str,
) -> None:
    if not rows:
        return
    latent_cols = sorted([key for key in rows[0] if key.startswith("z")], key=lambda value: int(value[1:]))
    if len(latent_cols) < 2:
        return
    groups = sorted({str(row[group_key]) for row in rows})
    cmap = plt.get_cmap("tab20", max(len(groups), 1))
    color_by_group = {group: cmap(idx % cmap.N) for idx, group in enumerate(groups)}

    fig = plt.figure(figsize=(8, 6))
    if len(latent_cols) >= 3:
        ax = fig.add_subplot(111, projection="3d")
        for group in groups:
            values = [row for row in rows if row[group_key] == group]
            ax.scatter(
                [float(row["z0"]) for row in values],
                [float(row["z1"]) for row in values],
                [float(row["z2"]) for row in values],
                s=12,
                alpha=0.72,
                label=group,
                color=color_by_group[group],
            )
        ax.set_zlabel("z2")
    else:
        ax = fig.add_subplot(111)
        for group in groups:
            values = [row for row in rows if row[group_key] == group]
            ax.scatter(
                [float(row["z0"]) for row in values],
                [float(row["z1"]) for row in values],
                s=12,
                alpha=0.72,
                label=group,
                color=color_by_group[group],
            )
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_rat_latents(rows: list[dict[str, int | float | str]], fig_dir: Path) -> dict[str, str]:
    specs = [
        ("condition", rows, "rat_latent_by_condition_all.png", "Rat latent by music/time condition"),
        ("condition", [row for row in rows if row["split"] == "test"], "rat_latent_by_condition_test.png", "Rat test latent by music/time condition"),
        ("rat_id", rows, "rat_latent_by_rat_all.png", "Rat latent by individual"),
        ("rat_id", [row for row in rows if row["split"] == "test"], "rat_latent_by_rat_test.png", "Rat test latent by individual"),
    ]
    paths: dict[str, str] = {}
    for group_key, selected, filename, title in specs:
        if not selected:
            continue
        path = fig_dir / filename
        _plot_rat_latent_group(selected, path, group_key=group_key, title=title)
        if path.exists():
            paths[path.stem] = str(path)
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


def _latent_grid(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    grid_size: int,
    grid_dims: tuple[int, int],
    grid_min: tuple[float, float] | None,
    grid_max: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate([traj.reshape(-1, traj.shape[-1]) for traj in sampled.values()], axis=0)
    sampled_latent = _encode_array(module, all_points)
    latent_dim = sampled_latent.shape[1]
    if grid_size < 2:
        raise ValueError("latent_grid_size must be at least 2")
    if any(dim < 0 or dim >= latent_dim for dim in grid_dims):
        raise ValueError(f"latent_grid_dims={grid_dims} is out of bounds for latent_dim={latent_dim}")

    if grid_min is None or grid_max is None:
        values = sampled_latent[:, list(grid_dims)]
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        grid_min = (float(mins[0] - 0.05 * span[0]), float(mins[1] - 0.05 * span[1]))
        grid_max = (float(maxs[0] + 0.05 * span[0]), float(maxs[1] + 0.05 * span[1]))
    if grid_min[0] >= grid_max[0] or grid_min[1] >= grid_max[1]:
        raise ValueError("latent grid min values must be smaller than max values")

    x = np.linspace(grid_min[0], grid_max[0], grid_size)
    y = np.linspace(grid_min[1], grid_max[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    latent_grid = np.zeros((grid_size * grid_size, latent_dim), dtype=sampled_latent.dtype)
    latent_grid[:, grid_dims[0]] = xx.reshape(-1)
    latent_grid[:, grid_dims[1]] = yy.reshape(-1)
    return xx, yy, latent_grid, grid_min, grid_max


def _plot_eigen_component_heatmaps(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    fig_dir: Path,
    grid_size: int,
    grid_dims: tuple[int, int],
    grid_min: tuple[float, float] | None,
    grid_max: tuple[float, float] | None,
) -> tuple[dict[str, str], dict[str, object]]:
    xx, yy, latent_grid, resolved_min, resolved_max = _latent_grid(module, sampled, grid_size, grid_dims, grid_min, grid_max)
    paths = {}
    for name, values in _omega_components(module, latent_grid).items():
        path = fig_dir / f"eigen_component_{name}_heatmap.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        image = values.reshape(grid_size, grid_size)
        sc = ax.pcolormesh(xx, yy, image, shading="auto", cmap="viridis")
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_xlabel(f"g{grid_dims[0] + 1}")
        ax.set_ylabel(f"g{grid_dims[1] + 1}")
        ax.set_title(name)
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"eigen_component_{name}_heatmap"] = str(path)
    metadata = {
        "grid_size": grid_size,
        "grid_dims": [int(grid_dims[0]), int(grid_dims[1])],
        "grid_min": [float(resolved_min[0]), float(resolved_min[1])],
        "grid_max": [float(resolved_max[0]), float(resolved_max[1])],
    }
    return paths, metadata


def _state_grid(
    sampled: dict[str, np.ndarray],
    grid_size: int,
    grid_min: tuple[float, float] | None,
    grid_max: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float], tuple[float, float]]:
    all_points = np.concatenate([traj.reshape(-1, traj.shape[-1]) for traj in sampled.values()], axis=0)
    if all_points.shape[1] > 2:
        raise ValueError("state grid heatmaps only support data dimensions up to 2")
    if grid_size < 2:
        raise ValueError("state_grid_size must be at least 2")
    if grid_min is None or grid_max is None:
        values = np.column_stack(
            [
                all_points[:, 0],
                all_points[:, 1] if all_points.shape[1] > 1 else np.zeros(all_points.shape[0]),
            ]
        )
        mins = values.min(axis=0)
        maxs = values.max(axis=0)
        span = np.maximum(maxs - mins, 1e-6)
        grid_min = (float(mins[0] - 0.05 * span[0]), float(mins[1] - 0.05 * span[1]))
        grid_max = (float(maxs[0] + 0.05 * span[0]), float(maxs[1] + 0.05 * span[1]))
    if grid_min[0] >= grid_max[0] or grid_min[1] >= grid_max[1]:
        raise ValueError("state grid min values must be smaller than max values")

    x = np.linspace(grid_min[0], grid_max[0], grid_size)
    y = np.linspace(grid_min[1], grid_max[1], grid_size)
    xx, yy = np.meshgrid(x, y)
    if all_points.shape[1] == 1:
        state_grid = xx.reshape(-1, 1)
    else:
        state_grid = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1)
    return xx, yy, state_grid, grid_min, grid_max


def _plot_eigenfunction_heatmaps(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    fig_dir: Path,
    grid_size: int,
    grid_min: tuple[float, float] | None,
    grid_max: tuple[float, float] | None,
) -> tuple[dict[str, str], dict[str, object] | None]:
    all_points = np.concatenate([traj.reshape(-1, traj.shape[-1]) for traj in sampled.values()], axis=0)
    if all_points.shape[1] > 2:
        return {}, None
    xx, yy, state_grid, resolved_min, resolved_max = _state_grid(sampled, grid_size, grid_min, grid_max)
    latent = _encode_array(module, state_grid)
    paths = {}
    for idx in range(latent.shape[1]):
        path = fig_dir / f"eigenfunction_{idx + 1}_heatmap.png"
        fig, ax = plt.subplots(figsize=(6, 5))
        image = latent[:, idx].reshape(grid_size, grid_size)
        sc = ax.pcolormesh(xx, yy, image, shading="auto", cmap="coolwarm")
        fig.colorbar(sc, ax=ax, label=f"g{idx + 1}")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"Eigenfunction g{idx + 1}")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"eigenfunction_{idx + 1}_heatmap"] = str(path)
    metadata = {
        "grid_size": grid_size,
        "grid_min": [float(resolved_min[0]), float(resolved_min[1])],
        "grid_max": [float(resolved_max[0]), float(resolved_max[1])],
    }
    return paths, metadata


def run_postprocess(
    run_dir: str | Path,
    *,
    data_dir: str | Path = "data",
    dataset: str | None = None,
    output_dir: str | Path | None = None,
    samples_per_split: int = 3,
    seed: int = 42,
    latent_grid_size: int = 100,
    latent_grid_dims: tuple[int, int] = (0, 1),
    latent_grid_min: tuple[float, float] | None = None,
    latent_grid_max: tuple[float, float] | None = None,
    state_grid_size: int = 100,
    state_grid_min: tuple[float, float] | None = None,
    state_grid_max: tuple[float, float] | None = None,
    rat_metadata: str | Path | None = None,
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
    data_dir = Path(data_dir)

    test_metrics = evaluate_data(module, splits["test"])
    (table_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    _write_single_row_csv(table_dir / "test_metrics.csv", test_metrics)

    sampled, sample_rows = sample_trajectories(splits, cfg.data.len_time, samples_per_split, seed)
    _write_rows(table_dir / "sampled_trajectories.csv", sample_rows)

    figures = {}
    figures.update(_plot_data_trajectories(sampled, fig_dir, cfg.data.delta_t))
    figures.update(_plot_latent_true_vs_pred(module, sampled, fig_dir))

    tables = {
        "test_metrics_json": str(table_dir / "test_metrics.json"),
        "test_metrics_csv": str(table_dir / "test_metrics.csv"),
        "sampled_trajectories": str(table_dir / "sampled_trajectories.csv"),
    }

    rat_metadata_path = Path(rat_metadata) if rat_metadata is not None else data_dir / f"{dataset}_window_metadata.csv"
    if rat_metadata_path.exists():
        rat_rows = _rat_latent_rows(module, sampled, sample_rows, rat_metadata_path)
        rat_table = table_dir / "rat_latent_samples.csv"
        _write_rows(rat_table, rat_rows)
        tables["rat_latent_samples"] = str(rat_table)
        figures.update(_plot_rat_latents(rat_rows, fig_dir))

    eigen_figures, eigen_grid = _plot_eigen_component_heatmaps(
        module,
        sampled,
        fig_dir,
        latent_grid_size,
        latent_grid_dims,
        latent_grid_min,
        latent_grid_max,
    )
    figures.update(eigen_figures)
    eigenfunction_figures, state_grid = _plot_eigenfunction_heatmaps(
        module,
        sampled,
        fig_dir,
        state_grid_size,
        state_grid_min,
        state_grid_max,
    )
    figures.update(eigenfunction_figures)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt),
        "dataset": dataset,
        "output_dir": str(out_dir),
        "tables": tables,
        "figures": figures,
        "eigen_component_grid": eigen_grid,
        "eigenfunction_state_grid": state_grid,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
