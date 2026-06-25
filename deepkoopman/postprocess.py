from __future__ import annotations

import csv
import json
from itertools import combinations, product
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
import numpy as np
import torch
from tqdm.auto import tqdm

from .data import WindowedTrajectoryDataset, read_conditions, read_trajectories, trajectory_count
from .io import load_split_data
from .io import TrajectoryData
from .lightning import DeepKoopmanLightningModule
from .losses import compute_losses


LOSS_NAMES = {
    "loss": "loss",
    "loss1": "reconstruction",
    "loss2": "prediction",
    "loss3": "latent_consistency",
    "loss_cov": "latent_covariance",
    "loss_linf": "linf",
    "loss_l1": "l1",
    "loss_l2": "l2",
}

MUSIC_TYPE_LABELS = {
    "gamma": "gamma_music",
    "conventional": "normal_music",
    "control": "gamma_click",
}
MUSIC_PERIOD_GROUPS = [
    "pre",
    "during-gamma_music",
    "during-normal_music",
    "during-gamma_click",
    "post-gamma_music",
    "post-normal_music",
    "post-gamma_click",
]
BAND_LABELS = ["delta", "theta", "alpha", "beta", "low gamma", "high gamma"]


def _shared_latent_limits(arrays: list[np.ndarray], *, padding: float = 0.05) -> tuple[float, float]:
    """Return one padded range shared by every latent coordinate in ``arrays``."""
    finite_arrays = [np.asarray(array)[np.isfinite(array)] for array in arrays]
    finite_arrays = [array for array in finite_arrays if array.size]
    if not finite_arrays:
        raise ValueError("Cannot determine latent limits without finite values")
    minimum = min(float(array.min()) for array in finite_arrays)
    maximum = max(float(array.max()) for array in finite_arrays)
    span = maximum - minimum
    if span < 1e-4:
        span = max(abs(minimum), abs(maximum), 1.0) * 1e-4
    return minimum - padding * span, maximum + padding * span


def _resolve_device(device: str | torch.device) -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for postprocessing, but torch.cuda.is_available() is false")
    return resolved


def _resolve_postprocess_checkpoint(run_dir: Path, checkpoint: str | Path) -> tuple[Path, str]:
    value = str(checkpoint)
    if value == "best":
        path = run_dir / "best_checkpoint.ckpt"
        if not path.exists():
            candidates = sorted(run_dir.glob("**/*.ckpt"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint found under {run_dir}")
            path = candidates[0]
        return path, "best"
    if value == "last":
        path = run_dir / "last.ckpt"
        if not path.exists():
            raise FileNotFoundError(f"Last checkpoint not found: {path}")
        return path, "last"

    path = Path(checkpoint)
    if not path.is_absolute() and not path.exists():
        path = run_dir / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    label = "last" if path.name == "last.ckpt" else "best" if path.name == "best_checkpoint.ckpt" else path.stem
    return path, label


def _as_trajectories(data: TrajectoryData, len_time: int) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        return read_trajectories(data, np.arange(trajectory_count(data, len_time)), len_time)
    if data.ndim == 1:
        data = data[:, None]
    usable = (data.shape[0] // len_time) * len_time
    return data[:usable].reshape(usable // len_time, len_time, data.shape[1])


def evaluate_data(
    module: DeepKoopmanLightningModule,
    data: TrajectoryData,
    *,
    show_progress: bool = True,
) -> dict[str, float]:
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
        for batch in tqdm(loader, desc="test metrics", unit="batch", disable=not show_progress):
            if isinstance(batch, (list, tuple)):
                moved = []
                for item in batch:
                    if torch.is_floating_point(item):
                        moved.append(item.to(module.device, dtype=dtype))
                    else:
                        moved.append(item.to(module.device))
                prepared = module._prepare_batch(tuple(moved))
            else:
                prepared = module._prepare_batch(batch.to(module.device, dtype=dtype))
            if isinstance(prepared, tuple):
                stacked, conditions = prepared
            else:
                stacked, conditions = prepared, None
            raw = compute_losses(module.model, stacked, cfg, conditions)
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


def sample_conditions(
    splits: dict[str, TrajectoryData],
    sample_rows: list[dict[str, int | str]],
    len_time: int,
) -> dict[str, np.ndarray]:
    sampled: dict[str, np.ndarray] = {}
    for split, data in splits.items():
        selected = [row for row in sample_rows if row["split"] == split]
        indices = np.asarray([int(row["trajectory_index"]) for row in selected], dtype=np.int64)
        conditions = read_conditions(data, indices, len_time)
        if conditions is not None:
            sampled[split] = conditions
    return sampled


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


def _predict_latent_array(
    module: DeepKoopmanLightningModule,
    x0: np.ndarray,
    steps: int,
    conditions: np.ndarray | None = None,
) -> np.ndarray:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(x0, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        g0 = module.model.encode(x)
        if getattr(module.model, "is_conditioned", False):
            condition_tensor = None
            if conditions is not None:
                condition_values = np.asarray(conditions)
                if condition_values.ndim == 1:
                    condition_values = condition_values[:steps, None]
                elif condition_values.ndim == 2:
                    condition_values = condition_values[:, :steps].T
                else:
                    raise ValueError(f"Expected conditions to be 1-D or 2-D, got {condition_values.shape}")
                condition_tensor = torch.as_tensor(condition_values, dtype=torch.long, device=module.device)
            latents = module.model.predict_latent(g0, steps, condition_tensor)
        else:
            latents = module.model.predict_latent(g0, steps)
    return torch.stack(latents, dim=0).detach().cpu().numpy()


def _reconstruct_array(module: DeepKoopmanLightningModule, values: np.ndarray) -> np.ndarray:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(values, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        return module.model.reconstruct(x).detach().cpu().numpy()


def _predict_state_array(
    module: DeepKoopmanLightningModule,
    x0: np.ndarray,
    steps: int,
    conditions: np.ndarray | None = None,
) -> np.ndarray:
    latent = _predict_latent_array(module, x0, steps, conditions)[-1]
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    z = torch.as_tensor(latent, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        return module.model.decoder(z).detach().cpu().numpy()


def _plot_spatial_reconstruction_comparison(
    observed: np.ndarray,
    reconstructed: np.ndarray,
    predicted: np.ndarray,
    path: Path,
    *,
    input_shape: tuple[int, int, int],
    title: str,
) -> None:
    observed_map = observed.reshape(input_shape)
    reconstructed_map = reconstructed.reshape(input_shape)
    predicted_map = predicted.reshape(input_shape)
    reconstruction_error = np.abs(reconstructed_map - observed_map)
    prediction_error = np.abs(predicted_map - observed_map)
    value_limit = max(
        float(np.max(np.abs(observed_map))),
        float(np.max(np.abs(reconstructed_map))),
        float(np.max(np.abs(predicted_map))),
        1e-6,
    )
    error_limit = max(float(np.max(reconstruction_error)), float(np.max(prediction_error)), 1e-6)

    fig = plt.figure(figsize=(15, 10), layout="constrained")
    grid = fig.add_gridspec(5, len(BAND_LABELS) + 2, width_ratios=[1] * len(BAND_LABELS) + [0.06, 0.06])
    axes = np.empty((5, len(BAND_LABELS)), dtype=object)
    row_labels = [
        "Observed target",
        "AE reconstruction",
        "AE absolute error",
        "Koopman prediction",
        "Prediction absolute error",
    ]
    value_image = None
    error_image = None
    for row_index in range(5):
        for band_index, band_label in enumerate(BAND_LABELS):
            ax = fig.add_subplot(grid[row_index, band_index])
            axes[row_index, band_index] = ax
            if row_index == 0:
                values = observed_map[:, :, band_index]
            elif row_index == 1:
                values = reconstructed_map[:, :, band_index]
            elif row_index == 2:
                values = reconstruction_error[:, :, band_index]
            elif row_index == 3:
                values = predicted_map[:, :, band_index]
            else:
                values = prediction_error[:, :, band_index]
            if row_index in {0, 1, 3}:
                value_image = ax.imshow(
                    values,
                    cmap="RdBu_r",
                    vmin=-value_limit,
                    vmax=value_limit,
                    interpolation="nearest",
                )
            else:
                error_image = ax.imshow(
                    values,
                    cmap="magma",
                    vmin=0.0,
                    vmax=error_limit,
                    interpolation="nearest",
                )
            if row_index == 0:
                ax.set_title(band_label, fontsize=10)
            if band_index == 0:
                ax.set_ylabel(row_labels[row_index], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    value_cax = fig.add_subplot(grid[:, -2])
    error_cax = fig.add_subplot(grid[:, -1])
    if value_image is not None:
        fig.colorbar(value_image, cax=value_cax, label="standardized log-bandpower")
    if error_image is not None:
        fig.colorbar(error_image, cax=error_cax, label="absolute error")
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_rat_reconstruction_heatmaps(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    sample_rows: list[dict[str, int | str]],
    fig_dir: Path,
    sampled_conditions: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, str], list[dict[str, int | float | str]]]:
    shape = module.config.data.input_shape
    if shape is None or len(shape) != 3 or tuple(shape) != (8, 8, 6):
        return {}, []
    feature_dims = {int(trajectories.shape[-1]) for trajectories in sampled.values()}
    if feature_dims != {int(np.prod(shape))}:
        return {}, []
    horizon = min(max(module.config.data.shifts), module.config.data.len_time - 1)
    paths: dict[str, str] = {}
    manifest: list[dict[str, int | float | str]] = []
    for split, trajectories in sampled.items():
        conditions = None if sampled_conditions is None else sampled_conditions.get(split)
        selected_rows = sorted(
            [row for row in sample_rows if row["split"] == split], key=lambda row: int(row["sample_order"])
        )
        selected_trajectories = trajectories[:2]
        observed_batch = selected_trajectories[:, horizon]
        reconstructed_batch = _reconstruct_array(module, observed_batch)
        selected_conditions = None if conditions is None else conditions[: len(selected_trajectories)]
        predicted_batch = _predict_state_array(
            module,
            selected_trajectories[:, :1].reshape(len(selected_trajectories), -1),
            horizon,
            selected_conditions,
        )
        for sample_index, trajectory in enumerate(selected_trajectories):
            observed = trajectory[horizon]
            reconstructed = reconstructed_batch[sample_index]
            predicted = predicted_batch[sample_index]
            path = fig_dir / f"{split}_sample_{sample_index + 1}_reconstruction_prediction_heatmap.png"
            horizon_seconds = horizon * module.config.data.delta_t
            _plot_spatial_reconstruction_comparison(
                observed,
                reconstructed,
                predicted,
                path,
                input_shape=(8, 8, 6),
                title=(
                    f"{split} sample {sample_index + 1}: observed vs AE reconstruction vs "
                    f"Koopman prediction at h={horizon} ({horizon_seconds:g} s)"
                ),
            )
            key = f"{split}_sample_{sample_index + 1}_reconstruction_prediction_heatmap"
            paths[key] = str(path)
            source = selected_rows[sample_index] if sample_index < len(selected_rows) else {}
            manifest.append(
                {
                    "split": split,
                    "sample_order": sample_index,
                    "trajectory_index": int(source.get("trajectory_index", sample_index)),
                    "horizon_steps": horizon,
                    "horizon_seconds": horizon_seconds,
                    "autoencoder_mse": float(np.mean((reconstructed - observed) ** 2)),
                    "prediction_mse": float(np.mean((predicted - observed) ** 2)),
                }
            )
    return paths, manifest


def _sample_music_period_trajectories(
    splits: dict[str, TrajectoryData],
    metadata_path: Path,
    len_time: int,
    seed: int,
) -> tuple[dict[tuple[str, str], dict[str, object]], list[dict[str, int | float | str]]]:
    metadata = _metadata_rows_by_split(metadata_path)
    rng = np.random.default_rng(seed)
    selected: dict[tuple[str, str], dict[str, object]] = {}
    manifest: list[dict[str, int | float | str]] = []
    missing: list[str] = []
    for split in ("train", "val", "test"):
        for group in MUSIC_PERIOD_GROUPS:
            candidates = [
                index
                for (row_split, index), row in metadata.items()
                if row_split == split and _music_period_group(row) == group
            ]
            if not candidates:
                missing.append(f"{split}:{group}")
                continue
            index = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
            trajectory = read_trajectories(splits[split], np.asarray([index]), len_time)[0]
            condition_values = read_conditions(splits[split], np.asarray([index]), len_time)
            info = metadata[(split, index)]
            if condition_values is None:
                condition_id = int(info.get("condition_id", 0) or 0)
                conditions = np.full(len_time, condition_id, dtype=np.int64)
            else:
                conditions = np.asarray(condition_values[0], dtype=np.int64)
            selected[(split, group)] = {
                "trajectory": trajectory,
                "conditions": conditions,
                "metadata": info,
                "trajectory_index": index,
            }
            manifest.append(
                {
                    "split": split,
                    "music_period": group,
                    "trajectory_index": index,
                    "rat_id": info.get("rat_id", ""),
                    "music_type": info.get("music_type", ""),
                    "section": info.get("section", ""),
                    "start_sec": float(info.get("start_sec", 0.0) or 0.0),
                }
            )
    if missing:
        raise ValueError(f"No trajectory candidates for required music-period groups: {', '.join(missing)}")
    return selected, manifest


def _prepare_trajectory_video_data(
    module: DeepKoopmanLightningModule,
    samples: dict[tuple[str, str], dict[str, object]],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    prepared: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    module.model.eval()
    for key, sample in samples.items():
        trajectory = np.asarray(sample["trajectory"])
        conditions = np.asarray(sample["conditions"], dtype=np.int64)
        true_latent = _encode_array(module, trajectory)
        predicted_latent = _predict_latent_array(
            module,
            trajectory[:1],
            trajectory.shape[0] - 1,
            conditions,
        )[:, 0, :]
        reconstructed = _reconstruct_array(module, trajectory)
        z = torch.as_tensor(predicted_latent, dtype=dtype, device=module.device)
        with torch.no_grad():
            predicted = module.model.decoder(z).detach().cpu().numpy()
        prepared[key] = {
            "observed": trajectory,
            "reconstructed": reconstructed,
            "predicted": predicted,
            "true_latent": true_latent,
            "predicted_latent": predicted_latent,
        }
    return prepared


def _write_reconstruction_video(
    values: dict[str, np.ndarray],
    path: Path,
    *,
    split: str,
    group: str,
    delta_t: float,
    fps: int,
) -> None:
    observed = values["observed"].reshape(-1, 8, 8, 6)
    reconstructed = values["reconstructed"].reshape(-1, 8, 8, 6)
    predicted = values["predicted"].reshape(-1, 8, 8, 6)
    value_limit = max(
        float(np.max(np.abs(observed))),
        float(np.max(np.abs(reconstructed))),
        float(np.max(np.abs(predicted))),
        1e-6,
    )
    fig = plt.figure(figsize=(14, 7), layout="constrained")
    grid = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.06])
    row_values = [observed, reconstructed, predicted]
    row_labels = ["Original data", "AE reconstruction", "Koopman prediction"]
    images = []
    for row_index, data in enumerate(row_values):
        row_images = []
        for band_index, band_label in enumerate(BAND_LABELS):
            ax = fig.add_subplot(grid[row_index, band_index])
            image = ax.imshow(
                data[0, :, :, band_index],
                cmap="RdBu_r",
                vmin=-value_limit,
                vmax=value_limit,
                interpolation="nearest",
            )
            row_images.append(image)
            if row_index == 0:
                ax.set_title(band_label, fontsize=10)
            if band_index == 0:
                ax.set_ylabel(row_labels[row_index], fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        images.append(row_images)
    color_ax = fig.add_subplot(grid[:, -1])
    fig.colorbar(images[0][0], cax=color_ax, label="standardized log-bandpower")
    title = fig.suptitle("")
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"title": f"{split} {group} reconstruction"},
        extra_args=["-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(path), dpi=100):
        for frame in range(observed.shape[0]):
            for row_index, data in enumerate(row_values):
                for band_index in range(6):
                    images[row_index][band_index].set_data(data[frame, :, :, band_index])
            title.set_text(f"{split} / {group} — frame {frame + 1}/{observed.shape[0]} ({frame * delta_t:g} s)")
            writer.grab_frame()
    plt.close(fig)


def _write_latent_trajectory_grid_video(
    prepared: dict[tuple[str, str], dict[str, np.ndarray]],
    path: Path,
    *,
    delta_t: float,
    fps: int,
    include_prediction: bool = False,
) -> None:
    fig = plt.figure(figsize=(28, 12))
    lines: list[tuple[object, object | None, np.ndarray, np.ndarray]] = []
    frame_count = min(values["true_latent"].shape[0] for values in prepared.values())
    limit_arrays = [values["true_latent"] for values in prepared.values()]
    if include_prediction:
        limit_arrays.extend(values["predicted_latent"] for values in prepared.values())
    latent_min, latent_max = _shared_latent_limits(limit_arrays)
    for row_index, split in enumerate(("train", "val", "test")):
        for col_index, group in enumerate(MUSIC_PERIOD_GROUPS):
            ax = fig.add_subplot(3, 7, row_index * 7 + col_index + 1, projection="3d")
            values = prepared[(split, group)]
            true_latent = values["true_latent"]
            predicted_latent = values["predicted_latent"]
            ax.set_xlim(latent_min, latent_max)
            ax.set_ylim(latent_min, latent_max)
            ax.set_zlim(latent_min, latent_max)
            true_line, = ax.plot([], [], [], color="tab:blue", linewidth=1.4)
            predicted_line = None
            if include_prediction:
                predicted_line, = ax.plot([], [], [], color="tab:red", linestyle="--", linewidth=1.2)
            lines.append((true_line, predicted_line, true_latent, predicted_latent))
            ax.set_title(f"{split}\n{group}", fontsize=8)
            ax.set_xlabel("g1", fontsize=7)
            ax.set_ylabel("g2", fontsize=7)
            ax.set_zlabel("g3", fontsize=7)
            ax.tick_params(labelsize=5, pad=0)
    legend_handles = [Line2D([0], [0], color="tab:blue", linewidth=1.4, label="Encoded data")]
    if include_prediction:
        legend_handles.append(
            Line2D([0], [0], color="tab:red", linestyle="--", linewidth=1.2, label="Koopman prediction")
        )
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9)
    title = fig.suptitle("")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.91, wspace=0.02, hspace=0.18)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={
            "title": (
                "Latent prediction by split and music-period"
                if include_prediction
                else "Observed latent trajectories by split and music-period"
            )
        },
        extra_args=["-pix_fmt", "yuv420p"],
    )
    with writer.saving(fig, str(path), dpi=80):
        for frame in range(frame_count):
            end = frame + 1
            for true_line, predicted_line, true_latent, predicted_latent in lines:
                true_line.set_data_3d(true_latent[:end, 0], true_latent[:end, 1], true_latent[:end, 2])
                if predicted_line is not None:
                    predicted_line.set_data_3d(
                        predicted_latent[:end, 0], predicted_latent[:end, 1], predicted_latent[:end, 2]
                    )
            prefix = "Latent prediction" if include_prediction else "Observed latent trajectories"
            title.set_text(f"{prefix} — frame {end}/{frame_count} ({frame * delta_t:g} s)")
            writer.grab_frame()
    plt.close(fig)


def _make_music_period_videos(
    module: DeepKoopmanLightningModule,
    splits: dict[str, TrajectoryData],
    metadata_path: Path,
    out_dir: Path,
    *,
    seed: int,
    fps: int,
    show_progress: bool,
) -> tuple[dict[str, str], list[dict[str, int | float | str]]]:
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required to generate postprocess videos")
    samples, manifest = _sample_music_period_trajectories(
        splits,
        metadata_path,
        module.config.data.len_time,
        seed,
    )
    prepared = _prepare_trajectory_video_data(module, samples)
    paths: dict[str, str] = {}
    iterator = tqdm(
        [(split, group) for split in ("train", "val", "test") for group in MUSIC_PERIOD_GROUPS],
        desc="reconstruction videos",
        unit="video",
        disable=not show_progress,
    )
    for split, group in iterator:
        path = out_dir / "reconstruction" / split / f"{group}.mp4"
        _write_reconstruction_video(
            prepared[(split, group)],
            path,
            split=split,
            group=group,
            delta_t=module.config.data.delta_t,
            fps=fps,
        )
        paths[f"reconstruction_{split}_{group}"] = str(path)
    latent_path = out_dir / "latent" / "latent_trajectory_grid.mp4"
    _write_latent_trajectory_grid_video(
        prepared,
        latent_path,
        delta_t=module.config.data.delta_t,
        fps=fps,
        include_prediction=False,
    )
    paths["latent_trajectory_grid"] = str(latent_path)
    prediction_path = out_dir / "latent" / "latent_prediction_grid.mp4"
    _write_latent_trajectory_grid_video(
        prepared,
        prediction_path,
        delta_t=module.config.data.delta_t,
        fps=fps,
        include_prediction=True,
    )
    paths["latent_prediction_grid"] = str(prediction_path)
    for row in manifest:
        key = (str(row["split"]), str(row["music_period"]))
        values = prepared[key]
        row["autoencoder_mse_all_frames"] = float(
            np.mean((values["reconstructed"] - values["observed"]) ** 2)
        )
        row["prediction_mse_all_frames"] = float(np.mean((values["predicted"] - values["observed"]) ** 2))
    return paths, manifest


def _plot_latent_true_vs_pred(
    module: DeepKoopmanLightningModule,
    sampled: dict[str, np.ndarray],
    fig_dir: Path,
    sampled_conditions: dict[str, np.ndarray] | None = None,
) -> dict[str, str]:
    paths = {}
    prepared: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for split, trajectories in sampled.items():
        conditions = None if sampled_conditions is None else sampled_conditions.get(split)
        prepared[split] = []
        for idx, traj in enumerate(trajectories):
            steps = min(traj.shape[0] - 1, 64)
            true_latent = _encode_array(module, traj[: steps + 1])
            cond = None if conditions is None else conditions[idx]
            pred_latent = _predict_latent_array(module, traj[:1], steps, cond)[:, 0, :]
            prepared[split].append((true_latent, pred_latent))
    latent_min, latent_max = _shared_latent_limits(
        [latent for pairs in prepared.values() for pair in pairs for latent in pair]
    )

    for split, trajectories in sampled.items():
        path = fig_dir / f"{split}_latent_true_vs_pred.png"
        first = prepared[split][0][0]
        latent_dim = first.shape[1]
        fig = plt.figure(figsize=(7, 5))
        colors = plt.get_cmap("tab10")
        if latent_dim >= 3:
            ax = fig.add_subplot(111, projection="3d")
            for idx, (true_latent, pred_latent) in enumerate(prepared[split]):
                color = colors(idx % 10)
                ax.plot(true_latent[:, 0], true_latent[:, 1], true_latent[:, 2], linewidth=1.5, color=color)
                ax.plot(pred_latent[:, 0], pred_latent[:, 1], pred_latent[:, 2], linestyle="--", linewidth=1.2, color=color)
            ax.set_xlabel("g1")
            ax.set_ylabel("g2")
            ax.set_zlabel("g3")
            ax.set_xlim(latent_min, latent_max)
            ax.set_ylim(latent_min, latent_max)
            ax.set_zlim(latent_min, latent_max)
        elif latent_dim == 2:
            ax = fig.add_subplot(111)
            for idx, (true_latent, pred_latent) in enumerate(prepared[split]):
                color = colors(idx % 10)
                ax.plot(true_latent[:, 0], true_latent[:, 1], linewidth=1.5, color=color)
                ax.plot(pred_latent[:, 0], pred_latent[:, 1], linestyle="--", linewidth=1.2, color=color)
            ax.set_xlabel("g1")
            ax.set_ylabel("g2")
            ax.set_xlim(latent_min, latent_max)
            ax.set_ylim(latent_min, latent_max)
        else:
            ax = fig.add_subplot(111)
            for idx, (true_latent, pred_latent) in enumerate(prepared[split]):
                steps = np.arange(true_latent.shape[0])
                color = colors(idx % 10)
                ax.plot(steps, true_latent[:, 0], linewidth=1.5, color=color)
                ax.plot(steps, pred_latent[:, 0], linestyle="--", linewidth=1.2, color=color)
            ax.set_xlabel("step")
            ax.set_ylabel("g1")
            ax.set_ylim(latent_min, latent_max)
        ax.set_title(f"{split} latent true vs predicted")
        style_handles = [
            Line2D([0], [0], color="black", linewidth=1.5, label="Encoded observation"),
            Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label="Koopman rollout"),
        ]
        sample_handles = [
            Line2D([0], [0], color=colors(idx % 10), linewidth=2, label=f"Trajectory {idx + 1}")
            for idx in range(len(trajectories))
        ]
        first_legend = ax.legend(handles=style_handles, title="Line style", fontsize=7, loc="upper left")
        ax.add_artist(first_legend)
        ax.legend(handles=sample_handles, title="Sample", fontsize=7, loc="upper right")
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
        latents = _encode_array(module, sampled[split][sample_order])
        for time_step, latent in enumerate(latents):
            row: dict[str, int | float | str] = {
                "split": split,
                "sample_order": sample_order,
                "trajectory_index": trajectory_index,
                "time_step": time_step,
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
    trajectory_count = len({(str(row["split"]), int(row["sample_order"])) for row in rows})
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
            trajectories = sorted({(str(row["split"]), int(row["sample_order"])) for row in values})
            for split, sample_order in trajectories:
                trajectory = sorted(
                    [row for row in values if row["split"] == split and int(row["sample_order"]) == sample_order],
                    key=lambda row: int(row["time_step"]),
                )
                ax.plot(
                    [float(row["z0"]) for row in trajectory],
                    [float(row["z1"]) for row in trajectory],
                    [float(row["z2"]) for row in trajectory],
                    linewidth=0.7,
                    alpha=0.45,
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
    ax.set_title(f"{title}\n{len(rows)} time points from {trajectory_count} sampled trajectories")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_rat_latents(rows: list[dict[str, int | float | str]], fig_dir: Path) -> dict[str, str]:
    specs = [
        ("condition", rows, "rat_latent_by_condition_all.png", "Sampled latent trajectories by music/time condition"),
        ("condition", [row for row in rows if row["split"] == "test"], "rat_latent_by_condition_test.png", "Sampled test latent trajectories by music/time condition"),
        ("rat_id", rows, "rat_latent_by_rat_all.png", "Sampled latent trajectories by rat (all splits)"),
        ("rat_id", [row for row in rows if row["split"] == "test"], "rat_latent_by_rat_test.png", "Sampled test latent trajectories by rat"),
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


def _condition_names(module: DeepKoopmanLightningModule) -> list[str]:
    names = list(module.config.data.condition_names)
    if not names:
        names = ["silence", "normal_music", "gamma_music", "gamma_click"]
    return names


def _metadata_rows_by_split(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_split.setdefault(row["split"], []).append(row)
    return {(split, idx): row for split, rows in rows_by_split.items() for idx, row in enumerate(rows)}


def _write_optional_parquet(csv_rows: list[dict[str, int | float | str]], path: Path) -> str | None:
    try:
        import pandas as pd

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(csv_rows).to_parquet(path, index=False)
        return str(path)
    except Exception:
        return None


def _rat_rollout_rows(
    module: DeepKoopmanLightningModule,
    data: TrajectoryData,
    metadata_path: Path,
    *,
    split: str = "test",
    chunk_size: int = 512,
    show_progress: bool = True,
) -> tuple[list[dict[str, int | float | str]], dict[str, np.ndarray]]:
    n = trajectory_count(data, module.config.data.len_time)
    metadata = _metadata_rows_by_split(metadata_path)
    condition_names = _condition_names(module)
    horizons = {2: "1s", 4: "2s", 10: "5s", 20: "10s"}
    max_horizon = max(horizons)
    rows: list[dict[str, int | float | str]] = []
    band_error_sum = np.zeros(6, dtype=np.float64)
    band_error_count = 0
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    module.model.eval()

    starts = range(0, n, chunk_size)
    for start in tqdm(starts, desc=f"{split} rat rollout", unit="chunk", disable=not show_progress):
        indices = np.arange(start, min(start + chunk_size, n), dtype=np.int64)
        trajectories = read_trajectories(data, indices, module.config.data.len_time)
        conditions = read_conditions(data, indices, module.config.data.len_time)
        if conditions is None:
            conditions = np.zeros((indices.size, module.config.data.len_time), dtype=np.int64)
            for local, index in enumerate(indices):
                info = metadata.get((split, int(index)))
                if info and info.get("condition_id", "") != "":
                    conditions[local, :] = int(info["condition_id"])

        x = torch.as_tensor(trajectories, dtype=dtype, device=module.device)
        cond = torch.as_tensor(conditions, dtype=torch.long, device=module.device)
        with torch.no_grad():
            x0 = x[:, 0]
            z = module.model.encode(x0)
            recon = module.model.decoder(z)
            params = module.model.eigen_params(z, cond[:, 0])
            rollout_conditions = cond[:, :max_horizon].T.contiguous()
            pred = module.model.predict(x0, max_horizon, rollout_conditions)

        z_np = z.detach().cpu().numpy()
        recon_np = recon.detach().cpu().numpy()
        params_np = params.detach().cpu().numpy()
        pred_np = pred.detach().cpu().numpy()
        x_np = trajectories
        rec_errors = ((recon_np - x_np[:, 0]) ** 2).mean(axis=1)
        band_errors = ((recon_np - x_np[:, 0]) ** 2).reshape(indices.size, 8, 8, 6).mean(axis=(1, 2))
        band_error_sum += band_errors.sum(axis=0)
        band_error_count += band_errors.shape[0]
        pred_errors = {
            label: ((pred_np[h] - x_np[:, h]) ** 2).mean(axis=1)
            for h, label in horizons.items()
            if h < x_np.shape[1]
        }
        lambda_r_mu = np.log(np.maximum(params_np[:, 0], 1e-12)) / module.config.data.delta_t
        mu = np.log(np.maximum(params_np[:, 1], 1e-12)) / module.config.data.delta_t
        frequency = params_np[:, 2] / (2 * np.pi * module.config.data.delta_t)

        for local, index in enumerate(indices):
            info = metadata.get((split, int(index)), {})
            cond_id = int(conditions[local, 0])
            row: dict[str, int | float | str] = {
                "split": split,
                "trajectory_index": int(index),
                "rat_id": info.get("rat_id", ""),
                "block_id": info.get("block_id", ""),
                "music_type": info.get("music_type", ""),
                "time_point": info.get("time_point", ""),
                "section": info.get("section", info.get("time_point", "")),
                "condition": info.get("condition", condition_names[cond_id] if cond_id < len(condition_names) else str(cond_id)),
                "condition_id": cond_id,
                "time_in_block": float(info.get("start_sec", 0.0) or 0.0),
                "global_window_id": int(info.get("global_window_id", index) or index),
                "z1": float(z_np[local, 0]),
                "z2": float(z_np[local, 1]),
                "z3": float(z_np[local, 2]),
                "lambda_r": float(params_np[local, 0]),
                "rho": float(params_np[local, 1]),
                "theta": float(params_np[local, 2]),
                "lambda_r_mu": float(lambda_r_mu[local]),
                "mu": float(mu[local]),
                "frequency": float(frequency[local]),
                "reconstruction_error": float(rec_errors[local]),
            }
            for label, values in pred_errors.items():
                row[f"prediction_error_{label}"] = float(values[local])
            rows.append(row)

    aggregates = {
        "reconstruction_error_by_band": band_error_sum / max(band_error_count, 1),
    }
    return rows, aggregates


def _sample_plot_rows(rows: list[dict[str, int | float | str]], limit: int = 5000) -> list[dict[str, int | float | str]]:
    if len(rows) <= limit:
        return rows
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(rows), size=limit, replace=False))
    return [rows[int(idx)] for idx in indices]


def _plot_latent_3d(rows: list[dict[str, int | float | str]], path: Path, *, group_key: str, title: str) -> None:
    rows = _sample_plot_rows(rows)
    if not rows:
        return
    groups = sorted({str(row[group_key]) for row in rows})
    cmap = plt.get_cmap("tab10", max(len(groups), 1))
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    for idx, group in enumerate(groups):
        selected = [row for row in rows if str(row[group_key]) == group]
        ax.scatter(
            [float(row["z1"]) for row in selected],
            [float(row["z2"]) for row in selected],
            [float(row["z3"]) for row in selected],
            s=10,
            alpha=0.65,
            label=group,
            color=cmap(idx % cmap.N),
        )
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")
    ax.set_zlabel("z3")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_true_latent_trajectories_by_split(
    rows_by_split: dict[str, list[dict[str, int | float | str]]],
    fig_dir: Path,
) -> dict[str, str]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    latent_min, latent_max = _shared_latent_limits(
        [
            np.asarray([[float(row["z1"]), float(row["z2"]), float(row["z3"])] for row in rows])
            for rows in rows_by_split.values()
        ]
    )
    for split in ("train", "val", "test"):
        rows = rows_by_split[split]
        fig = plt.figure(figsize=(28, 5.5))
        for group_index, group in enumerate(MUSIC_PERIOD_GROUPS):
            ax = fig.add_subplot(1, 7, group_index + 1, projection="3d")
            selected = [row for row in rows if _music_period_group(row) == group]
            blocks: dict[str, list[dict[str, int | float | str]]] = {}
            for row in selected:
                block_id = str(row.get("block_id", "")) or (
                    f"{row.get('rat_id', '')}:{row.get('music_type', '')}:{row.get('section', '')}"
                )
                blocks.setdefault(block_id, []).append(row)
            for trajectory in blocks.values():
                trajectory.sort(key=lambda row: (float(row.get("time_in_block", 0.0)), int(row["trajectory_index"])))
                ax.plot(
                    [float(row["z1"]) for row in trajectory],
                    [float(row["z2"]) for row in trajectory],
                    [float(row["z3"]) for row in trajectory],
                    color="tab:blue",
                    linewidth=0.55,
                    alpha=0.45,
                )
            rat_count = len({str(row["rat_id"]) for row in selected})
            ax.set_title(f"{group}\nN={rat_count} rats, {len(selected)} windows", fontsize=8)
            ax.set_xlabel("g1", fontsize=7)
            ax.set_ylabel("g2", fontsize=7)
            ax.set_zlabel("g3", fontsize=7)
            ax.set_xlim(latent_min, latent_max)
            ax.set_ylim(latent_min, latent_max)
            ax.set_zlim(latent_min, latent_max)
            ax.tick_params(labelsize=5, pad=0)
        fig.suptitle(f"{split}: observed latent trajectories (all available data)", fontsize=14)
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.05, top=0.86, wspace=0.02)
        path = fig_dir / f"{split}_latent_true_vs_pred.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths[path.stem] = str(path)
    return paths


def _rat_group_means(
    rows: list[dict[str, int | float | str]],
    metric: str,
    groups: list[str],
    *,
    group_key: str = "condition",
) -> dict[str, dict[str, float]]:
    values: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        group = _music_period_group(row) if group_key == "music_period" else str(row.get(group_key, ""))
        if group in groups and metric in row:
            values.setdefault((str(row["rat_id"]), group), []).append(float(row[metric]))
    return {
        group: {
            rat_id: float(np.mean(group_values))
            for (rat_id, selected_group), group_values in values.items()
            if selected_group == group
        }
        for group in groups
    }


def _music_period_group(row: dict[str, int | float | str]) -> str | None:
    section = str(row.get("section", ""))
    if section == "pre":
        return "pre"
    period = "during" if section in {"during1", "during2"} else "post" if section == "post" else None
    music = MUSIC_TYPE_LABELS.get(str(row.get("music_type", "")))
    if period is None or music is None:
        return None
    return f"{period}-{music}"


def _holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[int(index)]
        running = max(running, candidate)
        adjusted[int(index)] = min(running, 1.0)
    return adjusted.tolist()


def _paired_permutation_tests(
    grouped: dict[str, dict[str, float]],
    *,
    family: str,
    metric: str,
) -> list[dict[str, int | float | str]]:
    results: list[dict[str, int | float | str]] = []
    for group_a, group_b in combinations(grouped, 2):
        rats = sorted(set(grouped[group_a]) & set(grouped[group_b]))
        if len(rats) < 2:
            continue
        differences = np.asarray(
            [grouped[group_b][rat] - grouped[group_a][rat] for rat in rats], dtype=np.float64
        )
        observed = abs(float(differences.mean()))
        permuted = np.asarray(
            [abs(float(np.mean(differences * np.asarray(signs)))) for signs in product((-1.0, 1.0), repeat=len(rats))]
        )
        p_value = float(np.mean(permuted >= observed - 1e-15))
        results.append(
            {
                "family": family,
                "metric": metric,
                "group_a": group_a,
                "group_b": group_b,
                "n_rats": len(rats),
                "mean_difference_b_minus_a": float(differences.mean()),
                "p_value": p_value,
                "method": "two-sided exact paired sign-flip permutation",
            }
        )
    adjusted = _holm_adjust([float(result["p_value"]) for result in results])
    for result, p_adjusted in zip(results, adjusted):
        result["p_holm"] = p_adjusted
        difference = float(result["mean_difference_b_minus_a"])
        if difference > 0:
            result["relation"] = f"{result['group_a']} < {result['group_b']}"
        elif difference < 0:
            result["relation"] = f"{result['group_a']} > {result['group_b']}"
        else:
            result["relation"] = f"{result['group_a']} = {result['group_b']}"
        result["significant_holm_0.05"] = int(p_adjusted < 0.05)
    return results


def _plot_rat_level_box(
    ax: Axes,
    grouped: dict[str, dict[str, float]],
    groups: list[str],
    tests: list[dict[str, int | float | str]],
) -> None:
    labels = []
    letters = {group: chr(ord("A") + index) for index, group in enumerate(groups)}
    for group in groups:
        display_group = group.replace("-", "-\n")
        labels.append(f"{letters[group]}\n{display_group}\nN={len(grouped[group])}")
    ax.boxplot([[grouped[group][rat] for rat in sorted(grouped[group])] for group in groups], labels=labels, showfliers=False)
    rats = sorted(set().union(*(set(grouped[group]) for group in groups)))
    for rat in rats:
        positions = [index + 1 for index, group in enumerate(groups) if rat in grouped[group]]
        values = [grouped[group][rat] for group in groups if rat in grouped[group]]
        ax.plot(positions, values, color="0.65", linewidth=0.7, alpha=0.65, zorder=2)
        ax.scatter(positions, values, color="black", s=18, alpha=0.8, zorder=3)
    significant_tests = sorted(
        [test for test in tests if float(test["p_holm"]) < 0.05],
        key=lambda test: (float(test["p_holm"]), groups.index(str(test["group_b"])) - groups.index(str(test["group_a"]))),
    )
    if significant_tests:
        bottom, top = ax.get_ylim()
        span = max(top - bottom, 1e-8)
        base = top + 0.04 * span
        step = 0.09 * span
        occupied: list[list[tuple[int, int]]] = []
        for test in significant_tests:
            left = groups.index(str(test["group_a"]))
            right = groups.index(str(test["group_b"]))
            if left > right:
                left, right = right, left
            level = 0
            while level < len(occupied) and any(not (right < start or left > end) for start, end in occupied[level]):
                level += 1
            if level == len(occupied):
                occupied.append([])
            occupied[level].append((left, right))
            y = base + level * step
            ax.plot([left + 1, left + 1, right + 1, right + 1], [y, y + 0.02 * span, y + 0.02 * span, y], color="black", linewidth=0.8)
            difference = float(test["mean_difference_b_minus_a"])
            relation = (
                f"{letters[str(test['group_a'])]} < {letters[str(test['group_b'])]}"
                if difference > 0
                else f"{letters[str(test['group_a'])]} > {letters[str(test['group_b'])]}"
            )
            ax.text(
                (left + right) / 2 + 1,
                y + 0.024 * span,
                f"{relation}; pH={float(test['p_holm']):.3g}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
        ax.set_ylim(bottom, base + (len(occupied) + 0.45) * step)
    ax.text(
        0.5,
        -0.28,
        _rat_level_plot_detail(tests),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
    )


def _rat_level_plot_detail(tests: list[dict[str, int | float | str]]) -> str:
    significant = sum(float(test["p_holm"]) < 0.05 for test in tests)
    min_p = min((float(test["p_holm"]) for test in tests), default=float("nan"))
    detail = f"minimum Holm-adjusted p={min_p:.3g}" if tests else "insufficient paired rats"
    if tests and significant == 0:
        detail += "; no adjusted p < 0.05"
    return f"Rat-level means and paired observations. Exact sign-flip permutation; Holm correction; {detail}."


def _plot_metric_by_condition(
    rows: list[dict[str, int | float | str]],
    path: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> list[dict[str, int | float | str]]:
    grouped = _rat_group_means(rows, metric, MUSIC_PERIOD_GROUPS, group_key="music_period")
    groups = [group for group in MUSIC_PERIOD_GROUPS if grouped[group]]
    if not groups:
        return []
    grouped = {group: grouped[group] for group in groups}
    tests = _paired_permutation_tests(grouped, family=f"{metric}_by_music_period", metric=metric)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    _plot_rat_level_box(ax, grouped, groups, tests)
    bottom, top = ax.get_ylim()
    if bottom <= 0.0 <= top:
        ax.axhline(0.0, color="0.75", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return tests


def _plot_prediction_horizons(rows: list[dict[str, int | float | str]], path: Path) -> None:
    metrics = ["prediction_error_1s", "prediction_error_2s", "prediction_error_5s", "prediction_error_10s"]
    present = [metric for metric in metrics if any(metric in row for row in rows)]
    if not present:
        return
    grouped_by_metric = {
        metric: _rat_group_means(rows, metric, MUSIC_PERIOD_GROUPS, group_key="music_period")
        for metric in present
    }
    groups = [
        group
        for group in MUSIC_PERIOD_GROUPS
        if any(grouped_by_metric[metric][group] for metric in present)
    ]
    x = np.asarray([float(metric.removeprefix("prediction_error_").removesuffix("s")) for metric in present])
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = plt.get_cmap("tab10")
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    for idx, group in enumerate(groups):
        rat_ids = sorted(set.intersection(*(set(grouped_by_metric[metric][group]) for metric in present)))
        means = np.asarray([
            np.mean([grouped_by_metric[metric][group][rat] for rat in rat_ids]) for metric in present
        ])
        sem = np.asarray([
            np.std([grouped_by_metric[metric][group][rat] for rat in rat_ids], ddof=1) / np.sqrt(len(rat_ids))
            if len(rat_ids) > 1 else 0.0
            for metric in present
        ])
        ax.errorbar(
            x,
            means,
            yerr=sem,
            marker=markers[idx],
            linewidth=1.5,
            capsize=2,
            color=colors(idx % 10),
            label=f"{group} (N={len(rat_ids)})",
        )
    ax.set_xticks(x, [f"{value:g}s" for value in x])
    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("MSE")
    ax.set_title("Prediction error by music × period (rat mean ± SEM)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_reconstruction_by_band(values: np.ndarray, path: Path) -> None:
    labels = ["delta", "theta", "alpha", "beta", "low gamma", "high gamma"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(values)), values)
    ax.set_xticks(np.arange(len(values)), labels, rotation=20)
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction error by band")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_latent_displacement(
    rows: list[dict[str, int | float | str]], path: Path
) -> list[dict[str, int | float | str]]:
    grouped: dict[tuple[str, str, str], list[np.ndarray]] = {}
    for row in rows:
        key = (str(row["rat_id"]), str(row.get("music_type", "")), str(row.get("section", "")))
        grouped.setdefault(key, []).append(np.array([float(row["z1"]), float(row["z2"]), float(row["z3"])]))
    distances: dict[str, dict[str, float]] = {}
    music_to_condition = {"conventional": "normal_music", "gamma": "gamma_music", "control": "gamma_click"}
    rat_music = sorted({(rat_id, music_type) for rat_id, music_type, _ in grouped})
    for rat_id, music_type in rat_music:
        pre = grouped.get((rat_id, music_type, "pre"))
        during = grouped.get((rat_id, music_type, "during1"), []) + grouped.get((rat_id, music_type, "during2"), [])
        if not pre or not during:
            continue
        label = music_to_condition.get(music_type, music_type)
        pre_centroid = np.mean(pre, axis=0)
        during_centroid = np.mean(during, axis=0)
        distances.setdefault(label, {})[rat_id] = float(np.linalg.norm(during_centroid - pre_centroid))
    if not distances:
        return []
    labels = sorted(distances)
    tests = _paired_permutation_tests(
        distances,
        family="latent_displacement_by_condition",
        metric="latent_displacement_during_minus_pre",
    )
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _plot_rat_level_box(ax, distances, labels, tests)
    ax.set_ylabel("latent distance")
    ax.set_title("During-pre latent displacement")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return tests


def _plot_condition_flow_fields(
    module: DeepKoopmanLightningModule,
    rows: list[dict[str, int | float | str]],
    fig_dir: Path,
    *,
    grid_size: int = 25,
) -> tuple[dict[str, str], list[dict[str, int | float | str]]]:
    if not getattr(module.model, "is_conditioned", False):
        return {}, []
    fig_dir.mkdir(parents=True, exist_ok=True)
    planes = [((0, 1), "g1_g2"), ((0, 2), "g1_g3"), ((1, 2), "g2_g3")]
    condition_names = _condition_names(module)
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    paths: dict[str, str] = {}
    metadata: list[dict[str, int | float | str]] = []
    rng = np.random.default_rng(42)
    for condition_id, condition_name in enumerate(condition_names):
        selected = np.asarray(
            [
                [float(row["z1"]), float(row["z2"]), float(row["z3"])]
                for row in rows
                if str(row.get("condition", "")) == condition_name
            ],
            dtype=np.float32,
        )
        if selected.shape[0] < 2:
            continue
        center = np.median(selected, axis=0)
        scatter = selected
        if scatter.shape[0] > 3000:
            scatter = scatter[np.sort(rng.choice(scatter.shape[0], size=3000, replace=False))]
        for (dim_x, dim_y), plane_name in planes:
            minimum = selected[:, [dim_x, dim_y]].min(axis=0)
            maximum = selected[:, [dim_x, dim_y]].max(axis=0)
            span = np.maximum(maximum - minimum, 1e-5)
            minimum -= 0.03 * span
            maximum += 0.03 * span
            axis_x = np.linspace(minimum[0], maximum[0], grid_size)
            axis_y = np.linspace(minimum[1], maximum[1], grid_size)
            xx, yy = np.meshgrid(axis_x, axis_y)
            latent_grid = np.repeat(center[None, :], grid_size * grid_size, axis=0)
            latent_grid[:, dim_x] = xx.reshape(-1)
            latent_grid[:, dim_y] = yy.reshape(-1)
            z = torch.as_tensor(latent_grid, dtype=dtype, device=module.device)
            condition = torch.full(
                (latent_grid.shape[0],), condition_id, dtype=torch.long, device=module.device
            )
            module.model.eval()
            with torch.no_grad():
                displacement = (module.model.step_latent(z, condition) - z).detach().cpu().numpy()
            u = displacement[:, dim_x].reshape(grid_size, grid_size)
            v = displacement[:, dim_y].reshape(grid_size, grid_size)
            magnitude = np.sqrt(u**2 + v**2)

            path = fig_dir / f"flow_field_{condition_name}_{plane_name}.png"
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(
                scatter[:, dim_x],
                scatter[:, dim_y],
                s=3,
                color="0.45",
                alpha=0.12,
                rasterized=True,
                label="Observed latent states",
            )
            quiver = ax.quiver(
                xx,
                yy,
                u,
                v,
                magnitude,
                cmap="viridis",
                angles="xy",
                scale_units="xy",
                scale=None,
                width=0.003,
            )
            fig.colorbar(quiver, ax=ax, label=r"$\|K(y)y-y\|$")
            ax.set_xlabel(f"g{dim_x + 1}")
            ax.set_ylabel(f"g{dim_y + 1}")
            fixed_dim = ({0, 1, 2} - {dim_x, dim_y}).pop()
            ax.set_title(
                f"{condition_name}: latent displacement field\n"
                f"g{fixed_dim + 1} fixed at condition median ({center[fixed_dim]:.3g}); N={selected.shape[0]} windows"
            )
            ax.legend(fontsize=7, loc="best")
            ax.set_xlim(minimum[0], maximum[0])
            ax.set_ylim(minimum[1], maximum[1])
            fig.tight_layout()
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths[path.stem] = str(path)
            metadata.append(
                {
                    "condition": condition_name,
                    "condition_id": condition_id,
                    "plane": plane_name,
                    "n_windows": int(selected.shape[0]),
                    "grid_size": grid_size,
                    "x_min": float(minimum[0]),
                    "x_max": float(maximum[0]),
                    "y_min": float(minimum[1]),
                    "y_max": float(maximum[1]),
                    "fixed_dimension": f"g{fixed_dim + 1}",
                    "fixed_value": float(center[fixed_dim]),
                }
            )
    return paths, metadata


def _rat_analysis_outputs(
    module: DeepKoopmanLightningModule,
    splits: dict[str, TrajectoryData],
    metadata_path: Path,
    out_dir: Path,
    fig_dir: Path,
    *,
    show_progress: bool = True,
) -> tuple[dict[str, str], dict[str, str]]:
    latent_dir = out_dir.parent / "latents" / "fold_0"
    latent_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split: dict[str, list[dict[str, int | float | str]]] = {}
    band_errors: list[np.ndarray] = []
    split_weights: list[int] = []
    tables: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_rows, aggregates = _rat_rollout_rows(
            module,
            splits[split],
            metadata_path,
            split=split,
            show_progress=show_progress,
        )
        rows_by_split[split] = split_rows
        band_errors.append(aggregates["reconstruction_error_by_band"])
        split_weights.append(len(split_rows))
        csv_path = latent_dir / f"{split}_latents.csv"
        _write_rows(csv_path, split_rows)
        tables[f"{split}_latents_csv"] = str(csv_path)
        parquet = _write_optional_parquet(split_rows, latent_dir / f"{split}_latents.parquet")
        if parquet is not None:
            tables[f"{split}_latents_parquet"] = parquet

    rows = [row for split in ("train", "val", "test") for row in rows_by_split[split]]
    reconstruction_by_band = np.average(
        np.stack(band_errors, axis=0), axis=0, weights=np.asarray(split_weights, dtype=np.float64)
    )

    figures: dict[str, str] = {}
    figures.update(_plot_true_latent_trajectories_by_split(rows_by_split, fig_dir))
    statistics: list[dict[str, int | float | str]] = []
    metric_specs = [
        ("eigenvalue_lambda_r_by_condition.png", "lambda_r", "lambda_r", "Real-axis eigenvalue multiplier by music × period"),
        (
            "eigenvalue_lambda_r_mu_by_condition.png",
            "lambda_r_mu",
            "log(lambda_r) / delta_t",
            "Real-axis eigenvalue growth by music × period",
        ),
        ("eigenvalue_mu_by_condition.png", "mu", "mu", "Eigenvalue decay/growth by music × period"),
        ("eigenvalue_frequency_by_condition.png", "frequency", "Hz", "State-transition frequency by music × period"),
    ]
    for filename, metric, ylabel, title in metric_specs:
        path = fig_dir / filename
        statistics.extend(
            _plot_metric_by_condition(rows, path, metric=metric, ylabel=ylabel, title=title)
        )
        if path.exists():
            figures[path.stem] = str(path)

    displacement_path = fig_dir / "latent_displacement_by_condition.png"
    statistics.extend(_plot_latent_displacement(rows, displacement_path))
    if displacement_path.exists():
        figures[displacement_path.stem] = str(displacement_path)

    specs = [
        ("latent_3d_by_condition.png", lambda p: _plot_latent_3d(rows, p, group_key="condition", title="Latent space by condition (all splits)")),
        ("latent_3d_by_section.png", lambda p: _plot_latent_3d(rows, p, group_key="section", title="Latent space by section (all splits)")),
        ("prediction_error_by_horizon.png", lambda p: _plot_prediction_horizons(rows, p)),
        ("reconstruction_error_by_band.png", lambda p: _plot_reconstruction_by_band(reconstruction_by_band, p)),
    ]
    for filename, func in specs:
        path = fig_dir / filename
        func(path)
        if path.exists():
            figures[path.stem] = str(path)

    full_rat_path = fig_dir / "rat_latent_by_rat_test.png"
    _plot_latent_3d(
        rows_by_split["test"],
        full_rat_path,
        group_key="rat_id",
        title="Test latent space by rat (up to 5,000 windows)",
    )
    if full_rat_path.exists():
        figures[full_rat_path.stem] = str(full_rat_path)

    flow_figures, flow_metadata = _plot_condition_flow_fields(module, rows, fig_dir)
    figures.update(flow_figures)
    flow_csv = out_dir / "tables" / "flow_field_metadata.csv"
    flow_json = out_dir / "tables" / "flow_field_metadata.json"
    _write_rows(flow_csv, flow_metadata)
    flow_json.write_text(json.dumps(flow_metadata, indent=2), encoding="utf-8")
    tables["flow_field_metadata_csv"] = str(flow_csv)
    tables["flow_field_metadata_json"] = str(flow_json)

    statistics_csv = out_dir / "tables" / "statistical_tests.csv"
    statistics_json = out_dir / "tables" / "statistical_tests.json"
    _write_rows(statistics_csv, statistics)
    statistics_json.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    tables["statistical_tests_csv"] = str(statistics_csv)
    tables["statistical_tests_json"] = str(statistics_json)
    return tables, figures


def _omega_components(module: DeepKoopmanLightningModule, latent: np.ndarray) -> dict[str, np.ndarray]:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    g = torch.as_tensor(latent, dtype=dtype, device=module.device)
    module.model.eval()
    if getattr(module.model, "is_conditioned", False):
        condition = torch.zeros(g.shape[0], dtype=torch.long, device=module.device)
        with torch.no_grad():
            params = module.model.eigen_params(g, condition).detach().cpu().numpy()
        delta_t = module.config.data.delta_t
        return {
            "lambda_r_silence": params[:, 0],
            "rho_silence": params[:, 1],
            "theta_silence": params[:, 2],
            "mu_silence": np.log(np.maximum(params[:, 1], 1e-12)) / delta_t,
            "frequency_silence": params[:, 2] / (2 * np.pi * delta_t),
        }
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
    checkpoint: str | Path = "best",
    device: str | torch.device = "auto",
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
    show_progress: bool = True,
    make_videos: bool = True,
    video_fps: int = 4,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    ckpt, checkpoint_label = _resolve_postprocess_checkpoint(run_dir, checkpoint)
    default_output_name = "postprocess" if checkpoint_label == "best" else f"postprocess_{checkpoint_label}"
    out_dir = Path(output_dir) if output_dir is not None else run_dir / default_output_name
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    module = DeepKoopmanLightningModule.load_checkpoint(ckpt).to(resolved_device)
    cfg = module.config
    dataset = dataset or cfg.data.name
    splits = load_split_data(data_dir, dataset, cfg.data.train_files)
    data_dir = Path(data_dir)

    test_metrics = evaluate_data(module, splits["test"], show_progress=show_progress)
    (table_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    _write_single_row_csv(table_dir / "test_metrics.csv", test_metrics)

    sampled, sample_rows = sample_trajectories(splits, cfg.data.len_time, samples_per_split, seed)
    sampled_conditions = sample_conditions(splits, sample_rows, cfg.data.len_time)
    _write_rows(table_dir / "sampled_trajectories.csv", sample_rows)

    figures = {}
    videos: dict[str, str] = {}
    figures.update(_plot_data_trajectories(sampled, fig_dir, cfg.data.delta_t))
    if not getattr(module.model, "is_conditioned", False):
        figures.update(_plot_latent_true_vs_pred(module, sampled, fig_dir, sampled_conditions))

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
        if getattr(module.model, "is_conditioned", False):
            if make_videos:
                for stale in fig_dir.glob("*_sample_*_reconstruction_prediction_heatmap.png"):
                    stale.unlink()
                stale_manifest = table_dir / "reconstruction_heatmap_samples.csv"
                if stale_manifest.exists():
                    stale_manifest.unlink()
                videos, video_manifest = _make_music_period_videos(
                    module,
                    splits,
                    rat_metadata_path,
                    out_dir / "videos",
                    seed=seed,
                    fps=video_fps,
                    show_progress=show_progress,
                )
                video_manifest_path = table_dir / "video_samples.csv"
                _write_rows(video_manifest_path, video_manifest)
                tables["video_samples"] = str(video_manifest_path)
            rat_tables, rat_figures = _rat_analysis_outputs(
                module,
                splits,
                rat_metadata_path,
                out_dir,
                fig_dir,
                show_progress=show_progress,
            )
            tables.update(rat_tables)
            figures.update(rat_figures)

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
        "videos": videos,
        "eigen_component_grid": eigen_grid,
        "eigenfunction_state_grid": state_grid,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
