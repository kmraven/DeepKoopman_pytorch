from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from .data import stack_data
from .lightning import DeepKoopmanLightningModule
from .losses import compute_losses
from .reproduction import load_paper_manifest


SPLIT_LABELS = {
    "train": "train",
    "validation": "validation",
    "test": "test",
}


def _losses_to_float(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def evaluate_split(module: DeepKoopmanLightningModule, data: np.ndarray) -> dict[str, float]:
    cfg = module.config
    max_shift = max([1] + cfg.data.shifts + cfg.data.middle_shifts)
    stacked = stack_data(data, max_shift, cfg.data.len_time)
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    batch = torch.from_numpy(stacked).to(module.device, dtype=dtype)
    module.model.eval()
    with torch.no_grad():
        losses = _losses_to_float(compute_losses(module.model, batch, cfg))
    losses["pre_regularization_loss"] = losses["loss1"] + losses["loss2"] + losses["loss3"] + losses["loss_linf"]
    return losses


def compute_eigenvalue_ranges(module: DeepKoopmanLightningModule, data: np.ndarray, sample_rows: int = 1000) -> dict[str, dict[str, float]]:
    cfg = module.config
    sample = data[:sample_rows]
    dtype = torch.float32 if cfg.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(sample, dtype=dtype, device=module.device)
    module.model.eval()
    out: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        latent = module.model.encode(x)
        omegas = module.model._omega_net_apply(latent)
    for j in range(cfg.model.num_complex_pairs):
        omega = omegas[j].detach().cpu().numpy()
        out[f"complex_pair_{j + 1}"] = {
            "omega_min": float(np.min(omega[:, 0])),
            "omega_max": float(np.max(omega[:, 0])),
            "mu_min": float(np.min(omega[:, 1])),
            "mu_max": float(np.max(omega[:, 1])),
        }
    offset = cfg.model.num_complex_pairs
    for j in range(cfg.model.num_real):
        mu = omegas[offset + j].detach().cpu().numpy()
        out[f"real_{j + 1}"] = {
            "mu_min": float(np.min(mu[:, 0])),
            "mu_max": float(np.max(mu[:, 0])),
        }
    return out


def save_latent_tables(module: DeepKoopmanLightningModule, data: np.ndarray, out_dir: Path, sample_rows: int = 1000) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = data[:sample_rows]
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(sample, dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        latent = module.model.encode(x)
        omegas = module.model._omega_net_apply(latent)
    latent_np = latent.detach().cpu().numpy()
    omega_np = np.concatenate([om.detach().cpu().numpy() for om in omegas], axis=1) if omegas else np.empty((len(sample), 0))
    latent_path = out_dir / "latent_coordinates.csv"
    omega_path = out_dir / "omega_parameters.csv"
    np.savetxt(latent_path, latent_np, delimiter=",")
    np.savetxt(omega_path, omega_np, delimiter=",")
    return {"latent_coordinates": str(latent_path), "omega_parameters": str(omega_path)}


def compute_prediction_horizon(module: DeepKoopmanLightningModule, data: np.ndarray, threshold: float = 0.10) -> dict[str, float | int | None]:
    cfg = module.config
    if data.ndim == 1:
        data = data[:, None]
    num_traj = data.shape[0] // cfg.data.len_time
    traj = data[: num_traj * cfg.data.len_time].reshape(num_traj, cfg.data.len_time, data.shape[1])
    pred = module.predict_array(traj[:, 0, :], steps=cfg.data.len_time - 1)
    pred = np.moveaxis(pred, 0, 1)
    errors = np.linalg.norm(pred - traj, axis=2)
    denom = np.maximum(np.linalg.norm(traj, axis=2), 1e-12)
    relative = errors / denom
    mean_relative = np.mean(relative, axis=0)
    exceeded = np.flatnonzero(mean_relative > threshold)
    step = int(exceeded[0]) if exceeded.size else None
    return {
        "threshold": threshold,
        "first_step_exceeding_threshold": step,
        "time_exceeding_threshold": None if step is None else float(step * cfg.data.delta_t),
        "max_mean_relative_error": float(np.max(mean_relative)),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paper_dataset_meta(dataset: str, config_dir: str | Path | None = None) -> dict:
    return load_paper_manifest(config_dir)["datasets"][dataset]


def write_paper_tables(
    dataset: str,
    module: DeepKoopmanLightningModule,
    splits: dict[str, np.ndarray],
    out_dir: Path,
    *,
    config_dir: str | Path | None = None,
) -> dict[str, str]:
    cfg = module.config
    meta = _paper_dataset_meta(dataset, config_dir)
    table_dir = out_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    split_metrics = {name: evaluate_split(module, data) for name, data in splits.items()}
    table1_rows = []
    for split, losses in split_metrics.items():
        table1_rows.append(
            {
                "dataset": dataset,
                "split": SPLIT_LABELS.get(split, split),
                "reconstruction": losses["loss1"],
                "prediction": losses["loss2"],
                "linearity": losses["loss3"],
                "linf": losses["loss_linf"],
                "pre_regularization_loss": losses["pre_regularization_loss"],
                "regularized_loss": losses["loss"],
                "paper_reference_pre_regularization_loss": meta["table1_reference"].get(SPLIT_LABELS.get(split, split), ""),
            }
        )
    _write_csv(table_dir / "table1_metrics.csv", table1_rows)

    _write_csv(
        table_dir / "table2_dataset_sizes.csv",
        [
            {
                "dataset": dataset,
                "train_files": cfg.data.train_files,
                "len_time": cfg.data.len_time,
                "delta_t": cfg.data.delta_t,
                "batch_size": cfg.trainer.batch_size,
                "paper_reported_train_trajectories": meta["paper_reported_train_trajectories"],
                "notebook_observed_train_trajectories": meta["notebook_observed_train_trajectories"],
                "loaded_train_trajectories": int(splits["train"].shape[0] // cfg.data.len_time),
                "validation_trajectories": int(splits["validation"].shape[0] // cfg.data.len_time),
                "test_trajectories": int(splits["test"].shape[0] // cfg.data.len_time),
            }
        ],
    )

    encoder_hidden = cfg.model.widths[1 : 1 + ((len(cfg.model.widths) - 4) // 2)]
    _write_csv(
        table_dir / "table3_architecture.csv",
        [
            {
                "dataset": dataset,
                "main_network_widths": json.dumps(cfg.model.widths),
                "encoder_hidden_layers": len(encoder_hidden),
                "encoder_hidden_widths": json.dumps(encoder_hidden),
                "omega_hidden_layers": len(cfg.model.omega_hidden_widths),
                "omega_hidden_widths": json.dumps(cfg.model.omega_hidden_widths),
                "num_real": cfg.model.num_real,
                "num_complex_pairs": cfg.model.num_complex_pairs,
            }
        ],
    )

    _write_csv(
        table_dir / "table4_loss_hparams.csv",
        [
            {
                "dataset": dataset,
                "alpha_1_reconstruction_weight": cfg.loss.reconstruction_weight,
                "log10_alpha_1": float(np.log10(cfg.loss.reconstruction_weight)),
                "alpha_2_linf_weight": cfg.loss.linf_weight,
                "log10_alpha_2": float(np.log10(cfg.loss.linf_weight)) if cfg.loss.linf_weight > 0 else "",
                "alpha_3_l2_weight": cfg.loss.l2_weight,
                "log10_alpha_3": float(np.log10(cfg.loss.l2_weight)) if cfg.loss.l2_weight > 0 else "",
                "prediction_steps_penalized": max(cfg.data.shifts) if cfg.data.shifts else 0,
                "linearity_steps_penalized": max(cfg.data.middle_shifts) if cfg.data.middle_shifts else 0,
            }
        ],
    )

    eigen_ranges = compute_eigenvalue_ranges(module, splits["validation"])
    (table_dir / "eigenvalue_ranges.json").write_text(json.dumps(eigen_ranges, indent=2), encoding="utf-8")
    horizon = compute_prediction_horizon(module, splits["test"])
    (table_dir / "prediction_horizon_10pct.json").write_text(json.dumps(horizon, indent=2), encoding="utf-8")
    latent_paths = save_latent_tables(module, splits["test"], table_dir)

    return {
        "table1_metrics": str(table_dir / "table1_metrics.csv"),
        "table2_dataset_sizes": str(table_dir / "table2_dataset_sizes.csv"),
        "table3_architecture": str(table_dir / "table3_architecture.csv"),
        "table4_loss_hparams": str(table_dir / "table4_loss_hparams.csv"),
        "eigenvalue_ranges": str(table_dir / "eigenvalue_ranges.json"),
        "prediction_horizon_10pct": str(table_dir / "prediction_horizon_10pct.json"),
        **latent_paths,
    }


def _trajectory_view(data: np.ndarray, len_time: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    return data.reshape(data.shape[0] // len_time, len_time, data.shape[1])


def _plot_prediction_against_truth(module: DeepKoopmanLightningModule, data: np.ndarray, path: Path, title: str) -> None:
    cfg = module.config
    traj = _trajectory_view(data, cfg.data.len_time)
    truth = traj[0]
    pred = module.predict_array(truth[:1], steps=cfg.data.len_time - 1)[:, 0, :]
    t = np.arange(cfg.data.len_time) * cfg.data.delta_t
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for feat in range(truth.shape[1]):
        ax.plot(t, truth[:, feat], linewidth=2, label=f"x{feat} true")
        ax.plot(t, pred[:, feat], linestyle="--", label=f"x{feat} pred")
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_latent(module: DeepKoopmanLightningModule, data: np.ndarray, path: Path, title: str) -> None:
    dtype = torch.float32 if module.config.runtime.dtype == "float32" else torch.float64
    x = torch.as_tensor(data[:1000], dtype=dtype, device=module.device)
    module.model.eval()
    with torch.no_grad():
        latent = module.model.encode(x).detach().cpu().numpy()
    fig = plt.figure(figsize=(6, 5))
    if latent.shape[1] >= 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], s=4)
        ax.set_zlabel("g3")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(latent[:, 0], latent[:, 1] if latent.shape[1] > 1 else np.zeros(len(latent)), s=4)
    ax.set_title(title)
    ax.set_xlabel("g1")
    ax.set_ylabel("g2")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_paper_figures(dataset: str, module: DeepKoopmanLightningModule, splits: dict[str, np.ndarray], out_dir: Path) -> dict[str, str]:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figure_map = {
        "DiscreteSpectrumExample": "fig3_discrete.png",
        "Pendulum": "fig4_pendulum.png",
        "FluidFlowOnAttractor": "fig6_fluid_attractor.png",
        "FluidFlowBox": "fig6_fluid_box.png",
    }
    pred_path = fig_dir / figure_map[dataset]
    _plot_prediction_against_truth(module, splits["test"], pred_path, f"{dataset} prediction")
    outputs = {pred_path.stem: str(pred_path)}
    if dataset == "Pendulum":
        latent_path = fig_dir / "fig5_pendulum_phase_magnitude.png"
        _plot_latent(module, splits["test"], latent_path, "Pendulum latent coordinates")
        outputs[latent_path.stem] = str(latent_path)
    return outputs


def write_paper_manifest(out_dir: Path, *, config_dir: str | Path | None = None) -> str:
    manifest = load_paper_manifest(config_dir)
    path = out_dir / "manifest.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return str(path)


def save_paper_artifacts(
    dataset: str,
    module: DeepKoopmanLightningModule,
    splits: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    config_dir: str | Path | None = None,
) -> dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = write_paper_tables(dataset, module, splits, out_dir, config_dir=config_dir)
    figures = save_paper_figures(dataset, module, splits, out_dir)
    manifest = write_paper_manifest(out_dir, config_dir=config_dir)
    summary = {"tables": tables, "figures": figures, "manifest": manifest}
    (out_dir / "paper_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
