from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

_CACHE_DIR = Path.cwd() / ".cache"
(_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

from deepkoopman.model import DeepKoopmanModule
from deepkoopman.reproduction import PAPER_DATASETS, paper_best_params, paper_config, train_paths_for_dataset
from deepkoopman.data import DeepKoopmanDataModule
from deepkoopman.lightning import DeepKoopmanLightningModule, build_trainer
from deepkoopman.data import stack_data
from deepkoopman.losses import compute_losses
from deepkoopman.visualization import load_history, plot_losses, plot_prediction, plot_reconstruction, save_history_csv


def _losses_to_float(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in losses.items()}


def _evaluate_split(module: DeepKoopmanLightningModule, data: np.ndarray) -> dict[str, float]:
    cfg = module.config
    max_shift = max([1] + cfg.shifts + cfg.shifts_middle)
    stacked = stack_data(data, max_shift, cfg.len_time)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    batch = torch.from_numpy(stacked).to(module.device, dtype=dtype)
    module.model.eval()
    with torch.no_grad():
        return _losses_to_float(compute_losses(module.model, batch, cfg))


def _save_latent_tables(module: DeepKoopmanLightningModule, data: np.ndarray, out_dir: Path, sample_rows: int = 1000) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = data[:sample_rows]
    dtype = torch.float32 if module.config.dtype == "float32" else torch.float64
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


def _save_artifacts(
    module: DeepKoopmanLightningModule,
    dataset: str,
    run_dir: Path,
    val: np.ndarray,
    test: np.ndarray,
    run_summary: dict[str, object],
) -> dict[str, object]:
    fig_dir = run_dir / "figures"
    table_dir = run_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    history_file = next(run_dir.glob("logs/**/metrics.csv"), None)
    history = load_history(history_file) if history_file else []
    save_history_csv(history, table_dir / "history.csv")
    if history:
        plot_losses(history, fig_dir / "losses.png")

    sample = test[:1]
    recon = module.reconstruct_array(sample)
    pred = module.predict_array(sample, steps=min(30, max(module.config.shifts)))
    np.savetxt(table_dir / "sample_input.csv", sample, delimiter=",")
    np.savetxt(table_dir / "sample_recon.csv", recon, delimiter=",")
    np.savetxt(table_dir / "sample_pred.csv", pred.reshape(pred.shape[0], -1), delimiter=",")
    plot_reconstruction(sample, recon, fig_dir / "reconstruction.png", title=f"{dataset} Reconstruction")
    plot_prediction(pred, fig_dir / "prediction.png", title=f"{dataset} Multi-step Prediction")

    test_metrics = _evaluate_split(module, test)
    val_metrics = _evaluate_split(module, val)
    metrics_path = table_dir / "metrics.json"
    metrics = {"validation": val_metrics, "test": test_metrics}
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latent_paths = _save_latent_tables(module, test, table_dir)
    artifacts = {
        "figures": {
            "losses": str(fig_dir / "losses.png"),
            "reconstruction": str(fig_dir / "reconstruction.png"),
            "prediction": str(fig_dir / "prediction.png"),
        },
        "tables": {
            "history": str(table_dir / "history.csv"),
            "metrics": str(metrics_path),
            "sample_input": str(table_dir / "sample_input.csv"),
            "sample_recon": str(table_dir / "sample_recon.csv"),
            "sample_pred": str(table_dir / "sample_pred.csv"),
            **latent_paths,
        },
    }
    summary = {**run_summary, "metrics": metrics, "artifacts": artifacts}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _csv_rows(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _expected_steps(cfg, train_paths: list[Path]) -> int:
    total = 0
    row_counts = [_csv_rows(path) for path in train_paths]
    for file_pass in range(cfg.data_train_len * cfg.num_passes_per_file):
        file_num = file_pass % cfg.data_train_len
        max_shift = max([1] + cfg.shifts + cfg.shifts_middle)
        num_traj = row_counts[file_num] // cfg.len_time
        num_examples = num_traj * (cfg.len_time - max_shift)
        batch_size = cfg.batch_size if cfg.batch_size > 0 else num_examples
        num_batches = max(1, int(np.floor(num_examples / batch_size)))
        requested = cfg.num_steps_per_batch * num_batches
        if cfg.num_steps_per_file_pass is not None:
            requested = min(requested, cfg.num_steps_per_file_pass + 1)
        total += requested
    return total


def _progress_callback(dataset: str, cfg, train_paths: list[Path], enabled: bool):
    if not enabled:
        return None
    bar = tqdm(
        total=_expected_steps(cfg, train_paths),
        desc=dataset,
        unit="step",
        dynamic_ncols=True,
        leave=True,
    )
    state = {"best": None}

    def callback(event: dict[str, object]) -> None:
        if event["event"] == "step":
            bar.update(1)
        elif event["event"] == "file_start":
            bar.set_description(f"{dataset} train{event['file_num']} pass {int(event['file_pass']) + 1}")
        elif event["event"] == "eval":
            state["best"] = event["best_val_loss"]
            bar.set_postfix(
                val=f"{event['val_loss']:.3e}",
                best=f"{event['best_val_loss']:.3e}",
                elapsed=f"{event['elapsed_sec'] / 60:.1f}m",
                refresh=False,
            )
        elif event["event"] == "stop":
            bar.set_postfix(
                best=f"{event['best_val_loss']:.3e}",
                stop=str(event["stop_condition"]),
                refresh=False,
            )
            bar.close()

    callback.close = bar.close  # type: ignore[attr-defined]
    return callback


def run_dataset(dataset: str, args: argparse.Namespace) -> dict[str, object]:
    data_dir = Path(args.data_dir)
    cfg = paper_config(dataset, quick=args.quick, device=args.device)
    cfg.logging.save_dir = str(Path(args.output_dir) / dataset / "logs")
    cfg.trainer.enable_progress_bar = not getattr(args, "no_progress", False)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / dataset / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    params = paper_best_params()[dataset]
    (run_dir / "paper_best_params.yaml").write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    train_paths = train_paths_for_dataset(data_dir, dataset, cfg.data_train_len)
    missing = [str(p) for p in train_paths if not p.exists()]
    val_path = data_dir / f"{dataset}_val_x.csv"
    test_path = data_dir / f"{dataset}_test_x.csv"
    missing.extend(str(p) for p in [val_path, test_path] if not p.exists())
    if missing:
        raise FileNotFoundError(f"Missing required data files for {dataset}: {missing}")

    train = np.concatenate([np.loadtxt(path, delimiter=",", dtype=np.float64) for path in train_paths], axis=0)
    val = np.loadtxt(val_path, delimiter=",", dtype=np.float64)
    test = np.loadtxt(test_path, delimiter=",", dtype=np.float64)
    cfg.logging.save_dir = str(run_dir / "logs")
    module = DeepKoopmanLightningModule(cfg)
    datamodule = DeepKoopmanDataModule(train, val, cfg, test_data=test)
    trainer = build_trainer(cfg, default_root_dir=run_dir, checkpoint_dir=run_dir / "checkpoints", run_name=dataset)
    trainer.fit(module, datamodule=datamodule)
    checkpoint = Path(trainer.checkpoint_callback.best_model_path)
    module = DeepKoopmanLightningModule.load_checkpoint(checkpoint)
    train_summary = {
        "best_val_loss": float(trainer.callback_metrics["val/loss"].detach().cpu()),
        "steps": int(trainer.global_step),
        "stop_condition": "completed lightning trainer fit",
    }
    history_path = next(run_dir.glob("logs/**/metrics.csv"), None)
    run_summary = {
        "dataset": dataset,
        "quick": bool(args.quick),
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "history": str(history_path) if history_path else "",
        "config": cfg.to_dict(),
        "paper_best_params": params,
        **train_summary,
    }
    return _save_artifacts(module, dataset, run_dir, val, test, run_summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *PAPER_DATASETS], default="all")
    parser.add_argument("--output-dir", default="results/reproduction")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    datasets = PAPER_DATASETS if args.dataset == "all" else [args.dataset]
    summaries = [run_dataset(dataset, args) for dataset in datasets]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
