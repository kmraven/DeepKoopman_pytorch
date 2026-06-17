from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from deepkoopman import DeepKoopmanConfig, DeepKoopmanLightningModule, build_trainer
from deepkoopman.data import DeepKoopmanDataModule


def find_data_files(data_dir: Path, data_name: str) -> tuple[Path, Path]:
    train = data_dir / f"{data_name}_train1_x.csv"
    val = data_dir / f"{data_name}_val_x.csv"
    if not train.exists() or not val.exists():
        raise FileNotFoundError(
            f"Missing dataset CSVs for {data_name}. Expected {train.name} and {val.name}. "
            "Place generated CSV files under ./data (README参照)."
        )
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/discrete_train.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", default="results/example")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config
    config = DeepKoopmanConfig.from_yaml(config_path)
    if args.epochs is not None:
        config.trainer.max_epochs = args.epochs
    if args.batch_size is not None:
        config.trainer.batch_size = args.batch_size
    if args.wandb:
        config.logging.backend = "wandb"
    if args.wandb_project:
        config.logging.wandb.project = args.wandb_project
    if args.wandb_entity:
        config.logging.wandb.entity = args.wandb_entity
    if args.wandb_mode:
        config.logging.wandb.mode = args.wandb_mode
    if args.no_progress:
        config.trainer.enable_progress_bar = False

    data_dir = root / config.data.root
    train_path, val_path = find_data_files(data_dir, config.data.name)

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float64)
    val = np.loadtxt(val_path, delimiter=",", dtype=np.float64)

    run_dir = root / args.output_dir
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.logging.save_dir = str(run_dir / "logs")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")

    module = DeepKoopmanLightningModule(config)
    datamodule = DeepKoopmanDataModule(train, val, config)
    trainer = build_trainer(
        config,
        default_root_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.wandb,
        run_name=args.run_name,
    )
    trainer.fit(module, datamodule=datamodule)
    best_checkpoint = trainer.checkpoint_callback.best_model_path if trainer.checkpoint_callback else ""
    if best_checkpoint:
        module = DeepKoopmanLightningModule.load_checkpoint(best_checkpoint)

    sample = val[:1]
    pred = module.predict_array(sample, steps=5)
    recon = module.reconstruct_array(sample)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": best_checkpoint,
        "epochs": int(trainer.current_epoch),
        "predict_shape": list(pred.shape),
        "reconstruct_shape": list(recon.shape),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Checkpoint: {best_checkpoint}")
    print(f"predict shape: {pred.shape}, reconstruct shape: {recon.shape}")


if __name__ == "__main__":
    main()
