from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from deepkoopman import DeepKoopmanConfig, DeepKoopmanLightningModule, build_trainer
from deepkoopman.data import DeepKoopmanDataModule
from deepkoopman.io import h5_path_for_dataset, load_split_data


def find_val_file(data_dir: Path, data_name: str) -> Path:
    h5_path = h5_path_for_dataset(data_dir, data_name)
    if h5_path.exists():
        return h5_path
    val = data_dir / f"{data_name}_val_x.csv"
    if not val.exists():
        raise FileNotFoundError(
            f"Missing validation data for {data_name}. Expected {h5_path.name} or {val.name}. "
            "Place generated CSV files under ./data (README参照)."
        )
    return val


def run_training(args: argparse.Namespace) -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
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
    find_val_file(data_dir, config.data.name)
    splits = load_split_data(data_dir, config.data.name, config.data.train_files)

    run_dir = root / args.output_dir
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    config.logging.save_dir = str(run_dir / "logs")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )

    module = DeepKoopmanLightningModule(config)
    datamodule = DeepKoopmanDataModule(splits["train"], splits["val"], config)
    trainer = build_trainer(
        config,
        default_root_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        use_wandb=args.wandb,
        run_name=args.run_name,
    )
    trainer.fit(module, datamodule=datamodule)

    best_checkpoint = (
        trainer.checkpoint_callback.best_model_path
        if trainer.checkpoint_callback
        else ""
    )
    best_target = run_dir / "best_checkpoint.ckpt"
    if best_checkpoint:
        best_source = Path(best_checkpoint)
        shutil.copy2(best_source, best_target)
        if best_source.resolve() != best_target.resolve():
            best_source.unlink(missing_ok=True)
            try:
                best_source.parent.rmdir()
            except OSError:
                pass

    best_val = (
        trainer.checkpoint_callback.best_model_score
        if trainer.checkpoint_callback
        else None
    )
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_target) if best_checkpoint else "",
        "best_val_loss": None if best_val is None else float(best_val.detach().cpu()),
        "epochs": int(trainer.current_epoch),
        "global_step": int(trainer.global_step),
        "config_path": str(config_path),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train/discrete_spectrum.yaml")
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

    summary = run_training(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
