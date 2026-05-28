from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deepkoopman import DeepKoopmanConfig, DeepKoopmanModule, DeepKoopmanTrainer


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
    parser.add_argument("--save-path", default="results/deepkoopman.pt")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    config = DeepKoopmanConfig.from_yaml(config_path)
    if args.epochs is not None:
        config.max_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    data_dir = root / "data"
    train_path, val_path = find_data_files(data_dir, config.data_name)

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float64)
    val = np.loadtxt(val_path, delimiter=",", dtype=np.float64)

    model = DeepKoopmanModule(config)
    trainer = DeepKoopmanTrainer(model, config)
    history = trainer.fit(train, val)
    trainer.save(root / args.save_path)

    sample = val[:1]
    pred = trainer.predict(sample, steps=5)
    recon = trainer.reconstruct(sample)

    print(f"Trained epochs: {len(history)}")
    print(f"Last val loss: {history[-1]['loss']:.6e}")
    print(f"predict shape: {pred.shape}, reconstruct shape: {recon.shape}")


if __name__ == "__main__":
    main()
