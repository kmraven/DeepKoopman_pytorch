from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deepkoopman.lightning import DeepKoopmanLightningModule
from deepkoopman.paper import save_paper_artifacts
from deepkoopman.reproduction import PAPER_DATASETS, train_paths_for_dataset
from deepkoopman.visualization import (
    load_history,
    plot_losses,
    plot_prediction,
    plot_reconstruction,
    save_history_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--config-dir", default="configs/train")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "postprocess"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    ckpt = run_dir / "best_checkpoint.ckpt"
    if not ckpt.exists():
        candidates = list(run_dir.glob("**/*.ckpt"))
        if not candidates:
            raise FileNotFoundError("No checkpoint found in run-dir")
        ckpt = candidates[0]

    history_candidates = list(run_dir.glob("**/metrics.csv"))
    history_file = history_candidates[0] if history_candidates else ckpt.with_suffix(".history.json")
    if not history_file.exists():
        # fallback to first history json
        hist = list(run_dir.glob("*.history.json"))
        if not hist:
            raise FileNotFoundError("No history json found in run-dir")
        history_file = hist[0]

    history = load_history(history_file)
    save_history_csv(history, table_dir / "history.csv")
    plot_losses(history, fig_dir / "losses.png")

    data_dir = Path(args.data_dir)
    val = np.loadtxt(data_dir / f"{args.dataset}_val_x.csv", delimiter=",", dtype=np.float64)
    sample = val[:1]

    module = DeepKoopmanLightningModule.load_checkpoint(ckpt)
    recon = module.reconstruct_array(sample)
    pred = module.predict_array(sample, steps=args.steps)

    np.savetxt(table_dir / "sample_input.csv", sample, delimiter=",")
    np.savetxt(table_dir / "sample_recon.csv", recon, delimiter=",")
    np.savetxt(table_dir / "sample_pred.csv", pred.reshape(pred.shape[0], -1), delimiter=",")

    plot_reconstruction(sample, recon, fig_dir / "reconstruction.png", title=f"{args.dataset} Reconstruction")
    plot_prediction(pred, fig_dir / "prediction.png", title=f"{args.dataset} Multi-step Prediction")

    train_paths = train_paths_for_dataset(data_dir, args.dataset, module.config.data.train_files)
    test_path = data_dir / f"{args.dataset}_test_x.csv"
    if args.dataset in PAPER_DATASETS and all(path.exists() for path in train_paths) and test_path.exists():
        train = np.concatenate([np.loadtxt(path, delimiter=",", dtype=np.float64) for path in train_paths], axis=0)
        test = np.loadtxt(test_path, delimiter=",", dtype=np.float64)
        save_paper_artifacts(
            args.dataset,
            module,
            {"train": train, "validation": val, "test": test},
            out_dir / "paper",
            config_dir=args.config_dir,
        )

    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
