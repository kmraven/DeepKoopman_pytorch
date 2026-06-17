# DeepKoopman (PyTorch)

This repository provides a PyTorch Lightning-based DeepKoopman implementation.
It also includes random hyperparameter search, optional Weights & Biases monitoring, and postprocessing/visualization workflows.

Original Paper: https://doi.org/10.1038/s41467-018-07210-0

## Setup
```bash
uv sync
```

### NVIDIA GPU server setup
For NVIDIA driver 470.x / CUDA 11.x systems, use Python 3.10-3.12 and the CUDA 11.8 PyTorch wheel pinned in `pyproject.toml`.

```bash
uv python pin 3.11
uv sync --reinstall-package torch
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output includes a `+cu118` PyTorch build, CUDA `11.8`, and `True`.

## 1) Train a model (example)
`run_example.py` is config-driven (YAML).

```bash
uv run python scripts/run_example.py --config configs/discrete_train.yaml
```

Optional overrides:
```bash
uv run python scripts/run_example.py --config configs/discrete_train.yaml --epochs 1 --batch-size 128
```

Enable Weights & Biases explicitly:
```bash
uv run python scripts/run_example.py --config configs/discrete_train.yaml --wandb --wandb-project deepkoopman --wandb-mode offline
```

## 2) Hyperparameter search (random search)
Example search configs:
- `configs/discrete_search.yaml`
- `configs/pendulum_search.yaml`
- `configs/fluid_attractor_search.yaml`
- `configs/fluid_box_search.yaml`

Run search:
```bash
uv run python scripts/search_hparams.py --config configs/discrete_search.yaml
```

Search outputs:
- `results/search/<run_id>/trials.csv`
- `results/search/<run_id>/best_config.yaml`
- `results/search/<run_id>/best_checkpoint.ckpt`
- `results/search/<run_id>/summary.json`

## 3) Postprocessing (PNG/CSV)
```bash
uv run python scripts/postprocess.py --run-dir results/search/<run_id> --dataset DiscreteSpectrumExample
```

Outputs:
- `.../postprocess/figures/losses.png`
- `.../postprocess/figures/reconstruction.png`
- `.../postprocess/figures/prediction.png`
- `.../postprocess/tables/history.csv`
- `.../postprocess/tables/sample_*.csv`

## 4) marimo notebook
```bash
uv run marimo edit postprocessing_marimo/deepkoopman_postprocess.py
```

## Differences from the original TensorFlow repository
- The old random search flow in `*Experiment.py` is replaced by `scripts/search_hparams.py` + YAML configs.
- The hand-written PyTorch training loop is retired in favor of Lightning `LightningModule`, `DataModule`, callbacks, and `.ckpt` checkpoints.
- W&B monitoring is opt-in; default runs use local CSV logs.
- The old `postprocessing/*.ipynb` flow is replaced by a shared visualization module, CLI postprocessing, and a marimo notebook.

## Run tests
```bash
uv run pytest -q
```
