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
Training is config-driven with nested YAML sections for data, model, loss, optimizer, trainer, runtime, and logging.

```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete.yaml
```

Optional overrides:
```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete.yaml --epochs 1 --batch-size 128
```

Enable Weights & Biases explicitly:
```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete.yaml --wandb --wandb-project deepkoopman --wandb-mode offline
```

## 2) Hyperparameter search (random search)
Example search configs:
- `configs/search/discrete.yaml`
- `configs/search/pendulum.yaml`
- `configs/search/fluid_attractor.yaml`
- `configs/search/fluid_box.yaml`

Run search:
```bash
uv run python -m deepkoopman.cli.search --config configs/search/discrete.yaml
```

Search outputs:
- `results/search/<run_id>/trials.csv`
- `results/search/<run_id>/best_config.yaml`
- `results/search/<run_id>/best_checkpoint.ckpt`
- `results/search/<run_id>/summary.json`

## 3) Postprocessing (PNG/CSV)
```bash
uv run python -m deepkoopman.cli.postprocess --run-dir results/search/<run_id> --dataset DiscreteSpectrumExample
```

Outputs:
- `.../postprocess/figures/losses.png`
- `.../postprocess/figures/reconstruction.png`
- `.../postprocess/figures/prediction.png`
- `.../postprocess/tables/history.csv`
- `.../postprocess/tables/sample_*.csv`

## 4) Rat auditory cortex analysis
Rat analysis is config-driven. Preprocessing, model, loss, optimizer, trainer, runtime, cache, and output defaults live in `configs/rat_analysis/default.yaml`; CLI flags are reserved for execution-time overrides.
Rat metadata lives in `data/rat_id.csv`; the raw `.mat` root template is configured in `configs/rat_analysis/default.yaml` under `input.source.data_root_template`.

```bash
uv run python -m deepkoopman.cli.rat_analysis --config configs/rat_analysis/default.yaml --quick --no-progress
```

## Differences from the original TensorFlow repository
- The old random search flow in `*Experiment.py` is replaced by `deepkoopman.cli.search` + nested YAML configs.
- The old `postprocessing/*.ipynb` flow is replaced by a shared visualization module and CLI postprocessing.

## Run tests
```bash
uv run pytest -q
```
