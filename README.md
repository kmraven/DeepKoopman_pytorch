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

## 1) Train a model
Training is config-driven with nested YAML sections for data, model, loss, optimizer, trainer, runtime, and logging.

```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete_spectrum.yaml
```

Optional overrides:
```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete_spectrum.yaml --epochs 1 --batch-size 128
```

Enable Weights & Biases explicitly:
```bash
uv run python -m deepkoopman.cli.train --config configs/train/discrete_spectrum.yaml --output-dir results/discrete_spectrum --wandb --wandb-project deepkoopman_discrete_spectrum
```

Training outputs:
- `results/example/best_checkpoint.ckpt`
- `results/example/last.ckpt`
- `results/example/config.yaml`
- `results/example/logs/**/metrics.csv`
- `results/example/summary.json`

## 2) Hyperparameter search (random search)
Search configs live under `configs/search/*.yaml`.

Run search:
```bash
uv run python -m deepkoopman.cli.search --config configs/search/discrete_spectrum.yaml
```

Search outputs:
- `results/search/<run_id>/trials.csv`
- `results/search/<run_id>/best_config.yaml`
- `results/search/<run_id>/best_checkpoint.ckpt`
- `results/search/<run_id>/summary.json`

## 3) Postprocessing (PNG/CSV)
```bash
uv run python -m deepkoopman.cli.postprocess --run-dir results/example
```

Eigenvalue-component heatmaps use a latent-space mesh. Override the default auto-bounds and density with:
```bash
uv run python -m deepkoopman.cli.postprocess --run-dir results/example --latent-grid-min=-1,-1 --latent-grid-max=1,1 --latent-grid-size 100
```
Eigenfunction heatmaps for 1D/2D data use a state-space mesh. Override it with:
```bash
uv run python -m deepkoopman.cli.postprocess --run-dir results/example --state-grid-min=-2,-2 --state-grid-max=2,2 --state-grid-size 100
```

Outputs:
- `.../postprocess/tables/test_metrics.json`
- `.../postprocess/tables/test_metrics.csv`
- `.../postprocess/tables/sampled_trajectories.csv`
- `.../postprocess/tables/statistical_tests.csv`
- `.../postprocess/tables/statistical_tests.json`
- `.../postprocess/tables/video_samples.csv`
- `.../postprocess/videos/reconstruction/{split}/{music-period}.mp4`
- `.../postprocess/videos/latent/latent_trajectory_grid.mp4`
- `.../postprocess/videos/latent/latent_prediction_grid.mp4`
- `.../postprocess/figures/flow_field_{condition}_{plane}.png`
- `.../postprocess/tables/flow_field_metadata.csv`
- `.../postprocess/figures/*_data_trajectories.png`
- `.../postprocess/figures/*_latent_true_vs_pred.png`
- `.../postprocess/figures/eigen_component_*_heatmap.png`
- `.../postprocess/figures/eigenfunction_*_heatmap.png` for data dimensions up to 2

## 4) Rat auditory cortex analysis
Rat preprocessing is config-driven. Preprocessing and source defaults live in `configs/rat_analysis/default.yaml`; the model training defaults live in `configs/train/rat.yaml`.
Rat metadata lives in `data/rat_id.csv`; the raw `.mat` root template is configured in `configs/rat_analysis/default.yaml` under `input.source.data_root_template`.

```bash
uv run python -m deepkoopman.cli.rat_preprocess --config configs/rat_analysis/default.yaml
uv run python -m deepkoopman.cli.train --config configs/train/rat.yaml --output-dir results/rat --wandb --wandb-project deepkoopman_rat
uv run python -m deepkoopman.cli.postprocess --run-dir results/rat --dataset RatAuditoryCortex --data-dir /mnt/outputs/DeepKoopman_pytorch/data
```

Use `--checkpoint last` to process `last.ckpt`; its default output directory is `results/rat/postprocess_last`.

Postprocessing uses CUDA when available. Pass `--device cpu` to force CPU execution.
Progress bars are enabled by default; pass `--no-progress` to disable them.
Rat statistics aggregate train, validation, and test rats before rat-level paired permutation tests. Video generation is enabled by default and can be disabled with `--no-videos`.

Rat preprocessing performs bad-channel rejection, common-average re-referencing, spatial interpolation, 1-200 Hz filtering with line-noise notches, 2 s / 0.5 s windowing, and 6-band log-power feature extraction. It writes per-block feature arrays under `processed/<rat_id>/<block_id>/features.npy` with shape `[time, 8, 8, 6]`, and chunked HDF5 training sequences under `data/RatAuditoryCortex.h5` with flattened 64-step sequences plus `/split/condition` labels.

Rat training uses a shared 3-D latent space with a condition-dependent Koopman eigenvalue network. Postprocessing writes sampled latent plots plus full test latent tables under `results/rat/latents/fold_0/` and condition/section figures under `results/rat/postprocess/figures/`.

## Differences from the original TensorFlow repository
- The old random search flow in `*Experiment.py` is replaced by `deepkoopman.cli.search` + nested YAML configs.
- The old `postprocessing/*.ipynb` flow is replaced by a shared visualization module and CLI postprocessing.

## Run tests
```bash
uv run pytest -q
```
