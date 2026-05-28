# DeepKoopman (PyTorch)

This repository provides a PyTorch-based DeepKoopman implementation.
It also includes random hyperparameter search and postprocessing/visualization workflows that mirror the original repository's experiment style.

## Setup
```bash
uv sync
```

## 1) Train a model (example)
`run_example.py` is config-driven (YAML).

```bash
uv run python scripts/run_example.py --config configs/discrete_train.yaml
```

Optional overrides:
```bash
uv run python scripts/run_example.py --config configs/discrete_train.yaml --epochs 1 --batch-size 128
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
- `results/search/<run_id>/best_checkpoint.pt`
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
- The old `postprocessing/*.ipynb` flow is replaced by a shared visualization module, CLI postprocessing, and a marimo notebook.

## Run tests
```bash
uv run pytest -q
```
