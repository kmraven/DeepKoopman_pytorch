# DeepKoopman (PyTorch)

PyTorch版 DeepKoopman 実装です。元リポジトリのランダムハイパラ探索と後処理可視化を、YAML設定 + CLI + marimo notebook で再構成しています。

## セットアップ
```bash
uv sync
```

## 1) 通常学習（例）
```bash
uv run python scripts/run_example.py --data-name DiscreteSpectrumExample --epochs 2
```

## 2) ハイパラ探索（ランダム探索）
YAML設定例:
- `configs/discrete_search.yaml`
- `configs/pendulum_search.yaml`
- `configs/fluid_attractor_search.yaml`
- `configs/fluid_box_search.yaml`

実行:
```bash
uv run python scripts/search_hparams.py --config configs/discrete_search.yaml
```

出力:
- `results/search/<run_id>/trials.csv`
- `results/search/<run_id>/best_config.yaml`
- `results/search/<run_id>/best_checkpoint.pt`
- `results/search/<run_id>/summary.json`

## 3) 後処理可視化（PNG/CSV）
```bash
uv run python scripts/postprocess.py --run-dir results/search/<run_id> --dataset DiscreteSpectrumExample
```

出力:
- `.../postprocess/figures/losses.png`
- `.../postprocess/figures/reconstruction.png`
- `.../postprocess/figures/prediction.png`
- `.../postprocess/tables/history.csv`
- `.../postprocess/tables/sample_*.csv`

## 4) marimo notebook
```bash
uv run marimo edit postprocessing_marimo/deepkoopman_postprocess.py
```

## 旧リポとの差分
- 旧 `*Experiment.py` のランダム探索は、PyTorch版では `scripts/search_hparams.py` + YAMLに置換。
- 旧 `postprocessing/*.ipynb` の可視化は、PyTorch版では `scripts/postprocess.py` と marimo notebook に置換。

## テスト
```bash
uv run pytest -q
```
