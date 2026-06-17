from __future__ import annotations

import csv
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from .config import DeepKoopmanConfig
from .data import DeepKoopmanDataModule
from .lightning import DeepKoopmanLightningModule, build_trainer
from .io import train_paths_for_dataset


def _range_int_values(space: dict) -> list[int]:
    step = int(space.get("step", 1))
    if step <= 0:
        raise ValueError("range_int step must be positive")
    low = int(space["low"])
    high = int(space["high"])
    return list(range(low, high + 1, step))


def _sample_scalar_space(space: dict, rng: random.Random):
    kind = space["type"]
    if kind == "choice":
        return rng.choice(space["values"])
    if kind == "int":
        return rng.randint(int(space["low"]), int(space["high"]))
    if kind == "range_int":
        return rng.choice(_range_int_values(space))
    if kind == "float_log":
        low = float(space["low"])
        high = float(space["high"])
        return 10 ** rng.uniform(low, high)
    raise ValueError(f"Unknown scalar search space type: {kind}")


def _sample_depth_space(space: dict, rng: random.Random) -> tuple[int, int]:
    depth_space = rng.choice(space["depth_spaces"])
    depth = int(depth_space["depth"])
    width = int(_sample_scalar_space(depth_space["widths"], rng))
    return depth, width


def _sample_space(space: dict, rng: random.Random):
    kind = space["type"]
    if kind in {"choice", "int", "range_int", "float_log"}:
        return _sample_scalar_space(space, rng)
    if kind == "hidden_width_template":
        depth, width = _sample_depth_space(space, rng)
        return [width] * depth
    if kind == "symmetric_width_template":
        depth, width = _sample_depth_space(space, rng)
        input_dim = int(space["input_dim"])
        latent_dim = int(space["latent_dim"])
        hidden = [width] * depth
        return [input_dim, *hidden, latent_dim, latent_dim, *reversed(hidden), input_dim]
    raise ValueError(f"Unknown search space type: {kind}")


def _set_by_path(payload: dict, dotted_path: str, value) -> None:
    current = payload
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _load_training_data(data_dir: Path, data_name: str, train_files: int) -> np.ndarray:
    paths = train_paths_for_dataset(data_dir, data_name, train_files)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing training data files for {data_name}: {missing}")
    arrays = [np.loadtxt(path, delimiter=",", dtype=np.float64) for path in paths]
    return np.concatenate(arrays, axis=0)


def run_random_search(search_config_path: str | Path) -> dict:
    with open(search_config_path, "r", encoding="utf-8") as f:
        search_cfg = yaml.safe_load(f)

    fixed = deepcopy(search_cfg["fixed"])
    spaces = search_cfg["search_spaces"]
    num_trials = int(search_cfg.get("num_trials", 10))
    base_seed = int(search_cfg.get("seed", 42))
    metric = search_cfg.get("metric", "loss")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(search_cfg.get("output_dir", "results/search")) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    data_name = fixed["data"]["name"]
    data_dir = Path(fixed["data"].get("root", search_cfg.get("data_dir", "data")))
    val = np.loadtxt(data_dir / f"{data_name}_val_x.csv", delimiter=",", dtype=np.float64)

    rows = []
    best = None
    best_metric = float("inf")

    for trial in range(num_trials):
        rng = random.Random(base_seed + trial)
        params = deepcopy(fixed)
        for k, spec in spaces.items():
            _set_by_path(params, k, _sample_space(spec, rng))

        cfg = DeepKoopmanConfig(**params)
        trial_data_dir = Path(cfg.data.root)
        if not trial_data_dir.is_absolute():
            trial_data_dir = data_dir if trial_data_dir == data_dir or str(trial_data_dir) == str(data_dir) else Path(trial_data_dir)
        train_for_trial = _load_training_data(trial_data_dir, cfg.data.name, cfg.data.train_files)
        cfg.logging.backend = search_cfg.get("logging", {}).get("backend", "csv")
        cfg.logging.save_dir = str(out_dir / "logs")
        cfg.logging.name = f"trial_{trial:03d}"
        cfg.trainer.enable_progress_bar = bool(search_cfg.get("progress", False))
        module = DeepKoopmanLightningModule(cfg)
        datamodule = DeepKoopmanDataModule(train_for_trial, val, cfg)
        trial_dir = out_dir / f"trial_{trial:03d}"
        trainer = build_trainer(
            cfg,
            default_root_dir=trial_dir,
            checkpoint_dir=trial_dir / "checkpoints",
            use_wandb=cfg.logging.backend == "wandb",
            run_name=f"trial_{trial:03d}",
        )
        trainer.fit(module, datamodule=datamodule)
        metric_name = metric if "/" in metric else f"val/{metric}"
        if metric_name not in trainer.callback_metrics:
            raise KeyError(f"Metric {metric_name!r} not found. Available: {sorted(trainer.callback_metrics)}")
        score = float(trainer.callback_metrics[metric_name].detach().cpu())

        ckpt_path = Path(trainer.checkpoint_callback.best_model_path)

        row = {"trial": trial, "metric": score, **params}
        rows.append(row)

        if score < best_metric:
            best_metric = score
            best = {
                "trial": trial,
                "metric": score,
                "config": params,
                "checkpoint": str(ckpt_path),
            }

    fields = sorted({k for r in rows for k in r.keys()})
    with open(out_dir / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with open(out_dir / "best_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(best["config"], f, sort_keys=False)

    best_ckpt = Path(best["checkpoint"])
    best_target = out_dir / "best_checkpoint.ckpt"
    best_target.write_bytes(best_ckpt.read_bytes())

    summary = {
        "run_dir": str(out_dir),
        "num_trials": num_trials,
        "metric": metric,
        "best_trial": best["trial"],
        "best_metric": best["metric"],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
