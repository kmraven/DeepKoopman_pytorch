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


def _sample_space(space: dict, rng: random.Random):
    kind = space["type"]
    if kind == "choice":
        return rng.choice(space["values"])
    if kind == "int":
        return rng.randint(int(space["low"]), int(space["high"]))
    if kind == "float_log":
        low = float(space["low"])
        high = float(space["high"])
        return 10 ** rng.uniform(low, high)
    raise ValueError(f"Unknown search space type: {kind}")


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

    data_name = fixed["data_name"]
    data_dir = Path(search_cfg.get("data_dir", "data"))
    train = np.loadtxt(data_dir / f"{data_name}_train1_x.csv", delimiter=",", dtype=np.float64)
    val = np.loadtxt(data_dir / f"{data_name}_val_x.csv", delimiter=",", dtype=np.float64)

    rows = []
    best = None
    best_metric = float("inf")

    for trial in range(num_trials):
        rng = random.Random(base_seed + trial)
        params = deepcopy(fixed)
        for k, spec in spaces.items():
            params[k] = _sample_space(spec, rng)

        cfg = DeepKoopmanConfig(**params)
        cfg.logging.backend = search_cfg.get("logging", {}).get("backend", "csv")
        cfg.logging.save_dir = str(out_dir / "logs")
        cfg.logging.name = f"trial_{trial:03d}"
        cfg.trainer.enable_progress_bar = bool(search_cfg.get("progress", False))
        module = DeepKoopmanLightningModule(cfg)
        datamodule = DeepKoopmanDataModule(train, val, cfg)
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
