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
from .model import DeepKoopmanModule
from .trainer import DeepKoopmanTrainer


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
        model = DeepKoopmanModule(cfg)
        trainer = DeepKoopmanTrainer(model, cfg)
        history = trainer.fit(train, val)
        final = history[-1]
        score = float(final[metric])

        ckpt_path = out_dir / f"trial_{trial:03d}.pt"
        trainer.save(ckpt_path)

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
    best_target = out_dir / "best_checkpoint.pt"
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
