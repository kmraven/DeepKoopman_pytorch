from __future__ import annotations

import argparse
import json

from deepkoopman.postprocess import run_postprocess


def _parse_pair(value: str, *, cast=float) -> tuple:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values")
    try:
        return (cast(parts[0]), cast(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latent-grid-size", type=int, default=100)
    parser.add_argument("--latent-grid-dims", type=lambda value: _parse_pair(value, cast=int), default=(0, 1))
    parser.add_argument("--latent-grid-min", type=lambda value: _parse_pair(value, cast=float), default=None)
    parser.add_argument("--latent-grid-max", type=lambda value: _parse_pair(value, cast=float), default=None)
    parser.add_argument("--state-grid-size", type=int, default=100)
    parser.add_argument("--state-grid-min", type=lambda value: _parse_pair(value, cast=float), default=None)
    parser.add_argument("--state-grid-max", type=lambda value: _parse_pair(value, cast=float), default=None)
    args = parser.parse_args()
    if (args.latent_grid_min is None) != (args.latent_grid_max is None):
        parser.error("--latent-grid-min and --latent-grid-max must be specified together")
    if (args.state_grid_min is None) != (args.state_grid_max is None):
        parser.error("--state-grid-min and --state-grid-max must be specified together")

    summary = run_postprocess(
        args.run_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
        latent_grid_size=args.latent_grid_size,
        latent_grid_dims=args.latent_grid_dims,
        latent_grid_min=args.latent_grid_min,
        latent_grid_max=args.latent_grid_max,
        state_grid_size=args.state_grid_size,
        state_grid_min=args.state_grid_min,
        state_grid_max=args.state_grid_max,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
