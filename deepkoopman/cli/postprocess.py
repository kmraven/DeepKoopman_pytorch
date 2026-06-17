from __future__ import annotations

import argparse
import json

from deepkoopman.postprocess import run_postprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--samples-per-split", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = run_postprocess(
        args.run_dir,
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir,
        samples_per_split=args.samples_per_split,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
