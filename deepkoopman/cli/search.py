from __future__ import annotations

import argparse

from deepkoopman.search import run_random_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    summary = run_random_search(args.config)
    print(summary)


if __name__ == "__main__":
    main()
