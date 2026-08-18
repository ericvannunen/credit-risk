from __future__ import annotations

import argparse

from .pipeline import run_training_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare credit-risk models")
    parser.add_argument("--data-dir", default="data", help="Directory for UCI dataset cache")
    parser.add_argument("--data-path", default=None, help="Path to local 1year.arff file")
    args = parser.parse_args()
    print(run_training_json(data_dir=args.data_dir, data_path=args.data_path))


if __name__ == "__main__":
    main()
