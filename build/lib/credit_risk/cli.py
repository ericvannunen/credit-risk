from __future__ import annotations

import argparse

from .pipeline import TARGET_RECALL, run_training_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare credit-risk models")
    parser.add_argument("--data-dir", default="data", help="Directory for UCI dataset cache")
    parser.add_argument("--data-path", default=None, help="Path to local 5year.arff file")
    parser.add_argument(
        "--target-recall",
        type=float,
        default=TARGET_RECALL,
        help="Minimum validation recall used to select decision thresholds",
    )
    args = parser.parse_args()
    print(
        run_training_json(
            data_dir=args.data_dir,
            data_path=args.data_path,
            target_recall=args.target_recall,
        )
    )


if __name__ == "__main__":
    main()
