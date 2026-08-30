"""CLI for producing leak-free prompt-optimization dataset splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import create_grouped_splits, describe_split


HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split the QA dataset by based_on_template while stratifying by "
            "retrieval_operation."
        )
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "deep_distributed_graph_dataset.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    split = create_grouped_splits(
        args.dataset,
        args.output_dir,
        random_seed=args.seed,
    )
    for name, path in (
        ("train", split.train_path),
        ("validation", split.validation_path),
        ("test", split.test_path),
    ):
        print(f"{name}: {path} ({describe_split(path)})")


if __name__ == "__main__":
    main()
