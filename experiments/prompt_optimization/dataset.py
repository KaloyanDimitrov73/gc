"""Dataset preparation for the question-processing optimization experiment."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {
    "uid",
    "question",
    "source_ids",
    "golden_triples",
    "based_on_template",
    "retrieval_operation",
}


@dataclass(frozen=True)
class DatasetSplit:
    """Paths and row counts produced by a grouped dataset split."""

    train_path: Path
    validation_path: Path
    test_path: Path
    train_rows: int
    validation_rows: int
    test_rows: int


def _validate_dataset(data: pd.DataFrame) -> None:
    missing_columns = REQUIRED_COLUMNS - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    if data.empty:
        raise ValueError("Dataset must contain at least one row.")

    for column in ("question", "based_on_template", "retrieval_operation"):
        if data[column].isna().any() or data[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Dataset column {column!r} contains empty values.")

    operations_per_template = data.groupby("based_on_template")[
        "retrieval_operation"
    ].nunique()
    mixed_templates = operations_per_template[operations_per_template > 1]
    if not mixed_templates.empty:
        raise ValueError(
            "Each based_on_template group must have exactly one "
            "retrieval_operation. Mixed templates: "
            f"{mixed_templates.index.tolist()}"
        )


def create_grouped_splits(
    dataset_path: Path | str,
    output_dir: Path | str,
    *,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create deterministic 60/20/20 train/validation/test CSV files.

    Questions generated from the same ``based_on_template`` are kept in the
    same split. ``retrieval_operation`` is used only for stratification so that
    the query-operation distribution stays approximately balanced.
    """

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    data = pd.read_csv(dataset_path)
    _validate_dataset(data)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_seed,
    )
    fold_ids = pd.Series(-1, index=data.index, dtype="int64")

    for fold_id, (_, held_out_indices) in enumerate(
        splitter.split(
            X=data,
            y=data["retrieval_operation"],
            groups=data["based_on_template"],
        )
    ):
        fold_ids.iloc[held_out_indices] = fold_id

    if (fold_ids < 0).any():
        raise RuntimeError("Not every dataset row was assigned to a fold.")

    test_fold, validation_fold = _select_fold_roles(data, fold_ids)
    train = data.loc[
        (fold_ids != test_fold) & (fold_ids != validation_fold)
    ].copy()
    validation = data.loc[fold_ids == validation_fold].copy()
    test = data.loc[fold_ids == test_fold].copy()
    _assert_disjoint_templates(train, validation, test)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.csv"
    validation_path = output_dir / "validation.csv"
    test_path = output_dir / "test.csv"
    train.to_csv(train_path, index=False)
    validation.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    return DatasetSplit(
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        train_rows=len(train),
        validation_rows=len(validation),
        test_rows=len(test),
    )


def _select_fold_roles(
    data: pd.DataFrame,
    fold_ids: pd.Series,
) -> tuple[int, int]:
    """Choose validation/test folds with the best class balance.

    StratifiedGroupKFold balances each fold approximately. With a small grouped
    dataset, assigning fixed fold numbers can still omit a class from validation
    or test. We deterministically inspect all fold-role permutations using only
    labels and group sizes, never retrieval outcomes.
    """

    classes = sorted(data["retrieval_operation"].unique())
    total_counts = data["retrieval_operation"].value_counts()
    available_folds = sorted(fold_ids.unique())
    best: tuple[tuple[float, ...], int, int] | None = None

    for test_fold, validation_fold in permutations(available_folds, 2):
        masks = (
            (fold_ids != test_fold) & (fold_ids != validation_fold),
            fold_ids == validation_fold,
            fold_ids == test_fold,
        )
        splits = [data.loc[mask] for mask in masks]
        ratios = (0.6, 0.2, 0.2)
        class_counts = [
            split["retrieval_operation"].value_counts().reindex(
                classes, fill_value=0
            )
            for split in splits
        ]
        missing_class_count = sum(
            int(count == 0) for counts in class_counts for count in counts
        )
        class_deviation = sum(
            abs(counts[class_name] - total_counts[class_name] * ratio)
            for counts, ratio in zip(class_counts, ratios)
            for class_name in classes
        )
        size_deviation = sum(
            abs(len(split) - len(data) * ratio)
            for split, ratio in zip(splits, ratios)
        )
        score = (
            float(missing_class_count),
            float(class_deviation),
            float(size_deviation),
            float(test_fold),
            float(validation_fold),
        )
        candidate = (score, int(test_fold), int(validation_fold))
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        raise RuntimeError("Could not assign validation and test folds.")
    return best[1], best[2]


def _assert_disjoint_templates(*splits: pd.DataFrame) -> None:
    template_sets = [set(split["based_on_template"]) for split in splits]
    for left_index, left in enumerate(template_sets):
        for right in template_sets[left_index + 1 :]:
            overlap = left & right
            if overlap:
                raise RuntimeError(
                    "based_on_template leakage detected between splits: "
                    f"{sorted(overlap)}"
                )


def parse_string_list(value: Any) -> list[str]:
    """Parse list-valued CSV cells without executing arbitrary code."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError) as error:
            raise ValueError(f"Invalid list value: {value!r}") from error
    else:
        raise ValueError(f"Expected a string-encoded list, received {type(value)}")

    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"Expected a list of strings, received {parsed!r}")
    return parsed


def load_dspy_examples(
    csv_path: Path | str,
    *,
    limit: int | None = None,
) -> list[Any]:
    """Load a split as DSPy examples while keeping gold fields metric-only."""

    try:
        import dspy
    except ImportError as error:
        raise RuntimeError(
            "DSPy is not installed. Create the dedicated environment and run "
            "`pip install -r requirements.txt` from "
            "experiments/prompt_optimization."
        ) from error

    data = pd.read_csv(csv_path)
    _validate_dataset(data)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero.")
        data = data.head(limit)

    examples = []
    for row in data.to_dict(orient="records"):
        examples.append(
            dspy.Example(
                uid=str(row["uid"]),
                question=row["question"],
                expected_source_ids=parse_string_list(row["source_ids"]),
                expected_triples=parse_string_list(row["golden_triples"]),
                retrieval_operation=row["retrieval_operation"],
                based_on_template=row["based_on_template"],
            ).with_inputs("question")
        )
    return examples


def describe_split(csv_path: Path | str) -> str:
    """Return a compact human-readable split summary."""

    data = pd.read_csv(csv_path)
    distribution = data["retrieval_operation"].value_counts().sort_index()
    counts = ", ".join(f"{name}={count}" for name, count in distribution.items())
    return (
        f"rows={len(data)}, templates={data['based_on_template'].nunique()}, "
        f"operations=[{counts}]"
    )
