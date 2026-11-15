"""Load, validate, clean, and split the workload chronologically.

Preprocessing statistics that must not see future rows are fit on the
training split only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    ALL_COLUMNS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_ROWS,
    TRAIN_ROWS,
    VALIDATION_ROWS,
)


class TrainOnlyImputer:
    """Fill missing feature values with medians learned from training rows."""

    def __init__(self) -> None:
        self.medians_: dict[str, float] | None = None
        self.columns_: list[str] | None = None

    def fit(self, frame: pd.DataFrame) -> "TrainOnlyImputer":
        columns = [column for column in FEATURE_COLUMNS if column in frame.columns]
        medians = frame[columns].median(numeric_only=True)
        self.columns_ = columns
        self.medians_ = {column: float(medians[column]) for column in columns}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None or self.columns_ is None:
            raise RuntimeError("Fit the imputer on training data before calling transform.")
        transformed = frame.copy()
        for column, median in self.medians_.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(median)
        return transformed


def validate_columns(frame: pd.DataFrame) -> None:
    """Raise ValueError when required columns are missing."""
    missing = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(missing)
            + ". Expected columns: "
            + ", ".join(ALL_COLUMNS)
        )


def clean_workload(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, coerce types, and drop rows that cannot be used."""
    validate_columns(frame)
    cleaned = frame.loc[:, ALL_COLUMNS].copy()
    cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")
    for column in ["request_rate", "memory_usage", "response_time", TARGET_COLUMN]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned["active_vms"] = pd.to_numeric(cleaned["active_vms"], errors="coerce")
    cleaned["sla_violation"] = pd.to_numeric(cleaned["sla_violation"], errors="coerce")

    cleaned = cleaned.dropna(subset=ALL_COLUMNS)
    cleaned = cleaned.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    vm_is_whole = np.isclose(cleaned["active_vms"], np.rint(cleaned["active_vms"]))
    sla_is_binary = cleaned["sla_violation"].isin([0, 1])
    valid = (
        vm_is_whole
        & sla_is_binary
        & cleaned["request_rate"].between(0, 100_000)
        & cleaned["memory_usage"].between(0, 100)
        & cleaned["active_vms"].between(1, 1_000)
        & cleaned["response_time"].between(0, 100_000)
        & cleaned[TARGET_COLUMN].between(0, 100)
    )
    cleaned = cleaned.loc[valid].reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid workload rows remain after preprocessing.")

    cleaned["active_vms"] = cleaned["active_vms"].round().astype(int)
    cleaned["sla_violation"] = cleaned["sla_violation"].round().astype(int)
    return cleaned


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Read a CSV and return a cleaned chronological frame."""
    path = Path(path or DATASET_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Generate the simulated file with: python src\\data_generator.py --overwrite"
        )
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Dataset at {path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Dataset at {path} could not be parsed as CSV.") from exc
    return clean_workload(frame)


def chronological_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Split without shuffling.

    A dataset of at least 20,000 rows uses the first 14,000 for training,
    the next 3,000 for validation, and the next 3,000 for testing. Extra rows
    are not used. Smaller datasets use a 70/15/15 chronological split.
    """
    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("Split expects rows sorted by timestamp. Clean the dataset first.")

    row_count = len(frame)
    required = TRAIN_ROWS + VALIDATION_ROWS + TEST_ROWS
    if row_count >= required:
        train = frame.iloc[:TRAIN_ROWS].copy()
        validation = frame.iloc[TRAIN_ROWS : TRAIN_ROWS + VALIDATION_ROWS].copy()
        test = frame.iloc[TRAIN_ROWS + VALIDATION_ROWS : required].copy()
        unused = row_count - required
        note = (
            f"Chronological split {TRAIN_ROWS}/{VALIDATION_ROWS}/{TEST_ROWS}."
            + (f" {unused} extra rows were not used." if unused else "")
        )
        return train, validation, test, note

    train_end = max(1, int(row_count * 0.70))
    validation_end = max(train_end + 1, int(row_count * 0.85))
    if validation_end >= row_count:
        validation_end = row_count - 1
    train = frame.iloc[:train_end].copy()
    validation = frame.iloc[train_end:validation_end].copy()
    test = frame.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) < 1:
        raise ValueError("Dataset is too small for a train, validation, and test split.")
    note = (
        f"Dataset has {row_count} rows, fewer than {required}. "
        f"Used a chronological {len(train)}/{len(validation)}/{len(test)} split "
        "instead of 14000/3000/3000."
    )
    return train, validation, test, note


def prepare_matrices(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    imputer: TrainOnlyImputer | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, TrainOnlyImputer]:
    """Fit imputation on training features and transform every split."""
    imputer = imputer or TrainOnlyImputer()
    imputer.fit(train)
    train_x = imputer.transform(train)[FEATURE_COLUMNS]
    validation_x = imputer.transform(validation)[FEATURE_COLUMNS]
    test_x = imputer.transform(test)[FEATURE_COLUMNS]
    return (
        train_x,
        train[TARGET_COLUMN].astype(float),
        validation_x,
        validation[TARGET_COLUMN].astype(float),
        test_x,
        test[TARGET_COLUMN].astype(float),
        imputer,
    )
