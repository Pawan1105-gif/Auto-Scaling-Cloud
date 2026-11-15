"""Load the saved Random Forest and predict CPU utilization."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import FEATURE_COLUMNS, MODEL_PATH

try:
    import joblib
except ImportError as exc:  # pragma: no cover - joblib ships with scikit-learn
    raise ImportError("joblib is required to load the trained model.") from exc


class ModelNotFoundError(FileNotFoundError):
    """Raised when the serialized model file is missing."""


def load_artifact(path: Path | None = None) -> dict:
    """Load the joblib bundle produced by ``train_model.py``."""
    path = Path(path or MODEL_PATH)
    if not path.exists():
        raise ModelNotFoundError(
            f"Trained model not found at {path}. Train it with: python src\\train_model.py"
        )
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ValueError(
            f"Could not read the model at {path}. Retrain it with: python src\\train_model.py"
        ) from exc
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("The model file does not contain a training bundle. Retrain the model.")
    if "feature_columns" not in bundle:
        bundle["feature_columns"] = list(FEATURE_COLUMNS)
    return bundle


def validate_feature_frame(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Check interactive or batch features before prediction."""
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError("Missing features: " + ", ".join(missing))
    values = frame.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce")
    if values.isnull().any().any():
        bad_rows = values.isnull().any(axis=1)
        raise ValueError(
            "Features must be numeric. Invalid rows: "
            + ", ".join(str(index) for index in values.index[bad_rows][:5])
        )
    if (values["memory_usage"] < 0).any() or (values["memory_usage"] > 100).any():
        raise ValueError("memory_usage must be between 0 and 100.")
    if (values["request_rate"] < 0).any():
        raise ValueError("request_rate cannot be negative.")
    if (values["response_time"] < 0).any():
        raise ValueError("response_time cannot be negative.")
    if (~values["active_vms"].between(1, 1000)).any():
        raise ValueError("active_vms must be between 1 and 1000.")
    if (~values["sla_violation"].isin([0, 1])).any():
        raise ValueError("sla_violation must be 0 or 1.")
    return values


def predict_cpu(bundle: dict, features: pd.DataFrame) -> np.ndarray:
    """Predict CPU utilization for one or more feature rows."""
    columns = list(bundle["feature_columns"])
    values = validate_feature_frame(features, columns)
    model = bundle["model"]
    predictions = np.asarray(model.predict(values), dtype=float)
    if predictions.shape[0] != len(values):
        raise RuntimeError("The model returned an unexpected number of predictions.")
    return predictions


def predict_one(
    request_rate: float,
    memory_usage: float,
    active_vms: int,
    response_time: float,
    sla_violation: int,
    bundle: dict | None = None,
    model_path: Path | None = None,
) -> float:
    """Predict CPU utilization for a single simulated observation."""
    artifact = bundle if bundle is not None else load_artifact(model_path)
    frame = pd.DataFrame(
        [
            {
                "request_rate": request_rate,
                "memory_usage": memory_usage,
                "active_vms": active_vms,
                "response_time": response_time,
                "sla_violation": sla_violation,
            }
        ]
    )
    return float(predict_cpu(artifact, frame)[0])
