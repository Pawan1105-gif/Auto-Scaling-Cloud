"""Metrics, baselines, workload summaries, and a closed-loop scaling replay."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import FEATURE_COLUMNS, TARGET_COLUMN
from src.autoscaler import AutoScaler
from src.data_preprocessing import TrainOnlyImputer

# Illustrative on-demand price used only for the simulated cost figure.
HOURLY_VM_RATE_USD = 0.023


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Return MAE, RMSE, and R² for one set of predictions."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError("Actual and predicted arrays must have the same shape.")
    if actual.size == 0:
        raise ValueError("Cannot score an empty prediction set.")
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def persistence_predictions(previous_value: float, y_true) -> np.ndarray:
    """Forecast each point with the previous observed CPU value."""
    actual = np.asarray(y_true, dtype=float)
    if actual.size == 0:
        raise ValueError("Persistence needs at least one actual value.")
    if not np.isfinite(previous_value):
        raise ValueError("The previous observation must be finite.")
    shifted = np.empty_like(actual)
    shifted[0] = float(previous_value)
    if actual.size > 1:
        shifted[1:] = actual[:-1]
    return shifted


def mean_predictions(train_mean: float, size: int) -> np.ndarray:
    """Forecast every point with the training-set mean."""
    if size < 1:
        raise ValueError("Mean baseline needs at least one row.")
    if not np.isfinite(train_mean):
        raise ValueError("Training mean must be finite.")
    return np.full(size, float(train_mean), dtype=float)


def metric_row(model_name: str, split_name: str, metrics: dict[str, float], note: str) -> dict:
    return {
        "model": model_name,
        "split": split_name,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "data_origin": "simulated",
        "notes": note,
    }


def summarize_workload(frame: pd.DataFrame) -> dict[str, float]:
    """Summarize a simulated trace. The cost is illustrative, not a cloud bill."""
    if frame.empty:
        raise ValueError("Cannot summarize an empty workload.")
    minutes = float(len(frame))
    hours = minutes / 60.0
    average_vms = float(frame["active_vms"].mean())
    vm_changes = int((frame["active_vms"].diff().fillna(0) != 0).sum())
    return {
        "rows": minutes,
        "hours": hours,
        "average_cpu": float(frame[TARGET_COLUMN].mean()),
        "average_memory": float(frame["memory_usage"].mean()),
        "average_request_rate": float(frame["request_rate"].mean()),
        "average_response_time": float(frame["response_time"].mean()),
        "average_vms": average_vms,
        "sla_violation_percent": float(frame["sla_violation"].mean() * 100.0),
        "recorded_vm_changes": vm_changes,
        "estimated_cost_usd": average_vms * hours * HOURLY_VM_RATE_USD,
        "hourly_vm_rate_usd": HOURLY_VM_RATE_USD,
    }


def simulate_controller(
    frame: pd.DataFrame,
    model,
    imputer: TrainOnlyImputer,
    feature_columns: list[str] | None = None,
    min_vms: int = 1,
    max_vms: int = 10,
    scale_up_threshold: float = 80.0,
    scale_down_threshold: float = 30.0,
    cooldown_steps: int = 1,
    initial_vms: int | None = None,
) -> pd.DataFrame:
    """Replay the scaler over a trace.

    At each minute the model sees that minute's monitored features, except
    ``active_vms``, which is the controller's current simulated count. The
    recorded dataset VM count is not fed back in, so this is a what-if
    simulation rather than a replay of historical operator actions.
    """
    feature_columns = list(feature_columns or FEATURE_COLUMNS)
    data = frame.reset_index(drop=True)
    if data.empty:
        raise ValueError("Cannot simulate an empty frame.")
    missing = [column for column in feature_columns + ["timestamp"] if column not in data.columns]
    if missing:
        raise ValueError("Simulation frame is missing columns: " + ", ".join(missing))

    start_vms = int(data.loc[0, "active_vms"] if initial_vms is None else initial_vms)
    start_vms = int(np.clip(start_vms, min_vms, max_vms))
    scaler = AutoScaler(
        min_vms=min_vms,
        max_vms=max_vms,
        scale_up_threshold=scale_up_threshold,
        scale_down_threshold=scale_down_threshold,
        cooldown_steps=cooldown_steps,
        initial_vms=start_vms,
    )

    pieces = []
    base = data[feature_columns].copy()
    for vm_count in range(min_vms, max_vms + 1):
        candidate = base.copy()
        candidate["active_vms"] = vm_count
        pieces.append(candidate)
    stacked = imputer.transform(pd.concat(pieces, ignore_index=True))
    predictions = np.asarray(model.predict(stacked[feature_columns]), dtype=float)
    grid = predictions.reshape(max_vms - min_vms + 1, len(data))

    for index in range(len(data)):
        vm_count = scaler.current_vms
        predicted = float(grid[vm_count - min_vms, index])
        scaler.decide(
            predicted,
            current_vms=vm_count,
            timestamp=str(data.loc[index, "timestamp"]),
            mode="automatic-simulation",
        )
    return pd.DataFrame(decision.to_dict() for decision in scaler.history)


def count_scaling_actions(history: pd.DataFrame) -> dict[str, int]:
    """Count committed scale-up and scale-down actions."""
    if history.empty or "action" not in history.columns:
        return {"scale_up": 0, "scale_down": 0, "maintain": 0, "changes": 0}
    counts = history["action"].value_counts().to_dict()
    scale_up = int(counts.get("scale_up", 0))
    scale_down = int(counts.get("scale_down", 0))
    maintain = int(counts.get("maintain", 0))
    return {
        "scale_up": scale_up,
        "scale_down": scale_down,
        "maintain": maintain,
        "changes": scale_up + scale_down,
    }
