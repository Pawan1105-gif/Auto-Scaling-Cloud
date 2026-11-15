"""Train the Random Forest, score the baselines, and save the artifacts."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    COMPARISON_GRAPH_PATH,
    DATA_NOTE,
    DATA_ORIGIN,
    DATASET_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    SCALING_HISTORY_PATH,
    TARGET_COLUMN,
)
from src.data_preprocessing import load_dataset, chronological_split, prepare_matrices
from src.evaluation import (
    mean_predictions,
    metric_row,
    persistence_predictions,
    regression_metrics,
    simulate_controller,
)

DEFAULT_CANDIDATES = [
    {"n_estimators": 80, "max_depth": 8, "min_samples_leaf": 3},
    {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 2},
    {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 2},
]
METRIC_NOTE = "Computed on the synthetic workload. Not a published paper score."


def train_random_forest(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict,
    n_jobs: int = 1,
) -> RandomForestRegressor:
    """Fit one Random Forest. ``n_jobs=1`` avoids Windows process-spawn issues."""
    model = RandomForestRegressor(
        random_state=42,
        n_jobs=n_jobs,
        **params,
    )
    model.fit(x_train, y_train)
    return model


def select_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    candidates: list[dict] | None = None,
    n_jobs: int = 1,
) -> tuple[RandomForestRegressor, dict, dict[str, float], pd.DataFrame]:
    """Choose the candidate with the lowest validation RMSE."""
    candidates = candidates or DEFAULT_CANDIDATES
    best_model = None
    best_params: dict | None = None
    best_metrics: dict[str, float] | None = None
    rows = []
    for params in candidates:
        model = train_random_forest(x_train, y_train, params, n_jobs=n_jobs)
        validation_metrics = regression_metrics(y_validation, model.predict(x_validation))
        rows.append({"split": "validation", **params, **validation_metrics})
        if best_metrics is None or validation_metrics["rmse"] < best_metrics["rmse"]:
            best_model = model
            best_params = dict(params)
            best_metrics = validation_metrics
    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("Model selection did not produce a model.")
    return best_model, best_params, best_metrics, pd.DataFrame(rows)


def save_comparison_graph(
    metrics: pd.DataFrame,
    y_true,
    y_pred,
    path: Path,
) -> Path:
    """Save a bar chart of the test metrics and a short actual-versus-predicted trace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    step = max(1, len(actual) // 400)
    sample_index = np.arange(0, len(actual), step)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    models = test_metrics["model"].tolist()
    x_pos = np.arange(len(models))
    width = 0.36
    axes[0].bar(x_pos - width / 2, test_metrics["mae"], width, label="MAE", color="#2563eb")
    axes[0].bar(x_pos + width / 2, test_metrics["rmse"], width, label="RMSE", color="#d97706")
    axes[0].set_xticks(x_pos, models)
    axes[0].set_ylabel("Error (CPU percentage points)")
    axes[0].set_title("Test error on simulated data")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(sample_index, actual[sample_index], label="Actual", color="#64748b", linewidth=1.2)
    axes[1].plot(
        sample_index,
        predicted[sample_index],
        label="Random Forest",
        color="#0284c7",
        linewidth=1.2,
    )
    axes[1].set_xlabel("Test-row index")
    axes[1].set_ylabel("CPU utilization (%)")
    axes[1].set_title("Actual versus predicted CPU")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.suptitle("Simulated evaluation — not published research results", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def run_training(
    data_path: Path = DATASET_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
    history_path: Path = SCALING_HISTORY_PATH,
    graph_path: Path = COMPARISON_GRAPH_PATH,
    candidates: list[dict] | None = None,
    n_jobs: int = 1,
    history_rows: int | None = None,
) -> dict:
    """Train, evaluate, and write the model, metrics, graph, and scaling log."""
    frame = load_dataset(data_path)
    train, validation, test, split_note = chronological_split(frame)
    x_train, y_train, x_validation, y_validation, x_test, y_test, imputer = prepare_matrices(
        train, validation, test
    )
    model, params, validation_metrics, _selection = select_model(
        x_train,
        y_train,
        x_validation,
        y_validation,
        candidates=candidates,
        n_jobs=n_jobs,
    )
    forest_test = regression_metrics(y_test, model.predict(x_test))
    previous = float(validation[TARGET_COLUMN].iloc[-1])
    persistence = persistence_predictions(previous, y_test)
    mean_pred = mean_predictions(float(y_train.mean()), len(y_test))
    rows = [
        metric_row("Random Forest", "validation", validation_metrics, METRIC_NOTE),
        metric_row("Random Forest", "test", forest_test, METRIC_NOTE),
        metric_row("Persistence", "test", regression_metrics(y_test, persistence), METRIC_NOTE),
        metric_row("Mean", "test", regression_metrics(y_test, mean_pred), METRIC_NOTE),
    ]
    metrics = pd.DataFrame(rows)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    replay = test if history_rows is None else test.iloc[:history_rows]
    history = simulate_controller(replay, model, imputer, FEATURE_COLUMNS, cooldown_steps=1)
    history.to_csv(history_path, index=False)
    save_comparison_graph(metrics, y_test, model.predict(x_test), graph_path)

    importances = {
        column: float(value)
        for column, value in zip(FEATURE_COLUMNS, model.feature_importances_)
    }
    bundle = {
        "model": model,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "imputer": imputer,
        "imputer_medians": imputer.medians_,
        "train_target_mean": float(y_train.mean()),
        "selected_params": params,
        "feature_importances": importances,
        "metrics": rows,
        "split_note": split_note,
        "data_origin": DATA_ORIGIN,
        "data_note": DATA_NOTE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    return {
        "metrics": metrics,
        "split_note": split_note,
        "model_path": model_path,
        "params": params,
        "history_rows": int(len(history)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the simulated CPU Random Forest.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args(argv)
    result = run_training(data_path=args.data, model_path=args.model, n_jobs=args.n_jobs)
    print(result["split_note"])
    print(DATA_NOTE)
    print(f"Selected parameters: {result['params']}")
    print(result["metrics"].round(4).to_string(index=False))
    print(f"Model saved: {result['model_path']}")
    print(f"Scaling-history rows: {result['history_rows']}")


if __name__ == "__main__":
    main()
