"""Dataset generation, chronological preprocessing, and model scoring."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import ALL_COLUMNS, MODEL_PATH, TARGET_COLUMN, TEST_ROWS, TRAIN_ROWS, VALIDATION_ROWS
from src.data_generator import generate_workload, save_workload
from src.data_preprocessing import (
    TrainOnlyImputer,
    chronological_split,
    clean_workload,
    load_dataset,
    prepare_matrices,
)
from src.evaluation import mean_predictions, persistence_predictions, regression_metrics
from src.monitoring import snapshot_from_inputs
from src.prediction import load_artifact, predict_one
from src.train_model import run_training


def test_generator_is_reproducible_and_labeled_by_its_columns():
    first = generate_workload(240, seed=42)
    second = generate_workload(240, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ALL_COLUMNS
    assert first["cpu_utilization"].between(0, 100).all()
    assert first["memory_usage"].between(0, 100).all()
    assert first["active_vms"].between(1, 10).all()
    assert set(first["sla_violation"].unique()).issubset({0, 1})
    assert first["timestamp"].is_monotonic_increasing


def test_full_dataset_uses_the_required_chronological_sizes():
    frame = clean_workload(generate_workload(20_000, seed=42))
    train, validation, test, note = chronological_split(frame)
    assert len(train) == TRAIN_ROWS
    assert len(validation) == VALIDATION_ROWS
    assert len(test) == TEST_ROWS
    assert "14000/3000/3000" in note
    assert train["timestamp"].max() <= validation["timestamp"].min()
    assert validation["timestamp"].max() <= test["timestamp"].min()
    assert train["timestamp"].iloc[0] == frame["timestamp"].iloc[0]


def test_invalid_and_missing_rows_are_removed():
    frame = generate_workload(120, seed=1)
    frame.loc[0, "cpu_utilization"] = 140
    frame.loc[1, "request_rate"] = -4
    frame.loc[2, "memory_usage"] = np.nan
    frame.loc[3, "sla_violation"] = 7
    cleaned = clean_workload(frame)
    assert len(cleaned) == len(frame) - 4


def test_missing_column_is_rejected(tmp_path: Path):
    frame = generate_workload(80, seed=2).drop(columns=["response_time"])
    path = tmp_path / "bad.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="response_time"):
        load_dataset(path)


def test_imputer_uses_only_the_training_median():
    train = pd.DataFrame({"request_rate": [10.0, 30.0], "memory_usage": [20.0, 40.0]})
    later = pd.DataFrame({"request_rate": [np.nan, 1000.0], "memory_usage": [np.nan, 5.0]})
    transformed = TrainOnlyImputer().fit(train).transform(later)
    assert transformed.loc[0, "request_rate"] == 20.0
    assert transformed.loc[0, "memory_usage"] == 30.0


def test_smaller_dataset_does_not_shuffle():
    frame = clean_workload(generate_workload(200, seed=5))
    train, validation, test, _note = chronological_split(frame)
    combined = pd.concat([train, validation, test], ignore_index=True)
    pd.testing.assert_series_equal(
        combined["timestamp"].reset_index(drop=True),
        frame["timestamp"].reset_index(drop=True),
        check_names=False,
    )


def test_persistence_uses_the_previous_observation_not_the_current_one():
    predicted = persistence_predictions(10.0, [40.0, 50.0, 60.0])
    np.testing.assert_allclose(predicted, [10.0, 40.0, 50.0])
    mean = mean_predictions(12.5, 3)
    np.testing.assert_allclose(mean, [12.5, 12.5, 12.5])
    scores = regression_metrics([0.0, 2.0], [0.0, 2.0])
    assert scores["mae"] == 0.0
    assert scores["r2"] == 1.0


def test_training_pipeline_writes_artifacts_and_beats_the_mean(tmp_path: Path):
    data_path = tmp_path / "workload_data.csv"
    save_workload(generate_workload(360, seed=9), data_path)
    model_path = tmp_path / "random_forest.pkl"
    metrics_path = tmp_path / "model_metrics.csv"
    result = run_training(
        data_path=data_path,
        model_path=model_path,
        metrics_path=metrics_path,
        history_path=tmp_path / "scaling_history.csv",
        graph_path=tmp_path / "comparison_graph.png",
        candidates=[{"n_estimators": 12, "max_depth": 8, "min_samples_leaf": 1}],
        history_rows=30,
    )
    metrics = result["metrics"]
    forest = metrics[(metrics["model"] == "Random Forest") & (metrics["split"] == "test")].iloc[0]
    average = metrics[(metrics["model"] == "Mean") & (metrics["split"] == "test")].iloc[0]
    assert forest["rmse"] < average["rmse"]
    assert model_path.stat().st_size > 1000
    assert (tmp_path / "comparison_graph.png").stat().st_size > 1000
    bundle = load_artifact(model_path)
    prediction = predict_one(120, 40, 3, 140, 0, bundle=bundle)
    assert np.isfinite(prediction)
    frame = load_dataset(data_path)
    train, validation, test, _note = chronological_split(frame)
    x_train, y_train, _x_val, _y_val, x_test, _y_test, imputer = prepare_matrices(train, validation, test)
    assert imputer.medians_ is not None
    assert len(x_train) == len(y_train)
    assert list(x_test.columns) == list(bundle["feature_columns"])


def test_simulator_rejects_impossible_monitor_input():
    with pytest.raises(ValueError):
        snapshot_from_inputs(120, 40, 10, 80, 2)
    snapshot = snapshot_from_inputs(92, 70, 400, 700, 2)
    assert snapshot.sla_violation == 1
    assert snapshot.source == "simulator"


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Train the project model before this check.")
def test_shipped_model_metrics_are_computed_scores():
    from src import METRICS_PATH

    assert METRICS_PATH.exists()
    metrics = pd.read_csv(METRICS_PATH)
    forest = metrics[(metrics["model"] == "Random Forest") & (metrics["split"] == "test")].iloc[0]
    average = metrics[(metrics["model"] == "Mean") & (metrics["split"] == "test")].iloc[0]
    persistence = metrics[(metrics["model"] == "Persistence") & (metrics["split"] == "test")].iloc[0]
    assert forest["rmse"] < average["rmse"]
    assert np.isfinite(forest["mae"]) and np.isfinite(persistence["r2"])
    published = {
        ("Random Forest", "mae"): 2.944,
        ("Random Forest", "rmse"): 3.766,
        ("Random Forest", "r2"): 0.879,
        ("Persistence", "mae"): 6.439,
        ("Persistence", "rmse"): 8.621,
        ("Persistence", "r2"): 0.364,
        ("Mean", "mae"): 9.244,
        ("Mean", "rmse"): 10.929,
        ("Mean", "r2"): -0.023,
    }
    copied = True
    for _, row in metrics.loc[metrics["split"] == "test"].iterrows():
        if row["model"] in {"Random Forest", "Persistence", "Mean"}:
            for metric in ("mae", "rmse", "r2"):
                reference = published[(row["model"], metric)]
                if abs(float(row[metric]) - reference) > 1e-6:
                    copied = False
    assert not copied
    bundle = load_artifact(MODEL_PATH)
    assert bundle["data_origin"] == "simulated"
    assert TARGET_COLUMN == bundle["target_column"]
