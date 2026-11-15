"""Local academic prototype of an intelligent cloud auto-scaling platform.

This package simulates monitoring, CPU prediction, and rule-based scaling.
It does not provision virtual machines in any cloud provider.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
DATASET_PATH = DATASET_DIR / "workload_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "random_forest.pkl"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.pt"
Q_TABLE_PATH = MODEL_DIR / "q_table.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRICS_PATH = OUTPUT_DIR / "model_metrics.csv"
SCALING_HISTORY_PATH = OUTPUT_DIR / "scaling_history.csv"
COMPARISON_GRAPH_PATH = OUTPUT_DIR / "comparison_graph.png"
STRATEGY_COMPARISON_PATH = OUTPUT_DIR / "strategy_comparison.csv"
ONLINE_METRICS_PATH = OUTPUT_DIR / "online_learning_log.csv"

FEATURE_COLUMNS = [
    "request_rate",
    "memory_usage",
    "active_vms",
    "response_time",
    "sla_violation",
]
TARGET_COLUMN = "cpu_utilization"
ALL_COLUMNS = ["timestamp", *FEATURE_COLUMNS, TARGET_COLUMN]

TRAIN_ROWS = 14_000
VALIDATION_ROWS = 3_000
TEST_ROWS = 3_000

DATA_ORIGIN = "simulated"
DATA_NOTE = (
    "Synthetic workload generated for a local academic simulation. "
    "These are not measurements from a real cloud provider and do not "
    "reproduce a published research dataset."
)

__all__ = [
    "PROJECT_ROOT",
    "DATASET_DIR",
    "DATASET_PATH",
    "MODEL_DIR",
    "MODEL_PATH",
    "LSTM_MODEL_PATH",
    "Q_TABLE_PATH",
    "OUTPUT_DIR",
    "METRICS_PATH",
    "SCALING_HISTORY_PATH",
    "COMPARISON_GRAPH_PATH",
    "STRATEGY_COMPARISON_PATH",
    "ONLINE_METRICS_PATH",
    "FEATURE_COLUMNS",
    "TARGET_COLUMN",
    "ALL_COLUMNS",
    "TRAIN_ROWS",
    "VALIDATION_ROWS",
    "TEST_ROWS",
    "DATA_ORIGIN",
    "DATA_NOTE",
]
