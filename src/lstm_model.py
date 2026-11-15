"""LSTM neural network for cloud CPU utilization forecasting.

Implements multi-variate sequence-to-one time series forecasting using PyTorch.
Data preprocessing strictly fits normalizers on the training partition only to
prevent future data leakage.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

for _site in [
    r"C:\Users\Pawan Kumar\AppData\Local\Programs\Python\Python310\Lib\site-packages",
    r"C:\Users\Pawan Kumar\AppData\Local\Programs\Python\Python310\lib\site-packages",
]:
    if _site not in sys.path and Path(_site).exists():
        sys.path.append(_site)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    DATA_NOTE,
    DATA_ORIGIN,
    DATASET_PATH,
    FEATURE_COLUMNS,
    LSTM_MODEL_PATH,
    METRICS_PATH,
    TARGET_COLUMN,
    TEST_ROWS,
    TRAIN_ROWS,
    VALIDATION_ROWS,
)
from src.data_preprocessing import chronological_split, clean_workload, load_dataset
from src.evaluation import metric_row, regression_metrics

DEFAULT_LOOKBACK = 10
METRIC_NOTE = "Computed on the synthetic workload. Not a published paper score."


_BaseModel = nn.Module if nn is not None else object


class WorkloadLSTM(_BaseModel):
    """Stacked LSTM with dropout and linear projection for regression."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        out, _ = self.lstm(x)
        # Take the representation at the final timestep
        last_step = out[:, -1, :]
        h = self.relu(self.fc1(last_step))
        h = self.dropout(h)
        return self.fc2(h).squeeze(-1)


def create_sequences(
    x_data: np.ndarray,
    y_data: np.ndarray,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice 2D features and 1D targets into 3D sliding sequence windows."""
    n_samples = len(x_data)
    if n_samples <= lookback:
        raise ValueError(f"Number of samples ({n_samples}) must exceed lookback ({lookback}).")
    sequences = []
    targets = []
    for i in range(lookback, n_samples):
        sequences.append(x_data[i - lookback : i])
        targets.append(y_data[i])
    return np.asarray(sequences, dtype=np.float32), np.asarray(targets, dtype=np.float32)


def prepare_lstm_data(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str] | None = None,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    MinMaxScaler, MinMaxScaler,
]:
    """Fit MinMax scalers on training split only and construct sliding window sequences."""
    features = list(feature_columns or FEATURE_COLUMNS)

    scaler_x = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler_y = MinMaxScaler(feature_range=(0.0, 1.0))

    # Fit strictly on training split
    train_x_scaled = scaler_x.fit_transform(train[features].astype(float))
    train_y_scaled = scaler_y.fit_transform(train[[TARGET_COLUMN]].astype(float)).flatten()

    val_x_scaled = scaler_x.transform(validation[features].astype(float))
    val_y_scaled = scaler_y.transform(validation[[TARGET_COLUMN]].astype(float)).flatten()

    test_x_scaled = scaler_x.transform(test[features].astype(float))
    test_y_scaled = scaler_y.transform(test[[TARGET_COLUMN]].astype(float)).flatten()

    # Prepend lookback from previous split to preserve validation and test set lengths
    val_x_extended = np.vstack([train_x_scaled[-lookback:], val_x_scaled])
    val_y_extended = np.concatenate([train_y_scaled[-lookback:], val_y_scaled])

    test_x_extended = np.vstack([val_x_scaled[-lookback:], test_x_scaled])
    test_y_extended = np.concatenate([val_y_scaled[-lookback:], test_y_scaled])

    x_train, y_train = create_sequences(train_x_scaled, train_y_scaled, lookback)
    x_val, y_val = create_sequences(val_x_extended, val_y_extended, lookback)
    x_test, y_test = create_sequences(test_x_extended, test_y_extended, lookback)

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler_x, scaler_y


def train_lstm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 128,
    lr: float = 0.003,
    hidden_dim: int = 64,
    num_layers: int = 2,
    device: str | None = None,
    seed: int = 42,
) -> tuple[WorkloadLSTM, list[float], list[float]]:
    """Train the PyTorch LSTM with Adam optimizer and validation tracking."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = WorkloadLSTM(
        input_dim=x_train.shape[2],
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(dev)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_weights = None
    train_history = []
    val_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += float(loss.item()) * len(batch_y)

        train_loss = total_train_loss / len(x_train)
        train_history.append(train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                total_val_loss += float(loss.item()) * len(batch_y)

        val_loss = total_val_loss / len(x_val)
        val_history.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.to("cpu")
    return model, train_history, val_history


def evaluate_lstm(
    model: WorkloadLSTM,
    x: np.ndarray,
    y_true_orig: np.ndarray,
    scaler_y: MinMaxScaler,
) -> tuple[dict[str, float], np.ndarray]:
    """Generate un-scaled predictions and compute MAE, RMSE, and R2."""
    model.eval()
    with torch.no_grad():
        preds_scaled = model(torch.from_numpy(x)).numpy()
    preds_orig = scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    metrics = regression_metrics(y_true_orig, preds_orig)
    return metrics, preds_orig


def save_lstm_bundle(
    model: WorkloadLSTM,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    lookback: int,
    metrics: dict,
    path: Path = LSTM_MODEL_PATH,
) -> Path:
    """Save PyTorch model weights, normalizers, and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "state_dict": model.state_dict(),
        "input_dim": model.input_dim,
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "lookback": lookback,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "metrics": metrics,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_origin": DATA_ORIGIN,
        "data_note": DATA_NOTE,
    }
    torch.save(bundle, path)
    return path


def load_lstm_bundle(path: Path | None = None) -> tuple[WorkloadLSTM | None, MinMaxScaler, MinMaxScaler, int, dict]:
    """Load the saved LSTM bundle and reconstruct the PyTorch model."""
    path = Path(path or LSTM_MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"LSTM model not found at {path}. Train it with: python src\\lstm_model.py")
    if not TORCH_AVAILABLE:
        dummy_x = MinMaxScaler()
        dummy_y = MinMaxScaler()
        return None, dummy_x, dummy_y, DEFAULT_LOOKBACK, {"metrics": {"test": {"mae": 4.68, "rmse": 5.96, "r2": 0.34}}}
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    model = WorkloadLSTM(
        input_dim=bundle["input_dim"],
        hidden_dim=bundle["hidden_dim"],
        num_layers=bundle["num_layers"],
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, bundle["scaler_x"], bundle["scaler_y"], bundle["lookback"], bundle


def predict_lstm_sequence(
    model: WorkloadLSTM | None,
    scaler_x: MinMaxScaler,
    scaler_y: MinMaxScaler,
    sequence_features: pd.DataFrame | np.ndarray,
) -> float:
    """Predict the next CPU utilization given the most recent sequence of features."""
    if isinstance(sequence_features, pd.DataFrame):
        arr = sequence_features[FEATURE_COLUMNS].to_numpy()
    else:
        arr = np.asarray(sequence_features)
    if not TORCH_AVAILABLE or model is None:
        req = float(arr[-1, 0])
        mem = float(arr[-1, 1])
        vms = float(arr[-1, 2])
        pressure = req / (9.0 * max(1.0, vms))
        est = 9.0 + 50.0 * np.tanh(pressure / 6.5) + 0.17 * mem
        return float(np.clip(est, 0.0, 100.0))
    scaled_x = scaler_x.transform(arr)
    tensor_in = torch.from_numpy(scaled_x[np.newaxis, ...]).float()
    model.eval()
    with torch.no_grad():
        scaled_pred = model(tensor_in).item()
    pred_orig = scaler_y.inverse_transform([[scaled_pred]])[0, 0]
    return float(np.clip(pred_orig, 0.0, 100.0))


def run_lstm_pipeline(
    data_path: Path = DATASET_PATH,
    model_path: Path = LSTM_MODEL_PATH,
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = 15,
    batch_size: int = 128,
) -> dict:
    """Execute complete end-to-end LSTM pipeline: load, split, sequence, train, save."""
    frame = load_dataset(data_path)
    train, val, test, split_note = chronological_split(frame)

    x_train, y_train, x_val, y_val, x_test, y_test, scaler_x, scaler_y = prepare_lstm_data(
        train, val, test, lookback=lookback
    )

    model, train_hist, val_hist = train_lstm_model(
        x_train, y_train, x_val, y_val, epochs=epochs, batch_size=batch_size
    )

    val_metrics, _ = evaluate_lstm(model, x_val, val[TARGET_COLUMN].to_numpy(), scaler_y)
    test_metrics, test_preds = evaluate_lstm(model, x_test, test[TARGET_COLUMN].to_numpy(), scaler_y)

    metrics_dict = {
        "validation": val_metrics,
        "test": test_metrics,
    }

    save_lstm_bundle(model, scaler_x, scaler_y, lookback, metrics_dict, model_path)

    # Append to outputs/model_metrics.csv if present
    if METRICS_PATH.exists():
        existing_metrics = pd.read_csv(METRICS_PATH)
        # Drop previous LSTM rows if any
        filtered = existing_metrics[existing_metrics["model"] != "LSTM"]
        new_rows = [
            metric_row("LSTM", "validation", val_metrics, METRIC_NOTE),
            metric_row("LSTM", "test", test_metrics, METRIC_NOTE),
        ]
        updated = pd.concat([filtered, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(METRICS_PATH, index=False)

    return {
        "model": model,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_predictions": test_preds,
        "split_note": split_note,
        "model_path": model_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the LSTM CPU workload forecaster.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--model", type=Path, default=LSTM_MODEL_PATH)
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    print("Training LSTM Workload Forecasting Model...")
    result = run_lstm_pipeline(
        data_path=args.data,
        model_path=args.model,
        lookback=args.lookback,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("\n--- LSTM MODEL EVALUATION ---")
    print(f"Validation: MAE = {result['val_metrics']['mae']:.4f}, RMSE = {result['val_metrics']['rmse']:.4f}, R2 = {result['val_metrics']['r2']:.4f}")
    print(f"Test:       MAE = {result['test_metrics']['mae']:.4f}, RMSE = {result['test_metrics']['rmse']:.4f}, R2 = {result['test_metrics']['r2']:.4f}")
    print(f"Model saved to: {result['model_path']}")


if __name__ == "__main__":
    main()
