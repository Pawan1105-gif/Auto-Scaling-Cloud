"""Simulated cloud monitoring over the dataset and over operator inputs.

Snapshots are local Python objects. Nothing here connects to a cloud API.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src import TARGET_COLUMN


@dataclass
class ResourceSnapshot:
    """One minute of monitored application and infrastructure state."""

    timestamp: pd.Timestamp
    cpu_utilization: float
    memory_usage: float
    request_rate: float
    response_time: float
    active_vms: int
    sla_violation: int
    source: str

    def feature_dict(self) -> dict[str, float]:
        return {
            "request_rate": float(self.request_rate),
            "memory_usage": float(self.memory_usage),
            "active_vms": int(self.active_vms),
            "response_time": float(self.response_time),
            "sla_violation": int(self.sla_violation),
        }


def infer_sla_violation(response_time: float, cpu_utilization: float) -> int:
    """Mark a simulated SLA miss when latency or observed CPU is very high."""
    if response_time < 0 or cpu_utilization < 0:
        raise ValueError("Response time and CPU utilization cannot be negative.")
    return int(response_time >= 500.0 or cpu_utilization >= 90.0)


def snapshot_from_row(row: pd.Series, source: str = "dataset") -> ResourceSnapshot:
    """Convert one cleaned dataset row into a monitor snapshot."""
    return ResourceSnapshot(
        timestamp=pd.Timestamp(row["timestamp"]),
        cpu_utilization=float(row[TARGET_COLUMN]),
        memory_usage=float(row["memory_usage"]),
        request_rate=float(row["request_rate"]),
        response_time=float(row["response_time"]),
        active_vms=int(row["active_vms"]),
        sla_violation=int(row["sla_violation"]),
        source=source,
    )


def snapshot_from_inputs(
    cpu_utilization: float,
    memory_usage: float,
    request_rate: float,
    response_time: float,
    active_vms: int,
    sla_violation: int | None = None,
    timestamp: pd.Timestamp | None = None,
) -> ResourceSnapshot:
    """Build a snapshot from dashboard controls."""
    cpu = float(cpu_utilization)
    memory = float(memory_usage)
    requests = float(request_rate)
    latency = float(response_time)
    vms = int(active_vms)
    if not 0 <= cpu <= 100:
        raise ValueError("CPU utilization must be between 0 and 100.")
    if not 0 <= memory <= 100:
        raise ValueError("Memory utilization must be between 0 and 100.")
    if requests < 0:
        raise ValueError("Request rate cannot be negative.")
    if latency < 0:
        raise ValueError("Response time cannot be negative.")
    if vms < 1:
        raise ValueError("At least one virtual machine is required.")
    sla = infer_sla_violation(latency, cpu) if sla_violation is None else int(sla_violation)
    if sla not in (0, 1):
        raise ValueError("SLA violation must be 0 or 1.")
    return ResourceSnapshot(
        timestamp=pd.Timestamp.utcnow() if timestamp is None else pd.Timestamp(timestamp),
        cpu_utilization=cpu,
        memory_usage=memory,
        request_rate=requests,
        response_time=latency,
        active_vms=vms,
        sla_violation=sla,
        source="simulator",
    )


def scenario_presets() -> dict[str, dict[str, float]]:
    """Four classroom scenarios.

    The feature values are inputs to the trained model. Observed CPU is shown
    on the dashboard, while scaling uses the model's predicted CPU.
    """
    return {
        "Low workload": {
            "cpu_utilization": 18.0,
            "memory_usage": 22.0,
            "request_rate": 30.0,
            "response_time": 45.0,
        },
        "Normal workload": {
            "cpu_utilization": 50.0,
            "memory_usage": 55.0,
            "request_rate": 180.0,
            "response_time": 150.0,
        },
        "High workload": {
            "cpu_utilization": 86.0,
            "memory_usage": 82.0,
            "request_rate": 620.0,
            "response_time": 540.0,
        },
        "Sudden traffic spike": {
            "cpu_utilization": 95.0,
            "memory_usage": 93.0,
            "request_rate": 920.0,
            "response_time": 1400.0,
        },
    }


class DatasetMonitor:
    """Walk through the cleaned CSV one simulated minute at a time."""

    def __init__(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("Dataset monitor requires at least one row.")
        self.frame = frame.reset_index(drop=True)
        self.index = 0

    def __len__(self) -> int:
        return len(self.frame)

    def seek(self, index: int) -> ResourceSnapshot:
        if len(self.frame) == 0:
            raise ValueError("Dataset monitor is empty.")
        self.index = int(min(max(index, 0), len(self.frame) - 1))
        return self.current()

    def advance(self, steps: int = 1) -> ResourceSnapshot:
        return self.seek(self.index + int(steps))

    def current(self) -> ResourceSnapshot:
        return snapshot_from_row(self.frame.iloc[self.index], source="dataset")

    def window(self, lookback: int = 120) -> pd.DataFrame:
        start = max(0, self.index - lookback + 1)
        return self.frame.iloc[start : self.index + 1].copy()
