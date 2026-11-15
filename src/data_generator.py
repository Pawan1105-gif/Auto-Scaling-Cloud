"""Generate a reproducible synthetic cloud-workload dataset.

Every value written by this module is simulated. Do not describe the file as
a real cloud trace or as a reproduction of a published research dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import DATA_NOTE, DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN

SEED = 42
N_RECORDS = 20_000
MIN_RECORDS = 32


def generate_workload(n_records: int = N_RECORDS, seed: int = SEED) -> pd.DataFrame:
    """Build a chronological workload with low, normal, high, and spike periods.

    CPU utilization is a noisy, nonlinear function of request rate, memory,
    response time, SLA state, and the number of active VMs. The noise is
    intentional so a model cannot fit the series perfectly.
    """
    if n_records < MIN_RECORDS:
        raise ValueError(f"n_records must be at least {MIN_RECORDS}.")

    rng = np.random.default_rng(seed)
    t = np.arange(n_records, dtype=float)
    hour = (t % (24 * 60)) / 60.0
    day = (t // (24 * 60)) % 7

    diurnal = 0.62 + 0.30 * np.sin((hour - 9.0) / 24.0 * 2.0 * np.pi)
    weekend = np.where(day >= 5, 0.78, 1.0)

    regime = np.empty(n_records, dtype=float)
    block = max(40, n_records // 20)
    cursor = 0
    while cursor < n_records:
        end = min(cursor + block, n_records)
        level = rng.choice(
            np.array([0.45, 1.0, 1.55]),
            p=np.array([0.28, 0.47, 0.25]),
        )
        regime[cursor:end] = level
        cursor = end

    spike = np.zeros(n_records, dtype=float)
    margin = 15
    population = np.arange(margin, n_records - margin)
    n_spikes = min(len(population), max(2, n_records // 500))
    spike_starts = rng.choice(population, size=n_spikes, replace=False)
    for start in np.sort(spike_starts):
        available = n_records - int(start)
        if available <= 6:
            continue
        length = int(rng.integers(6, min(36, available)))
        amplitude = float(rng.uniform(0.9, 2.4))
        shape = np.linspace(0.35, 1.0, length)
        spike[int(start) : int(start) + length] += amplitude * shape

    demand = np.clip(diurnal * weekend * regime + spike, 0.12, 4.0)

    request_rate = np.clip(25.0 + demand * 210.0 + rng.normal(0.0, 10.0, n_records), 5.0, 980.0)

    kernel = np.ones(21) / 21.0
    smooth_demand = np.convolve(demand, kernel, mode="same")
    active_vms = np.clip(
        np.rint(1.0 + smooth_demand * 2.4 + rng.normal(0.0, 0.28, n_records)),
        1,
        10,
    ).astype(int)

    memory_usage = np.clip(
        14.0 + 0.042 * request_rate + 1.6 * active_vms + rng.normal(0.0, 2.8, n_records),
        5.0,
        98.0,
    )
    load_per_vm = request_rate / active_vms
    response_time = np.clip(
        35.0 + 0.62 * load_per_vm + 0.35 * memory_usage + rng.normal(0.0, 14.0, n_records),
        12.0,
        2400.0,
    )

    sla_score = (response_time - 280.0) / 90.0 + (load_per_vm - 55.0) / 40.0
    sla_violation = (sla_score + rng.normal(0.0, 0.35, n_records) > 1.15).astype(int)

    pressure = request_rate / (9.0 * active_vms)
    signal = (
        9.0
        + 50.0 * np.tanh(pressure / 6.5)
        + 0.17 * memory_usage
        + 0.007 * response_time
        + 2.8 * sla_violation
    )
    white = rng.normal(0.0, 2.6, n_records)
    noise = np.zeros(n_records, dtype=float)
    for index in range(1, n_records):
        noise[index] = 0.42 * noise[index - 1] + white[index]
    cpu_utilization = np.clip(signal + noise, 1.0, 99.0)

    timestamps = pd.date_range("2024-01-01", periods=n_records, freq="min")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "request_rate": np.round(request_rate, 2),
            "memory_usage": np.round(memory_usage, 2),
            "active_vms": active_vms,
            "response_time": np.round(response_time, 2),
            "sla_violation": sla_violation,
            "cpu_utilization": np.round(cpu_utilization, 2),
        }
    )
    return frame


def save_workload(frame: pd.DataFrame, path: Path = DATASET_PATH, overwrite: bool = False) -> Path:
    """Write the simulated dataset to CSV."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass overwrite=True to replace it. "
            "An existing file is left untouched so a supplied research CSV is not destroyed."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate the simulated workload CSV.")
    parser.add_argument("--records", type=int, default=N_RECORDS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing CSV. Omit this flag if you supplied your own dataset.",
    )
    args = parser.parse_args(argv)

    frame = generate_workload(args.records, args.seed)
    destination = save_workload(frame, args.output, overwrite=args.overwrite)
    print("SIMULATED DATASET")
    print(DATA_NOTE)
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(frame.columns)}")
    print(f"Seed: {args.seed}")
    print(f"Saved: {destination}")
    print(frame[FEATURE_COLUMNS + [TARGET_COLUMN]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
