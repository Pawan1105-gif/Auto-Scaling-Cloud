"""Benchmark and Comparative Evaluation of Four Cloud Auto-Scaling Strategies.

Evaluates under identical workload conditions:
- Strategy A: Traditional Threshold-Based Auto-Scaling
- Strategy B: Forecast-Only Auto-Scaling
- Strategy C: Reinforcement-Learning-Only Auto-Scaling
- Strategy D: Hybrid LSTM + Q-Learning Auto-Scaling
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

from src import (
    DATASET_PATH,
    LSTM_MODEL_PATH,
    MODEL_PATH,
    Q_TABLE_PATH,
    STRATEGY_COMPARISON_PATH,
)
from src.data_preprocessing import chronological_split, load_dataset
from src.evaluation import HOURLY_VM_RATE_USD
from src.hybrid_autoscaler import HybridAutoScaler
from src.lstm_model import load_lstm_bundle, predict_lstm_sequence
from src.prediction import load_artifact, predict_one
from src.q_learning import QLearningAgent

STRATEGIES = [
    "Strategy A: Traditional Threshold",
    "Strategy B: Forecast-Only",
    "Strategy C: RL-Only",
    "Strategy D: Hybrid LSTM + Q-Learning",
]


def run_strategy_simulation(
    workload_df: pd.DataFrame,
    strategy_name: str,
    q_agent: QLearningAgent | None = None,
    forecaster_fn=None,
    initial_vms: int = 3,
    min_vms: int = 1,
    max_vms: int = 10,
    scale_up_th: float = 80.0,
    scale_down_th: float = 30.0,
    cooldown_steps: int = 1,
    provisioning_delay: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate a single auto-scaling strategy through the entire workload slice."""
    scaler = HybridAutoScaler(
        q_agent=q_agent,
        forecaster_fn=forecaster_fn,
        strategy=strategy_name,
        min_vms=min_vms,
        max_vms=max_vms,
        scale_up_threshold=scale_up_th,
        scale_down_threshold=scale_down_th,
        cooldown_steps=cooldown_steps,
        provisioning_delay=provisioning_delay,
        initial_vms=initial_vms,
    )

    records = []
    df = workload_df.reset_index(drop=True)
    prev_req = float(df.loc[0, "request_rate"])

    for idx in range(len(df)):
        row = df.iloc[idx]
        req_rate = float(row["request_rate"])
        mem = float(row["memory_usage"])
        vms = scaler.current_vms

        # Physical CPU response under current allocated VMs
        pressure = req_rate / (9.0 * vms)
        curr_cpu = float(np.clip(9.0 + 50.0 * np.tanh(pressure / 6.5) + 0.17 * mem, 2.0, 99.0))
        load_per_vm = req_rate / vms
        resp_time = float(np.clip(35.0 + 0.62 * load_per_vm + 0.35 * mem, 15.0, 2000.0))
        sla = 1 if (resp_time >= 500.0 or curr_cpu >= 90.0) else 0
        trend = req_rate - prev_req
        prev_req = req_rate

        decision = scaler.decide(
            current_cpu=curr_cpu,
            memory_usage=mem,
            request_rate=req_rate,
            response_time=resp_time,
            sla_violation=sla,
            workload_trend=trend,
            timestamp=str(row["timestamp"]),
        )

        records.append({
            "step": idx,
            "timestamp": row["timestamp"],
            "request_rate": req_rate,
            "memory_usage": mem,
            "cpu_utilization": curr_cpu,
            "predicted_cpu": decision.predicted_cpu,
            "response_time": resp_time,
            "sla_violation": sla,
            "active_vms": vms,
            "new_vms": decision.new_vms,
            "action": decision.action,
            "strategy": strategy_name,
        })

    trace_df = pd.DataFrame(records)

    # Compute comprehensive metrics
    total_minutes = len(trace_df)
    total_hours = total_minutes / 60.0
    avg_cpu = float(trace_df["cpu_utilization"].mean())
    std_cpu = float(trace_df["cpu_utilization"].std())
    sla_count = int(trace_df["sla_violation"].sum())
    sla_pct = float((sla_count / total_minutes) * 100.0)
    avg_latency = float(trace_df["response_time"].mean())
    scale_ups = int((trace_df["action"] == "SCALE_UP").sum())
    scale_downs = int((trace_df["action"] == "SCALE_DOWN").sum())
    total_actions = scale_ups + scale_downs
    avg_vms = float(trace_df["active_vms"].mean())
    vm_hours = avg_vms * total_hours
    estimated_cost = vm_hours * HOURLY_VM_RATE_USD

    metrics = {
        "strategy": strategy_name,
        "avg_cpu": round(avg_cpu, 2),
        "std_cpu": round(std_cpu, 2),
        "sla_count": sla_count,
        "sla_percent": round(sla_pct, 2),
        "avg_latency_ms": round(avg_latency, 2),
        "scale_up_count": scale_ups,
        "scale_down_count": scale_downs,
        "total_scaling_actions": total_actions,
        "avg_vms": round(avg_vms, 2),
        "vm_hours": round(vm_hours, 2),
        "estimated_cost_usd": round(estimated_cost, 3),
    }

    return trace_df, metrics


def compare_all_strategies(
    workload_df: pd.DataFrame,
    rf_bundle: dict | None = None,
    lstm_bundle: tuple | None = None,
    q_agent: QLearningAgent | None = None,
    output_path: Path = STRATEGY_COMPARISON_PATH,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Run all four auto-scaling strategies on the exact same workload."""
    # Define forecaster functions
    def rf_forecaster(req, mem, vms, resp, sla):
        if rf_bundle is None:
            return 50.0
        return predict_one(req, mem, vms, resp, sla, bundle=rf_bundle)

    def lstm_forecaster(req, mem, vms, resp, sla):
        if lstm_bundle is None:
            return 50.0
        model, sx, sy, lookback, _ = lstm_bundle
        seq = np.tile([req, mem, vms, resp, sla], (lookback, 1))
        seq_df = pd.DataFrame(seq, columns=FEATURE_COLUMNS)
        return predict_lstm_sequence(model, sx, sy, seq_df)

    summary_rows = []
    trace_dict = {}

    for strat in STRATEGIES:
        forecaster = lstm_forecaster if "LSTM" in strat else rf_forecaster
        trace, metrics = run_strategy_simulation(
            workload_df=workload_df,
            strategy_name=strat,
            q_agent=q_agent,
            forecaster_fn=forecaster,
        )
        summary_rows.append(metrics)
        trace_dict[strat] = trace

    summary_df = pd.DataFrame(summary_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    return summary_df, trace_dict


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate four cloud auto-scaling strategies.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--rows", type=int, default=1000, help="Number of test records to evaluate")
    args = parser.parse_args(argv)

    print("Loading test workload slice...")
    df = load_dataset(args.data)
    _, _, test_df, _ = chronological_split(df)
    eval_slice = test_df.iloc[:args.rows]

    print("Loading models (RF, LSTM, Q-Table)...")
    rf_bundle = load_artifact(MODEL_PATH) if MODEL_PATH.exists() else None
    lstm_bundle = load_lstm_bundle(LSTM_MODEL_PATH) if LSTM_MODEL_PATH.exists() else None
    q_agent = QLearningAgent.load(Q_TABLE_PATH) if Q_TABLE_PATH.exists() else None

    print(f"Running 4-Strategy benchmark across {len(eval_slice)} minutes...")
    summary, _ = compare_all_strategies(eval_slice, rf_bundle, lstm_bundle, q_agent)

    print("\n--- STRATEGY COMPARISON SUMMARY ---")
    print(summary.to_string(index=False))
    print(f"\nSaved comparison to: {STRATEGY_COMPARISON_PATH}")


if __name__ == "__main__":
    main()
