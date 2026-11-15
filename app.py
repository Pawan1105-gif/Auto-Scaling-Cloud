"""Interactive Web Dashboard for the Intelligent Cloud Auto-Scaling Platform.

Implements all 13 research demonstration modules:
1. Project Overview
2. Live Workload Monitoring
3. Random Forest Prediction
4. LSTM Prediction
5. Forecasting Model Comparison
6. Q-Learning Training
7. Reinforcement Learning Decisions
8. Hybrid Auto-Scaling
9. Four-Strategy Comparison
10. Uncertainty-Aware Prediction
11. Explainable AI
12. Performance Evaluation
13. Scaling History and Logs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _candidate in [
    r"C:\Users\Pawan Kumar\AppData\Local\Programs\Python\Python310\Lib\site-packages",
    r"C:\Users\Pawan Kumar\AppData\Local\Programs\Python\Python310\lib\site-packages",
]:
    if _candidate not in sys.path and Path(_candidate).exists():
        sys.path.append(_candidate)

import src
if not hasattr(src, "LSTM_MODEL_PATH"):
    import importlib
    importlib.reload(src)

from src import (
    COMPARISON_GRAPH_PATH,
    DATA_NOTE,
    DATASET_PATH,
    FEATURE_COLUMNS,
    LSTM_MODEL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    ONLINE_METRICS_PATH,
    Q_TABLE_PATH,
    SCALING_HISTORY_PATH,
    STRATEGY_COMPARISON_PATH,
    TARGET_COLUMN,
)
from src.autoscaler import AutoScaler
from src.data_generator import generate_workload, save_workload
from src.data_preprocessing import chronological_split, load_dataset
from src.evaluation import count_scaling_actions, summarize_workload
from src.explainable_ai import (
    explain_lstm_temporal_sensitivity,
    explain_q_learning_decision,
    explain_random_forest,
)
from src.hybrid_autoscaler import HybridAutoScaler
from src.lstm_model import (
    load_lstm_bundle,
    predict_lstm_sequence,
    run_lstm_pipeline,
)
from src.monitoring import scenario_presets, snapshot_from_inputs, snapshot_from_row
from src.online_learning import OnlineModelManager
from src.prediction import ModelNotFoundError, load_artifact, predict_cpu, predict_one
from src.q_learning import (
    ACTION_NAMES,
    QLearningAgent,
    extract_state_tuple,
    train_q_learning,
)
from src.strategy_comparison import STRATEGIES, compare_all_strategies
from src.train_model import run_training
from src.uncertainty_forecasting import (
    apply_uncertainty_aware_guard,
    estimate_lstm_uncertainty,
    estimate_rf_uncertainty,
)

PAGES = [
    "1. Project Overview",
    "2. Live Workload Monitoring",
    "3. Random Forest Prediction",
    "4. LSTM Prediction",
    "5. Forecasting Model Comparison",
    "6. Q-Learning Training",
    "7. Reinforcement Learning Decisions",
    "8. Hybrid Auto-Scaling",
    "9. Four-Strategy Comparison",
    "10. Uncertainty-Aware Prediction",
    "11. Explainable AI",
    "12. Performance Evaluation",
    "13. Scaling History and Logs",
]

REFERENCE_METRICS = pd.DataFrame([
    {"model": "Random Forest", "mae": 2.944, "rmse": 3.766, "r2": 0.879, "source": "Paper Reference"},
    {"model": "Persistence", "mae": 6.439, "rmse": 8.621, "r2": 0.364, "source": "Paper Reference"},
    {"model": "Mean", "mae": 9.244, "rmse": 10.929, "r2": -0.023, "source": "Paper Reference"},
])


def inject_css() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 0.6rem 0.8rem;
            border-radius: 0.75rem;
        }
        .status-badge {
            display: inline-block;
            padding: 0.4rem 0.85rem;
            border-radius: 0.5rem;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .callout-box {
            background: rgba(30, 41, 59, 0.7);
            border-left: 4px solid #38bdf8;
            padding: 0.8rem 1.1rem;
            border-radius: 0.4rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("follow_dataset", True)
    st.session_state.setdefault("row_index", 0)
    st.session_state.setdefault("control_mode", "Automatic")
    st.session_state.setdefault("th_up", 80.0)
    st.session_state.setdefault("th_down", 30.0)
    st.session_state.setdefault("th_cool", 1)
    st.session_state.setdefault("th_min", 1)
    st.session_state.setdefault("th_max", 10)
    st.session_state.setdefault("sim_cpu", 50.0)
    st.session_state.setdefault("sim_memory", 55.0)
    st.session_state.setdefault("sim_requests", 180.0)
    st.session_state.setdefault("sim_latency", 150.0)
    st.session_state.setdefault("last_decision", None)
    if "scaler" not in st.session_state:
        st.session_state.scaler = AutoScaler()
    if "hybrid_scaler" not in st.session_state:
        st.session_state.hybrid_scaler = HybridAutoScaler()
    if "online_manager" not in st.session_state:
        st.session_state.online_manager = OnlineModelManager()


@st.cache_data(show_spinner=False)
def get_dataset(modified: float | None) -> pd.DataFrame:
    del modified
    return load_dataset(DATASET_PATH)


@st.cache_resource(show_spinner=False)
def get_rf_model(modified: float | None) -> dict | None:
    del modified
    if not MODEL_PATH.exists():
        return None
    try:
        return load_artifact(MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_lstm_bundle(modified: float | None):
    del modified
    if not LSTM_MODEL_PATH.exists():
        return None
    try:
        return load_lstm_bundle(LSTM_MODEL_PATH)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def get_q_agent(modified: float | None) -> QLearningAgent | None:
    del modified
    if not Q_TABLE_PATH.exists():
        return None
    try:
        return QLearningAgent.load(Q_TABLE_PATH)
    except Exception:
        return None


def styled_chart(fig: go.Figure, title: str, height: int = 280) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        height=height,
        margin=dict(l=10, r=10, t=38, b=10),
        template="plotly_dark",
    )
    return fig


def status_badge(action: str) -> None:
    act = action.upper()
    colors = {
        "SCALE_UP": ("#7c2d12", "#fdba74", "SCALE UP (+1 VM)"),
        "SCALE_DOWN": ("#064e3b", "#6ee7b7", "SCALE DOWN (-1 VM)"),
        "MAINTAIN": ("#0c4a6e", "#7dd3fc", "MAINTAIN CURRENT VMS"),
    }
    bg, fg, label = colors.get(act, ("#1e293b", "#cbd5e1", act))
    st.markdown(
        f"<span class='status-badge' style='background:{bg};color:{fg};'>{label}</span>",
        unsafe_allow_html=True,
    )


# =========================================================================
# 1. Project Overview
# =========================================================================
def page_overview(frame: pd.DataFrame, rf_bundle, lstm_bundle, q_agent) -> None:
    st.title("INTELLIGENT CLOUD AUTO-SCALING PLATFORM")
    st.caption("Hybrid Machine Learning & Reinforcement Learning Cloud Resource Optimization")

    st.markdown(
        """
        <div class='callout-box'>
        <b>Academic Research Prototype:</b> This platform demonstrates proactive, predictive cloud auto-scaling.
        It integrates <b>Random Forest</b> regression, <b>Stacked LSTM</b> sequential forecasting, and a <b>Q-Learning</b>
        reinforcement learning decision controller to eliminate reactive provisioning latency and prevent SLA violations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top KPI Metrics
    idx = int(st.session_state.row_index)
    curr_row = frame.iloc[idx]
    c_cpu = float(curr_row[TARGET_COLUMN])
    c_mem = float(curr_row["memory_usage"])
    c_vms = int(st.session_state.scaler.current_vms)

    # Predictions
    rf_pred = None
    if rf_bundle is not None:
        try:
            feats = curr_row[FEATURE_COLUMNS].to_dict()
            feats["active_vms"] = c_vms
            rf_pred = float(predict_cpu(rf_bundle, pd.DataFrame([feats]))[0])
        except Exception:
            pass

    lstm_pred = None
    if lstm_bundle is not None:
        try:
            model, sx, sy, lookback, _ = lstm_bundle
            start = max(0, idx - lookback + 1)
            window = frame.iloc[start : idx + 1]
            if len(window) < lookback:
                window = pd.concat([frame.iloc[: lookback - len(window)], window], ignore_index=True)
            lstm_pred = predict_lstm_sequence(model, sx, sy, window)
        except Exception:
            pass

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Observed CPU", f"{c_cpu:.1f}%")
    k2.metric("RF Predicted CPU", f"{rf_pred:.1f}%" if rf_pred else "—")
    k3.metric("LSTM Predicted CPU", f"{lstm_pred:.1f}%" if lstm_pred else "—")
    k4.metric("Memory Usage", f"{c_mem:.1f}%")
    k5.metric("Active VMs", c_vms)

    st.subheader("System Architecture")
    st.markdown(
        """
        ```text
        +-------------------------------------------------------------+
        |                 Monitoring Layer (Telemetry)                |
        |        Request Rate, Memory, Latency, Active VMs, SLA       |
        +-------------------------------------------------------------+
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        +-----------------------+             +-----------------------+
        |  Random Forest Model  |             |  Stacked LSTM Model   |
        |  Tabular Tree Regr.   |             |  Time-Series Sequence |
        +-----------------------+             +-----------------------+
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                        [Forecasted CPU Utilization]
                                      │
                                      ▼
        +-------------------------------------------------------------+
        |             Reinforcement Learning Controller               |
        |         Q-Learning Agent (Multi-Objective Reward)           |
        |     State: [CPU, Pred CPU, Memory, VMs, Trend, SLA]         |
        |     Action: [MAINTAIN, SCALE_UP, SCALE_DOWN]                |
        +-------------------------------------------------------------+
                                      │
                                      ▼
        +-------------------------------------------------------------+
        |          Guardrails: Cooldown & Capacity Clamps             |
        |               [Min VMs = 1, Max VMs = 10]                   |
        +-------------------------------------------------------------+
        ```
        """
    )


# =========================================================================
# 2. Live Workload Monitoring
# =========================================================================
def page_live_monitoring(frame: pd.DataFrame, rf_bundle, lstm_bundle, q_agent) -> None:
    st.title("Live Workload Monitoring")
    st.caption("Chronological time-series monitoring of cloud infrastructure telemetry.")

    max_idx = len(frame) - 1
    st.slider("Simulated Minute (Cursor)", 0, max_idx, key="row_index")

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        if st.button("Advance +1 Min", type="primary"):
            st.session_state.row_index = min(max_idx, st.session_state.row_index + 1)
            st.rerun()

    idx = int(st.session_state.row_index)
    start = max(0, idx - 120)
    window = frame.iloc[start : idx + 1]

    c1, c2 = st.columns(2)
    c1.plotly_chart(styled_chart(px.line(window, x="timestamp", y=TARGET_COLUMN), "CPU Utilization (%)"), use_container_width=True)
    c2.plotly_chart(styled_chart(px.line(window, x="timestamp", y="memory_usage"), "Memory Utilization (%)"), use_container_width=True)

    c3, c4 = st.columns(2)
    c3.plotly_chart(styled_chart(px.line(window, x="timestamp", y="request_rate"), "Request Arrival Rate (req/s)"), use_container_width=True)
    c4.plotly_chart(styled_chart(px.line(window, x="timestamp", y="response_time"), "End-to-End Latency (ms)"), use_container_width=True)


# =========================================================================
# 3. Random Forest Prediction
# =========================================================================
def page_random_forest(frame: pd.DataFrame, rf_bundle) -> None:
    st.title("Random Forest CPU Prediction")
    st.caption("Scikit-learn RandomForestRegressor fitted on chronological workload splits.")

    if rf_bundle is None:
        st.warning("Random Forest model is not trained yet.")
        if st.button("Train Random Forest Now"):
            with st.spinner("Training Random Forest..."):
                run_training()
            st.rerun()
        return

    _, _, test_df, note = chronological_split(frame)
    sample_slice = test_df.iloc[:400]
    preds = predict_cpu(rf_bundle, sample_slice)

    plot_df = pd.DataFrame({
        "timestamp": sample_slice["timestamp"],
        "Actual CPU": sample_slice[TARGET_COLUMN].to_numpy(),
        "Predicted CPU (RF)": preds,
    }).melt(id_vars="timestamp", var_name="Series", value_name="CPU (%)")

    fig = px.line(plot_df, x="timestamp", y="CPU (%)", color="Series", title="Actual vs. Random Forest Predicted CPU")
    st.plotly_chart(styled_chart(fig, "Actual vs. Random Forest Predicted CPU", height=380), use_container_width=True)

    # Feature Importance
    importances = rf_bundle.get("feature_importances", {})
    if importances:
        imp_df = pd.DataFrame({"Feature": list(importances.keys()), "Importance": list(importances.values())})
        st.subheader("Random Forest Feature Importance")
        st.plotly_chart(styled_chart(px.bar(imp_df, x="Importance", y="Feature", orientation="h"), "Feature Importances (MDI)", height=240), use_container_width=True)


# =========================================================================
# 4. LSTM Prediction
# =========================================================================
def page_lstm(frame: pd.DataFrame, lstm_bundle) -> None:
    st.title("LSTM Workload Forecasting")
    st.caption("Multi-variate stacked LSTM sequence-to-one forecasting built with PyTorch.")

    if lstm_bundle is None:
        st.warning("LSTM model is not trained yet.")
        if st.button("Train LSTM Model Now"):
            with st.spinner("Training Stacked LSTM on chronological partitions..."):
                run_lstm_pipeline()
            st.rerun()
        return

    model, sx, sy, lookback, metadata = lstm_bundle
    st.markdown(f"**Lookback Window:** `{lookback}` timesteps | **Architecture:** Stacked PyTorch LSTM (2 Layers, 64 Units) with Dropout")

    _, _, test_df, _ = chronological_split(frame)
    sample_slice = test_df.iloc[:300]

    preds = []
    for i in range(len(sample_slice)):
        row_idx = len(test_df) - 300 + i
        start = max(0, row_idx - lookback)
        seq = frame.iloc[start : row_idx]
        if len(seq) < lookback:
            seq = pd.concat([frame.iloc[: lookback - len(seq)], seq], ignore_index=True)
        preds.append(predict_lstm_sequence(model, sx, sy, seq))

    plot_df = pd.DataFrame({
        "timestamp": sample_slice["timestamp"],
        "Actual CPU": sample_slice[TARGET_COLUMN].to_numpy(),
        "Predicted CPU (LSTM)": preds,
    }).melt(id_vars="timestamp", var_name="Series", value_name="CPU (%)")

    fig = px.line(plot_df, x="timestamp", y="CPU (%)", color="Series", title="Actual vs. LSTM Forecasted CPU")
    st.plotly_chart(styled_chart(fig, "Actual vs. LSTM Forecasted CPU", height=380), use_container_width=True)

    metrics = metadata.get("metrics", {}).get("test", {})
    if metrics:
        m1, m2, m3 = st.columns(3)
        m1.metric("LSTM Test MAE", f"{metrics.get('mae', 0.0):.4f}")
        m2.metric("LSTM Test RMSE", f"{metrics.get('rmse', 0.0):.4f}")
        m3.metric("LSTM Test R²", f"{metrics.get('r2', 0.0):.4f}")


# =========================================================================
# 5. Forecasting Model Comparison
# =========================================================================
def page_forecasting_comparison() -> None:
    st.title("Forecasting Model Comparison")
    st.caption("Empirical benchmark: Random Forest vs. Stacked LSTM vs. Persistence vs. Mean Baselines.")

    if not METRICS_PATH.exists():
        st.warning("Metrics file not found. Train both Random Forest and LSTM first.")
        return

    metrics_df = pd.read_csv(METRICS_PATH)
    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()

    st.subheader("Model Evaluation Results on Unseen Test Split")
    st.dataframe(test_metrics[["model", "mae", "rmse", "r2", "notes"]], use_container_width=True, hide_index=True)

    # Bar chart comparison
    bar_df = test_metrics.melt(id_vars="model", value_vars=["mae", "rmse"], var_name="Metric", value_name="Error Value")
    fig = px.bar(bar_df, x="model", y="Error Value", color="Metric", barmode="group", title="Test Error Comparison (MAE & RMSE)")
    st.plotly_chart(styled_chart(fig, "Test Error Comparison", height=340), use_container_width=True)

    with st.expander("Reference Results from Original Research Paper"):
        st.dataframe(REFERENCE_METRICS, use_container_width=True, hide_index=True)
        st.caption("These reference metrics from the literature are shown for academic context and are not fabricated into our local run.")


# =========================================================================
# 6. Q-Learning Training
# =========================================================================
def page_q_learning_training(frame: pd.DataFrame, q_agent) -> None:
    st.title("Reinforcement Learning: Q-Learning Training")
    st.caption("Model-free tabular Q-learning agent optimizing cloud resource scaling.")

    st.markdown(
        """
        <div class='callout-box'>
        <b>Multi-Objective Reward Function:</b>
        <br>
        <code>R = - w_sla * SLA - w_cost * (VMs / MaxVMs) - w_osc * ScaleAction - w_dev * |CPU - Target| + w_stable * InBand</code>
        <br>
        Balancing QoS preservation against infrastructure cloud expenditure.
        </div>
        """,
        unsafe_allow_html=True,
    )

    episodes = st.slider("Training Episodes", 3, 20, 5)
    if st.button("Train Q-Learning Agent"):
        train_df, _, _, _ = chronological_split(frame)
        with st.spinner(f"Training Q-agent across {episodes} episodes..."):
            agent, rewards, slas = train_q_learning(train_df, episodes=episodes)
            agent.save(Q_TABLE_PATH, metadata={"episodes": episodes})
        st.success(f"Training completed! Explored {len(agent.q_table)} distinct states.")
        st.session_state.rewards_hist = rewards
        st.session_state.slas_hist = slas
        st.rerun()

    if "rewards_hist" in st.session_state:
        c1, c2 = st.columns(2)
        c1.plotly_chart(styled_chart(px.line(y=st.session_state.rewards_hist, title="Cumulative Reward per Episode"), "Training Reward Curve"), use_container_width=True)
        c2.plotly_chart(styled_chart(px.line(y=st.session_state.slas_hist, title="SLA Violation % per Episode"), "SLA Violation Rate"), use_container_width=True)


# =========================================================================
# 7. Reinforcement Learning Decisions
# =========================================================================
def page_rl_decisions(frame: pd.DataFrame, q_agent) -> None:
    st.title("Reinforcement Learning Decisions & State Space")
    st.caption("Real-time state discretization and Q-value action evaluation.")

    if q_agent is None:
        st.warning("Q-learning agent not trained yet. Train it from section 6.")
        return

    idx = int(st.session_state.row_index)
    row = frame.iloc[idx]
    c_cpu = float(row[TARGET_COLUMN])
    c_mem = float(row["memory_usage"])
    vms = int(st.session_state.scaler.current_vms)

    state = extract_state_tuple(c_cpu, c_cpu, c_mem, vms, 0.0, 0)
    q_vals = q_agent.get_q_values(state)
    best_act = int(np.argmax(q_vals))

    st.subheader("Current State Breakdown")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("CPU Bin Index", state[0])
    s2.metric("Memory Bin Index", state[2])
    s3.metric("VM Bin Index", state[3])
    s4.metric("Workload Trend Bin", state[4])

    st.subheader("Action-Value (Q-Table) Estimates")
    q_df = pd.DataFrame([
        {"Action": ACTION_NAMES[0], "Q-Value": round(float(q_vals[0]), 3), "Optimal": "★" if best_act == 0 else ""},
        {"Action": ACTION_NAMES[1], "Q-Value": round(float(q_vals[1]), 3), "Optimal": "★" if best_act == 1 else ""},
        {"Action": ACTION_NAMES[2], "Q-Value": round(float(q_vals[2]), 3), "Optimal": "★" if best_act == 2 else ""},
    ])
    st.dataframe(q_df, use_container_width=True, hide_index=True)
    status_badge(ACTION_NAMES[best_act])


# =========================================================================
# 8. Hybrid Auto-Scaling
# =========================================================================
def page_hybrid_autoscaling(frame: pd.DataFrame, rf_bundle, lstm_bundle, q_agent) -> None:
    st.title("Hybrid Predictive Auto-Scaling (Forecasting + RL)")
    st.caption("Live simulation coupling neural forecasting and reinforcement learning policies.")

    presets = scenario_presets()
    st.subheader("Select Workload Scenario")
    cols = st.columns(4)
    for col, name in zip(cols, presets):
        if col.button(name, use_container_width=True):
            p = presets[name]
            st.session_state.sim_cpu = p["cpu_utilization"]
            st.session_state.sim_memory = p["memory_usage"]
            st.session_state.sim_requests = p["request_rate"]
            st.session_state.sim_latency = p["response_time"]
            st.rerun()

    s1, s2, s3, s4 = st.columns(4)
    s1.slider("Workload CPU (%)", 0.0, 100.0, key="sim_cpu")
    s2.slider("Workload Memory (%)", 0.0, 100.0, key="sim_memory")
    s3.slider("Request Rate (req/s)", 0.0, 1000.0, key="sim_requests")
    s4.slider("Latency (ms)", 0.0, 2000.0, key="sim_latency")

    # Forecast with LSTM
    lstm_pred = float(st.session_state.sim_cpu)
    if lstm_bundle is not None:
        try:
            model, sx, sy, lookback, _ = lstm_bundle
            seq = np.tile([st.session_state.sim_requests, st.session_state.sim_memory, st.session_state.scaler.current_vms, st.session_state.sim_latency, 0], (lookback, 1))
            seq_df = pd.DataFrame(seq, columns=FEATURE_COLUMNS)
            lstm_pred = predict_lstm_sequence(model, sx, sy, seq_df)
        except Exception:
            pass

    dec = st.session_state.hybrid_scaler.decide(
        current_cpu=st.session_state.sim_cpu,
        memory_usage=st.session_state.sim_memory,
        request_rate=st.session_state.sim_requests,
        response_time=st.session_state.sim_latency,
        sla_violation=1 if st.session_state.sim_latency >= 500 or st.session_state.sim_cpu >= 90 else 0,
        predicted_cpu=lstm_pred,
        timestamp="Interactive",
    )

    st.subheader("Decision Outcome")
    status_badge(dec.action)
    st.info(f"**Justification:** {dec.reason}")
    st.caption(f"Observed CPU: {dec.observed_cpu:.1f}% | Predicted CPU: {dec.predicted_cpu:.1f}% | VMs: {dec.current_vms} → {dec.new_vms}")


# =========================================================================
# 9. Four-Strategy Comparison
# =========================================================================
def page_four_strategy_comparison(frame: pd.DataFrame, rf_bundle, lstm_bundle, q_agent) -> None:
    st.title("Four Auto-Scaling Strategies Benchmark")
    st.caption("Direct empirical evaluation under identical workload traces, VM bounds, and SLA targets.")

    sample_size = st.slider("Workload Evaluation Length (Minutes)", 100, 2000, 500, step=100)

    if st.button("Run 4-Strategy Benchmark", type="primary"):
        _, _, test_df, _ = chronological_split(frame)
        eval_slice = test_df.iloc[:sample_size]
        with st.spinner("Simulating all 4 auto-scaling strategies under identical workload..."):
            summary_df, traces = compare_all_strategies(eval_slice, rf_bundle, lstm_bundle, q_agent)
            st.session_state.strategy_summary = summary_df
            st.session_state.strategy_traces = traces
        st.success("Benchmark completed successfully!")

    if "strategy_summary" in st.session_state:
        st.subheader("Comparative Benchmark Metrics")
        st.dataframe(st.session_state.strategy_summary, use_container_width=True, hide_index=True)

        # Plot traces
        traces = st.session_state.strategy_traces
        combined = pd.concat([traces[s] for s in traces], ignore_index=True)
        st.plotly_chart(styled_chart(px.line(combined, x="step", y="cpu_utilization", color="strategy"), "CPU Utilization Comparison", height=320), use_container_width=True)
        st.plotly_chart(styled_chart(px.line(combined, x="step", y="active_vms", color="strategy"), "Active VM Count Over Time", height=320), use_container_width=True)


# =========================================================================
# 10. Uncertainty-Aware Prediction
# =========================================================================
def page_uncertainty(frame: pd.DataFrame, rf_bundle, lstm_bundle) -> None:
    st.title("Uncertainty-Aware Workload Forecasting")
    st.caption("Quantifying epistemic uncertainty via tree-variance and Monte Carlo Dropout.")

    if rf_bundle is None:
        st.warning("Random Forest model is required.")
        return

    idx = int(st.session_state.row_index)
    sample = frame.iloc[[idx]]

    rf_unc = estimate_rf_uncertainty(rf_bundle, sample)

    u1, u2, u3, u4 = st.columns(4)
    u1.metric("Predicted CPU (Mean)", f"{rf_unc.mean}%")
    u2.metric("Epistemic Std Dev (σ)", f"±{rf_unc.std}%")
    u3.metric("95% Confidence Band", f"[{rf_unc.lower_bound_95}%, {rf_unc.upper_bound_95}%]")
    u4.metric("Confidence Score", f"{rf_unc.confidence_score * 100:.1f}%")

    if rf_unc.is_high_uncertainty:
        st.error(f"HIGH UNCERTAINTY DETECTED: Forecast standard deviation (±{rf_unc.std}%) exceeds safe scaling threshold.")
    else:
        st.success("FORECAST CONFIDENCE HIGH: Prediction variance is within narrow bounds.")

    st.subheader("Uncertainty-Damped Scaling Policy")
    test_action = st.radio("Test Scaling Action Candidate", ["SCALE_DOWN", "SCALE_UP", "MAINTAIN"])
    damped_act, damp_reason = apply_uncertainty_aware_guard(test_action, rf_unc)
    st.markdown(f"**Adjusted Action:** `{damped_act}`")
    st.caption(damp_reason)


# =========================================================================
# 11. Explainable AI
# =========================================================================
def page_xai(frame: pd.DataFrame, rf_bundle, lstm_bundle, q_agent) -> None:
    st.title("Explainable AI (XAI) for Cloud Auto-Scaling")
    st.caption("Interpretable explanations for tabular trees, sequential recurrence, and RL policies.")

    tab_rf, tab_lstm, tab_rl = st.tabs(["Random Forest (Tabular)", "LSTM (Sequential Saliency)", "Q-Learning (Action-Values)"])

    with tab_rf:
        if rf_bundle:
            rf_exp = explain_random_forest(rf_bundle, frame.iloc[[int(st.session_state.row_index)]])
            st.subheader("Global Feature Importance (MDI)")
            imp = rf_exp["global_importance"]
            st.plotly_chart(styled_chart(px.bar(x=list(imp.values()), y=list(imp.keys()), orientation="h"), "Feature Contribution"), use_container_width=True)

    with tab_lstm:
        if lstm_bundle:
            model, sx, sy, lookback, _ = lstm_bundle
            idx = int(st.session_state.row_index)
            start = max(0, idx - lookback + 1)
            window = frame.iloc[start : idx + 1]
            if len(window) < lookback:
                window = pd.concat([frame.iloc[: lookback - len(window)], window], ignore_index=True)

            lstm_exp = explain_lstm_temporal_sensitivity(model, sx, sy, window)
            st.subheader("Temporal Lag Saliency Analysis")
            st.caption(lstm_exp["academic_disclosure"])
            ts_imp = lstm_exp["timestep_importance"]
            st.plotly_chart(styled_chart(px.bar(x=list(ts_imp.keys()), y=list(ts_imp.values())), "Timestep Sensitivity (% Gradient Impact)"), use_container_width=True)

    with tab_rl:
        if q_agent:
            idx = int(st.session_state.row_index)
            row = frame.iloc[idx]
            q_exp = explain_q_learning_decision(
                q_agent, float(row[TARGET_COLUMN]), float(row[TARGET_COLUMN]), float(row["memory_usage"]), int(st.session_state.scaler.current_vms), 0.0, int(row["sla_violation"])
            )
            st.subheader("Q-Value Decomposition")
            st.json(q_exp["q_values"])
            st.info(f"**Policy Choice Justification:** {q_exp['justification']}")


# =========================================================================
# 12. Performance Evaluation
# =========================================================================
def page_performance_eval(frame: pd.DataFrame) -> None:
    st.title("Performance Evaluation & Online Learning")
    st.caption("Telemetry metrics, long-range trends, and concept drift monitoring.")

    summary = summarize_workload(frame)
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("SLA Violation Rate", f"{summary['sla_violation_percent']:.2f}%")
    p2.metric("Average Active VMs", f"{summary['average_vms']:.2f}")
    p3.metric("Simulated Cost", f"${summary['estimated_cost_usd']:.2f}")
    p4.metric("Recorded VM Shifts", int(summary["recorded_vm_changes"]))

    st.subheader("Full-Trace Workload Trends")
    st.plotly_chart(styled_chart(px.line(frame.iloc[::20], x="timestamp", y=TARGET_COLUMN), "CPU Utilization Trend"), use_container_width=True)

    if ONLINE_METRICS_PATH.exists():
        st.subheader("Online Learning & Drift Event Log")
        log_df = pd.read_csv(ONLINE_METRICS_PATH)
        st.dataframe(log_df, use_container_width=True, hide_index=True)


# =========================================================================
# 13. Scaling History and Logs
# =========================================================================
def page_scaling_history() -> None:
    st.title("Scaling Decision Audit Logs")
    st.caption("Immutable chronological record of committed scaling operations.")

    if SCALING_HISTORY_PATH.exists():
        hist_df = pd.read_csv(SCALING_HISTORY_PATH)
        action_filter = st.multiselect("Filter by Action", ["scale_up", "scale_down", "maintain"], default=["scale_up", "scale_down"])
        filtered = hist_df[hist_df["action"].isin(action_filter)]
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button("Download CSV Log", hist_df.to_csv(index=False).encode("utf-8"), "scaling_history.csv", "text/csv")
    else:
        st.info("No persistent scaling history found. Run model training or strategy simulation to generate logs.")


# =========================================================================
# Main Router
# =========================================================================
def main() -> None:
    st.set_page_config(
        page_title="Intelligent Cloud Auto-Scaling Platform",
        page_icon="☁",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()
    init_state()

    frame = get_dataset(DATASET_PATH.stat().st_mtime if DATASET_PATH.exists() else None)
    rf_bundle = get_rf_model(MODEL_PATH.stat().st_mtime if MODEL_PATH.exists() else None)
    lstm_bundle = get_lstm_bundle(LSTM_MODEL_PATH.stat().st_mtime if LSTM_MODEL_PATH.exists() else None)
    q_agent = get_q_agent(Q_TABLE_PATH.stat().st_mtime if Q_TABLE_PATH.exists() else None)

    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Select View", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.metric("Controller VMs", st.session_state.scaler.current_vms)

    if choice == PAGES[0]:
        page_overview(frame, rf_bundle, lstm_bundle, q_agent)
    elif choice == PAGES[1]:
        page_live_monitoring(frame, rf_bundle, lstm_bundle, q_agent)
    elif choice == PAGES[2]:
        page_random_forest(frame, rf_bundle)
    elif choice == PAGES[3]:
        page_lstm(frame, lstm_bundle)
    elif choice == PAGES[4]:
        page_forecasting_comparison()
    elif choice == PAGES[5]:
        page_q_learning_training(frame, q_agent)
    elif choice == PAGES[6]:
        page_rl_decisions(frame, q_agent)
    elif choice == PAGES[7]:
        page_hybrid_autoscaling(frame, rf_bundle, lstm_bundle, q_agent)
    elif choice == PAGES[8]:
        page_four_strategy_comparison(frame, rf_bundle, lstm_bundle, q_agent)
    elif choice == PAGES[9]:
        page_uncertainty(frame, rf_bundle, lstm_bundle)
    elif choice == PAGES[10]:
        page_xai(frame, rf_bundle, lstm_bundle, q_agent)
    elif choice == PAGES[11]:
        page_performance_eval(frame)
    elif choice == PAGES[12]:
        page_scaling_history()


if __name__ == "__main__":
    main()
