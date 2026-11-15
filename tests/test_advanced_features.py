"""Comprehensive Automated Test Suite for Advanced Cloud Auto-Scaling Features.

Tests:
1. LSTM model training, sequence creation, inference, and metrics.
2. Q-Learning environment, reward function, Q-table updates, and provisioning delay.
3. Hybrid controller with 4 strategies and guardrail constraints.
4. Uncertainty-aware estimation (Tree variance & Monte Carlo Dropout) and damping.
5. Explainable AI (RF feature importance, LSTM gradient saliency, Q-value breakdown).
6. Online learning error monitoring, concept drift detection, and adaptation.
7. Workload scenarios 1 through 7 (Low, Normal, High, Spike, Increasing, Decreasing, Fluctuating).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src import FEATURE_COLUMNS, LSTM_MODEL_PATH, MODEL_PATH, Q_TABLE_PATH
from src.data_generator import generate_workload
from src.data_preprocessing import load_dataset
from src.explainable_ai import (
    explain_lstm_temporal_sensitivity,
    explain_q_learning_decision,
    explain_random_forest,
)
from src.hybrid_autoscaler import HybridAutoScaler
from src.lstm_model import (
    WorkloadLSTM,
    create_sequences,
    evaluate_lstm,
    load_lstm_bundle,
    predict_lstm_sequence,
    prepare_lstm_data,
)
from src.online_learning import OnlineModelManager
from src.prediction import load_artifact
from src.q_learning import (
    ACTION_MAINTAIN,
    ACTION_SCALE_DOWN,
    ACTION_SCALE_UP,
    CloudAutoScalingEnv,
    EnvironmentConfig,
    QLearningAgent,
    extract_state_tuple,
)
from src.strategy_comparison import compare_all_strategies, run_strategy_simulation
from src.uncertainty_forecasting import (
    apply_uncertainty_aware_guard,
    estimate_lstm_uncertainty,
    estimate_rf_uncertainty,
)


# =========================================================================
# 1. LSTM Tests
# =========================================================================

def test_lstm_sequence_creation():
    x = np.random.randn(50, 5).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    lookback = 10
    seq_x, seq_y = create_sequences(x, y, lookback=lookback)
    assert seq_x.shape == (40, 10, 5)
    assert seq_y.shape == (40,)
    assert np.allclose(seq_x[0], x[:10])
    assert seq_y[0] == y[10]


def test_lstm_forward_pass_and_inference():
    model = WorkloadLSTM(input_dim=5, hidden_dim=32, num_layers=1)
    x = torch.randn(4, 10, 5)
    out = model(x)
    assert out.shape == (4,)

    # Test predict_lstm_sequence function with mock scalers
    class MockScaler:
        def transform(self, a):
            return np.asarray(a, dtype=float)
        def inverse_transform(self, a):
            return np.asarray(a, dtype=float)

    seq = pd.DataFrame(np.random.uniform(10, 50, (10, 5)), columns=FEATURE_COLUMNS)
    pred = predict_lstm_sequence(model, MockScaler(), MockScaler(), seq)
    assert 0.0 <= pred <= 100.0


# =========================================================================
# 2. Q-Learning & Environment Tests
# =========================================================================

def test_q_learning_state_discretization():
    # Low CPU, High memory, 3 VMs, positive trend, no SLA
    state = extract_state_tuple(20.0, 25.0, 80.0, 3, 15.0, 0)
    assert len(state) == 6
    assert state[0] == 0  # CPU < 35
    assert state[1] == 0  # Pred CPU < 35
    assert state[2] == 2  # Memory >= 75
    assert state[3] == 2  # VM 3 -> index 2
    assert state[4] == 2  # Trend > 10
    assert state[5] == 0  # No SLA


def test_cloud_env_provisioning_delay_and_clamping():
    df = generate_workload(100, seed=12)
    cfg = EnvironmentConfig(min_vms=1, max_vms=5, cooldown_steps=0, provisioning_delay=2)
    env = CloudAutoScalingEnv(df, config=cfg)
    env.reset(initial_vms=2)

    # Scale up action
    state, reward, done, info = env.step(ACTION_SCALE_UP)
    assert info["active_vms"] == 2  # Delayed, not active immediately
    assert len(env.provisioning_queue) == 1

    # Step 2: Still pending
    state, reward, done, info = env.step(ACTION_MAINTAIN)
    # Step 3: Now ready
    state, reward, done, info = env.step(ACTION_MAINTAIN)
    assert env.current_vms == 3


def test_q_learning_agent_update_and_saving(tmp_path: Path):
    agent = QLearningAgent(learning_rate=0.1, discount_factor=0.9, seed=1)
    s = (1, 1, 1, 2, 1, 0)
    ns = (2, 2, 1, 3, 1, 0)
    initial_q = agent.get_q_values(s).copy()
    agent.update(s, ACTION_SCALE_UP, reward=-2.0, next_state=ns, done=False)
    updated_q = agent.get_q_values(s)
    assert updated_q[ACTION_SCALE_UP] != initial_q[ACTION_SCALE_UP]

    # Save and load
    save_file = tmp_path / "test_q.pkl"
    agent.save(save_file)
    loaded = QLearningAgent.load(save_file)
    assert np.allclose(loaded.get_q_values(s), updated_q)


# =========================================================================
# 3. Hybrid Controller & 4-Strategy Tests
# =========================================================================

def test_hybrid_autoscaler_guardrails():
    scaler = HybridAutoScaler(min_vms=1, max_vms=4, cooldown_steps=2, initial_vms=4)

    # Scale up when at max capacity
    d1 = scaler.decide(current_cpu=95.0, memory_usage=80.0, request_rate=500.0, response_time=200.0, sla_violation=0)
    assert d1.action == "MAINTAIN"
    assert "max capacity" in d1.reason

    # Cooldown enforcement
    scaler.current_vms = 2
    scaler.steps_since_scale = 0
    d2 = scaler.decide(current_cpu=95.0, memory_usage=80.0, request_rate=500.0, response_time=200.0, sla_violation=0)
    assert d2.action == "MAINTAIN"
    assert "cooldown active" in d2.reason


def test_four_strategies_benchmark():
    df = generate_workload(100, seed=7)
    q_agent = QLearningAgent()
    summary, traces = compare_all_strategies(df, q_agent=q_agent)
    assert len(summary) == 4
    for strat in ["Strategy A: Traditional Threshold", "Strategy B: Forecast-Only", "Strategy C: RL-Only", "Strategy D: Hybrid LSTM + Q-Learning"]:
        assert strat in summary["strategy"].values
        assert strat in traces
        assert len(traces[strat]) == len(df)


# =========================================================================
# 4. Uncertainty-Aware Forecasting Tests
# =========================================================================

def test_rf_uncertainty_estimation():
    bundle = load_artifact(MODEL_PATH)
    sample = pd.DataFrame([{
        "request_rate": 200.0, "memory_usage": 40.0, "active_vms": 3, "response_time": 100.0, "sla_violation": 0
    }])
    est = estimate_rf_uncertainty(bundle, sample)
    assert 0.0 <= est.mean <= 100.0
    assert est.std >= 0.0
    assert est.lower_bound_95 <= est.upper_bound_95
    assert 0.0 <= est.confidence_score <= 1.0


def test_uncertainty_damping_logic():
    from src.uncertainty_forecasting import UncertaintyEstimate
    # High uncertainty where upper CI reaches 38%
    high_unc = UncertaintyEstimate(
        mean=28.0, std=5.5, lower_bound_95=17.2, upper_bound_95=38.8,
        confidence_score=0.45, is_high_uncertainty=True, method="Test"
    )
    action, reason = apply_uncertainty_aware_guard("SCALE_DOWN", high_unc, scale_down_threshold=30.0)
    assert action == "MAINTAIN"
    assert "UNCERTAINTY DAMPING" in reason


# =========================================================================
# 5. Explainable AI Tests
# =========================================================================

def test_explainable_ai_modules():
    # RF
    bundle = load_artifact(MODEL_PATH)
    sample = pd.DataFrame([{
        "request_rate": 200.0, "memory_usage": 40.0, "active_vms": 3, "response_time": 100.0, "sla_violation": 0
    }])
    rf_exp = explain_random_forest(bundle, sample)
    assert "global_importance" in rf_exp
    assert "local_feature_deviation" in rf_exp

    # Q-Learning
    agent = QLearningAgent.load(Q_TABLE_PATH)
    q_exp = explain_q_learning_decision(agent, 88.0, 92.0, 75.0, 3, 20.0, 1)
    assert "q_values" in q_exp
    assert "selected_action" in q_exp
    assert "justification" in q_exp

    # LSTM
    model, sx, sy, lookback, _ = load_lstm_bundle(LSTM_MODEL_PATH)
    seq = pd.DataFrame(np.random.uniform(10, 50, (lookback, 5)), columns=FEATURE_COLUMNS)
    lstm_exp = explain_lstm_temporal_sensitivity(model, sx, sy, seq)
    assert "timestep_importance" in lstm_exp
    assert "feature_temporal_sensitivity" in lstm_exp


# =========================================================================
# 6. Online Learning Tests
# =========================================================================

def test_online_learning_drift_and_adaptation(tmp_path: Path):
    manager = OnlineModelManager(window_size=20, drift_error_threshold=5.0, log_path=tmp_path / "test_log.csv")
    feats = {"request_rate": 100.0, "memory_usage": 30.0, "active_vms": 2, "response_time": 80.0, "sla_violation": 0}

    # Inject small errors
    for _ in range(15):
        manager.record_observation(feats, actual_cpu=50.0, predicted_cpu=51.0)
    assert not manager.drift_active

    # Inject large error spike
    for _ in range(15):
        status = manager.record_observation(feats, actual_cpu=80.0, predicted_cpu=40.0)
    assert status["drift_detected"]

    # Trigger adaptation
    model, sx, sy, _, _ = load_lstm_bundle(LSTM_MODEL_PATH)
    adapt_res = manager.trigger_adaptation(model, sx, sy, epochs=1)
    assert adapt_res["status"] == "success"
    assert manager.model_version == 2


# =========================================================================
# 7. Verification Across 7 Required Workload Scenarios
# =========================================================================

@pytest.mark.parametrize("scenario_type", [
    "Low workload",
    "Normal workload",
    "High workload",
    "Sudden traffic spike",
    "Gradually increasing workload",
    "Gradually decreasing workload",
    "Highly fluctuating workload",
])
def test_all_seven_workload_scenarios(scenario_type: str):
    """Verify that predictions, actions, and VM clamps function properly across all 7 scenarios."""
    scaler = HybridAutoScaler(min_vms=1, max_vms=10, initial_vms=3)
    n_steps = 25

    if scenario_type == "Low workload":
        requests = np.full(n_steps, 35.0)
    elif scenario_type == "Normal workload":
        requests = np.full(n_steps, 200.0)
    elif scenario_type == "High workload":
        requests = np.full(n_steps, 650.0)
    elif scenario_type == "Sudden traffic spike":
        requests = np.full(n_steps, 200.0)
        requests[10:15] = 950.0  # Acute spike
    elif scenario_type == "Gradually increasing workload":
        requests = np.linspace(50.0, 850.0, n_steps)
    elif scenario_type == "Gradually decreasing workload":
        requests = np.linspace(850.0, 50.0, n_steps)
    else:  # Highly fluctuating workload
        rng = np.random.default_rng(42)
        requests = rng.uniform(40.0, 900.0, n_steps)

    for step, req in enumerate(requests):
        cpu = float(np.clip(req / 10.0, 5.0, 95.0))
        dec = scaler.decide(
            current_cpu=cpu,
            memory_usage=30.0,
            request_rate=req,
            response_time=70.0,
            sla_violation=1 if cpu >= 90 else 0,
            predicted_cpu=cpu,
        )
        assert 1 <= scaler.current_vms <= 10
        assert dec.action in ["MAINTAIN", "SCALE_UP", "SCALE_DOWN"]
