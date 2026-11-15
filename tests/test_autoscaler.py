"""Scaling rules for the simulated VM controller."""

import numpy as np
import pandas as pd

from src.autoscaler import AutoScaler
from src.data_generator import generate_workload
from src.data_preprocessing import TrainOnlyImputer
from src.evaluation import simulate_controller


def _scaler(vms: int, cooldown: int = 0) -> AutoScaler:
    return AutoScaler(initial_vms=vms, cooldown_steps=cooldown)


def test_low_cpu_scales_down_when_more_than_one_vm_is_running():
    scaler = _scaler(4)
    decision = scaler.decide(22)
    assert decision.action == "scale_down"
    assert decision.new_vms == 3
    assert "below" in decision.reason.lower()


def test_low_cpu_maintains_when_only_one_vm_remains():
    scaler = _scaler(1)
    decision = scaler.decide(15)
    assert decision.action == "maintain"
    assert decision.new_vms == 1
    assert "minimum" in decision.reason.lower()


def test_predicted_cpu_above_80_scales_up_by_one():
    scaler = _scaler(3)
    decision = scaler.decide(85)
    assert decision.action == "scale_up"
    assert decision.new_vms == 4
    assert decision.previous_vms == 3


def test_predicted_cpu_between_30_and_80_maintains():
    scaler = _scaler(5)
    decision = scaler.decide(55)
    assert decision.action == "maintain"
    assert decision.new_vms == 5
    assert "maintain band" in decision.reason.lower()


def test_thresholds_are_exclusive_at_the_boundaries():
    scaler = _scaler(4)
    at_upper = scaler.decide(80)
    assert at_upper.action == "maintain"
    at_lower = scaler.decide(30)
    assert at_lower.action == "maintain"


def test_predicted_cpu_below_30_scales_down_above_minimum():
    scaler = _scaler(4)
    decision = scaler.decide(29.9)
    assert decision.action == "scale_down"
    assert decision.new_vms == 3


def test_maximum_capacity_blocks_another_vm():
    scaler = _scaler(10)
    decision = scaler.decide(96)
    assert decision.action == "maintain"
    assert decision.new_vms == 10
    assert "maximum" in decision.reason.lower()


def test_minimum_capacity_blocks_further_removal():
    scaler = _scaler(1)
    decision = scaler.decide(5)
    assert decision.action == "maintain"
    assert decision.new_vms == 1
    assert "minimum" in decision.reason.lower()


def test_cooldown_blocks_the_next_scale_until_quiet_decisions_elapse():
    scaler = _scaler(3, cooldown=2)
    first = scaler.decide(91)
    second = scaler.decide(92)
    third = scaler.decide(93)
    fourth = scaler.decide(94)
    assert first.action == "scale_up" and first.new_vms == 4
    assert second.action == "maintain" and "cooldown" in second.reason.lower()
    assert third.action == "maintain" and third.new_vms == 4
    assert fourth.action == "scale_up" and fourth.new_vms == 5


def test_plan_action_does_not_change_vms_or_history():
    scaler = _scaler(4)
    action, new_vms, _reason = scaler.plan_action(95, scaler.current_vms, scaler.steps_since_scale)
    assert action == "scale_up"
    assert new_vms == 5
    assert scaler.current_vms == 4
    assert scaler.history == []


def test_manual_scale_respects_capacity_and_ignores_thresholds():
    scaler = _scaler(10)
    blocked = scaler.manual("scale_up", predicted_cpu=10)
    assert blocked.new_vms == 10
    assert "maximum" in blocked.reason.lower()
    scaler.current_vms = 2
    changed = scaler.manual("scale_down", predicted_cpu=99)
    assert changed.action == "scale_down"
    assert changed.new_vms == 1


def test_non_finite_prediction_is_rejected():
    scaler = _scaler(3)
    try:
        scaler.decide(float("nan"))
    except ValueError as exc:
        assert "finite" in str(exc).lower()
    else:
        raise AssertionError("A NaN prediction should be rejected.")


class _ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value)


def test_closed_loop_never_leaves_the_vm_bounds():
    frame = generate_workload(80, seed=3)
    imputer = TrainOnlyImputer().fit(frame)
    high = simulate_controller(
        frame,
        _ConstantModel(95),
        imputer,
        cooldown_steps=0,
        initial_vms=2,
    )
    low = simulate_controller(
        frame,
        _ConstantModel(10),
        imputer,
        cooldown_steps=0,
        initial_vms=6,
    )
    assert high["new_vms"].between(1, 10).all()
    assert low["new_vms"].between(1, 10).all()
    assert int(high["new_vms"].iloc[-1]) == 10
    assert int(low["new_vms"].iloc[-1]) == 1
    assert "maximum" in " ".join(high["reason"].tail(3)).lower()
    assert "minimum" in " ".join(low["reason"].tail(3)).lower()


# =========================================================================
# Explicit Verification for Section 12 Scenarios (1 through 6)
# =========================================================================

def test_scenario_1_low_cpu_maintain_or_scale_down():
    """Scenario 1: CPU utilization is low. Expected: Maintain or scale down."""
    # Scale down when multiple VMs running
    scaler_multi = _scaler(4)
    d1 = scaler_multi.decide(15.0)
    assert d1.action == "scale_down"
    assert d1.new_vms == 3

    # Maintain when at minimum capacity (1 VM)
    scaler_single = _scaler(1)
    d2 = scaler_single.decide(15.0)
    assert d2.action == "maintain"
    assert d2.new_vms == 1


def test_scenario_2_high_cpu_above_80_scales_up():
    """Scenario 2: Predicted CPU utilization is above 80%. Expected: Scale up by one VM."""
    scaler = _scaler(3)
    decision = scaler.decide(85.0)
    assert decision.action == "scale_up"
    assert decision.new_vms == 4
    assert decision.previous_vms == 3
    assert "above the scale-up threshold" in decision.reason


def test_scenario_3_normal_cpu_between_30_and_80_maintains():
    """Scenario 3: Predicted CPU utilization is between 30% and 80%. Expected: Maintain resources."""
    scaler = _scaler(4)
    decision = scaler.decide(55.0)
    assert decision.action == "maintain"
    assert decision.new_vms == 4
    assert "maintain band" in decision.reason


def test_scenario_4_cpu_below_30_scales_down_if_vms_greater_than_one():
    """Scenario 4: Predicted CPU utilization is below 30%. Expected: Scale down if VM count > 1."""
    scaler = _scaler(5)
    decision = scaler.decide(25.0)
    assert decision.action == "scale_down"
    assert decision.new_vms == 4
    assert "below the scale-down threshold" in decision.reason


def test_scenario_5_max_capacity_prevents_additional_vm():
    """Scenario 5: VM count reaches maximum capacity. Expected: Prevent additional VM creation."""
    scaler = _scaler(10)  # maximum capacity
    decision = scaler.decide(95.0)
    assert decision.action == "maintain"
    assert decision.new_vms == 10
    assert "already at the maximum" in decision.reason


def test_scenario_6_min_capacity_prevents_further_removal():
    """Scenario 6: VM count reaches minimum capacity. Expected: Prevent further VM removal."""
    scaler = _scaler(1)  # minimum capacity
    decision = scaler.decide(10.0)
    assert decision.action == "maintain"
    assert decision.new_vms == 1
    assert "already at the minimum" in decision.reason

