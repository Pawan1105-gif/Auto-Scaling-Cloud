"""Hybrid Predictive Auto-Scaling Controller combining ML Forecasting and Q-Learning.

Integrates real-time workload monitoring, LSTM/Random Forest CPU forecasting,
and reinforcement learning decision-making with guardrails (cooldown, min/max VM bounds,
and provisioning delay).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.autoscaler import AutoScaler, ScalingDecision
from src.q_learning import (
    ACTION_MAINTAIN,
    ACTION_NAMES,
    ACTION_SCALE_DOWN,
    ACTION_SCALE_UP,
    EnvironmentConfig,
    QLearningAgent,
    extract_state_tuple,
)


@dataclass
class HybridDecision:
    timestamp: str
    observed_cpu: float
    predicted_cpu: float
    current_vms: int
    new_vms: int
    action: str
    reason: str
    strategy: str
    q_values: dict[str, float] | None = None
    sla_violation: int = 0


class HybridAutoScaler:
    """Unified auto-scaler supporting Rule-based, Forecast-based, RL-based, and Hybrid modes."""

    def __init__(
        self,
        q_agent: QLearningAgent | None = None,
        forecaster_fn: Callable | None = None,
        strategy: str = "Hybrid LSTM + Q-Learning",
        min_vms: int = 1,
        max_vms: int = 10,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_steps: int = 1,
        provisioning_delay: int = 1,
        initial_vms: int = 3,
    ) -> None:
        self.q_agent = q_agent
        self.forecaster_fn = forecaster_fn
        self.strategy = strategy
        self.min_vms = int(min_vms)
        self.max_vms = int(max_vms)
        self.scale_up_threshold = float(scale_up_threshold)
        self.scale_down_threshold = float(scale_down_threshold)
        self.cooldown_steps = int(cooldown_steps)
        self.provisioning_delay = int(provisioning_delay)

        self.current_vms = int(initial_vms)
        self.steps_since_scale = self.cooldown_steps
        self.provisioning_queue: list[int] = []
        self.history: list[HybridDecision] = []

    def reset(self, initial_vms: int = 3) -> None:
        self.current_vms = int(np.clip(initial_vms, self.min_vms, self.max_vms))
        self.steps_since_scale = self.cooldown_steps
        self.provisioning_queue = []
        self.history = []

    def decide(
        self,
        current_cpu: float,
        memory_usage: float,
        request_rate: float,
        response_time: float,
        sla_violation: int,
        workload_trend: float = 0.0,
        predicted_cpu: float | None = None,
        timestamp: str = "",
    ) -> HybridDecision:
        """Make a scaling decision based on the configured strategy."""
        # Process any pending provisioning completions
        if self.provisioning_queue:
            ready_vms = self.provisioning_queue.pop(0)
            self.current_vms = ready_vms

        # Compute prediction if forecaster provided and not explicitly passed
        pred_cpu = predicted_cpu
        if pred_cpu is None:
            if self.forecaster_fn is not None:
                try:
                    pred_cpu = float(
                        self.forecaster_fn(
                            request_rate, memory_usage, self.current_vms, response_time, sla_violation
                        )
                    )
                except Exception:
                    pred_cpu = float(current_cpu)
            else:
                pred_cpu = float(current_cpu)

        action_name = "MAINTAIN"
        new_vms = self.current_vms
        reason = ""
        q_vals_dict = None

        # Strategy A: Traditional Threshold-Based (Reactive on current CPU)
        if "Traditional" in self.strategy or "Threshold" in self.strategy:
            if current_cpu > self.scale_up_threshold:
                desired = "scale_up"
                base_reason = f"Current CPU {current_cpu:.1f}% > threshold {self.scale_up_threshold:.1f}%"
            elif current_cpu < self.scale_down_threshold:
                desired = "scale_down"
                base_reason = f"Current CPU {current_cpu:.1f}% < threshold {self.scale_down_threshold:.1f}%"
            else:
                desired = "maintain"
                base_reason = f"Current CPU {current_cpu:.1f}% within normal band"

            action_name, new_vms, reason = self._apply_guardrails(desired, base_reason)

        # Strategy B: Forecast-Only Auto-Scaling (Predictive thresholds)
        elif "Forecast-Only" in self.strategy:
            if pred_cpu > self.scale_up_threshold:
                desired = "scale_up"
                base_reason = f"Predicted CPU {pred_cpu:.1f}% > threshold {self.scale_up_threshold:.1f}%"
            elif pred_cpu < self.scale_down_threshold:
                desired = "scale_down"
                base_reason = f"Predicted CPU {pred_cpu:.1f}% < threshold {self.scale_down_threshold:.1f}%"
            else:
                desired = "maintain"
                base_reason = f"Predicted CPU {pred_cpu:.1f}% within normal band"

            action_name, new_vms, reason = self._apply_guardrails(desired, base_reason)

        # Strategy C: Reinforcement-Learning-Only (no future forecast)
        elif "Reinforcement-Learning-Only" in self.strategy or "RL-Only" in self.strategy:
            if self.q_agent is not None:
                # RL-only does not see future prediction, uses current_cpu for prediction slot
                state = extract_state_tuple(
                    current_cpu, current_cpu, memory_usage, self.current_vms, workload_trend, sla_violation
                )
                q_vals = self.q_agent.get_q_values(state)
                q_vals_dict = {ACTION_NAMES[i]: float(q_vals[i]) for i in range(3)}
                raw_act = int(np.argmax(q_vals))
                desired = "scale_up" if raw_act == ACTION_SCALE_UP else ("scale_down" if raw_act == ACTION_SCALE_DOWN else "maintain")
                base_reason = f"Q-Learning policy selected {ACTION_NAMES[raw_act]} (Q={q_vals[raw_act]:.2f})"
            else:
                desired = "maintain"
                base_reason = "Q-Learning agent not loaded"

            action_name, new_vms, reason = self._apply_guardrails(desired, base_reason)

        # Strategy D: Hybrid LSTM + Q-Learning
        else:
            if self.q_agent is not None:
                # Uses BOTH current state and actual forecast
                state = extract_state_tuple(
                    current_cpu, pred_cpu, memory_usage, self.current_vms, workload_trend, sla_violation
                )
                q_vals = self.q_agent.get_q_values(state)
                q_vals_dict = {ACTION_NAMES[i]: float(q_vals[i]) for i in range(3)}
                raw_act = int(np.argmax(q_vals))
                desired = "scale_up" if raw_act == ACTION_SCALE_UP else ("scale_down" if raw_act == ACTION_SCALE_DOWN else "maintain")
                base_reason = f"Hybrid Agent selected {ACTION_NAMES[raw_act]} based on predicted CPU {pred_cpu:.1f}% (Q={q_vals[raw_act]:.2f})"
            else:
                # Fallback to forecast threshold if agent not loaded
                if pred_cpu > self.scale_up_threshold:
                    desired = "scale_up"
                    base_reason = f"Predicted CPU {pred_cpu:.1f}% > {self.scale_up_threshold:.1f}%"
                elif pred_cpu < self.scale_down_threshold:
                    desired = "scale_down"
                    base_reason = f"Predicted CPU {pred_cpu:.1f}% < {self.scale_down_threshold:.1f}%"
                else:
                    desired = "maintain"
                    base_reason = f"Predicted CPU {pred_cpu:.1f}% in normal band"

            action_name, new_vms, reason = self._apply_guardrails(desired, base_reason)

        decision = HybridDecision(
            timestamp=timestamp,
            observed_cpu=float(current_cpu),
            predicted_cpu=float(pred_cpu),
            current_vms=self.current_vms,
            new_vms=new_vms,
            action=action_name,
            reason=reason,
            strategy=self.strategy,
            q_values=q_vals_dict,
            sla_violation=int(sla_violation),
        )
        self.history.append(decision)
        return decision

    def _apply_guardrails(self, desired_action: str, base_reason: str) -> tuple[str, int, str]:
        """Apply min/max VM clamping and cooldown period."""
        if desired_action == "scale_up":
            if self.current_vms >= self.max_vms:
                return "MAINTAIN", self.current_vms, f"{base_reason} | Blocked: at max capacity ({self.max_vms} VMs)"
            if self.steps_since_scale < self.cooldown_steps:
                return "MAINTAIN", self.current_vms, f"{base_reason} | Blocked: cooldown active ({self.steps_since_scale}/{self.cooldown_steps})"
            
            target_vms = self.current_vms + 1
            if self.provisioning_delay > 0:
                self.provisioning_queue.append(target_vms)
                self.steps_since_scale = 0
                return "SCALE_UP", target_vms, f"{base_reason} | Scale-up queued (delay={self.provisioning_delay} step)"
            else:
                self.current_vms = target_vms
                self.steps_since_scale = 0
                return "SCALE_UP", self.current_vms, f"{base_reason} | Scaled up to {self.current_vms} VMs"

        elif desired_action == "scale_down":
            if self.current_vms <= self.min_vms:
                return "MAINTAIN", self.current_vms, f"{base_reason} | Blocked: at min capacity ({self.min_vms} VM)"
            if self.steps_since_scale < self.cooldown_steps:
                return "MAINTAIN", self.current_vms, f"{base_reason} | Blocked: cooldown active ({self.steps_since_scale}/{self.cooldown_steps})"
            
            self.current_vms -= 1
            self.steps_since_scale = 0
            return "SCALE_DOWN", self.current_vms, f"{base_reason} | Scaled down to {self.current_vms} VMs"

        else:
            self.steps_since_scale += 1
            return "MAINTAIN", self.current_vms, base_reason
