"""Q-Learning Reinforcement Learning Controller for Cloud Auto-Scaling.

Defines the Markov Decision Process (MDP) environment, state discretization,
action space, multi-objective reward function, and tabular Q-learning agent.
Includes scaling cooldown, VM capacity clamps, and provisioning delay.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    DATA_NOTE,
    DATA_ORIGIN,
    DATASET_PATH,
    FEATURE_COLUMNS,
    Q_TABLE_PATH,
    TARGET_COLUMN,
)
from src.data_preprocessing import chronological_split, load_dataset

# Action definitions
ACTION_MAINTAIN = 0
ACTION_SCALE_UP = 1
ACTION_SCALE_DOWN = 2
ACTION_NAMES = {
    ACTION_MAINTAIN: "MAINTAIN",
    ACTION_SCALE_UP: "SCALE_UP",
    ACTION_SCALE_DOWN: "SCALE_DOWN",
}

# Reward weights (Configurable)
DEFAULT_WEIGHTS = {
    "w_sla": 10.0,       # Penalty for SLA violation
    "w_cost": 2.5,       # Penalty for resource over-provisioning (fraction of max VMs)
    "w_osc": 0.8,        # Penalty for scaling action to avoid thrashing
    "w_dev": 3.0,        # Penalty for deviation from target CPU
    "w_stable": 2.0,     # Bonus for keeping CPU within optimal band [45%, 75%]
    "target_cpu": 60.0,  # Target optimal CPU percentage
    "cpu_low": 40.0,     # Lower bound of optimal band
    "cpu_high": 75.0,    # Upper bound of optimal band
}


@dataclass
class EnvironmentConfig:
    min_vms: int = 1
    max_vms: int = 10
    cooldown_steps: int = 1
    provisioning_delay: int = 1  # Steps until a provisioned VM is fully active
    reward_weights: dict = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))


def discretize_metric(val: float, bins: list[float]) -> int:
    """Return the 0-indexed bin index for a continuous value."""
    return int(np.digitize(val, bins))


def extract_state_tuple(
    current_cpu: float,
    predicted_cpu: float,
    memory_usage: float,
    active_vms: int,
    workload_trend: float,
    sla_violation: int,
) -> tuple[int, int, int, int, int, int]:
    """Discretize continuous monitoring features into a compact discrete state tuple.

    Bins:
    - cpu: <35 (0), 35-65 (1), 65-85 (2), >=85 (3)
    - pred_cpu: <35 (0), 35-65 (1), 65-85 (2), >=85 (3)
    - memory: <45 (0), 45-75 (1), >=75 (2)
    - active_vms: 1 to 10 (clamped index 0 to 9)
    - trend: < -10 (0), -10 to +10 (1), > +10 (2)
    - sla_violation: 0 or 1
    """
    cpu_b = discretize_metric(current_cpu, [35.0, 65.0, 85.0])
    pred_b = discretize_metric(predicted_cpu, [35.0, 65.0, 85.0])
    mem_b = discretize_metric(memory_usage, [45.0, 75.0])
    vm_b = int(np.clip(active_vms, 1, 10)) - 1
    trend_b = discretize_metric(workload_trend, [-10.0, 10.0])
    sla_b = int(1 if sla_violation > 0 else 0)
    return (cpu_b, pred_b, mem_b, vm_b, trend_b, sla_b)


class CloudAutoScalingEnv:
    """Simulated Cloud Infrastructure Environment for Reinforcement Learning."""

    def __init__(
        self,
        workload_df: pd.DataFrame,
        config: EnvironmentConfig | None = None,
        forecaster=None,
    ) -> None:
        self.workload_df = workload_df.reset_index(drop=True)
        self.config = config or EnvironmentConfig()
        self.forecaster = forecaster
        self.n_steps = len(self.workload_df)
        self.step_idx = 0

        self.current_vms = 3
        self.provisioning_queue: list[int] = []  # VMs pending provisioning
        self.steps_since_scale = self.config.cooldown_steps
        self.prev_request_rate = float(self.workload_df.loc[0, "request_rate"])

    def reset(self, initial_vms: int = 3) -> tuple[tuple, dict]:
        self.step_idx = 0
        self.current_vms = int(np.clip(initial_vms, self.config.min_vms, self.config.max_vms))
        self.provisioning_queue = []
        self.steps_since_scale = self.config.cooldown_steps
        self.prev_request_rate = float(self.workload_df.loc[0, "request_rate"])
        return self._get_state_and_info()

    def _calculate_cpu(self, request_rate: float, memory_usage: float, vms: int) -> float:
        """Simulate realistic CPU utilization under current allocated VMs."""
        vms = max(1, vms)
        pressure = request_rate / (9.0 * vms)
        simulated_cpu = (
            9.0
            + 50.0 * np.tanh(pressure / 6.5)
            + 0.17 * memory_usage
        )
        return float(np.clip(simulated_cpu, 2.0, 99.0))

    def _get_state_and_info(self) -> tuple[tuple, dict]:
        row = self.workload_df.iloc[self.step_idx]
        req_rate = float(row["request_rate"])
        mem = float(row["memory_usage"])
        trend = req_rate - self.prev_request_rate

        curr_cpu = self._calculate_cpu(req_rate, mem, self.current_vms)
        load_per_vm = req_rate / self.current_vms
        resp_time = float(np.clip(35.0 + 0.62 * load_per_vm + 0.35 * mem, 15.0, 2000.0))
        sla = 1 if (resp_time >= 500.0 or curr_cpu >= 90.0) else 0

        # Forecast next CPU
        if self.forecaster is not None:
            pred_cpu = float(self.forecaster(req_rate, mem, self.current_vms, resp_time, sla))
        else:
            # Baseline forecast (trend extrapolation)
            pred_cpu = float(np.clip(curr_cpu + 0.15 * trend, 5.0, 98.0))

        state = extract_state_tuple(curr_cpu, pred_cpu, mem, self.current_vms, trend, sla)
        info = {
            "step": self.step_idx,
            "timestamp": str(row["timestamp"]),
            "cpu_utilization": curr_cpu,
            "predicted_cpu": pred_cpu,
            "memory_usage": mem,
            "request_rate": req_rate,
            "response_time": resp_time,
            "active_vms": self.current_vms,
            "sla_violation": sla,
            "steps_since_scale": self.steps_since_scale,
        }
        return state, info

    def step(self, action: int) -> tuple[tuple, float, bool, dict]:
        """Apply action, advance simulation by 1 timestep, compute reward."""
        # 1. Update pending VMs from provisioning delay
        if self.provisioning_queue:
            ready_vms = self.provisioning_queue.pop(0)
            self.current_vms = ready_vms

        # 2. Check scaling validity (bounds & cooldown)
        executed_action = ACTION_MAINTAIN
        reason = "Maintained"

        if action == ACTION_SCALE_UP:
            if self.current_vms >= self.config.max_vms:
                reason = "Scale up blocked: at maximum capacity"
            elif self.steps_since_scale < self.config.cooldown_steps:
                reason = "Scale up blocked: cooldown active"
            else:
                executed_action = ACTION_SCALE_UP
                target_vms = self.current_vms + 1
                if self.config.provisioning_delay > 0:
                    # Queue the VM for delayed activation
                    self.provisioning_queue.append(target_vms)
                    reason = f"Scale up scheduled (delay={self.config.provisioning_delay})"
                else:
                    self.current_vms = target_vms
                    reason = "Scaled up by 1 VM"
                self.steps_since_scale = 0

        elif action == ACTION_SCALE_DOWN:
            if self.current_vms <= self.config.min_vms:
                reason = "Scale down blocked: at minimum capacity"
            elif self.steps_since_scale < self.config.cooldown_steps:
                reason = "Scale down blocked: cooldown active"
            else:
                executed_action = ACTION_SCALE_DOWN
                self.current_vms -= 1
                reason = "Scaled down by 1 VM"
                self.steps_since_scale = 0

        if executed_action == ACTION_MAINTAIN:
            self.steps_since_scale += 1

        # 3. Advance to next timestep
        self.prev_request_rate = float(self.workload_df.loc[self.step_idx, "request_rate"])
        self.step_idx += 1
        done = self.step_idx >= self.n_steps - 1

        next_state, info = self._get_state_and_info()
        info["action_taken"] = ACTION_NAMES[executed_action]
        info["action_reason"] = reason

        # 4. Compute reward
        reward = self._compute_reward(info, executed_action)
        info["reward"] = reward

        return next_state, reward, done, info

    def _compute_reward(self, info: dict, action: int) -> float:
        """Calculate multi-objective reward:

        R = - w_sla * SLA
            - w_cost * (VMs / MaxVMs)
            - w_osc * (action != MAINTAIN)
            - w_dev * |CPU - TargetCPU| / 100
            + w_stable * (CPU_low <= CPU <= CPU_high)
        """
        w = self.config.reward_weights
        cpu = info["cpu_utilization"]
        vms = info["active_vms"]
        sla = info["sla_violation"]

        sla_penalty = w["w_sla"] * float(sla)
        cost_penalty = w["w_cost"] * (float(vms) / float(self.config.max_vms))
        osc_penalty = w["w_osc"] * float(1.0 if action != ACTION_MAINTAIN else 0.0)
        dev_penalty = w["w_dev"] * (abs(cpu - w["target_cpu"]) / 100.0)
        stable_bonus = w["w_stable"] * float(1.0 if (w["cpu_low"] <= cpu <= w["cpu_high"]) else 0.0)

        total_reward = -sla_penalty - cost_penalty - osc_penalty - dev_penalty + stable_bonus
        return float(total_reward)


class QLearningAgent:
    """Tabular Q-learning agent with epsilon-greedy exploration."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple, np.ndarray] = {}

    def get_q_values(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            # Initialize with small optimistic values
            self.q_table[state] = np.zeros(3, dtype=float)
        return self.q_table[state]

    def select_action(self, state: tuple, explore: bool = True) -> int:
        if explore and (self.rng.random() < self.epsilon):
            return int(self.rng.integers(0, 3))
        q_vals = self.get_q_values(state)
        # Random tie breaking
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(self.rng.choice(best_actions))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> float:
        current_q = self.get_q_values(state)[action]
        next_max_q = np.max(self.get_q_values(next_state)) if not done else 0.0
        td_target = reward + self.gamma * next_max_q
        td_error = td_target - current_q
        self.q_table[state][action] = current_q + self.lr * td_error
        return float(td_error)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path = Q_TABLE_PATH, metadata: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_table": self.q_table,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "metadata": metadata or {},
        }
        joblib.dump(payload, path, compress=3)
        return path

    @classmethod
    def load(cls, path: Path = Q_TABLE_PATH) -> "QLearningAgent":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Q-table not found at {path}. Train it first.")
        payload = joblib.load(path)
        agent = cls(
            learning_rate=payload["lr"],
            discount_factor=payload["gamma"],
            epsilon=payload.get("epsilon_min", 0.05),
        )
        agent.q_table = payload["q_table"]
        return agent


def train_q_learning(
    workload_df: pd.DataFrame,
    episodes: int = 10,
    forecaster=None,
    config: EnvironmentConfig | None = None,
    learning_rate: float = 0.1,
    discount_factor: float = 0.95,
    seed: int = 42,
) -> tuple[QLearningAgent, list[float], list[float]]:
    """Train the Q-learning agent across multiple simulated episodes."""
    config = config or EnvironmentConfig()
    env = CloudAutoScalingEnv(workload_df, config=config, forecaster=forecaster)
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        seed=seed,
    )

    episode_rewards = []
    sla_percentages = []

    for ep in range(1, episodes + 1):
        state, info = env.reset(initial_vms=3)
        total_reward = 0.0
        sla_count = 0
        steps = 0

        while True:
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if info["sla_violation"] > 0:
                sla_count += 1
            steps += 1
            if done:
                break

        agent.decay_epsilon()
        sla_pct = (sla_count / max(1, steps)) * 100.0
        episode_rewards.append(total_reward)
        sla_percentages.append(sla_pct)

    return agent, episode_rewards, sla_percentages


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Q-Learning Cloud Auto-Scaling Agent.")
    parser.add_argument("--data", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=Q_TABLE_PATH)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args(argv)

    print("Loading dataset...")
    df = load_dataset(args.data)
    train_df, _, _, _ = chronological_split(df)

    print(f"Training Q-Learning Agent on {len(train_df)} chronological records ({args.episodes} episodes)...")
    agent, rewards, slas = train_q_learning(train_df, episodes=args.episodes)

    metadata = {
        "episodes": args.episodes,
        "final_reward": rewards[-1],
        "final_sla_pct": slas[-1],
        "q_table_size": len(agent.q_table),
    }
    agent.save(args.output, metadata=metadata)
    print("\n--- Q-LEARNING TRAINING COMPLETE ---")
    print(f"States explored: {len(agent.q_table)}")
    print(f"Episode 1 Reward: {rewards[0]:.1f} | Final Episode Reward: {rewards[-1]:.1f}")
    print(f"Episode 1 SLA: {slas[0]:.2f}% | Final Episode SLA: {slas[-1]:.2f}%")
    print(f"Saved Q-Table to: {args.output}")


if __name__ == "__main__":
    main()
