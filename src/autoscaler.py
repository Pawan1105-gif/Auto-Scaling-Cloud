"""Rule-based predictive auto-scaler.

Decisions use predicted CPU utilization and fixed thresholds. This module
does not implement reinforcement learning and does not create real virtual
machines. VM counts are simulated integers inside a configured range.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ScalingDecision:
    """One recorded scaling decision."""

    timestamp: str
    predicted_cpu: float
    previous_vms: int
    new_vms: int
    action: str
    reason: str
    mode: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        if isinstance(payload["predicted_cpu"], float) and not np.isfinite(payload["predicted_cpu"]):
            payload["predicted_cpu"] = None
        return payload


class AutoScaler:
    """Add, remove, or keep one simulated VM based on predicted CPU."""

    def __init__(
        self,
        min_vms: int = 1,
        max_vms: int = 10,
        scale_up_threshold: float = 80.0,
        scale_down_threshold: float = 30.0,
        cooldown_steps: int = 1,
        initial_vms: int = 3,
    ) -> None:
        self.min_vms = int(min_vms)
        self.max_vms = int(max_vms)
        if self.min_vms < 1:
            raise ValueError("Minimum VM count must be at least 1.")
        if self.max_vms < self.min_vms:
            raise ValueError("Maximum VM count must be greater than or equal to the minimum.")
        self.scale_up_threshold = float(scale_up_threshold)
        self.scale_down_threshold = float(scale_down_threshold)
        self._validate_thresholds()
        if cooldown_steps < 0:
            raise ValueError("Cooldown steps cannot be negative.")
        self.cooldown_steps = int(cooldown_steps)
        if not self.min_vms <= int(initial_vms) <= self.max_vms:
            raise ValueError(
                f"Initial VM count must be between {self.min_vms} and {self.max_vms}."
            )
        self.current_vms = int(initial_vms)
        self._steps_since_scale = self.cooldown_steps
        self.history: list[ScalingDecision] = []

    @property
    def steps_since_scale(self) -> int:
        return self._steps_since_scale

    def configure(
        self,
        min_vms: int | None = None,
        max_vms: int | None = None,
        scale_up_threshold: float | None = None,
        scale_down_threshold: float | None = None,
        cooldown_steps: int | None = None,
    ) -> None:
        """Update limits without clearing history. Invalid settings leave the old ones in place."""
        new_min = self.min_vms if min_vms is None else int(min_vms)
        new_max = self.max_vms if max_vms is None else int(max_vms)
        new_up = self.scale_up_threshold if scale_up_threshold is None else float(scale_up_threshold)
        new_down = self.scale_down_threshold if scale_down_threshold is None else float(scale_down_threshold)
        new_cooldown = self.cooldown_steps if cooldown_steps is None else int(cooldown_steps)
        if new_min < 1 or new_max < new_min:
            raise ValueError("VM limits must satisfy 1 <= minimum <= maximum.")
        if not np.isfinite(new_up) or not np.isfinite(new_down) or new_down >= new_up:
            raise ValueError("Scale-down threshold must be lower than the scale-up threshold.")
        if new_cooldown < 0:
            raise ValueError("Cooldown steps cannot be negative.")
        self.min_vms = new_min
        self.max_vms = new_max
        self.scale_up_threshold = new_up
        self.scale_down_threshold = new_down
        self.cooldown_steps = new_cooldown
        self.current_vms = int(np.clip(self.current_vms, self.min_vms, self.max_vms))

    def plan_action(
        self,
        predicted_cpu: float,
        current_vms: int,
        steps_since_scale: int,
    ) -> tuple[str, int, str]:
        """Return the action, next VM count, and reason without changing state."""
        cpu = self._check_cpu(predicted_cpu)
        vms = int(current_vms)
        if vms < self.min_vms or vms > self.max_vms:
            raise ValueError(
                f"Current VM count {vms} is outside {self.min_vms}–{self.max_vms}."
            )

        if cpu > self.scale_up_threshold:
            desired = "scale_up"
            reason = (
                f"Predicted CPU {cpu:.2f}% is above the scale-up threshold "
                f"({self.scale_up_threshold:.1f}%)."
            )
        elif cpu < self.scale_down_threshold:
            desired = "scale_down"
            reason = (
                f"Predicted CPU {cpu:.2f}% is below the scale-down threshold "
                f"({self.scale_down_threshold:.1f}%)."
            )
        else:
            return (
                "maintain",
                vms,
                (
                    f"Predicted CPU {cpu:.2f}% is inside the maintain band "
                    f"({self.scale_down_threshold:.1f}% to {self.scale_up_threshold:.1f}%)."
                ),
            )

        if desired == "scale_up" and vms >= self.max_vms:
            return (
                "maintain",
                vms,
                reason + f" Scale-up blocked: already at the maximum of {self.max_vms} VMs.",
            )
        if desired == "scale_down" and vms <= self.min_vms:
            return (
                "maintain",
                vms,
                reason + f" Scale-down blocked: already at the minimum of {self.min_vms} VM.",
            )
        if steps_since_scale < self.cooldown_steps:
            return (
                "maintain",
                vms,
                reason
                + (
                    " Cooldown is active "
                    f"({steps_since_scale} of {self.cooldown_steps} quiet decisions elapsed), "
                    "so the VM count is unchanged."
                ),
            )
        if desired == "scale_up":
            return "scale_up", vms + 1, reason + " Adding one virtual machine."
        return "scale_down", vms - 1, reason + " Removing one virtual machine."

    def decide(
        self,
        predicted_cpu: float,
        current_vms: int | None = None,
        timestamp: str = "",
        mode: str = "automatic",
    ) -> ScalingDecision:
        """Apply one automatic decision and store it in the history."""
        if current_vms is not None:
            self.current_vms = int(current_vms)
        action, new_vms, reason = self.plan_action(
            predicted_cpu,
            self.current_vms,
            self._steps_since_scale,
        )
        decision = self._commit(
            predicted_cpu=float(predicted_cpu),
            action=action,
            new_vms=new_vms,
            reason=reason,
            timestamp=timestamp,
            mode=mode,
        )
        return decision

    def manual(
        self,
        action: str,
        timestamp: str = "",
        predicted_cpu: float | None = None,
    ) -> ScalingDecision:
        """Apply an operator action. Manual changes ignore the cooldown."""
        if action not in {"scale_up", "scale_down", "maintain"}:
            raise ValueError("Manual action must be scale_up, scale_down, or maintain.")
        previous = self.current_vms
        cpu_value = float("nan") if predicted_cpu is None else float(predicted_cpu)
        if action == "maintain":
            chosen = "maintain"
            new_vms = previous
            reason = "Operator chose to maintain the current number of virtual machines."
        elif action == "scale_up":
            if previous >= self.max_vms:
                chosen = "maintain"
                new_vms = previous
                reason = (
                    f"Manual scale-up blocked: already at the maximum of {self.max_vms} VMs."
                )
            else:
                chosen = "scale_up"
                new_vms = previous + 1
                reason = "Operator requested a scale-up. Adding one virtual machine."
        else:
            if previous <= self.min_vms:
                chosen = "maintain"
                new_vms = previous
                reason = (
                    f"Manual scale-down blocked: already at the minimum of {self.min_vms} VM."
                )
            else:
                chosen = "scale_down"
                new_vms = previous - 1
                reason = "Operator requested a scale-down. Removing one virtual machine."
        return self._commit(cpu_value, chosen, new_vms, reason, timestamp, mode="manual")

    def _commit(
        self,
        predicted_cpu: float,
        action: str,
        new_vms: int,
        reason: str,
        timestamp: str,
        mode: str,
    ) -> ScalingDecision:
        previous = self.current_vms
        if action in {"scale_up", "scale_down"}:
            self._steps_since_scale = 0
        else:
            self._steps_since_scale += 1
        self.current_vms = new_vms
        decision = ScalingDecision(
            timestamp=timestamp,
            predicted_cpu=float(predicted_cpu) if np.isfinite(predicted_cpu) else float("nan"),
            previous_vms=previous,
            new_vms=new_vms,
            action=action,
            reason=reason,
            mode=mode,
        )
        self.history.append(decision)
        return decision

    def _validate_thresholds(self) -> None:
        if not np.isfinite(self.scale_up_threshold) or not np.isfinite(self.scale_down_threshold):
            raise ValueError("CPU thresholds must be finite numbers.")
        if self.scale_down_threshold >= self.scale_up_threshold:
            raise ValueError("Scale-down threshold must be lower than the scale-up threshold.")

    @staticmethod
    def _check_cpu(predicted_cpu: float) -> float:
        cpu = float(predicted_cpu)
        if not np.isfinite(cpu):
            raise ValueError("Predicted CPU utilization must be a finite number.")
        return cpu
