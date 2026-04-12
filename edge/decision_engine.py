"""Prediction-aware multi-objective offloading decision engine."""

from __future__ import annotations

import math


class DecisionEngine:
    def __init__(
        self,
        alpha: float = 0.55,
        beta: float = 0.20,
        gamma: float = 0.25,
        offload_margin: float = 0.08,
        uncertainty_weight: float = 0.35,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.offload_margin = offload_margin
        self.uncertainty_weight = uncertainty_weight

    def _queue_pressure(self, backlog: float, scale: float = 1.0) -> float:
        return 1.0 - math.exp(-max(0.0, backlog) / max(scale, 1e-6))

    def _deadline_penalty(self, delay: float, latency_req: float) -> float:
        if latency_req <= 0:
            return 1.0
        excess_ratio = max(0.0, delay - latency_req) / latency_req
        return 1.0 + (1.75 * excess_ratio)

    def _energy_reference(self, task) -> float:
        return max(0.2, 0.18 * task.size + 0.015 * task.compute)

    def _uncertainty_ratio(self, task, estimate: dict) -> float:
        return estimate.get("uncertainty", 0.0) / max(task.latency_req, 1e-3)

    def _risk_penalty(self, task, estimate: dict) -> float:
        if estimate.get("location") != "cloud":
            return 0.0

        uncertainty_ratio = self._uncertainty_ratio(task, estimate)
        deadline_tightness = 1.0 / max(task.latency_req, 0.2)
        return self.uncertainty_weight * (
            uncertainty_ratio + estimate.get("risk", 0.0) * deadline_tightness
        )

    def compute_cost(self, task, estimate: dict, effective_backlog: float) -> tuple[float, dict]:
        effective_delay = estimate["delay"] + estimate.get("uncertainty", 0.0)
        delay_ratio = effective_delay / max(task.latency_req, 1e-3)
        energy_ratio = estimate["energy"] / self._energy_reference(task)
        congestion = self._queue_pressure(effective_backlog, scale=max(task.latency_req, 0.25))
        penalty = self._deadline_penalty(effective_delay, task.latency_req)
        risk_penalty = self._risk_penalty(task, estimate)

        raw_cost = (
            self.alpha * delay_ratio
            + self.beta * energy_ratio
            + self.gamma * congestion
            + risk_penalty
        )
        return penalty * raw_cost, {
            "delay_ratio": delay_ratio,
            "energy_ratio": energy_ratio,
            "congestion": congestion,
            "penalty": penalty,
            "uncertainty": estimate.get("uncertainty", 0.0),
            "risk_penalty": risk_penalty,
        }

    def decide(
        self,
        task,
        edge_estimate,
        cloud_estimate,
        current_edge_backlog: float,
        current_cloud_backlog: float,
        predicted_edge_backlog: float | None = None,
        predicted_cloud_backlog: float | None = None,
    ) -> str:
        decision, _, _, _ = self.decide_with_reason(
            task,
            edge_estimate,
            cloud_estimate,
            current_edge_backlog=current_edge_backlog,
            current_cloud_backlog=current_cloud_backlog,
            predicted_edge_backlog=predicted_edge_backlog,
            predicted_cloud_backlog=predicted_cloud_backlog,
        )
        return decision

    def decide_with_reason(
        self,
        task,
        edge_estimate,
        cloud_estimate,
        current_edge_backlog: float,
        current_cloud_backlog: float = 0.0,
        predicted_edge_backlog: float | None = None,
        predicted_cloud_backlog: float | None = None,
    ):
        edge_backlog = current_edge_backlog
        if predicted_edge_backlog is not None:
            edge_backlog = max(
                current_edge_backlog,
                0.85 * predicted_edge_backlog + 0.15 * current_edge_backlog,
            )

        cloud_backlog = current_cloud_backlog
        if predicted_cloud_backlog is not None:
            cloud_backlog = max(
                current_cloud_backlog,
                0.75 * predicted_cloud_backlog + 0.25 * current_cloud_backlog,
            )

        edge_cost, edge_terms = self.compute_cost(task, edge_estimate, edge_backlog)
        cloud_cost, cloud_terms = self.compute_cost(task, cloud_estimate, cloud_backlog)

        if cloud_cost < (1.0 - self.offload_margin) * edge_cost:
            decision = "cloud"
        else:
            decision = "edge"

        chosen_cost = edge_cost if decision == "edge" else cloud_cost
        other_cost = cloud_cost if decision == "edge" else edge_cost
        chosen_estimate = edge_estimate if decision == "edge" else cloud_estimate
        chosen_terms = edge_terms if decision == "edge" else cloud_terms
        margin = other_cost - chosen_cost

        reason = (
            f"{decision.upper()} | margin={margin:.3f} "
            f"delay={chosen_estimate['delay']:.3f}s "
            f"wait={chosen_estimate['wait_time']:.3f}s "
            f"energy={chosen_estimate['energy']:.3f}J "
            f"cong={chosen_terms['congestion']:.2f} "
            f"unc={chosen_terms['uncertainty']:.3f}s "
            f"penalty={chosen_terms['penalty']:.2f}"
        )

        diagnostics = {
            "edge_cost": edge_cost,
            "cloud_cost": cloud_cost,
            "effective_edge_backlog": edge_backlog,
            "effective_cloud_backlog": cloud_backlog,
            "edge_terms": edge_terms,
            "cloud_terms": cloud_terms,
        }
        return decision, reason, diagnostics, {
            "edge": edge_estimate,
            "cloud": cloud_estimate,
        }
