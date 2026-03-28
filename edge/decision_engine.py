"""Decision Engine — multi-objective cost-based offloading"""


class DecisionEngine:

    def __init__(self, alpha: float = 0.5,
                       beta:  float = 0.3,
                       gamma: float = 0.2):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.alpha = alpha   # delay weight
        self.beta  = beta    # energy weight
        self.gamma = gamma   # congestion weight

    # ── physical models ───────────────────────────────────────────────
    def _edge_energy(self, execution_time: float) -> float:
        """Edge: local CPU power ~0.5W (Raspberry Pi class)"""
        return 0.5 * execution_time

    def _cloud_energy(self, task) -> float:
        """Cloud: WiFi uplink TX power ~1W over transmission time"""
        P_tx, bw = 1.0, 10.0
        tx_time  = (task.size * 8) / (bw * 1000)
        return P_tx * tx_time

    def _latency_penalty(self, delay: float, latency_req: float) -> float:
        """Proportional penalty for missing deadline."""
        if latency_req <= 0:
            return 1.0
        excess = max(0.0, delay - latency_req)
        return 1.0 + (excess / latency_req)

    def _congestion_cost(self, location: str,
                         edge_q: float, cloud_q: float) -> float:
        """
        Asymmetric congestion: edge queue affects edge cost,
        cloud upload congestion affects cloud cost.
        Normalised to [0, 1].
        """
        MAX_Q = 20.0
        raw = edge_q if location == "edge" else cloud_q
        return min(1.0, raw / MAX_Q)

    def compute_cost(self, delay, energy, congestion, latency_req):
        """
        Weighted cost with proportional latency penalty.
        REF values normalise delay and energy to comparable scales:
          - REF_DELAY  = 1.0s  (typical task completion target)
          - REF_ENERGY = 0.5J  (mid-range between edge ~0.25J and cloud ~0.8J)
        """
        REF_DELAY  = 1.0
        REF_ENERGY = 0.5
        penalty = self._latency_penalty(delay, latency_req)
        return penalty * (
            self.alpha * (delay  / REF_DELAY)  +
            self.beta  * (energy / REF_ENERGY) +
            self.gamma * congestion
        )

    # ── original decide (kept for compatibility) ──────────────────────
    def decide(self, task, edge_estimate, cloud_estimate,
               edge_queue, cloud_queue=0.0) -> str:
        decision, _ = self.decide_with_reason(
            task, edge_estimate, cloud_estimate, edge_queue, cloud_queue)
        return decision

    # ── decide + human-readable reason ───────────────────────────────
    def decide_with_reason(self, task, edge_estimate, cloud_estimate,
                           edge_queue, cloud_queue=0.0):
        """Returns (decision: str, reason: str)"""
        ed   = edge_estimate["delay"]
        cd   = cloud_estimate["delay"]
        ee   = self._edge_energy(ed)
        ce   = self._cloud_energy(task)
        ec   = self._congestion_cost("edge",  edge_queue, cloud_queue)
        cc   = self._congestion_cost("cloud", edge_queue, cloud_queue)
        lreq = task.latency_req

        edge_cost  = self.compute_cost(ed, ee, ec, lreq)
        cloud_cost = self.compute_cost(cd, ce, cc, lreq)

        decision = "edge" if edge_cost <= cloud_cost else "cloud"

        # human-readable reason
        if decision == "edge":
            margin = cloud_cost - edge_cost
            if ed < cd:
                reason = (f"edge faster ({ed:.4f}s vs {cd:.4f}s), "
                          f"cost margin={margin:.4f}")
            elif ec < cc:
                reason = (f"cloud more congested, edge cheaper "
                          f"by {margin:.4f}")
            else:
                reason = (f"edge cost={edge_cost:.4f} < "
                          f"cloud cost={cloud_cost:.4f} "
                          f"(margin={margin:.4f})")
        else:
            margin = edge_cost - cloud_cost
            ep = self._latency_penalty(ed, lreq)
            if ep > 1.2:
                reason = (f"edge misses deadline "
                          f"({ed:.4f}s > req={lreq:.2f}s, "
                          f"penalty={ep:.2f}x)")
            elif ec > 0.6:
                reason = (f"edge congestion high "
                          f"(queue={edge_queue:.1f}, "
                          f"cong={ec:.2f}), cloud cheaper by {margin:.4f}")
            elif cd < ed:
                reason = (f"cloud faster ({cd:.4f}s vs {ed:.4f}s), "
                          f"cost margin={margin:.4f}")
            else:
                reason = (f"cloud cost={cloud_cost:.4f} < "
                          f"edge cost={edge_cost:.4f} "
                          f"(margin={margin:.4f})")

        return decision, reason