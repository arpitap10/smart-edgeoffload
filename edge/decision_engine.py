class DecisionEngine:
    def __init__(self, alpha: float = 0.6,
                       beta:  float = 0.3,
                       gamma: float = 0.1):

        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            "Weights must sum to 1.0"

        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def _edge_energy(self, execution_time: float) -> float:
        P_cpu = 2.0
        return P_cpu * execution_time

    def _cloud_energy(self, task) -> float:
        P_tx = 0.5
        bandwidth = 10.0
        tx_time = (task.size * 8) / (bandwidth * 1000)
        return P_tx * tx_time

    def _latency_penalty(self, delay: float, latency_req: float) -> float:
        if latency_req <= 0:
            return 1.0
        excess = max(0.0, delay - latency_req)
        return 1.0 + (excess / latency_req)

    def _congestion_cost(self, location: str,
                         edge_queue: float,
                         cloud_queue: float) -> float:

        MAX_QUEUE = 20.0

        if location == "edge":
            return min(1.0, edge_queue / MAX_QUEUE)
        else:
            return min(1.0, cloud_queue / MAX_QUEUE)

    def compute_cost(self, delay, energy, congestion, latency_req):

        REF_DELAY = 2.0
        REF_ENERGY = 1.0

        delay_norm = delay / REF_DELAY
        energy_norm = energy / REF_ENERGY
        penalty = self._latency_penalty(delay, latency_req)

        return penalty * (
            self.alpha * delay_norm +
            self.beta  * energy_norm +
            self.gamma * congestion
        )

    def decide(self, task, edge_estimate, cloud_estimate,
               edge_queue, cloud_queue=0.0):

        edge_delay = edge_estimate["delay"]
        cloud_delay = cloud_estimate["delay"]

        edge_energy = self._edge_energy(edge_delay)
        cloud_energy = self._cloud_energy(task)

        edge_cong = self._congestion_cost("edge", edge_queue, cloud_queue)
        cloud_cong = self._congestion_cost("cloud", edge_queue, cloud_queue)

        latency_req = task.latency_req

        edge_cost = self.compute_cost(edge_delay, edge_energy, edge_cong, latency_req)
        cloud_cost = self.compute_cost(cloud_delay, cloud_energy, cloud_cong, latency_req)

        return "edge" if edge_cost <= cloud_cost else "cloud"