from shared.data_models import ExecutionResult, IoTTask


class CloudExecutor:
    """
    Cloud executor with explicit uplink delay and finite queueing capacity.

    The cloud is faster than the edge, but not congestion-free and not free of
    network overhead. This makes the edge/cloud tradeoff defensible in a paper.
    """

    def __init__(
        self,
        compute_power: float = 900.0,
        tx_power_w: float = 1.35,
        base_rtt: float = 0.045,
    ):
        self.compute_power = compute_power
        self.tx_power_w = tx_power_w
        self.base_rtt = base_rtt

    def service_time(self, task: IoTTask) -> float:
        return task.compute_demand / self.compute_power

    def transmission_time(self, task: IoTTask, bandwidth_mbps: float) -> float:
        safe_bandwidth = max(0.5, bandwidth_mbps)
        return (task.size * 8.0) / safe_bandwidth

    def _cloud_energy(self, task: IoTTask, bandwidth_mbps: float, rtt: float) -> float:
        tx_time = self.transmission_time(task, bandwidth_mbps)
        return (self.tx_power_w * tx_time) + (0.12 * rtt)

    def estimate(
        self,
        task: IoTTask,
        queue_backlog: float = 0.0,
        bandwidth_mbps: float = 30.0,
        rtt: float | None = None,
    ) -> dict:
        service_time = self.service_time(task)
        wait_time = max(0.0, queue_backlog)
        effective_rtt = self.base_rtt if rtt is None else max(0.0, rtt)
        tx_time = self.transmission_time(task, bandwidth_mbps)
        delay = tx_time + effective_rtt + wait_time + service_time
        energy = self._cloud_energy(task, bandwidth_mbps, effective_rtt)
        return {
            "location": "cloud",
            "delay": delay,
            "energy": energy,
            "wait_time": wait_time,
            "service_time": service_time,
            "queue_backlog": wait_time,
            "transmission_time": tx_time,
            "rtt": effective_rtt,
            "uncertainty": 0.02,
            "risk": 0.01,
        }

    def execute(
        self,
        task: IoTTask,
        queue_backlog: float = 0.0,
        bandwidth_mbps: float = 30.0,
        rtt: float | None = None,
    ) -> ExecutionResult:
        est = self.estimate(
            task,
            queue_backlog=queue_backlog,
            bandwidth_mbps=bandwidth_mbps,
            rtt=rtt,
        )
        return ExecutionResult(
            task_id=task.task_id,
            location="cloud",
            execution_time=est["delay"],
            energy=est["energy"],
        )
