from shared.data_models import ExecutionResult, IoTTask


class EdgeExecutor:
    """
    Local edge executor.

    The queue state is expressed as backlog in seconds, which makes it easy to
    forecast and directly comparable to application latency constraints.
    """

    def __init__(self, compute_power: float = 240.0, active_power_w: float = 1.05):
        self.compute_power = compute_power
        self.active_power_w = active_power_w

    def service_time(self, task: IoTTask) -> float:
        return task.compute_demand / self.compute_power

    def estimate(self, task: IoTTask, queue_backlog: float = 0.0) -> dict:
        service_time = self.service_time(task)
        wait_time = max(0.0, queue_backlog)
        delay = wait_time + service_time
        energy = self.active_power_w * service_time
        return {
            "location": "edge",
            "delay": delay,
            "energy": energy,
            "wait_time": wait_time,
            "service_time": service_time,
            "queue_backlog": wait_time,
            "transmission_time": 0.0,
            "rtt": 0.0,
            "uncertainty": 0.0,
            "risk": 0.0,
        }

    def execute(self, task: IoTTask, queue_backlog: float = 0.0) -> ExecutionResult:
        est = self.estimate(task, queue_backlog=queue_backlog)
        return ExecutionResult(
            task_id=task.task_id,
            location="edge",
            execution_time=est["delay"],
            energy=est["energy"],
        )
