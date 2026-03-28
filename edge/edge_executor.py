from shared.data_models import ExecutionResult, IoTTask


class EdgeExecutor:
    """
    Executes tasks locally on the edge node.
    Advantage over cloud: ZERO transmission delay.
    Disadvantage: lower compute power than cloud data centre.
    """

    def __init__(self, compute_power: float = 20.0):
        # Same raw compute speed as cloud — edge advantage is no network RTT
        self.compute_power = compute_power

    def estimate(self, task: IoTTask) -> dict:
        """
        delay  = (size × compute) / compute_power   — pure local CPU time
        energy = P_cpu × delay, but P_cpu is tiny for embedded hardware (~0.5W)
                 compared to WiFi transmission power, so energy stays low.
        """
        delay  = (task.size * task.compute) / self.compute_power
        energy = 0.5 * delay   # 0.5 W local CPU (Raspberry Pi class)
        return {"delay": delay, "energy": energy}

    def execute(self, task: IoTTask) -> ExecutionResult:
        est = self.estimate(task)
        return ExecutionResult(
            task_id        = task.task_id,
            location       = "edge",
            execution_time = est["delay"],
            energy         = est["energy"],
        )