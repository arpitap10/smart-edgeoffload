from shared.data_models import ExecutionResult, IoTTask
import random

class CloudExecutor:
    def __init__(self, compute_power=20):
        self.compute_power = compute_power

    def estimate(self, task: IoTTask):
        # network conditions
        bandwidth = random.uniform(5, 20)   # Mbps
        snr = random.uniform(1, 10)

        transmission = task.size / (bandwidth * (1 + snr))
        compute_time = (task.size * task.compute) / self.compute_power

        delay = transmission + compute_time

        # energy (transmission)
        energy = 0.5 * transmission

        return {
            "delay": delay,
            "energy": energy
        }

    def execute(self, task: IoTTask) -> ExecutionResult:
        est = self.estimate(task)

        return ExecutionResult(
            task_id=task.task_id,
            location="cloud",
            execution_time=est["delay"],
            energy=est["energy"]
        )