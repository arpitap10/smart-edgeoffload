from shared.data_models import ExecutionResult, IoTTask

class EdgeExecutor:
    def __init__(self, compute_power=5):
        self.compute_power = compute_power

    def estimate(self, task: IoTTask):
        # delay model
        delay = (task.size * task.compute) / self.compute_power

        # energy model (CPU)
        energy = 2.0 * delay   # P_cpu * time

        return {
            "delay": delay,
            "energy": energy
        }

    def execute(self, task: IoTTask) -> ExecutionResult:
        est = self.estimate(task)

        return ExecutionResult(
            task_id=task.task_id,
            location="edge",
            execution_time=est["delay"],
            energy=est["energy"]
        )