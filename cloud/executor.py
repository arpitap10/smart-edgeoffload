from shared.data_models import ExecutionResult, IoTTask
import random


class CloudExecutor:
    """
    Executes tasks on cloud server.
    Advantage: high compute power.
    Disadvantage: transmission delay (must upload data over network first).
    """

    def __init__(self, compute_power: float = 20.0):
        self.compute_power = compute_power

    def _cloud_energy(self, task: IoTTask) -> float:
        """Transmission energy dominates cloud energy cost at device side."""
        P_tx = 1.0     # watts — WiFi/4G uplink transmit power
        bw   = 10.0    # Mbps  — realistic uplink bandwidth
        tx_time = (task.size * 8) / (bw * 1000)
        return P_tx * tx_time

    def estimate(self, task: IoTTask) -> dict:
        """
        Cloud delay = transmission delay + compute time.
        Transmission uses realistic uplink: 5–15 Mbps with variable SNR.
        This means cloud is SLOWER than edge for small tasks (transmission
        overhead dominates), but may be worth it when edge is congested.
        """
        bandwidth    = random.uniform(5, 15)     # Mbps uplink
        snr          = random.uniform(1, 10)
        transmission = task.size / (bandwidth * (1 + snr / 10))  # seconds
        compute_time = (task.size * task.compute) / self.compute_power
        delay  = transmission + compute_time
        energy = self._cloud_energy(task)
        return {"delay": delay, "energy": energy}

    def execute(self, task: IoTTask) -> ExecutionResult:
        est = self.estimate(task)
        return ExecutionResult(
            task_id        = task.task_id,
            location       = "cloud",
            execution_time = est["delay"],
            energy         = est["energy"],
        )