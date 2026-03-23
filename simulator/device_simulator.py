from shared.data_models import IOTTask
import random
import time


class IoTSimulator:
    """
    Simulates IoT devices generating tasks.
    """

    def __init__(self):
        # simulate multiple IoT devices
        self.devices = [f"sensor_{i}" for i in range(1, 21)]

    def generate_task(self) -> IOTTask:
        """
        Generate a single IoT task.
        """

        device_id = random.choice(self.devices)
        timestamp = time.time()
        task_size = round(random.uniform(1.0, 10.0), 2)

        queue_length = random.randint(1, 50)
        latency = round(random.uniform(20, 100), 2)

        # simulate congestion effect
        if queue_length > 35:
            latency += random.uniform(10, 30)

        payload = {
            "latency_ms": latency,
            "packet_loss": round(random.uniform(0.0, 0.05), 4),
            "queue_length": queue_length,
            "cpu_usage": round(random.uniform(10, 90), 2)
        }

        return IOTTask(
            device_id=device_id,
            timestamp=timestamp,
            task_size=task_size,
            payload=payload
        )
    