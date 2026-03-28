"""IoT Device Simulator — generates IoTTask objects matching data_models.py"""

from shared.data_models import IoTTask
import random


class IoTSimulator:
    """Simulates IoT devices generating tasks"""

    def __init__(self, num_devices: int = 20):
        self.devices         = [f"sensor_{i}" for i in range(1, num_devices + 1)]
        self.task_id_counter = 0

    def generate_task(self) -> IoTTask:
        self.task_id_counter += 1
        # congestion effect: large tasks get tighter deadlines
        size        = round(random.uniform(1.0, 10.0), 2)
        compute     = round(random.uniform(1.0,  5.0), 2)
        latency_req = round(random.uniform(0.5,  5.0), 2)

        return IoTTask(
            task_id     = self.task_id_counter,
            size        = size,
            compute     = compute,
            latency_req = latency_req,
        )

    def generate_tasks_batch(self, count: int) -> list:
        return [self.generate_task() for _ in range(count)]