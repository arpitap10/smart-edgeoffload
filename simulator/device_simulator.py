"""Workload-driven IoT task generator used by the simulator and experiments."""

import random

from shared.data_models import IoTTask


class IoTSimulator:
    """Generates mixed IoT workloads with realistic deadline classes."""

    def __init__(self, num_devices: int = 20, seed: int | None = None):
        self.devices = [f"sensor_{i}" for i in range(1, num_devices + 1)]
        self.task_id_counter = 0
        self.rng = random.Random(seed)

    def _sample_profile(self) -> tuple[float, float, float]:
        profile = self.rng.choices(
            population=["vision", "telemetry", "analytics"],
            weights=[0.35, 0.40, 0.25],
            k=1,
        )[0]

        if profile == "vision":
            size = self.rng.uniform(3.0, 8.0)
            compute = self.rng.uniform(18.0, 34.0)
            latency_req = self.rng.uniform(1.2, 2.6)
        elif profile == "telemetry":
            size = self.rng.uniform(0.2, 1.1)
            compute = self.rng.uniform(2.0, 8.0)
            latency_req = self.rng.uniform(0.2, 0.9)
        else:
            size = self.rng.uniform(1.0, 4.0)
            compute = self.rng.uniform(10.0, 22.0)
            latency_req = self.rng.uniform(1.3, 3.4)

        return round(size, 2), round(compute, 2), round(latency_req, 2)

    def generate_task(self) -> IoTTask:
        self.task_id_counter += 1
        size, compute, latency_req = self._sample_profile()
        return IoTTask(
            task_id=self.task_id_counter,
            size=size,
            compute=compute,
            latency_req=latency_req,
        )

    def generate_tasks_batch(self, count: int) -> list[IoTTask]:
        return [self.generate_task() for _ in range(count)]
