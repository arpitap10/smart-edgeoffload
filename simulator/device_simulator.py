"""IoT Device Simulator for generating tasks"""

from shared.data_models import IoTTask
import random
import time


class IoTSimulator:
    """Simulates IoT devices generating tasks"""

    def __init__(self, num_devices: int = 20):
        self.devices = [f"sensor_{i}" for i in range(1, num_devices + 1)]
        self.task_id_counter = 0

    def generate_task(self) -> IoTTask:
        """Generate a single IoT task with network metrics"""
        self.task_id_counter += 1
        device_id = random.choice(self.devices)
        timestamp = time.time()
        task_size = round(random.uniform(1.0, 10.0), 2)  # MB
        
        # Simulate network conditions
        queue_length = random.randint(1, 50)
        latency = round(random.uniform(20, 100), 2)
        
        # Congestion effect
        if queue_length > 35:
            latency += random.uniform(10, 30)
        
        payload = {
            "task_id": self.task_id_counter,
            "latency_ms": latency,
            "packet_loss": round(random.uniform(0.0, 0.05), 4),
            "queue_length": queue_length,
            "cpu_usage": round(random.uniform(10, 90), 2)
        }
        
        return IoTTask(
            device_id=device_id,
            timestamp=timestamp,
            task_size=task_size,
            payload=payload
        )

    def generate_tasks_batch(self, count: int) -> list:
        """Generate multiple tasks"""
        return [self.generate_task() for _ in range(count)]