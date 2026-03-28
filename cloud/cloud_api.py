"""Cloud API — sends tasks to real cloud server, falls back to local simulation"""

import time
import requests
from shared.data_models import IoTTask, ExecutionResult
from cloud.executor import CloudExecutor


class CloudAPI:
    CLOUD_URL = "http://13.53.132.84:8000/execute_task"

    def __init__(self):
        self._fallback = CloudExecutor()

    def estimate(self, task: IoTTask) -> dict:
        """Delegate estimate to fallback (used by decision engine)."""
        return self._fallback.estimate(task)

    def execute(self, task: IoTTask) -> ExecutionResult:
        task_data = {
            "task_id":     task.task_id,
            "size":        task.size,
            "compute":     task.compute,
            "latency_req": task.latency_req,
        }
        print("[CloudAPI] Sending:", task_data)
        start = time.time()
        try:
            resp = requests.post(self.CLOUD_URL, json=task_data, timeout=10)
            resp.raise_for_status()
            result = resp.json()
            rtt    = time.time() - start
            print(f"[CloudAPI] Response: {result} | RTT: {rtt:.3f}s")
            return ExecutionResult(
                task_id        = task.task_id,
                location       = "cloud",
                execution_time = rtt,
                energy         = self._fallback._cloud_energy(task),
            )
        except Exception as e:
            print(f"[CloudAPI] WARNING: {e} — using local simulation.")
            return self._fallback.execute(task)