"""Cloud API - sends tasks to the remote cloud server with graceful fallback"""

import time
import requests
from shared.data_models import IoTTask, ExecutionResult
from cloud.executor import CloudExecutor


class CloudAPI:
    """
    Sends tasks to the remote cloud server.
    Falls back to local CloudExecutor simulation if the server is unreachable,
    so the pipeline never crashes due to network issues.
    """

    CLOUD_URL = "http://13.53.132.84:8000/execute_task"

    def __init__(self):
        # FIX: fallback executor used when the real server is down
        self._fallback = CloudExecutor()

    def send_to_cloud(self, task: IoTTask) -> ExecutionResult:
        task_data = task.__dict__
        print("[CloudAPI] Sending payload:", task_data)

        # FIX: measure wall-clock time locally so RTT is always captured,
        # regardless of what the server returns in its "latency" field
        start_time = time.time()

        try:
            response = requests.post(self.CLOUD_URL, json=task_data, timeout=10)
            response.raise_for_status()
            result = response.json()

            completion_time = time.time() - start_time
            print(f"[CloudAPI] Response: {result} | RTT: {completion_time:.3f}s")

            return ExecutionResult(
                status=result.get("status", "completed"),
                location="cloud",
                completion_time=completion_time  # was result.get("latency", 0.0)
            )

        # FIX: catch network/HTTP errors and fall back to local simulation
        except requests.exceptions.ConnectionError:
            print("[CloudAPI] WARNING: Cannot reach cloud server — using local simulation.")
            return self._fallback.execute(task)
        except requests.exceptions.Timeout:
            print("[CloudAPI] WARNING: Cloud server timed out — using local simulation.")
            return self._fallback.execute(task)
        except requests.exceptions.HTTPError as e:
            print(f"[CloudAPI] WARNING: HTTP error {e} — using local simulation.")
            return self._fallback.execute(task)

    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task on cloud server (or fallback)."""
        return self.send_to_cloud(task)