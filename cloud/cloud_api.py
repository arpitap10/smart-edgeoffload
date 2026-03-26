import requests
from shared.data_models import IoTTask, ExecutionResult


class CloudAPI:

    def send_to_cloud(self, task: IoTTask) -> ExecutionResult:

        url = "http://13.53.132.84:8000/execute_task"

        # convert task object to dictionary
        task_data = task.__dict__

        print("[CloudAPI] send_to_cloud payload:", task_data)

        response = requests.post(url, json=task_data, timeout=10)
        response.raise_for_status()

        result = response.json()

        print("[CloudAPI] cloud response:", result)

        return ExecutionResult(
            status=result.get("status", "completed"),
            location=result.get("location", "cloud"),
            completion_time=result.get("latency", 0.0)
        )
