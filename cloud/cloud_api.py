import requests
from shared.data_models import IoTTask, ExecutionResult


class CloudAPI:

    def send_to_cloud(self, task: IoTTask) -> ExecutionResult:

        url = "http://13.60.34.204:8000/execute_task"

        # convert task object to dictionary
        task_data = task.__dict__

        response = requests.post(url, json=task_data)

        result = response.json()

        return ExecutionResult(
            status=result.get("status", "completed"),
            location=result.get("location", "cloud"),
            latency=result.get("latency", 0.0)
        )
