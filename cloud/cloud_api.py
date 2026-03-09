from shared.data_models import IoTTask, ExecutionResult


class CloudAPI:

    def send_to_cloud(self, task: IoTTask) -> ExecutionResult:
        """
        Offload task to cloud.
        """
        raise NotImplementedError