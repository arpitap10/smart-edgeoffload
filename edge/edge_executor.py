from shared.data_models import IoTTask, ExecutionResult


class EdgeExecutor:

    def execute_task(self, task: IoTTask) -> ExecutionResult:
        """
        Execute task on edge node.
        """
        raise NotImplementedError