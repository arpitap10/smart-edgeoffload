import time
from shared.data_models import IoTTask, ExecutionResult


class EdgeExecutor:
    """
    Simulates execution of tasks on an edge node.
    """

    def __init__(self, processing_factor: float = 0.05):
        """
        Args:
            processing_factor : multiplier controlling how long
                                edge processing takes relative to task size
        """
        self.processing_factor = processing_factor

    def execute_task(self, task: IoTTask) -> ExecutionResult:
        """
        Execute a task on the edge node.

        Args:
            task : IoTTask

        Returns:
            ExecutionResult
        """

        # Simulated processing time
        processing_time = task.task_size * self.processing_factor
        time.sleep(processing_time)

        result = ExecutionResult(
            status="completed",
            location="edge",
            completion_time=time.time()
        )

        return result