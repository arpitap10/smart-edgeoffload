from shared.data_models import IoTTask, ExecutionResult


def process_task(task: IoTTask) -> ExecutionResult:
    """
    Process task in cloud worker.
    """
    raise NotImplementedError