import time

from shared.data_models import ExecutionResult, IoTTask


def process_task(task: IoTTask) -> ExecutionResult:
    """Minimal cloud worker helper used by legacy entry points."""
    start = time.time()
    time.sleep(0.05)
    latency = time.time() - start
    return ExecutionResult(
        task_id=task.task_id,
        location="cloud",
        execution_time=latency,
        energy=0.0,
    )
