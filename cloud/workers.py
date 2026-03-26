import time
from shared.data_models import IoTTask, ExecutionResult


def process_task(task: IoTTask) -> ExecutionResult:
    """
    Process task in cloud worker.
    """

    start = time.time()

    # simulate heavy computation
    time.sleep(0.3)

    latency = time.time() - start

    return ExecutionResult(
        status="completed",
        location="cloud",
        completion_time=latency
    )
