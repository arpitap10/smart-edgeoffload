"""Cloud Executor - Handles task execution in cloud"""

from shared.data_models import ExecutionResult, IoTTask
import time


class CloudExecutor:
    """Executes tasks in cloud (unlimited capacity)"""
    
    def __init__(self):
        self.tasks_executed = 0
        self.total_execution_time = 0
    
    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task in cloud"""
        start_time = time.time()
        
        # Simulate cloud execution (faster but with network latency)
        execution_time = 0.005 * task.task_size  # seconds (faster)
        time.sleep(execution_time)
        
        completion_time = time.time() - start_time
        self.tasks_executed += 1
        self.total_execution_time += completion_time
        
        return ExecutionResult(
            status="success",
            location="cloud",
            completion_time=completion_time
        )