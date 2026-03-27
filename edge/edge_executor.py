"""Edge Executor - Handles task execution at edge nodes"""

from shared.data_models import ExecutionResult, IoTTask
from shared.config import Config
import time


class EdgeExecutor:
    """Executes tasks on edge nodes"""

    def __init__(self,
                 cpu_capacity: int = Config.EDGE_CPU_CAPACITY,
                 memory_gb: int = Config.EDGE_MEMORY_GB):
        self.cpu_capacity = cpu_capacity
        self.memory_gb = memory_gb
        self.current_cpu_usage = 0  # percentage (0–100 per core)
        self.tasks_executed = 0
        self.total_execution_time = 0

    def _task_cpu_percent(self, task: IoTTask) -> float:
        """Return CPU % required for a task."""
        return (task.task_size / 10.0) * 100.0

    def can_execute(self, task: IoTTask) -> bool:
        """
        Check if edge can accept the task.
        Also prints reasoning.
        """
        required_cpu = self._task_cpu_percent(task)
        total_capacity = self.cpu_capacity * 100.0

        print(f"[EdgeExecutor] Current CPU={self.current_cpu_usage:.2f}/{total_capacity:.2f}, "
              f"Required={required_cpu:.2f}")

        if (self.current_cpu_usage + required_cpu) > total_capacity:
            print(f"[EdgeExecutor] REJECT → Not enough CPU capacity")
            return False

        print(f"[EdgeExecutor] ACCEPT → Enough capacity")
        return True

    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task on edge"""

        start_time = time.time()

        required_cpu = self._task_cpu_percent(task)

        print(f"[EdgeExecutor] Executing task (size={task.task_size:.2f}MB)")

        # Acquire CPU
        self.current_cpu_usage += required_cpu

        # Simulate execution
        execution_time = 0.01 * task.task_size
        time.sleep(execution_time)

        # Release CPU
        self.current_cpu_usage = max(0.0, self.current_cpu_usage - required_cpu)

        completion_time = time.time() - start_time

        self.tasks_executed += 1
        self.total_execution_time += completion_time

        print(f"[EdgeExecutor] Completed in {completion_time:.4f}s")

        return ExecutionResult(
            status="success",
            location="edge",
            completion_time=completion_time
        )

    def get_utilization(self) -> float:
        """Return utilization (0–1)."""
        total_capacity = self.cpu_capacity * 100.0
        return self.current_cpu_usage / total_capacity