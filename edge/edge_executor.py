"""Edge Executor - Handles task execution at edge nodes"""

from shared.data_models import ExecutionResult, IoTTask
import time


class EdgeExecutor:
    """Executes tasks on edge nodes"""
    
    def __init__(self, cpu_capacity: int = 4, memory_gb: int = 8):
        self.cpu_capacity = cpu_capacity
        self.memory_gb = memory_gb
        self.current_cpu_usage = 0
        self.tasks_executed = 0
        self.total_execution_time = 0
    
    def can_execute(self, task: IoTTask) -> bool:
        """Check if edge can execute task"""
        # Simple check: if CPU usage + task size <= capacity
        required_cpu = (task.task_size / 10) * 100  # normalize to CPU %
        return (self.current_cpu_usage + required_cpu) <= self.cpu_capacity * 100
    
    def execute(self, task: IoTTask) -> ExecutionResult:
        """Execute task on edge"""
        start_time = time.time()
        
        # Simulate execution time (proportional to task size)
        execution_time = 0.01 * task.task_size  # seconds
        time.sleep(execution_time)
        
        completion_time = time.time() - start_time
        
        # Update CPU usage (simulate resource consumption)
        required_cpu = (task.task_size / 10) * 100  # normalize to CPU %
        self.current_cpu_usage = min(100, self.current_cpu_usage + required_cpu)
        
        self.tasks_executed += 1
        self.total_execution_time += completion_time
        
        return ExecutionResult(
            status="success",
            location="edge",
            completion_time=completion_time
        )
    
    def get_utilization(self) -> float:
        """Get current edge utilization %"""
        return self.current_cpu_usage / (self.cpu_capacity * 100)