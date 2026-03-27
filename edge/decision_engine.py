"""Decision Engine - congestion-aware task offloading policy"""

from shared.data_models import IoTTask
from shared.config import Config


class DecisionEngine:
    """
    Determines whether a task should run on the edge node
    or be offloaded to the cloud.
    """

    def __init__(self,
                 queue_threshold: float = Config.QUEUE_LENGTH_THRESHOLD,
                 large_task_threshold: float = Config.LARGE_TASK_THRESHOLD):

        self.queue_threshold = queue_threshold
        self.large_task_threshold = large_task_threshold

    def decide_execution(self, predicted_queue: float, task: IoTTask):
        """
        Decide where a task should execute.

        Returns:
            (decision, reason)
        """

        print(f"[Decision] Predicted queue={predicted_queue:.2f}, "
              f"Task size={task.task_size:.2f}MB")

        # Rule 1: Congestion-based offloading
        if predicted_queue >= self.queue_threshold:
            reason = f"Predicted queue {predicted_queue:.2f} >= threshold {self.queue_threshold}"
            print(f"[Decision] CLOUD → {reason}")
            return "cloud", reason

        # Rule 2: Large task offloading
        if task.task_size >= self.large_task_threshold:
            reason = f"Task size {task.task_size:.2f}MB >= threshold {self.large_task_threshold}MB"
            print(f"[Decision] CLOUD → {reason}")
            return "cloud", reason

        # Default: Edge execution
        reason = "Low predicted congestion and small task"
        print(f"[Decision] EDGE → {reason}")
        return "edge", reason