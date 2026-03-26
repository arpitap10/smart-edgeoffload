"""Decision Engine - congestion-aware task offloading policy"""

from shared.data_models import IoTTask
from shared.config import Config


class DecisionEngine:
    """
    Determines whether a task should run on the edge node
    or be offloaded to the cloud.
    """

    def __init__(self,
                 # FIX: default thresholds now come from Config instead of magic numbers
                 queue_threshold: float = Config.QUEUE_LENGTH_THRESHOLD,
                 large_task_threshold: float = Config.LARGE_TASK_THRESHOLD):
        """
        Args:
            queue_threshold       : predicted queue length above which tasks offload
            large_task_threshold  : task size (MB) above which tasks offload
        """
        self.queue_threshold      = queue_threshold
        self.large_task_threshold = large_task_threshold

    def decide_execution(self, predicted_queue: float, task: IoTTask) -> str:
        """
        Decide where a task should execute.

        Returns:
            "edge" or "cloud"
        """
        # Offload if network congestion is predicted
        if predicted_queue >= self.queue_threshold:
            return "cloud"

        # Offload if task is too large for edge
        if task.task_size >= self.large_task_threshold:
            return "cloud"

        return "edge"