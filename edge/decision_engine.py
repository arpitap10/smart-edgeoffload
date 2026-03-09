from shared.data_models import IoTTask


class DecisionEngine:
    """
    Decision engine for congestion-aware task offloading.
    Determines whether a task should run on the edge node
    or be offloaded to the cloud.
    """

    def __init__(self,
                 queue_threshold: float = 15,
                 large_task_threshold: float = 8):
        """
        Args:
            queue_threshold : predicted queue length above which tasks
                              should be offloaded to cloud
            large_task_threshold : task size above which tasks should
                                   be offloaded
        """
        self.queue_threshold = queue_threshold
        self.large_task_threshold = large_task_threshold

    def decide_execution(self, predicted_queue: float, task: IoTTask) -> str:
        """
        Decide where a task should execute.

        Args:
            predicted_queue : predicted queue length from congestion predictor
            task : IoTTask object

        Returns:
            "edge" or "cloud"
        """

        # Offload if congestion is predicted
        if predicted_queue >= self.queue_threshold:
            return "cloud"

        # Offload if task is large
        if task.task_size >= self.large_task_threshold:
            return "cloud"

        # Otherwise run locally
        return "edge"