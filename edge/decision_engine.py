class DecisionEngine:

    def decide_execution(self, predicted_queue: float, task_size: float) -> str:
        """
        Decide whether to execute on edge or cloud.

        Returns:
            "edge" or "cloud"
        """
        raise NotImplementedError