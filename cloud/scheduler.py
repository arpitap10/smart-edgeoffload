class CloudScheduler:

    def schedule(self, task):
        """
        Decide which worker processes the task.
        For now, always send to cloud worker.
        """
        return "cloud_worker"
