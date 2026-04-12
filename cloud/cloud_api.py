"""Cloud API wrapper with a simulation-first fallback path."""

import time

import requests

from cloud.executor import CloudExecutor
from shared.data_models import ExecutionResult, IoTTask


class CloudAPI:
    CLOUD_URL = "http://13.53.132.84:8000/execute_task"

    def __init__(self, use_remote: bool = True):
        self.use_remote = use_remote
        self._fallback = CloudExecutor()

    def estimate(
        self,
        task: IoTTask,
        queue_backlog: float = 0.0,
        bandwidth_mbps: float = 30.0,
        rtt: float | None = None,
    ) -> dict:
        return self._fallback.estimate(
            task,
            queue_backlog=queue_backlog,
            bandwidth_mbps=bandwidth_mbps,
            rtt=rtt,
        )

    def execute(
        self,
        task: IoTTask,
        queue_backlog: float = 0.0,
        bandwidth_mbps: float = 30.0,
        rtt: float | None = None,
    ) -> ExecutionResult:
        if not self.use_remote:
            return self._fallback.execute(
                task,
                queue_backlog=queue_backlog,
                bandwidth_mbps=bandwidth_mbps,
                rtt=rtt,
            )

        task_data = {
            "task_id": task.task_id,
            "size": task.size,
            "compute": task.compute,
            "latency_req": task.latency_req,
        }

        start = time.time()
        try:
            resp = requests.post(self.CLOUD_URL, json=task_data, timeout=10)
            resp.raise_for_status()
            rtt_observed = time.time() - start
            return ExecutionResult(
                task_id=task.task_id,
                location="cloud",
                execution_time=rtt_observed,
                energy=self._fallback._cloud_energy(
                    task,
                    bandwidth_mbps=bandwidth_mbps,
                    rtt=rtt or self._fallback.base_rtt,
                ),
            )
        except Exception:
            return self._fallback.execute(
                task,
                queue_backlog=queue_backlog,
                bandwidth_mbps=bandwidth_mbps,
                rtt=rtt,
            )
