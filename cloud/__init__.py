"""Cloud API wrapper with a simulation-first fallback path."""

import time

import requests

from cloud.executor import CloudExecutor
from shared.data_models import ExecutionResult, IoTTask


class CloudAPI:
    CLOUD_URL = "http://98.70.25.204:8000/execute_task"

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

        # Locally-tracked congestion state (this client's simulated edge/cloud
        # queue backlog). The remote server has no visibility into this - it
        # only ever sees one task at a time - so it must be added back on top
        # of whatever the server/network round-trip measures. Dropping this
        # (as the previous version did) silently bypassed the congestion
        # model for every real-cloud-routed task, making "predictive" vs.
        # "reactive" etc. no longer a fair like-for-like comparison once
        # USE_REAL_CLOUD=True.
        wait_time = max(0.0, queue_backlog)

        start = time.time()
        try:
            resp = requests.post(self.CLOUD_URL, json=task_data, timeout=10)
            resp.raise_for_status()
            # rtt_observed is the real end-to-end transmission + propagation +
            # server-side compute time for this task - i.e. it already stands
            # in for Eq. 3's (transmission_time + rtt + service_time) using a
            # genuine network measurement instead of the simulated formula.
            rtt_observed = time.time() - start
            execution_time = wait_time + rtt_observed
            return ExecutionResult(
                task_id=task.task_id,
                location="cloud",
                execution_time=execution_time,
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