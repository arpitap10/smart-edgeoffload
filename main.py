"""Smart Edge Offload Framework — main orchestrator"""

from edge.monitoring           import EdgeMonitor
from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine      import DecisionEngine
from edge.edge_executor        import EdgeExecutor
from cloud.executor            import CloudExecutor


class SmartEdgeOffloadFramework:

    def __init__(self, simulator, edge_executor, cloud_executor):
        self.simulator      = simulator
        self.edge_executor  = edge_executor   # EdgeExecutor
        self.cloud_executor = cloud_executor  # CloudExecutor or CloudAPI

        self.monitor         = EdgeMonitor()
        self.predictor       = CongestionPredictor()
        self.decision_engine = DecisionEngine()
        self.task_counter    = 0

    def decide_offloading(self, task) -> str:
        self.task_counter += 1

        print(f"\n{'='*60}")
        print(f"[Task {self.task_counter:04d}] "
              f"id={task.task_id}  size={task.size:.2f}MB  "
              f"compute={task.compute:.2f}  lat_req={task.latency_req:.2f}s")

        # Step 1 — collect edge metrics
        self.monitor.collect_metrics()
        queue_series = self.monitor.queue_series[-10:] or [0]
        print(f"[Monitor]   queue_series (last 10): {queue_series}")

        # Step 2 — predict congestion
        predicted_queue = self.predictor.predict_congestion(queue_series)
        edge_queue  = predicted_queue
        cloud_queue = 0.0   # cloud channel assumed uncongested by default

        # Step 3 — cost-based decision with reason
        edge_est  = self.edge_executor.estimate(task)
        cloud_est = self.cloud_executor.estimate(task)

        decision, reason = self.decision_engine.decide_with_reason(
            task, edge_est, cloud_est, edge_queue, cloud_queue
        )

        print(f"[Decision]  {decision.upper()} — {reason}")
        return decision

    def run_task(self, task):
        decision = self.decide_offloading(task)
        if decision == "edge":
            return self.edge_executor.execute(task)
        else:
            return self.cloud_executor.execute(task)