"""Smart Edge Offload Framework - main orchestrator"""

from edge.monitoring import EdgeMonitor
from edge.congestion_predictor import CongestionPredictor
from edge.decision_engine import DecisionEngine


class SmartEdgeOffloadFramework:

    def __init__(self, simulator, edge_executor, cloud_executor):
        self.simulator      = simulator
        self.edge_executor  = edge_executor
        self.cloud_executor = cloud_executor

        self.monitor         = EdgeMonitor()
        self.predictor       = CongestionPredictor()
        self.decision_engine = DecisionEngine()

        # ✅ Task counter (NEW)
        self.task_counter = 0

    def decide_offloading(self, task):
        # ✅ Increment task number
        self.task_counter += 1

        print("\n==============================")
        print(f"[Task {self.task_counter:02d}] "
              f"device={task.device_id} size={task.task_size:.2f}MB")

        # Step 1: Collect metrics
        self.monitor.collect_metrics()

        queue_series = self.monitor.queue_series[-10:]
        if not queue_series:
            queue_series = [0]

        print(f"[Monitor] Recent queue={queue_series}")

        # Step 2: Predict congestion
        predicted_queue = self.predictor.predict_congestion(queue_series)
        print(f"[Predictor] Predicted queue={predicted_queue:.2f}")

        # Step 3: Decision engine
        decision, reason = self.decision_engine.decide_execution(
            predicted_queue, task
        )

        # Step 4: Edge capacity override
        if decision == "edge":
            if not self.edge_executor.can_execute(task):
                print("[Override] Edge cannot execute → switching to CLOUD")
                final_decision = "cloud"
                final_reason = "Edge capacity exceeded"
            else:
                final_decision = "edge"
                final_reason = reason
        else:
            final_decision = "cloud"
            final_reason = reason

        print(f"[Final Decision] {final_decision.upper()} → {final_reason}")

        return final_decision

    def run_task(self, task):
        decision = self.decide_offloading(task)

        if decision == "edge":
            return self.edge_executor.execute(task)
        else:
            return self.cloud_executor.execute(task)