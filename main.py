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

    def decide_offloading(self, task):
        # Collect current metrics (also appends to queue_series)
        self.monitor.collect_metrics()

        # FIX: guard against empty series on very first call;
        # predictor needs at least 1 point, handles <5 internally
        queue_series = self.monitor.queue_series[-10:]
        if not queue_series:
            queue_series = [0]

        predicted_queue = self.predictor.predict_congestion(queue_series)

        # Hard capacity check first — if edge is full, always go to cloud
        if not self.edge_executor.can_execute(task):
            return "cloud"

        return self.decision_engine.decide_execution(predicted_queue, task)

    def run_task(self, task):
        decision = self.decide_offloading(task)
        if decision == "edge":
            return self.edge_executor.execute(task)
        else:
            return self.cloud_executor.execute(task)